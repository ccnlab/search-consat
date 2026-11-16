// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package csenv

import (
	"fmt"
	"slices"
	"testing"

	"cogentcore.org/lab/tensor"
)

// TestSpaceSizes reports the space sizes for different settings.
func TestSpaceSizes(t *testing.T) {
	for _, c := range []struct{ ns, n, G, g, m int }{
		{3, 24, 8, 2, 1}, {3, 24, 8, 2, 2}, {3, 24, 8, 3, 1}, {3, 24, 8, 3, 2},
		{3, 32, 8, 3, 1}, {3, 32, 8, 3, 2}, {3, 32, 12, 3, 1}, {3, 32, 12, 3, 2}, {4, 32, 8, 3, 1}, {4, 32, 8, 3, 2}, {4, 32, 16, 8, 2},
	} {
		var cfg Config
		cfg.Defaults()
		cfg.NSlots, cfg.NInputs, cfg.NGroups, cfg.GroupsPer, cfg.NValid = c.ns, c.n, c.G, c.g, c.m
		sp := cfg.makeItemSpace()
		t.Logf("slots %d n %d G %d g %d m %d: items %d train %d test %d coverage %.3g", c.ns, c.n, c.G, c.g, c.m, len(sp.Items), len(sp.Train), len(sp.Test), sp.Coverage)
	}
}

func TestSpace(t *testing.T) {
	for _, m := range []int{1, 2} {
		var cfg Config
		cfg.Defaults()
		cfg.NValid = m
		sp := cfg.MakeSpace()
		t.Log("\n" + cfg.SpaceString(sp))
		if len(sp.Items) == 0 {
			t.Fatalf("m=%d: no items", m)
		}
		seen := map[string]bool{}
		for _, it := range sp.Items {
			k := fmt.Sprint(it.Inputs)
			if seen[k] {
				t.Fatalf("duplicate item: %v", it.Inputs)
			}
			seen[k] = true
			if len(it.Valid) != m || !slices.Equal(it.Valid, sp.ValidGroups(it.Inputs)) {
				t.Fatalf("bad valid groups: %s", it.String())
			}
		}
		if len(sp.Train)+len(sp.Test) != len(sp.Items) {
			t.Errorf("train + test != items")
		}
		all := append(slices.Clone(sp.Train), sp.Test...)
		slices.Sort(all)
		if len(slices.Compact(all)) != len(sp.Items) {
			t.Errorf("train and test overlap")
		}
		for o := range cfg.NOutputs() {
			in, g, mem := sp.OutputSource(&cfg, o)
			if sp.OutputIndex(&cfg, in, sp.GroupIndex(in, g), mem) != cfg.NInputs+o {
				t.Fatalf("OutputSource / OutputIndex mismatch for %d", o)
			}
		}
	}
}

func TestVocab(t *testing.T) {
	var cfg Config
	cfg.Defaults()
	sp := cfg.MakeSpace()
	non := cfg.NOn()
	maxOv := non - cfg.MinDiff()
	nv := cfg.NVocab()
	worst := 0
	for i := range nv {
		pv := sp.Vocab.SubSpace(i).(*tensor.Float32).Values
		if n := countOn(pv); n != non {
			t.Fatalf("pattern %d: %d on, want %d", i, n, non)
		}
		for j := i + 1; j < nv; j++ {
			ov := 0
			for k, v := range sp.Vocab.SubSpace(j).(*tensor.Float32).Values {
				if v > 0 && pv[k] > 0 {
					ov++
				}
			}
			worst = max(worst, ov)
		}
	}
	t.Logf("%d patterns, %d on, max overlap %d (allowed %d)", nv, non, worst, maxOv)
	if worst > maxOv {
		t.Errorf("max overlap %d > %d", worst, maxOv)
	}
}

// slotChoice returns a synthetic output that produces the given
// output pattern in each slot.
func slotChoice(ev *CSEnv, outs []int) []float32 {
	sp := ev.Space()
	nu := ev.Configs.PoolUnits()
	out := make([]float32, len(outs)*nu)
	for s, o := range outs {
		copy(out[s*nu:], sp.Vocab.SubSpace(o).(*tensor.Float32).Values)
	}
	return out
}

func newEnv(cfg *Config, test bool) *CSEnv {
	ev := &CSEnv{Configs: *cfg}
	ev.Config(100, 1, 0, test, 0)
	ev.Init(0)
	return ev
}

// TestStats checks the stats on the target outputs, and reports baselines
// for choosing a group independently in each slot (no coordination).
func TestStats(t *testing.T) {
	for _, m := range []int{1, 2} {
		var cfg Config
		cfg.Defaults()
		cfg.NValid = m
		ev := newEnv(&cfg, true)
		sp := ev.Space()
		var ts TrialStats
		n := 2000
		var indep, cohRand [5]float64
		for range n {
			ev.Step()
			ev.TrialStats(slotChoice(ev, ev.Outputs), &ts)
			if ts.Solved != 1 || ts.Miss != 0 || ts.Valid != 1 {
				t.Fatalf("target not solved: %s: %v", ev.String(), ts.Values())
			}
			it := ev.CurItem()
			// independent: each slot picks a random valid group
			outs := make([]int, cfg.NSlots)
			for s, in := range it.Inputs {
				g := it.Valid[ev.Rand.Intn(len(it.Valid))]
				outs[s] = sp.OutputIndex(&cfg, in, sp.GroupIndex(in, g), 0)
			}
			ev.TrialStats(slotChoice(ev, outs), &ts)
			for i, v := range ts.Values() {
				indep[i] += v / float64(n)
			}
			// random: each slot picks a random one of its input's groups
			for s, in := range it.Inputs {
				outs[s] = sp.OutputIndex(&cfg, in, ev.Rand.Intn(cfg.GroupsPer), 0)
			}
			ev.TrialStats(slotChoice(ev, outs), &ts)
			for i, v := range ts.Values() {
				cohRand[i] += v / float64(n)
			}
		}
		t.Logf("m=%d %v", m, TrialStatNames)
		t.Logf("  random group per slot:      %.3f", cohRand)
		t.Logf("  random valid group per slot: %.3f", indep)
	}
}

func countOn(pv []float32) int {
	n := 0
	for _, v := range pv {
		if v > 0 {
			n++
		}
	}
	return n
}
