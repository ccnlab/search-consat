// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package otmenv

import (
	"testing"

	"cogentcore.org/lab/tensor"
)

func countOn(pv []float32) int {
	n := 0
	for _, v := range pv {
		if v > 0 {
			n++
		}
	}
	return n
}

func TestVocab(t *testing.T) {
	for _, no := range []int{2, 3} {
		var cfg Config
		cfg.Defaults()
		cfg.NOut = no
		vc := cfg.MakeVocab()
		non := cfg.NOn()
		maxOv := non - cfg.MinDiff()
		nv := cfg.NVocab()
		worst := 0
		for i := range nv {
			pv := vc.SubSpace(i).(*tensor.Float32).Values
			if n := countOn(pv); n != non {
				t.Fatalf("pattern %d: %d on, want %d", i, n, non)
			}
			for j := i + 1; j < nv; j++ {
				ov := 0
				for k, v := range vc.SubSpace(j).(*tensor.Float32).Values {
					if v > 0 && pv[k] > 0 {
						ov++
					}
				}
				worst = max(worst, ov)
			}
		}
		t.Logf("o=%d: %d patterns, %d on, max overlap %d (allowed %d)", no, nv, non, worst, maxOv)
		if worst > maxOv {
			t.Errorf("max overlap %d > %d", worst, maxOv)
		}
	}
}

// TestStats checks the stats on the target outputs, the other outputs of
// the input, an output of another input, and an equal blend of two
// outputs of the input.
func TestStats(t *testing.T) {
	var cfg Config
	cfg.Defaults()
	ev := &OTMEnv{Configs: cfg}
	ev.Config(100, 0)
	ev.Init(0)
	vc := ev.Vocab()
	pat := func(vi int) []float32 { return vc.SubSpace(vi).(*tensor.Float32).Values }
	var ts TrialStats
	for range 200 {
		ev.Step()
		ev.TrialStats(ev.States["Output"].Values, &ts)
		if ts.Valid != 1 || ts.Out != ev.Out || ts.Margin < 0.49 {
			t.Fatalf("target not valid: %s: %v %d", ev.String(), ts.Values(), ts.Out)
		}
		other := (ev.Out + 1) % cfg.NOut
		ev.TrialStats(pat(cfg.OutputIndex(ev.Input, other)), &ts)
		if ts.Valid != 1 || ts.Out != other {
			t.Fatalf("other output not valid: %s: %v", ev.String(), ts.Values())
		}
		ev.TrialStats(pat(cfg.OutputIndex((ev.Input+1)%cfg.NInputs, 0)), &ts)
		if ts.Valid != 0 {
			t.Fatalf("output of another input is valid: %s: %v", ev.String(), ts.Values())
		}
		b := make([]float32, cfg.PoolUnits())
		p0, p1 := pat(cfg.OutputIndex(ev.Input, 0)), pat(cfg.OutputIndex(ev.Input, 1))
		for k := range b {
			b[k] = 0.5 * (p0[k] + p1[k])
		}
		ev.TrialStats(b, &ts)
		if ts.Valid != 0 || ts.Margin > 0.01 {
			t.Fatalf("blend is valid: %s: %v", ev.String(), ts.Values())
		}
	}
}
