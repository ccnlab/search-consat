// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcombenv

import (
	"fmt"
	"math"
	"testing"

	"cogentcore.org/core/math32"
)

// TestSpaceSizes reports the item space sizes for different settings.
func TestSpaceSizes(t *testing.T) {
	var cfg Config
	cfg.Defaults()
	t.Log("\n" + cfg.SpaceString())
	for _, na := range [][2]int{{2, 3}, {2, 4}, {2, 5}, {2, 6}, {2, 8}} {
		cfg.NActive.Set(na[0], na[1])
		tot := int64(0)
		for n := na[0]; n <= na[1]; n++ {
			tot += cfg.NCombos(n)
		}
		t.Logf("NActive %d-%d: %d items", na[0], na[1], tot)
	}
}

func TestSpace(t *testing.T) {
	var cfg Config
	cfg.Defaults()
	sp := cfg.MakeSpace()
	want := int64(0)
	for n := cfg.NActive.Min; n <= cfg.NActive.Max; n++ {
		want += cfg.NCombos(n)
	}
	if int64(len(sp.Items)) != want {
		t.Fatalf("items: %d, want %d", len(sp.Items), want)
	}
	seen := map[string]bool{}
	for _, it := range sp.Items {
		s := fmt.Sprint(it.Cues)
		if seen[s] {
			t.Fatalf("duplicate item: %s", s)
		}
		seen[s] = true
	}
	inTest := map[int]bool{}
	for k := range sp.Test {
		n := cfg.NActive.Min + k
		nt := len(sp.Train[k]) + len(sp.Test[k])
		if int64(nt) != cfg.NDistinct(n) {
			t.Errorf("n=%d: train+test = %d, want distinct %d", n, nt, cfg.NDistinct(n))
		}
		if int64(nt+len(sp.Agree[k])) != cfg.NCombos(n) {
			t.Errorf("n=%d: train+test+agree = %d", n, nt+len(sp.Agree[k]))
		}
		for _, i := range sp.Test[k] {
			inTest[i] = true
		}
		for _, i := range sp.Agree[k] {
			if !sp.Items[i].HasRepeat() {
				t.Fatalf("agree item %d has no repeated values", i)
			}
		}
	}
	for _, trn := range sp.Train {
		for _, i := range trn {
			if inTest[i] {
				t.Fatalf("item %d in both train and test", i)
			}
			if sp.Items[i].HasRepeat() {
				t.Fatalf("train item %d has repeated values", i)
			}
		}
	}
}

// TestBaselines reports the expected stats for reference strategies,
// computed with [PCombEnv.TrialStats] on synthetic output patterns:
// for the test env (distinct items for ChoseP, PGain, OutPErr, and
// agreement items for AgreeBoost).
func TestBaselines(t *testing.T) {
	ev := &PCombEnv{}
	ev.Defaults()
	ev.Config(100, 1, 0, true, 1)
	ev.Init(0)
	cfg := &ev.Configs
	npu := cfg.PopCodeUnits
	type strat struct {
		name string
		// choose returns index of chosen value, or -1 for the probability
		// weighted blend (outputs the blend instead of one bump).
		choose func(probs []float32, counts []int) int
		blend  func(probs []float32, counts []int) []float32
	}
	uniform := func(probs []float32, counts []int) []float32 {
		w := make([]float32, len(probs))
		for j := range w {
			w[j] = 1 / float32(len(w))
		}
		return w
	}
	strats := []strat{
		{name: "uniform choice", choose: func(probs []float32, counts []int) int { return ev.Rand.Intn(len(probs)) }},
		{name: "calibrated sampling", choose: func(probs []float32, counts []int) int {
			r := ev.Rand.Float32()
			for j, p := range probs {
				if r < p {
					return j
				}
				r -= p
			}
			return len(probs) - 1
		}},
		{name: "most probable", choose: func(probs []float32, counts []int) int {
			mi := 0
			for j, p := range probs {
				if p > probs[mi] {
					mi = j
				}
			}
			return mi
		}},
		{name: "most repeated (no probs)", choose: func(probs []float32, counts []int) int {
			mi := 0
			for j, c := range counts {
				if c > counts[mi] {
					mi = j
				}
			}
			return mi
		}},
		{name: "uniform blend (no probs)", blend: uniform},
		{name: "probability blend", blend: func(probs []float32, counts []int) []float32 { return probs }},
	}
	out := make([]float32, npu)
	sal := make([]float32, npu)
	var ts TrialStats
	n := 20000
	for _, st := range strats {
		sums := make([]float64, len(TrialStatNames))
		cnts := make([]float64, len(TrialStatNames))
		for range n {
			ev.Step()
			vals, probs, counts := ev.ValueCounts()
			clear(out)
			if st.choose != nil {
				cfg.ValuePop.Encode(&out, vals[st.choose(probs, counts)], npu, false)
			} else {
				w := st.blend(probs, counts)
				bump := make([]float32, npu)
				for j, v := range vals {
					cfg.ValuePop.Encode(&bump, v, npu, false)
					for i := range out {
						out[i] += w[j] * bump[i]
					}
				}
			}
			ev.TrialStats(out, sal, &ts)
			for i, v := range ts.Values() {
				if !math.IsNaN(v) {
					sums[i] += v
					cnts[i]++
				}
			}
		}
		res := ""
		for i, nm := range TrialStatNames {
			switch nm {
			case "Miss", "ChoseP", "PGain", "OutPErr", "AgreeBoost":
				res += fmt.Sprintf("  %s %.3f", nm, sums[i]/cnts[i])
			}
		}
		t.Logf("%-26s%s", st.name, res)
	}
}

// TestSampling checks that the chosen cue frequencies match the
// normalized probabilities, for a fixed item.
func TestSampling(t *testing.T) {
	ev := &PCombEnv{}
	ev.Defaults()
	ev.Config(100, 1, 0, false, 1)
	ev.Init(0)
	cfg := &ev.Configs
	// probs 0.2, 0.4, 1.0 with distinct values
	it := &Item{Cues: []Cue{{P: 0, V: 0}, {P: 1, V: 4}, {P: 4, V: 8}}}
	counts := map[float32]int{}
	n := 100000
	var vals, probs []float32
	for range n {
		ev.SetItem(it)
		counts[ev.ChosenValue()]++
		vals, probs = ev.ValueProbs()
	}
	for i, v := range vals {
		f := float32(counts[v]) / float32(n)
		if math32.Abs(f-probs[i]) > 0.01 {
			t.Errorf("value %g: freq %.3f, want %.3f", v, f, probs[i])
		}
	}
	_ = cfg
}

func TestRender(t *testing.T) {
	ev := &PCombEnv{}
	ev.Defaults()
	ev.Config(100, 2, 1, true, 1)
	ev.Init(0)
	cfg := &ev.Configs
	t.Logf("pop code units: %d", cfg.PopCodeUnits)
	for range 5 {
		ev.Step()
		vals, probs := ev.ValueProbs()
		t.Logf("%s  values: %v probs: %v", ev.String(), vals, probs)
		out := ev.States["Output"].Values
		// DecodePeak returns the peak unit's value, so it is accurate
		// to within half the unit spacing.
		pc := &cfg.ValuePop
		tol := 0.5 * (pc.Max - pc.Min) / float32(cfg.PopCodeUnits-1)
		if d := math32.Abs(ev.Decode(out) - ev.ChosenValue()); d > tol {
			t.Errorf("decoded output off by %g > %g", d, tol)
		}
	}
	act := func(pat []float32) int {
		n := 0
		for _, v := range pat {
			if v > 0.1 {
				n++
			}
		}
		return n
	}
	pat := make([]float32, cfg.PopCodeUnits)
	for vi := range cfg.NValues() {
		v := cfg.Value(vi)
		cfg.ValuePop.Encode(&pat, v, len(pat), false)
		nv := act(pat)
		ev.SaliencePattern(pat, v)
		ns := act(pat)
		t.Logf("value %+.2f: value units > 0.1: %d / %d, salience units: %d", v, nv, len(pat), ns)
		switch cfg.Salience(v) {
		case 0:
			if ns != 0 {
				t.Errorf("value %g: expected no salience", v)
			}
		case 0.5:
			if ns == 0 || ns >= nv {
				t.Errorf("value %g: expected half salience", v)
			}
		case 1:
			if ns != nv {
				t.Errorf("value %g: expected full salience", v)
			}
		}
	}
}
