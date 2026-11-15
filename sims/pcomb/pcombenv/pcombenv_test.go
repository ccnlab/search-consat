// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcombenv

import (
	"fmt"
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
		n := len(sp.Train[k]) + len(sp.Test[k])
		if int64(n) != cfg.NCombos(cfg.NActive.Min+k) {
			t.Errorf("n=%d: train+test = %d", cfg.NActive.Min+k, n)
		}
		for _, i := range sp.Test[k] {
			inTest[i] = true
		}
	}
	for _, trn := range sp.Train {
		for _, i := range trn {
			if inTest[i] {
				t.Fatalf("item %d in both train and test", i)
			}
		}
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
