// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package otmenv

import (
	"math"
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

// TestVocab checks the overlap among the input patterns and among the
// output patterns (MaxOverlap), and reports the number of exclusive units
// of each output (not shared with the input's other outputs).
func TestVocab(t *testing.T) {
	for _, c := range []struct{ n, o int }{{32, 2}, {32, 3}, {10, 4}, {20, 2}} {
		var cfg Config
		cfg.Defaults()
		cfg.NInputs, cfg.NOut = c.n, c.o
		vc := cfg.MakeVocab()
		pat := func(vi int) []float32 { return vc.SubSpace(vi).(*tensor.Float32).Values }
		overlap := func(a, b []float32) int {
			ov := 0
			for k, v := range a {
				if v > 0 && b[k] > 0 {
					ov++
				}
			}
			return ov
		}
		non := cfg.NOn()
		for i := range cfg.NVocab() {
			if n := countOn(pat(i)); n != non {
				t.Fatalf("pattern %d: %d on, want %d", i, n, non)
			}
		}
		inWorst, outWorst := 0, 0
		for i := range cfg.NInputs {
			for j := i + 1; j < cfg.NInputs; j++ {
				inWorst = max(inWorst, overlap(pat(i), pat(j)))
			}
		}
		for i := cfg.NInputs; i < cfg.NVocab(); i++ {
			for j := i + 1; j < cfg.NVocab(); j++ {
				outWorst = max(outWorst, overlap(pat(i), pat(j)))
			}
		}
		exMin, exSum := non, 0
		for in := range cfg.NInputs {
			for o := range cfg.NOut {
				p := pat(cfg.OutputIndex(in, o))
				ex := 0
				for k, v := range p {
					shared := false
					for o2 := range cfg.NOut {
						if o2 != o && pat(cfg.OutputIndex(in, o2))[k] > 0 {
							shared = true
						}
					}
					if v > 0 && !shared {
						ex++
					}
				}
				exMin, exSum = min(exMin, ex), exSum+ex
			}
		}
		t.Logf("n=%d o=%d: max overlap among inputs %d, among outputs %d (allowed %d); exclusive units per output min %d, mean %.1f of %d", c.n, c.o, inWorst, outWorst, non-cfg.MinDiff(), exMin, float64(exSum)/float64(cfg.NOutputs()), non)
		if inWorst > non-cfg.MinDiff() || outWorst > non-cfg.MinDiff() {
			t.Errorf("overlap above allowed")
		}
	}
}

// TestStats checks the stats for clean and partial choices, blends,
// intrusions, and low-level activity below ActThr.
func TestStats(t *testing.T) {
	for _, no := range []int{1, 2} {
		var cfg Config
		cfg.Defaults()
		cfg.NOut = no
		ev := &OTMEnv{Configs: cfg}
		ev.Config(100, 0)
		ev.Init(0)
		vc := ev.Vocab()
		pat := func(vi int) []float32 { return vc.SubSpace(vi).(*tensor.Float32).Values }
		// mix returns the given output patterns of the input with the given
		// activities, plus nOther units outside all of the input's outputs
		// at act otherAct, plus low-level activity everywhere.
		mix := func(acts map[int]float32, nOther int, otherAct float32) []float32 {
			out := make([]float32, cfg.PoolUnits())
			for i := range out {
				out[i] = 0.05
			}
			for o, a := range acts {
				for i, v := range pat(cfg.OutputIndex(ev.Input, o)) {
					if v > 0 {
						out[i] = max(out[i], a)
					}
				}
			}
			for i := range out {
				if nOther == 0 {
					break
				}
				in := false
				for o := range no {
					if pat(cfg.OutputIndex(ev.Input, o))[i] > 0 {
						in = true
					}
				}
				if !in {
					out[i] = otherAct
					nOther--
				}
			}
			return out
		}
		var ts TrialStats
		for range 200 {
			ev.Step()
			tg := ev.Out
			ev.TrialStats(mix(map[int]float32{tg: 0.9}, 0, 0), &ts)
			if ts.Valid != 1 || ts.Out != tg || ts.OtherRatio != 0 || ts.MatchCos < 0.95 {
				t.Fatalf("o=%d target: %s: %v %d", no, ev.String(), ts.Values(), ts.Out)
			}
			if no == 1 && !math.IsNaN(ts.SecondRatio) || no == 2 && ts.SecondRatio != 0 {
				t.Fatalf("o=%d target SecondRatio: %v", no, ts.Values())
			}
			// other activity below and above the intrusion ratio
			ev.TrialStats(mix(map[int]float32{tg: 0.6}, 3, 0.6), &ts)
			if ts.Valid != 1 || ts.Intrusion != 0 || math.Abs(ts.OtherRatio-0.3) > 1e-5 {
				t.Fatalf("o=%d other below ratio: %v", no, ts.Values())
			}
			ev.TrialStats(mix(map[int]float32{tg: 0.6}, 6, 0.6), &ts)
			if ts.Valid != 0 || ts.Intrusion != 1 {
				t.Fatalf("o=%d intrusion: %v", no, ts.Values())
			}
			ev.TrialStats(mix(nil, 0, 0), &ts)
			if ts.Valid != 0 || ts.Blend != 1 || !math.IsNaN(ts.OtherRatio) {
				t.Fatalf("o=%d no activity: %v", no, ts.Values())
			}
			if no == 1 {
				continue
			}
			other := 1 - tg
			ev.TrialStats(mix(map[int]float32{other: 0.9}, 0, 0), &ts)
			if ts.Valid != 1 || ts.Out != other {
				t.Fatalf("other output: %v", ts.Values())
			}
			// partial activity of the other output, above the blend ratio
			ev.TrialStats(mix(map[int]float32{tg: 0.8, other: 0.35}, 0, 0), &ts)
			if ts.Valid != 1 || ts.Out != tg || math.Abs(ts.SecondRatio-0.35/0.8) > 1e-5 {
				t.Fatalf("partial: %v", ts.Values())
			}
			// equal blend
			ev.TrialStats(mix(map[int]float32{tg: 0.6, other: 0.6}, 0, 0), &ts)
			if ts.Valid != 0 || ts.Blend != 1 || ts.SecondRatio != 1 {
				t.Fatalf("blend: %v", ts.Values())
			}
		}
	}
}
