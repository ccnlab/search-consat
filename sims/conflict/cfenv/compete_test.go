// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfenv

import (
	"math"
	"testing"
)

func TestCompete(t *testing.T) {
	const iv = 10 // 15 samples over 150 cycles
	ev := &CFEnv{}
	ev.Cur = &Item{Groups: []int{2, 0}, Probs: []float32{2. / 3, 1. / 3}, NConflict: 1}
	// drive builds samples from per-sample (a, b) activity of groups 2 and 0.
	drive := func(ab ...[2]float32) [][]float32 {
		d := make([][]float32, len(ab))
		for s, v := range ab {
			d[s] = make([]float32, 4)
			d[s][2], d[s][0] = v[0], v[1]
		}
		return d
	}
	var cs CompeteStats

	// majority wins from the start, smoothly: no swaps, commit at cycle 10
	var smooth [][2]float32
	for s := range 15 {
		smooth = append(smooth, [2]float32{float32(s), float32(s) / 4})
	}
	smooth[0] = [2]float32{0, 0} // no activity in the first sample
	ev.Compete(drive(smooth...), iv, 2, &cs)
	if cs.LeadSwaps != 0 || cs.CommitCyc != 20 || cs.Agree50 != 1 || cs.Agree100 != 1 {
		t.Fatalf("smooth: %+v", cs)
	}
	if math.Abs(cs.LoserFrac-0.2) > 1e-6 || cs.AltCorr <= 0.99 {
		t.Fatalf("smooth LoserFrac / AltCorr: %+v", cs)
	}
	if cs.LeadSwapsMajor != 0 || !math.IsNaN(cs.LeadSwapsMinor) || !math.IsNaN(cs.CommitCycMinor) {
		t.Fatalf("smooth splits: %+v", cs)
	}

	// alternating lead, minority (0) wins at the end
	var alt [][2]float32
	for s := range 15 {
		if s%2 == 0 {
			alt = append(alt, [2]float32{3, 1})
		} else {
			alt = append(alt, [2]float32{1, 3})
		}
	}
	ev.Compete(drive(alt...), iv, 0, &cs)
	if cs.LeadSwaps != 14 || cs.CommitCyc != 150 || cs.AltCorr > -0.99 {
		t.Fatalf("alternating: %+v", cs)
	}
	// leader at cycle 50 (sample 4) is group 2, at 100 (sample 9) is group 0
	if cs.Agree50 != 0 || cs.Agree100 != 1 {
		t.Fatalf("alternating agree: %+v", cs)
	}
	if cs.LeadSwapsMinor != 14 || cs.CommitCycMinor != 150 || !math.IsNaN(cs.LeadSwapsMajor) {
		t.Fatalf("alternating splits: %+v", cs)
	}

	// ties: no minority / majority split
	ev.Cur.Probs = []float32{0.5, 0.5}
	ev.Compete(drive(alt...), iv, 0, &cs)
	if !math.IsNaN(cs.LeadSwapsMinor) || !math.IsNaN(cs.LeadSwapsMajor) || cs.LeadSwaps != 14 {
		t.Fatalf("tie: %+v", cs)
	}

	// choice not an input group (intrusion) or blend: all NaN
	for _, ch := range []int{1, -1} {
		ev.Compete(drive(alt...), iv, ch, &cs)
		for i, v := range cs.Values() {
			if !math.IsNaN(v) {
				t.Fatalf("choice %d: %s = %g", ch, CompeteNames[i], v)
			}
		}
	}

	// consistent item: all NaN
	ev.Cur = &Item{Groups: []int{2}, Probs: []float32{1}}
	ev.Compete(drive(alt...), iv, 2, &cs)
	if !math.IsNaN(cs.LeadSwaps) {
		t.Fatalf("consistent: %+v", cs)
	}
	if len(cs.Values()) != len(CompeteNames) {
		t.Fatal("Values and CompeteNames differ in length")
	}
}
