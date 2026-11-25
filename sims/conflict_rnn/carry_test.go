// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflictrnn

import (
	"math"
	"testing"

	"github.com/emer/emergent/v2/egui"
)

// TestCarry runs a short training run with and without carryover
// (Decay 0.5 and 1), checking that the _CarryEnd stats are computed for
// all layers, are ~0 for the clamped Input, are copied to the train log,
// and that the carryover arithmetic is correct.
func TestCarry(t *testing.T) {
	for _, decay := range []float64{0.5, 1} {
		cfg, _ := egui.NewConfig[Config]()
		cfg.GUI = false
		cfg.Run.NData = 8
		cfg.Run.Trials = 64
		cfg.Run.Epochs = 2
		cfg.Run.Runs = 1
		cfg.Run.TestInterval = 1
		cfg.Params.NCycles = 3
		cfg.Params.Decay = decay
		cfg.Env.Env = map[string]any{"TestPerLevel": 4, "TestReps": 1}
		ss := &Sim{}
		ss.SetConfig(cfg)
		ss.ConfigSim()
		ss.Loops.Run(Train)
		trn := ss.Stats.Dir("Train/Epoch")
		tst := ss.Stats.Dir("Test/Epoch")
		for _, name := range ss.carryNames() {
			tr := trn.Float64(name)
			te := tst.Float64(name)
			cp := trn.Float64("Test_" + name)
			// tests: at the start of epoch 1, and the final test at the end of the run
			if tr.Len() != 2 || te.Len() != 2 || cp.Len() != 2 {
				t.Fatalf("decay %g %s: lens train %d test %d copy %d", decay, name, tr.Len(), te.Len(), cp.Len())
			}
			for _, v := range append(tr.Values, te.Values...) {
				if math.IsNaN(v) {
					t.Errorf("decay %g %s: NaN", decay, name)
				}
			}
			if cp.Values[1] != te.Values[0] {
				t.Errorf("decay %g %s: Test_ copy %g != test %g", decay, name, cp.Values[1], te.Values[0])
			}
			if name == "Input_CarryEnd" && math.Abs(tr.Values[1]) > 0.1 {
				t.Errorf("decay %g %s: input carryover %g, expected ~0", decay, name, tr.Values[1])
			}
			t.Logf("decay %g %s: train %.3f %.3f, test %.3f", decay, name, tr.Values[0], tr.Values[1], te.Values[1])
		}
	}

	cur := []float32{1, 0, 0}
	if v := carryover(cur, [][]float32{{1, 0, 0}}, 0); !math.IsNaN(v) {
		t.Errorf("one stream: %g, expected NaN", v)
	}
	if v := carryover(cur, [][]float32{{1, 0, 0}, nil}, 0); !math.IsNaN(v) {
		t.Errorf("missing prev: %g, expected NaN", v)
	}
	prev := [][]float32{{1, 0, 0}, {0, 1, 0}, {1, 1, 0}}
	want := 1 - (0+1/math.Sqrt2)/2
	if v := carryover(cur, prev, 0); math.Abs(v-want) > 1e-6 {
		t.Errorf("carryover %g, expected %g", v, want)
	}
}
