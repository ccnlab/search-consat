// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package onetomany

import (
	"math"
	"testing"

	"github.com/emer/emergent/v2/egui"
)

// TestSecondHist runs a short training epoch on the CPU and checks that
// the SecondRatio histogram is computed as proportions of trials.
func TestSecondHist(t *testing.T) {
	cfg, _ := egui.NewConfig[Config]()
	cfg.GPU = false
	cfg.GUI = false
	cfg.Run.NData = 4
	cfg.Run.Trials = 16
	cfg.Run.Epochs = 4
	cfg.Run.Runs = 1
	cfg.Run.TestInterval = 0
	ss := &Sim{}
	ss.SetConfig(cfg)
	ss.ConfigSim()
	ss.Loops.Run(Train)
	dir := ss.Stats.Dir("Train/SecondRatioHist")
	props := dir.Float64("Prop").Values
	trl := ss.Stats.Dir("Train/Trial").Float64("SecondRatio").Values
	n := 0
	for _, v := range trl {
		if !math.IsNaN(v) {
			n++
		}
	}
	sum := 0.0
	for _, p := range props {
		sum += p
	}
	t.Logf("SecondRatio hist (%d of %d trials): %v", n, len(trl), props)
	if len(props) != 10 || (n > 0 && math.Abs(sum-1) > 1e-6) {
		t.Errorf("histogram proportions sum to %g", sum)
	}
}
