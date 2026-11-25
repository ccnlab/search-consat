// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflictrnn

import (
	"math"
	"strings"
	"testing"

	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/plot/plots"
	"cogentcore.org/lab/tensorfs"

	"github.com/ccnlab/search-consat/sims/conflict/cfenv"
	"github.com/emer/emergent/v2/egui"
)

// TestCompete runs a short training run with a final test, and checks
// that the competition stats and the settling microscope are computed
// from the per-cycle activity (RecordCycles), on the axon time scale.
func TestCompete(t *testing.T) {
	cfg, _ := egui.NewConfig[Config]()
	cfg.GUI = false
	cfg.Run.NData = 8
	cfg.Run.Trials = 256
	cfg.Run.Epochs = 20 // enough to make choices of the input groups (not all intrusions)
	cfg.Run.Runs = 1
	cfg.Params.NCycles = 10
	cfg.Params.Decay = 0.5
	cfg.Env.Env = map[string]any{"TestPerLevel": 16, "TestReps": 1}
	ss := &Sim{}
	ss.SetConfig(cfg)
	ss.ConfigSim()
	ss.Loops.Run(Train)
	if iv := ss.cycleInterval(); iv != 15 {
		t.Fatalf("cycleInterval = %d, want 15", iv)
	}
	ep := ss.Stats.Dir("Test/Epoch")
	for _, nm := range cfenv.CompeteNames {
		v := ep.Float64(nm).Float1D(-1)
		t.Logf("%s: %.3f", nm, v)
		switch nm {
		case "LeadSwaps", "CommitCyc", "LoserFrac", "Agree50", "Agree100":
			if math.IsNaN(v) {
				t.Errorf("%s is NaN", nm)
			}
		}
	}
	if v := ep.Float64("CommitCyc").Float1D(-1); v < 15 || v > 150 {
		t.Errorf("CommitCyc %g not in [15, 150]", v)
	}
	mic := ss.Stats.Dir("Test/Microscope")
	if c := mic.Float64("Cycle").Values; len(c) != 10 || c[9] != 150 {
		t.Errorf("Microscope Cycle: %v", c)
	}
	for _, nm := range []string{"Hidden1", "Output", "OutWin", "OutLose"} {
		v := mic.Float64(nm).Values
		t.Logf("Microscope %s: %.2f", nm, v)
		if len(v) != 10 || math.IsNaN(v[9]) {
			t.Errorf("Microscope %s not computed", nm)
		}
	}
	checkMicroscopeSEM(t, mic, 9)
	trl := ss.Stats.Dir("Test/MicroscopeTrial").StringValue("TrialName")
	if want := ss.Envs.ByModeDi(Test, 0).(*cfenv.CFEnv).TrialLabel(); trl.Len() != 10 || trl.String1D(0) != want || trl.String1D(10-1) != want || !strings.Contains(want, "choice:") {
		t.Errorf("MicroscopeTrial TrialName %q (%d rows), want %q", trl.String1D(0), trl.Len(), want)
	}
	t.Logf("MicroscopeTrial TrialName: %s", trl.String1D(0))
	if v := mic.Float64("Output").Float1D(9); math.Abs(v-1) > 1e-5 {
		t.Errorf("Microscope Output final cosine = %g, want 1", v)
	}
	if v := ss.Stats.Dir("Train/Run").Float64("Test_LeadSwaps").Float1D(-1); math.IsNaN(v) {
		t.Errorf("Test_LeadSwaps not copied to Train Run")
	}
}

// checkMicroscopeSEM checks the SEM columns of the Test/Microscope average
// (about 0 for the final Output cosine, which is always 1), and that its
// plot draws an SEM Band for each line that is On.
func checkMicroscopeSEM(t *testing.T, mic *tensorfs.Node, last int) {
	t.Helper()
	if v := mic.Float64("OutWinSEM").Float1D(last / 2); math.IsNaN(v) || v < 0 || v > 0.5 {
		t.Errorf("Microscope OutWinSEM = %g", v)
	}
	if v := mic.Float64("OutputSEM").Float1D(last); math.IsNaN(v) || v > 1e-4 {
		t.Errorf("Microscope final OutputSEM = %g, want 0", v)
	}
	plt, err := plot.NewTablePlot(tensorfs.DirTable(mic, nil))
	if err != nil {
		t.Fatal(err)
	}
	nline, nband := 0, 0
	for _, p := range plt.Plotters {
		switch p.(type) {
		case *plots.Band:
			nband++
		case *plots.XY:
			nline++
		}
	}
	if nline == 0 || nband != nline {
		t.Errorf("Microscope plot: %d lines, %d SEM bands", nline, nband)
	}
}
