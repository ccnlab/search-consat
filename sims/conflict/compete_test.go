// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflict

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

// TestCompete runs a short test (untrained network) and checks that the
// competition samples cover the minus phase (the last sample has Output
// activity) and that the competition stats are computed on some trials.
func TestCompete(t *testing.T) {
	cfg, _ := egui.NewConfig[Config]()
	cfg.GPU = false
	cfg.GUI = false
	cfg.Run.NData = 4
	cfg.Run.Trials = 8
	cfg.Run.Epochs = 1
	cfg.Run.Runs = 1
	cfg.Env.Env = map[string]any{"TestPerLevel": 16, "TestReps": 1}
	ss := &Sim{}
	ss.SetConfig(cfg)
	ss.ConfigSim()
	ss.Loops.Run(Train) // final test at the end of the run
	nsamp := cfg.Run.MinusCycles / cfg.Run.CompeteInterval
	if len(ss.competeDrive) != cfg.Run.NData || len(ss.competeDrive[0]) != nsamp {
		t.Fatalf("competeDrive not sampled: %d data", len(ss.competeDrive))
	}
	last := float32(0)
	for _, d := range ss.competeDrive {
		for _, v := range d[nsamp-1] {
			last += v
		}
	}
	if last == 0 {
		t.Errorf("no Output CaM in the last sample")
	}
	trl := ss.Stats.Dir("Test/Trial")
	n := 0
	for _, v := range trl.Float64("LeadSwaps").Values {
		if !math.IsNaN(v) {
			n++
		}
	}
	ep := ss.Stats.Dir("Test/Epoch")
	for _, nm := range []string{"LeadSwaps", "CommitCyc", "LoserFrac", "AltCorr", "Agree50", "Agree100"} {
		t.Logf("%s: %.3f", nm, ep.Float64(nm).Float1D(-1))
	}
	t.Logf("conflict trials with a choice: %d of %d", n, trl.Float64("LeadSwaps").Len())
	if n == 0 {
		t.Errorf("no trials with competition stats")
	}
	for _, nm := range cfenv.MatchNames {
		v := ep.Float64(nm).Float1D(-1)
		t.Logf("%s: %.3f", nm, v)
		if math.IsNaN(v) {
			t.Errorf("%s is NaN", nm)
		}
	}
	mic := ss.Stats.Dir("Test/Microscope")
	for _, nm := range []string{"Output", "Hidden1", "OutWin", "OutLose", "OutWinMajor"} {
		v := mic.Float64(nm).Values
		t.Logf("Microscope %s: %.2f", nm, v)
		if len(v) != nsamp || math.IsNaN(v[nsamp-1]) {
			t.Errorf("Microscope %s not computed", nm)
		}
	}
	checkMicroscopeSEM(t, mic, nsamp-1)
	tnm := ss.Stats.Dir("Test/MicroscopeTrial").StringValue("TrialName")
	if want := ss.Envs.ByModeDi(Test, 0).(*cfenv.CFEnv).TrialLabel(); tnm.Len() != nsamp || tnm.String1D(0) != want || tnm.String1D(nsamp-1) != want || !strings.Contains(want, "choice:") {
		t.Errorf("MicroscopeTrial TrialName %q (%d rows), want %q", tnm.String1D(0), tnm.Len(), want)
	}
	t.Logf("MicroscopeTrial TrialName: %s", tnm.String1D(0))
	if v := mic.Float64("Output").Float1D(nsamp - 1); math.Abs(v-1) > 1e-6 {
		t.Errorf("Microscope Output final cosine = %g, want 1", v)
	}
	if v := mic.Float64("OutWin").Float1D(nsamp-1) - mic.Float64("OutLose").Float1D(nsamp-1); v <= 0 {
		t.Errorf("Microscope final OutWin not above OutLose: %g", v)
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
