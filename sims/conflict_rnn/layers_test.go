// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflictrnn

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/emer/egomlx/egomlx"
	"github.com/emer/emergent/v2/egui"
)

// TestLayerConfigs builds and runs the network with different numbers of
// encoder and hidden layers (including none) and the optional pathways,
// checking the layers and pathways, and that a short training run and
// test produce the output stats.
func TestLayerConfigs(t *testing.T) {
	type lc struct {
		nenc, nhid                     int
		encLat, short, encBack, outLat bool
		layers                         []string
		npaths                         int
	}
	cases := []lc{
		{nenc: 0, nhid: 0, layers: []string{"Input", "Output"}, npaths: 1},
		{nenc: 0, nhid: 1, layers: []string{"Input", "Hidden1", "Output"}, npaths: 3},
		{nenc: 1, nhid: 0, encLat: true, short: true, encBack: true, layers: []string{"Input", "Encoder1", "Output"}, npaths: 4},
		{nenc: 2, nhid: 1, encLat: true, short: true, encBack: true, outLat: true, layers: []string{"Input", "Encoder1", "Encoder2", "Hidden1", "Output"}, npaths: 12},
		{nenc: 1, nhid: 2, layers: []string{"Input", "Encoder1", "Hidden1", "Hidden2", "Output"}, npaths: 7},
	}
	for _, c := range cases {
		cfg, _ := egui.NewConfig[Config]()
		cfg.GUI = false
		cfg.Run.NData = 4
		cfg.Run.Trials = 16
		cfg.Run.Epochs = 1
		cfg.Run.Runs = 1
		cfg.Params.NEncoder, cfg.Params.NHidden = c.nenc, c.nhid
		cfg.Params.EncoderLateral, cfg.Params.Shortcut, cfg.Params.EncoderBack = c.encLat, c.short, c.encBack
		cfg.Params.OutputLateral = c.outLat
		cfg.Params.NCycles = 3
		cfg.Env.Env = map[string]any{"TestPerLevel": 4, "TestReps": 1}
		ss := &Sim{}
		ss.SetConfig(cfg)
		ss.ConfigSim()
		var names []string
		npaths := 0
		for _, ly := range ss.Net.Layers {
			names = append(names, ly.Name)
			npaths += len(ly.RecvPaths)
		}
		name := fmt.Sprintf("enc %d hid %d", c.nenc, c.nhid)
		if !slices.Equal(names, c.layers) || npaths != c.npaths {
			t.Errorf("%s: layers %v, paths %d, expected %v, %d", name, names, npaths, c.layers, c.npaths)
		}
		ss.Loops.Run(Train)
		p := ss.Stats.Dir("Train/Run").Float64("Test_P100")
		if p.Len() != 1 || math.IsNaN(p.Values[0]) {
			t.Errorf("%s: no test results: %v", name, p.Values)
		}
		t.Logf("%s: layers %v, %d paths, Test_P100 %v", name, names, npaths, p.Values)
	}
}

// TestSearch checks that the parameter search jobs apply their values to
// the config, and from there to the network, as RunNoGUI does.
func TestSearch(t *testing.T) {
	lbls := Search.Labels()
	if len(lbls) != 24 || len(lbls) != Search.NumParams() || lbls[0] != "Search 1: Baseline" || lbls[1] != "Search 2: Params.Decay=0" || lbls[7] != "Search 8: Decay=0.7,Noise=1" || lbls[18] != "Search 19: Decay=0.7,TestDropout=0.8" || lbls[23] != "Search 24: Noise=2,TestDropout=0.8" {
		t.Fatalf("Search labels: %q", lbls)
	}
	if n := SearchPS1.NumParams(); n != 30 {
		t.Errorf("SearchPS1 has %d jobs, want 30", n)
	}
	if n := Search1.NumParams(); n != 43 {
		t.Errorf("Search1 has %d jobs, want 43", n)
	}
	// job index: Decay, Noise, TestDropout
	want := map[int][3]float64{0: {1, 0.2, 0}, 1: {0, 0.2, 0}, 7: {0.7, 1, 0}, 18: {0.7, 0.2, 0.8}, 23: {1, 2, 0.8}}
	for at, w := range want {
		cfg, _ := egui.NewConfig[Config]()
		cfg.GUI = false
		cfg.Run.NData = 2
		cfg.Params.SearchAt = at + 1 // applied in ConfigSim, as with -search-at
		ss := &Sim{}
		ss.SetConfig(cfg)
		ss.ConfigSim()
		if ss.searchErr != nil || ss.searchLabel != lbls[at] {
			t.Fatalf("search job %d: %q %v", at+1, ss.searchLabel, ss.searchErr)
		}
		ss.Init()
		if ss.Net.Params.Decay != w[0] {
			t.Errorf("job %d: Decay = %g, want %g", at+1, ss.Net.Params.Decay, w[0])
		}
		for _, ly := range ss.Net.Layers {
			if ly.Type != egomlx.Hidden {
				continue
			}
			if n, d := ly.Params.Act.Noise, ly.Params.Act.TestDropout; n != w[1] || d != w[2] {
				t.Errorf("job %d: %s Noise, TestDropout = %g, %g, want %g, %g", at+1, ly.Name, n, d, w[1], w[2])
			}
		}
	}
}

// TestBackend checks that Config.Backend selects the compute backend
// (the pure-Go one here), through GOMLX_BACKEND.
func TestBackend(t *testing.T) {
	t.Setenv("GOMLX_BACKEND", "") // restored after the test
	cfg, _ := egui.NewConfig[Config]()
	cfg.GUI = false
	cfg.Run.NData = 2
	if cfg.Backend != "xla:cpu" {
		t.Errorf("default Backend %q, want xla:cpu", cfg.Backend)
	}
	cfg.Backend = "go"
	ss := &Sim{}
	ss.SetConfig(cfg)
	ss.ConfigSim()
	defer ss.Net.Backend.Finalize()
	if nm := ss.Net.Backend.Name(); nm != "Go Backend" {
		t.Errorf("Backend %q (%s), want Go Backend", nm, ss.Net.Backend.Description())
	}
}

// TestStable checks that the default hidden layers are Tanh, whose
// activity stays bounded over training: with ReLU, the recurrent loops
// can amplify the activity until the loss is NaN.
func TestStable(t *testing.T) {
	cfg, _ := egui.NewConfig[Config]()
	cfg.GUI = false
	cfg.Run.NData = 16
	cfg.Run.Trials = 512
	cfg.Run.Epochs = 5
	cfg.Run.Runs = 2
	cfg.Run.TestInterval = 1000
	cfg.Env.Env = map[string]any{"TestPerLevel": 4, "TestReps": 1}
	if cfg.Params.HiddenFn != egomlx.Tanh {
		t.Errorf("default HiddenFn = %v, want Tanh", cfg.Params.HiddenFn)
	}
	ss := &Sim{}
	ss.SetConfig(cfg)
	ss.ConfigSim()
	ss.Loops.Run(Train)
	for _, ly := range ss.Net.Layers {
		if ly.Type == egomlx.Hidden && ly.Params.Act.Fn != egomlx.Tanh {
			t.Errorf("%s Act.Fn = %v, want Tanh", ly.Name, ly.Params.Act.Fn)
		}
	}
	run := ss.Stats.Dir("Train/Run")
	for _, nm := range []string{"Hidden1_ActMax", "Hidden2_ActMax", "MatchCos"} {
		vals := run.Float64(nm).Values
		if len(vals) != 2 {
			t.Fatalf("%s: %d runs, want 2", nm, len(vals))
		}
		for ri, v := range vals {
			t.Logf("run %d %s: %.3f", ri, nm, v)
			if math.IsNaN(v) || math.Abs(v) > 1 {
				t.Errorf("run %d %s = %g, want within [-1, 1]", ri, nm, v)
			}
		}
	}
	if v := run.Float64("MatchCos").Values; v[0] < 0.9 || v[1] < 0.9 {
		t.Errorf("MatchCos %v: not learning", v)
	}
}
