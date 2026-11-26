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
	if len(lbls) != 30 || len(lbls) != Search.NumParams() || lbls[0] != "Search 1: Baseline" || lbls[1] != "Search 2: Params.Decay=0.5" || lbls[14] != "Search 15: Params.TestDropout=0.4" || lbls[26] != "Search 27: Params.OutputLateral=1" {
		t.Fatalf("Search labels: %q", lbls)
	}
	if n := Search1.NumParams(); n != 43 {
		t.Errorf("Search1 has %d jobs, want 43", n)
	}
	for _, at := range []int{0, 1, 14, 26} {
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
		if want := map[int]float64{0: 1, 1: 0.5}[at]; at < 2 && ss.Net.Params.Decay != want {
			t.Errorf("job %d: Decay = %g, want %g", at+1, ss.Net.Params.Decay, want)
		}
		for _, ly := range ss.Net.Layers {
			if ly.Type != egomlx.Hidden {
				continue
			}
			want := 0.0
			if at == 14 {
				want = 0.4
			}
			if d := ly.Params.Act.TestDropout; d != want {
				t.Errorf("job %d: %s TestDropout = %g, want %g", at+1, ly.Name, d, want)
			}
		}
		lat := false // OutputLateral: an Output -> Output path, built in ConfigSim
		for _, pt := range ss.Net.LayerByName("Output").RecvPaths {
			lat = lat || pt.Send.Name == "Output"
		}
		if lat != (at == 26) {
			t.Errorf("job %d: Output lateral path %v", at+1, lat)
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
