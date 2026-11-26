// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflictff

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
		{nenc: 0, nhid: 1, layers: []string{"Input", "Hidden1", "Output"}, npaths: 2},
		{nenc: 1, nhid: 0, short: true, layers: []string{"Input", "Encoder1", "Output"}, npaths: 2},
		{nenc: 2, nhid: 1, short: true, layers: []string{"Input", "Encoder1", "Encoder2", "Hidden1", "Output"}, npaths: 5},
		{nenc: 1, nhid: 2, layers: []string{"Input", "Encoder1", "Hidden1", "Hidden2", "Output"}, npaths: 4},
	}
	for _, c := range cases {
		cfg, _ := egui.NewConfig[Config]()
		cfg.GUI = false
		cfg.Run.NData = 4
		cfg.Run.Trials = 16
		cfg.Run.Epochs = 1
		cfg.Run.Runs = 1
		cfg.Params.NEncoder, cfg.Params.NHidden = c.nenc, c.nhid
		cfg.Params.Shortcut = c.short
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
	if len(lbls) != 19 || len(lbls) != Search.NumParams() || lbls[0] != "Search 1: Baseline" || lbls[11] != "Search 12: Params.OutNoise=1" || lbls[15] != "Search 16: Params.HiddenSize=5" {
		t.Fatalf("Search labels: %q", lbls)
	}
	if n := Search1.NumParams(); n != 33 {
		t.Errorf("Search1 has %d jobs, want 33", n)
	}
	// job at (0-based), with the base Args -test-dropout 0.5, applied as
	// in a run with -search-at (in ConfigSim, before the network is built)
	sim := func(at int) *Sim {
		cfg, _ := egui.NewConfig[Config]()
		cfg.GUI = false
		cfg.Run.NData = 2
		cfg.Params.TestDropout = 0.5
		cfg.Params.SearchAt = at + 1
		ss := &Sim{}
		ss.SetConfig(cfg)
		ss.ConfigSim()
		if ss.searchErr != nil || ss.searchLabel != lbls[at] {
			t.Fatalf("search job %d: %q %v", at+1, ss.searchLabel, ss.searchErr)
		}
		ss.Init()
		return ss
	}
	ss := sim(11)
	n := 0
	for _, ly := range ss.Net.Layers {
		switch ly.Type {
		case egomlx.Hidden:
			n++
			if a := ly.Params.Act; a.TestDropout != 0.5 || a.Noise != 0 {
				t.Errorf("%s TestDropout %g, Noise %g, want .5, 0", ly.Name, a.TestDropout, a.Noise)
			}
		case egomlx.Target:
			if nz := ly.Params.Act.Noise; nz != 1 {
				t.Errorf("%s Noise = %g, want 1 (OutNoise)", ly.Name, nz)
			}
		}
	}
	if n == 0 {
		t.Errorf("no hidden layers")
	}
	ss = sim(15)
	if nu := ss.Net.LayerByName("Hidden1").NumUnits(); nu != 25 {
		t.Errorf("Hidden1 has %d units, want 25 (HiddenSize 5)", nu)
	}
	ss = sim(18)
	if ss.Net.LayerByName("Hidden1") == nil || ss.Net.LayerByName("Hidden2") != nil {
		t.Errorf("NHidden 1: want only Hidden1")
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
