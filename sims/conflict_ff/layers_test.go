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
	if len(lbls) != Search.NumParams() || lbls[0] != "Search 1: Baseline" || lbls[4] != "Search 5: Params.Dropout=0.2" {
		t.Fatalf("Search labels: %q", lbls)
	}
	cfg, _ := egui.NewConfig[Config]()
	cfg.GUI = false
	cfg.Run.NData = 2
	lbl, err := Search.ApplyTo(cfg, 4)
	if err != nil || lbl != lbls[4] || cfg.Params.Dropout != 0.2 || cfg.Params.WeightDecay != 0 {
		t.Fatalf("ApplyTo: %q %v, Dropout %g, WeightDecay %g", lbl, err, cfg.Params.Dropout, cfg.Params.WeightDecay)
	}
	ss := &Sim{}
	ss.SetConfig(cfg)
	ss.ConfigSim()
	ss.Init()
	n := 0
	for _, ly := range ss.Net.Layers {
		if ly.Type == egomlx.Hidden {
			n++
			if d := ly.Params.Act.Dropout; d != 0.2 {
				t.Errorf("%s Dropout = %g, want 0.2", ly.Name, d)
			}
		}
	}
	if n == 0 {
		t.Errorf("no hidden layers")
	}
}
