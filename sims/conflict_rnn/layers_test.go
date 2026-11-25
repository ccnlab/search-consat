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
	if len(lbls) != Search.NumParams() || lbls[0] != "Search 1: Baseline" || lbls[1] != "Search 2: Params.Decay=0.5" || lbls[24] != "Search 25: Params.TestDropout=0.4" {
		t.Fatalf("Search labels: %q", lbls)
	}
	for _, at := range []int{0, 1, 24} {
		cfg, _ := egui.NewConfig[Config]()
		cfg.GUI = false
		cfg.Run.NData = 2
		lbl, err := Search.ApplyTo(cfg, at)
		if err != nil || lbl != lbls[at] {
			t.Fatalf("ApplyTo %d: %q %v", at, lbl, err)
		}
		ss := &Sim{}
		ss.SetConfig(cfg)
		ss.ConfigSim()
		ss.Init()
		if want := map[int]float64{0: 1, 1: 0.5}[at]; at < 2 && ss.Net.Params.Decay != want {
			t.Errorf("job %d: Decay = %g, want %g", at+1, ss.Net.Params.Decay, want)
		}
		for _, ly := range ss.Net.Layers {
			if ly.Type != egomlx.Hidden {
				continue
			}
			want := 0.0
			if at == 24 {
				want = 0.4
			}
			if d := ly.Params.Act.TestDropout; d != want {
				t.Errorf("job %d: %s TestDropout = %g, want %g", at+1, ly.Name, d, want)
			}
		}
	}
}
