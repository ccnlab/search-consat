// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflict

import (
	"math"
	"slices"
	"testing"

	"github.com/ccnlab/search-consat/sims/conflict/cfenv"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/egui"
)

// newTestSim returns a configured sim with the default config,
// on the CPU, without the GUI.
func newTestSim(t *testing.T, ndata int) *Sim {
	cfg, _ := egui.NewConfig[Config]()
	cfg.GPU = false
	cfg.GUI = false
	cfg.Run.NData = ndata
	ss := &Sim{}
	ss.SetConfig(cfg)
	ss.ConfigSim()
	ss.Init()
	return ss
}

// TestNetInputs checks that the values applied to the network's Input and
// Output layers for each data-parallel index are exactly the env states,
// over a number of training and testing trials.
func TestNetInputs(t *testing.T) {
	ndata := 4
	ss := newTestSim(t, ndata)
	inLy := ss.Net.LayerByName("Input")
	outLy := ss.Net.LayerByName("Output")
	var vals []float32
	for _, mode := range []Modes{Train, Test} {
		for range 20 {
			ss.ApplyInputs(mode)
			for di := range ndata {
				ev := ss.Envs.ByModeDi(mode, di).(*cfenv.CFEnv)
				vals = layerValues(inLy, axon.Ext, di, vals)
				if !slices.Equal(vals, ev.States["Input"].Values) {
					t.Fatalf("%s di %d: Input Ext != env Input: %s", mode, di, ev.String())
				}
				vals = layerValues(outLy, axon.Target, di, vals)
				if !slices.Equal(vals, ev.States["Output"].Values) {
					t.Fatalf("%s di %d: Output Target != env Output: %s", mode, di, ev.String())
				}
			}
		}
	}
}

// TestNetTrials runs training trials and checks that the plus phase
// activity of the Output layer is the target bar (the clamping works), and
// that the Input activity is the input pattern. It reports the mean plus
// phase activity of the target and other output units.
func TestNetTrials(t *testing.T) {
	ndata := 4
	ss := newTestSim(t, ndata)
	inLy := ss.Net.LayerByName("Input")
	outLy := ss.Net.LayerByName("Output")
	var vals []float32
	var onSum, offSum, offMax, inOn, inOff float32
	var nOn, nOff, nInOn, nInOff int
	ntrials := 10
	for range ntrials {
		ss.Loops.Step(Train, 1, Trial)
		for di := range ndata {
			ev := ss.Envs.ByModeDi(Train, di).(*cfenv.CFEnv)
			tg := ev.States["Output"].Values
			vals = layerValues(outLy, axon.ActP, di, vals)
			for i, v := range vals {
				if tg[i] > 0 {
					onSum += v
					nOn++
				} else {
					offSum += v
					offMax = max(offMax, v)
					nOff++
				}
			}
			ip := ev.States["Input"].Values
			vals = layerValues(inLy, axon.ActP, di, vals)
			for i, v := range vals {
				if ip[i] > 0 {
					inOn += v
					nInOn++
				} else {
					inOff += v
					nInOff++
				}
			}
		}
	}
	on, off := onSum/float32(nOn), offSum/float32(nOff)
	t.Logf("Output ActP: target units %.3f, other units %.3f (max %.3f)", on, off, offMax)
	t.Logf("Input ActP: pattern units %.3f, other units %.3f", inOn/float32(nInOn), inOff/float32(nInOff))
	if on < 0.5 || off > 0.05 {
		t.Errorf("Output plus phase is not the target bar: target units %.3f, other units %.3f", on, off)
	}
	if inOn/float32(nInOn) < 0.5 || inOff/float32(nInOff) > 0.05 {
		t.Errorf("Input activity is not the input pattern")
	}
}

// TestTestCopy runs a few short training epochs with a test after each,
// and checks that the test stats are copied into the Train Epoch and Run
// stats, and that the pattern distance matrix is computed.
func TestTestCopy(t *testing.T) {
	cfg, _ := egui.NewConfig[Config]()
	cfg.GPU = false
	cfg.GUI = false
	cfg.Run.NData = 4
	cfg.Run.Trials = 8
	cfg.Run.Epochs = 3
	cfg.Run.Runs = 1
	cfg.Run.TestInterval = 1
	cfg.Env.Env = map[string]any{"TestPerLevel": 8, "TestReps": 1}
	ss := &Sim{}
	ss.SetConfig(cfg)
	ss.ConfigSim()
	ss.Loops.Run(Train)
	ep := ss.Stats.Dir("Train/Epoch").Float64("Test_P100")
	t.Logf("Train Epoch Test_P100: %v, tests run: %d", ep.Values, ss.TestCount)
	if ep.Len() != 3 || !math.IsNaN(ep.Values[0]) || math.IsNaN(ep.Values[1]) || math.IsNaN(ep.Values[2]) {
		t.Errorf("Test_P100 not copied to Train Epoch: %v", ep.Values)
	}
	run := ss.Stats.Dir("Train/Run").Float64("Test_P100")
	if run.Len() != 1 || math.IsNaN(run.Values[0]) {
		t.Errorf("Test_P100 not copied to Train Run: %v", run.Values)
	}
	dm := ss.Stats.Dir("Patterns").Float64("DistMatrix")
	ni := ss.Envs.ByModeDi(Train, 0).(*cfenv.CFEnv).Configs.NInputs()
	if !slices.Equal(dm.ShapeSizes(), []int{ni, ni}) || dm.Float(0, 0) > 1e-6 || dm.Float(0, ni-1) < 0.5 {
		t.Errorf("DistMatrix: %v", dm.ShapeSizes())
	}
	if ss.Root.Dir("Patterns").Dir("Items").Node("G0") == nil {
		t.Errorf("patterns not saved")
	}
	props := ss.Stats.Dir("Test/SecondRatioHist").Float64("Prop").Values
	sum := 0.0
	for _, p := range props {
		sum += p
	}
	t.Logf("Test SecondRatio hist: %v", props)
	if len(props) != 10 || math.Abs(sum-1) > 1e-6 {
		t.Errorf("histogram proportions sum to %g", sum)
	}
}
