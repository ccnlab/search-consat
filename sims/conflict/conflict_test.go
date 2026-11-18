// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflict

import (
	"fmt"
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
	for _, nm := range []string{"Test_In100", "Test_InAbsDiff"} {
		v := ss.Stats.Dir("Train/Run").Float64(nm)
		if v.Len() != 1 || math.IsNaN(v.Values[0]) {
			t.Errorf("%s not copied to Train Run: %v", nm, v.Values)
		}
	}
	in100 := ss.Stats.Dir("Test/Epoch").Float64("In100").Float1D(-1)
	p100 := ss.Stats.Dir("Test/Epoch").Float64("P100").Float1D(-1)
	if in100 < p100 {
		t.Errorf("In100 %g < P100 %g: leaving out blends and intrusions cannot lower it", in100, p100)
	}
	// the slot stats for each slot are logged, and SlotRange is copied
	nslots := ss.Envs.ByModeDi(Test, 0).(*cfenv.CFEnv).Configs.NSlots
	for sl := range nslots {
		if ss.Stats.Dir("Test/Epoch").Float64(cfenv.SlotName(sl)).Len() == 0 {
			t.Errorf("no %s stat", cfenv.SlotName(sl))
		}
	}
	if v := ss.Stats.Dir("Train/Run").Float64("Test_SlotRange"); v.Len() != 1 || math.IsNaN(v.Values[0]) {
		t.Errorf("Test_SlotRange not copied to Train Run: %v", v.Values)
	}
	// silent layer and test activity stats, and their copies
	te := ss.Stats.Dir("Test/Epoch")
	for _, nm := range []string{"Hidden1_Silent", "Output_Silent", "Encoder1_Silent", "Hidden1_ActMAvg", "Hidden1_ActMMax", "Hidden1_MaxGeM"} {
		v := te.Float64(nm).Float1D(-1)
		cp := ss.Stats.Dir("Train/Run").Float64("Test_" + nm)
		t.Logf("Test %s = %.3f", nm, v)
		if math.IsNaN(v) || v < 0 || cp.Len() != 1 || cp.Values[0] != v {
			t.Errorf("%s: %g, Train Run copy %v", nm, v, cp.Values)
		}
	}
	if te.Float64("Hidden1_MaxGeM").Float1D(-1) <= 0 {
		t.Errorf("Hidden1_MaxGeM is 0 in testing")
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

// synState returns copies of the LWt and SWt values of all synapses,
// and the mask of failed (zero) weights, syncing from the GPU if used.
func synState(nsyn int) (lwt, swt []float32, fail []bool) {
	if axon.UseGPU {
		axon.RunOneSynFailSyn(nsyn, axon.SynapsesVar) // one more resample, and sync
	}
	lwt, swt, fail = make([]float32, nsyn), make([]float32, nsyn), make([]bool, nsyn)
	for i := range nsyn {
		lwt[i] = axon.Synapses.Value2D(i, int(axon.LWt))
		swt[i] = axon.Synapses.Value2D(i, int(axon.SWt))
		fail[i] = axon.Synapses.Value2D(i, int(axon.Wt)) == 0
	}
	return
}

// TestSynFail checks that testing with TestSynFail resamples the synaptic
// failures without any learning or slow adaptation: the learned weights,
// the layer inhibition multipliers and the slow counters are unchanged by a
// full test, and repeated resampling gives new failures at the expected rate.
func TestSynFail(t *testing.T) {
	// add PFail to the Base sheet (applied for both Train and Test),
	// independent of the current params.
	base := PathParams["Base"]
	orig := *base
	*base = append(*base, &axon.PathSel{Sel: "Path", Doc: "TestSynFail",
		Set: func(pt *axon.PathParams) {
			pt.Com.PFail = 0.02
			pt.Com.PFailSWt.SetBool(true)
		}})
	defer func() { *base = orig }()
	for _, gpu := range []bool{false, true} {
		cfg, _ := egui.NewConfig[Config]()
		cfg.GPU = gpu
		cfg.GUI = false
		cfg.Run.NData = 4
		cfg.Run.Trials = 64
		cfg.Params.TestSynFail = true
		cfg.Env.Env = map[string]any{"TestPerLevel": 8, "TestReps": 4}
		ss := &Sim{}
		ss.SetConfig(cfg)
		ss.ConfigSim()
		ss.Init()
		ss.Loops.Step(Train, 2, Epoch)
		nsyn := int(ss.Net.NetIxs().NSyns)
		ctx := ss.Net.Context()
		slow, adapt := ctx.SlowCounter, ctx.AdaptGiCounter
		var gis []float32
		for li := range ss.Net.Layers {
			gis = append(gis, axon.LayerStates.Value(li, 0, int(axon.LayerGiMult)))
		}
		lwt0, swt0, fail0 := synState(nsyn)
		ss.TestAll()
		lwt1, swt1, fail1 := synState(nsyn)
		if !slices.Equal(lwt0, lwt1) || !slices.Equal(swt0, swt1) {
			t.Errorf("GPU %v: learned weights changed by testing", gpu)
		}
		if ctx.SlowCounter != slow || ctx.AdaptGiCounter != adapt {
			t.Errorf("GPU %v: slow counters changed by testing: %d %d -> %d %d", gpu, slow, adapt, ctx.SlowCounter, ctx.AdaptGiCounter)
		}
		for li := range ss.Net.Layers {
			if g := axon.LayerStates.Value(li, 0, int(axon.LayerGiMult)); g != gis[li] {
				t.Errorf("GPU %v: layer %d GiMult changed by testing: %g -> %g", gpu, li, gis[li], g)
			}
		}
		// expected proportion of failures: PFail * (1 - SWt) for PFailSWt
		var expect float64
		nfail, ndiff := 0, 0
		for i := range nsyn {
			pti := axon.SynapseIxs.Value2D(i, int(axon.SynPathIndex))
			expect += float64(ss.Net.Paths[pti].Params.Com.WtFailP(swt1[i]))
			if fail1[i] {
				nfail++
			}
			if fail0[i] != fail1[i] {
				ndiff++
			}
		}
		expect /= float64(nsyn)
		got := float64(nfail) / float64(nsyn)
		t.Logf("GPU %v: %d synapses, failed %.4f (expected %.4f), changed between samples %.4f", gpu, nsyn, got, expect, float64(ndiff)/float64(nsyn))
		if expect == 0 || math.Abs(got-expect) > 0.2*expect || ndiff == 0 {
			t.Errorf("GPU %v: failures %.4f, expected %.4f, changed %d", gpu, got, expect, ndiff)
		}
		if gpu {
			axon.GPURelease()
		}
	}
}

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
		cfg.GPU = false
		cfg.GUI = false
		cfg.Run.NData = 4
		cfg.Run.Trials = 16
		cfg.Run.Epochs = 1
		cfg.Run.Runs = 1
		cfg.Params.NEncoder, cfg.Params.NHidden = c.nenc, c.nhid
		cfg.Params.EncoderLateral, cfg.Params.Shortcut, cfg.Params.EncoderBack = c.encLat, c.short, c.encBack
		cfg.Params.OutputLateral = c.outLat
		cfg.Params.Lateral = false
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
