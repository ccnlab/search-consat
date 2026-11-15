// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// package pcombff is a feedforward backprop baseline for the pcomb
// probabilistic combination task, built with the egomlx (gomlx) framework,
// using the same pcombenv environment. It replicates the axon pcomb
// network without the top-down back pathways: Value and Prob cue inputs
// project to their own encoding layers, which project to a Hidden layer,
// which projects to the Output value and Salience target layers.
package pcombff

//go:generate core generate -add-types -add-funcs

import (
	"fmt"
	"os"
	"reflect"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/metadata"
	"cogentcore.org/core/base/reflectx"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/base/mpi"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorfs"
	"github.com/ccnlab/search-consat/sims/pcomb/pcombenv"
	"github.com/emer/egomlx/egomlx"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/paths"
)

// Modes are the looping modes (Stacks) for running and statistics.
type Modes int32 //enums:enum

const (
	// Train mode: run the training pass of the model.
	Train Modes = iota

	// Test mode: run the testing pass of the model.
	Test
)

// Levels are the looping levels for running and statistics.
type Levels int32 //enums:enum

const (
	// Cycle is the shortest time-scale: a single forward pass.
	Cycle Levels = iota

	// Trial is one input pattern (or batch of NData patterns) processed.
	Trial

	// Epoch is one complete pass through the full set of trials.
	Epoch

	// Run is one complete training run, with its own random seed.
	Run

	// Expt is the outermost experiment level.
	Expt
)

const (
	// Start initializes stats, in start arg of StatFuncs call.
	Start = true
	// Step is an iteration of stats, in start arg of StatFuncs call.
	Step = false
)

// see params.go for params, config.go for config

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// simulation configuration parameters -- set by .toml config file and / or args
	Config *Config `new-window:"+"`

	// Net is the network: click to view / edit parameters for layers, paths, etc.
	Net *egomlx.Network `new-window:"+" display:"no-inline"`

	// Params manages network parameter setting.
	Params egomlx.Params `display:"inline"`

	// Loops are the control loops for running the sim, in different Modes
	// across stacks of Levels.
	Loops *looper.Stacks `new-window:"+" display:"no-inline"`

	// Envs provides mode-string based storage of environments.
	Envs env.Envs `new-window:"+" display:"no-inline"`

	// TrainUpdate has Train mode netview update parameters.
	TrainUpdate egomlx.NetViewUpdate `display:"inline"`

	// TestUpdate has Test mode netview update parameters.
	TestUpdate egomlx.NetViewUpdate `display:"inline"`

	// Root is the root tensorfs directory, where all stats and other misc sim data goes.
	Root *tensorfs.Node `display:"-"`

	// Stats has the stats directory within Root.
	Stats *tensorfs.Node `display:"-"`

	// Current has the current stats values within Stats.
	Current *tensorfs.Node `display:"-"`

	// StatFuncs are statistics functions called at given mode and level,
	// to perform all stats computations. phase = Start does init at start of given level,
	// and all intialization / configuration (called during Init too).
	StatFuncs []func(mode enums.Enum, level enums.Enum, start bool) `display:"-"`

	// GUI manages all the GUI elements
	GUI egui.GUI `display:"-"`

	// RandSeeds is a list of random seeds to use for each run.
	RandSeeds randx.Seeds `display:"-"`
}

// SetConfig sets the sim config.
func (ss *Sim) SetConfig(cfg *Config) { ss.Config = cfg }

// Body returns the [core.Body] used by the sim.
func (ss *Sim) Body() *core.Body { return ss.GUI.Body }

// ConfigSim initializes the config-dependent parts of the sim.
func (ss *Sim) ConfigSim() {
	ss.Root, _ = tensorfs.NewDir("Root")
	tensorfs.CurRoot = ss.Root
	ss.Net = egomlx.NewNetwork(ss.Config.Name)
	ss.Params.Config(NetworkParams, LayerParams, PathParams, ss.Config.Params.Sheet, ss.Config.Params.Tag, "github.com/ccnlab/search-consat/sims/pcomb_ff", reflect.ValueOf(ss))
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLoops()
	ss.ConfigStats()
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
}

// ConfigEnv configures the train and test environments: one per
// data-parallel index, exactly as in the axon pcomb model.
func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	newEnv := (len(ss.Envs) == 0)

	ndata := ss.Config.Run.NData

	for di := range ndata {
		var trn, tst *pcombenv.PCombEnv
		if newEnv {
			trn = &pcombenv.PCombEnv{}
			tst = &pcombenv.PCombEnv{}
		} else {
			trn = ss.Envs.ByModeDi(Train, di).(*pcombenv.PCombEnv)
			tst = ss.Envs.ByModeDi(Test, di).(*pcombenv.PCombEnv)
		}

		// note: names must be standard here!
		trn.Name = env.ModeDi(Train, di)
		trn.Defaults()
		if ss.Config.Env.Env != nil {
			reflectx.SetFieldsFromMap(&trn.Configs, ss.Config.Env.Env)
		}
		trn.Config(ss.Config.Run.Trials, ndata, di, false, 73+int64(di)*73)

		tst.Name = env.ModeDi(Test, di)
		tst.Defaults()
		if ss.Config.Env.Env != nil {
			reflectx.SetFieldsFromMap(&tst.Configs, ss.Config.Env.Env)
		}
		tst.Config(ss.Config.Run.Trials, ndata, di, true, 181+int64(di)*181)

		trn.Init(0)
		tst.Init(0)

		// note: names must be in place when adding
		ss.Envs.Add(trn, tst)
	}
}

// ConfigNet configures the network structure, replicating the axon pcomb
// network with feedforward paths only: Value -> ValueEnc and Prob -> ProbEnc,
// both encoders -> Hidden, and Hidden -> Output and Salience targets.
func (ss *Sim) ConfigNet(net *egomlx.Network) {
	net.SetMaxData(ss.Config.Run.NData)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	ev := ss.Envs.ByModeDi(Train, 0).(*pcombenv.PCombEnv)
	cfg := &ev.Configs
	nc := cfg.NCues
	npu := cfg.PopCodeUnits
	esz := ss.Config.Params.EncodeSize
	hsz := ss.Config.Params.HiddenSize

	val := net.AddLayer4D("Value", egomlx.Input, nc, 1, 1, npu)
	prob := net.AddLayer4D("Prob", egomlx.Input, nc, 1, 1, npu)
	venc := net.AddLayer2D("ValueEnc", egomlx.Hidden, esz.Y, esz.X).AddLayerClass("Encode")
	penc := net.AddLayer2D("ProbEnc", egomlx.Hidden, esz.Y, esz.X).AddLayerClass("Encode")
	hid := net.AddLayer2D("Hidden", egomlx.Hidden, hsz.Y, hsz.X)
	out := net.AddLayer2D("Output", egomlx.Target, 1, npu)
	sal := net.AddLayer2D("Salience", egomlx.Target, 1, npu)

	prob.PlaceRightOf(val, 2)
	venc.PlaceAbove(val)
	penc.PlaceRightOf(venc, 2)
	hid.PlaceAbove(venc)
	out.PlaceAbove(hid)
	sal.PlaceBehind(out, 2)

	full := paths.NewFull()

	net.ConnectLayers(val, venc, full, egomlx.Forward)
	// net.ConnectLayers(prob, penc, full, egomlx.Forward)
	net.ConnectLayers(venc, hid, full, egomlx.Forward)
	// net.ConnectLayers(penc, hid, full, egomlx.Forward)
	net.ConnectLayers(hid, out, full, egomlx.Forward)
	net.ConnectLayers(hid, sal, full, egomlx.Forward)

	err := net.Build()
	if errors.Log(err) != nil {
		return
	}
	net.Defaults()
	ss.ApplyParams()
	net.InitWeights()
}

// ApplyParams applies the network parameters, including any script.
func (ss *Sim) ApplyParams() {
	ss.Params.Script = ss.Config.Params.Script
	ss.Params.ApplyAll(ss.Net)
}

////////  Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Loops.ResetCounters()
	ss.SetRunName()
	ss.InitRandSeed(0)
	ss.ApplyParams()
	ss.StatsInit()
	ss.NewRun()
	ss.TrainUpdate.RecordSyns()
	ss.TrainUpdate.Update(Train, Trial)
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run, ss.Net.Rand)
}

// NetViewUpdater returns the NetViewUpdate for given mode.
func (ss *Sim) NetViewUpdater(mode enums.Enum) *egomlx.NetViewUpdate {
	if mode.Int64() == Train.Int64() {
		return &ss.TrainUpdate
	}
	return &ss.TestUpdate
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	trials := int(math32.IntMultipleGE(float32(ss.Config.Run.Trials), float32(ss.Config.Run.NData)))
	cycles := ss.Config.Run.Cycles

	ls.AddStack(Train, Trial).
		AddLevel(Expt, 1).
		AddLevel(Run, ss.Config.Run.Runs).
		AddLevel(Epoch, ss.Config.Run.Epochs).
		AddLevelIncr(Trial, trials, ss.Config.Run.NData).
		AddLevel(Cycle, cycles)

	ls.AddStack(Test, Trial).
		AddLevel(Epoch, 1).
		AddLevelIncr(Trial, trials, ss.Config.Run.NData).
		AddLevel(Cycle, cycles)

	egomlx.LooperStandard(ls, ss.Net, ss.NetViewUpdater, Cycle, Trial, Train)

	ls.Stacks[Train].OnInit.Add("Init", ss.Init)

	ls.AddOnStartToLoop(Trial, "ApplyInputs", func(mode enums.Enum) {
		ss.ApplyInputs(mode.(Modes))
	})

	ls.Loop(Train, Run).OnStart.Add("NewRun", ss.NewRun)

	trainEpoch := ls.Loop(Train, Epoch)
	trainEpoch.OnStart.Add("TestAtInterval", func() {
		if (ss.Config.Run.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.Run.TestInterval == 0) {
			ss.TestAll()
		}
	})

	ls.AddOnStartToAll("StatsStart", ss.StatsStart)
	ls.AddOnEndToAll("StatsStep", ss.StatsStep)

	ls.Loop(Train, Run).OnEnd.Add("SaveWeights", func() {
		if !ss.Config.Log.SaveWeights {
			return
		}
		ctrString := fmt.Sprintf("%03d_%05d", ls.Loop(Train, Run).Counter.Cur, ls.Loop(Train, Epoch).Counter.Cur)
		fn := core.Filename("weights_" + ctrString + "_" + ss.RunName() + ".json")
		if err := ss.Net.SaveWeightsJSON(fn); errors.Log(err) != nil {
			return
		}
		mpi.Printf("Saving weights to %s\n", fn)
	})

	if ss.Config.GUI {
		egomlx.LooperUpdateNetView(ls, Cycle, Trial, ss.NetViewUpdater)

		ls.Stacks[Train].OnInit.Add("GUI-Init", ss.GUI.UpdateWindow)
		ls.Stacks[Test].OnInit.Add("GUI-Init", ss.GUI.UpdateWindow)
	}

	if ss.Config.Debug {
		mpi.Println(ls.DocString())
	}
	ss.Loops = ls
}

// ApplyInputs applies input patterns from given environment for given mode,
// for all the data-parallel items: the env Value and Prob states go to the
// input layers, and the Output and Salience states are the targets.
func (ss *Sim) ApplyInputs(mode Modes) {
	net := ss.Net
	ndata := net.NData
	curModeDir := ss.Current.Dir(mode.String())
	lays := net.LayersByType(egomlx.Input, egomlx.Target)
	net.InitExt()
	for di := range ndata {
		ev := ss.Envs.ByModeDi(mode, di)
		ev.Step()
		curModeDir.StringValue("TrialName", ndata).SetString1D(ev.String(), di)
		for _, lnm := range lays {
			ly := net.LayerByName(lnm)
			st := ev.State(lnm)
			if st != nil {
				ly.ApplyExt(di, st)
			}
		}
	}
	net.ApplyExts()
}

// NewRun initializes a new Run level of the model.
func (ss *Sim) NewRun() {
	run := ss.Loops.Loop(Train, Run).Counter.Cur
	ss.InitRandSeed(run)
	for di := range ss.Net.NData {
		ss.Envs.ByModeDi(Train, di).Init(run)
		ss.Envs.ByModeDi(Test, di).Init(run)
	}
	ss.Net.InitWeights()
	if ss.Config.Run.StartWeights != "" {
		if err := ss.Net.OpenWeightsJSON(core.Filename(ss.Config.Run.StartWeights)); errors.Log(err) != nil {
			return
		}
		mpi.Printf("Starting with initial weights from: %s\n", ss.Config.Run.StartWeights)
	}
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	for di := range ss.Net.NData {
		ss.Envs.ByModeDi(Test, di).Init(0)
	}
	ss.Loops.ResetAndRun(Test)
	ss.Loops.Mode = Train // important because this is called from Train Run: go back.
}

//////// Stats

// AddStatStd adds a standard stat compute function.
func (ss *Sim) AddStatStd(f func(mode enums.Enum, level enums.Enum, start bool)) {
	ss.StatFuncs = append(ss.StatFuncs, f)
}

// AddStat adds a custom stat compute function.
func (ss *Sim) AddStat(f func(mode Modes, level Levels, start bool)) {
	ss.AddStatStd(func(mode enums.Enum, level enums.Enum, start bool) {
		f(mode.(Modes), level.(Levels), start)
	})
}

// StatsStart is called by Looper at the start of given level, for each iteration.
// It needs to call RunStats Start at the next level down.
// e.g., each Epoch is the start of the full set of Trial Steps.
func (ss *Sim) StatsStart(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	if level <= Trial {
		return
	}
	unlock := ss.GUI.LockGUI()
	ss.RunStats(mode, level-1, Start)
	unlock()
}

// StatsStep is called by Looper at each step of iteration,
// where it accumulates the stat results.
func (ss *Sim) StatsStep(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	if level == Cycle {
		return
	}
	unlock := ss.GUI.LockGUI()
	ss.RunStats(mode, level, Step)
	unlock()
	tensorfs.DirTable(egomlx.StatsNode(ss.Stats, mode, level), nil).WriteToLog()
	ss.UpdatePlots(mode, level)
}

// RunStats runs the StatFuncs for given mode, level and phase.
func (ss *Sim) RunStats(mode Modes, level Levels, start bool) {
	for _, sf := range ss.StatFuncs {
		sf(mode, level, start)
	}
}

// UpdatePlots updates the plots for given mode and level.
// Must be called outside of lockGUI, as the plot update takes the lock.
func (ss *Sim) UpdatePlots(mode Modes, level Levels) {
	if ss.GUI.Tabs == nil {
		return
	}
	tbs := ss.GUI.Tabs.AsLab()
	_, idx := tbs.CurrentTab()
	nm := mode.String() + " " + level.String() + " Plot"
	tbs.GoUpdatePlot(nm)
	if level == Run {
		tbs.GoUpdatePlot("Train RunAll Plot")
	}
	tbs.SelectTabIndex(idx)
}

// SetRunName sets the overall run name, used for naming output logs and weight files
// based on params extra sheets and tag, and starting run number (for distributed runs).
func (ss *Sim) SetRunName() string {
	runName := ss.Params.RunName(ss.Config.Run.Run)
	ss.Current.StringValue("RunName", 1).SetString1D(runName, 0)
	return runName
}

// RunName returns the overall run name, used for naming output logs and weight files.
func (ss *Sim) RunName() string {
	return ss.Current.StringValue("RunName", 1).String1D(0)
}

// StatsInit initializes all the stats by calling Start across all modes and levels,
// and makes the standard plot tabs for the run stats.
func (ss *Sim) StatsInit() {
	for md, st := range ss.Loops.Stacks {
		mode := md.(Modes)
		for _, lev := range st.Order {
			level := lev.(Levels)
			if level == Cycle {
				continue
			}
			ss.RunStats(mode, level, Start)
		}
	}
	if ss.GUI.Tabs != nil {
		tbs := ss.GUI.Tabs.AsLab()
		_, idx := tbs.CurrentTab()
		tbs.PlotTensorFS(egomlx.StatsNode(ss.Stats, Train, Epoch))
		tbs.PlotTensorFS(egomlx.StatsNode(ss.Stats, Train, Run))
		tbs.PlotTensorFS(egomlx.StatsNode(ss.Stats, Test, Epoch))
		tbs.PlotTensorFS(egomlx.StatsNode(ss.Stats, Test, Trial))
		tbs.PlotTensorFS(ss.Stats.Dir("Train/RunAll"))
		tbs.SelectTabIndex(idx)
	}
}

// ConfigStats configures the functions that do all the stats computation
// in the tensorfs system, mirroring the axon pcomb stats.
func (ss *Sim) ConfigStats() {
	net := ss.Net
	ss.Stats = ss.Root.Dir("Stats")
	ss.Current = ss.Stats.Dir("Current")

	ss.SetRunName()

	// last arg(s) are levels to exclude
	ss.AddStatStd(egomlx.StatLoopCounters(ss.Stats, ss.Current, ss.Loops, net, Trial, Cycle))
	ss.AddStatStd(egomlx.StatRunName(ss.Stats, ss.Current, ss.Loops, net, Trial, Cycle))
	ss.AddStatStd(egomlx.StatTrialName(ss.Stats, ss.Current, ss.Loops, net, Trial))
	ss.AddStatStd(egomlx.StatPerTrialMSec(ss.Stats, Train, Trial))

	ss.AddStat(ss.StatSampling())

	// get test stats in train log
	ss.AddStatStd(egomlx.StatCopyLevel(ss.Stats, Train, Test, Epoch, Run, func(s *plot.Style, stName string) {
		s.On = true
	}, "CueErr", "Miss", "PGain", "OutPErr", "CalibErr", "AgreeBoost"))

	lays := net.LayersByType(egomlx.Hidden, egomlx.Target)
	ss.AddStatStd(egomlx.StatLayerAct(ss.Stats, net, Train, Trial, Run, lays...))

	hidLays := net.LayersByType(egomlx.Hidden)
	pcaFunc := egomlx.StatPCA(ss.Stats, ss.Current, net, ss.Config.Run.PCAInterval, Train, Trial, Run, hidLays...)
	ss.AddStat(func(mode Modes, level Levels, start bool) {
		trnEpc := ss.Loops.Loop(Train, Epoch).Counter.Cur
		pcaFunc(mode, level, start, trnEpc)
	})

	ss.AddStatStd(egomlx.StatLevelAll(ss.Stats, Train, Run, func(s *plot.Style, cl tensor.Values) {
		name := metadata.Name(cl)
		switch name {
		case "CueErr", "Miss", "CalibErr", "Test_CueErr", "Test_Miss", "Test_CalibErr", "Test_PGain", "Test_AgreeBoost":
			s.On = true
			s.Range.SetMin(0)
		}
	}))
}

// StatCounters returns counters string to show at bottom of netview.
func (ss *Sim) StatCounters(mode, level enums.Enum) string {
	counters := ss.Loops.Stacks[mode].CountersString()
	vu := ss.NetViewUpdater(mode)
	if vu == nil || vu.View == nil {
		return counters
	}
	di := vu.View.Di
	counters += fmt.Sprintf(" Di: %d", di)
	curModeDir := ss.Current.Dir(mode.String())
	if curModeDir.Node("TrialName") == nil {
		return counters
	}
	counters += fmt.Sprintf(" TrialName: %s", curModeDir.StringValue("TrialName").String1D(di))
	statNames := []string{"Value", "CueErr", "Miss", "ChoseP"}
	if level == Cycle || curModeDir.Node(statNames[0]) == nil {
		return counters
	}
	for _, name := range statNames {
		counters += fmt.Sprintf(" %s: %.4g", name, curModeDir.Float64(name).Float1D(di))
	}
	return counters
}

//////// GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI(b tree.Node) {
	ss.GUI.MakeBody(b, ss, ss.Root, ss.Config.Name, ss.Config.Title, ss.Config.Doc)
	ss.GUI.StopLevel = Trial
	nv := ss.GUI.AddNetView("Network")
	nv.Settings.MaxRecs = 2 * ss.Config.Run.Cycles
	nv.Settings.Raster.Max = ss.Config.Run.Cycles
	nv.SetNet(ss.Net)
	ss.TrainUpdate.Config(nv, egomlx.Trial, ss.StatCounters)
	ss.TestUpdate.Config(nv, egomlx.Trial, ss.StatCounters)
	ss.GUI.OnStop = func(mode, level enums.Enum) {
		vu := ss.NetViewUpdater(mode)
		vu.UpdateWhenStopped(mode, level)
	}

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1, 2.75)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	ss.StatsInit()
	ss.GUI.FinalizeGUI(false)
}

// MakeToolbar makes the custom toolbar items.
func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "New Seed",
		Icon:    icons.Add,
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RandSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL(ss.Config.URL)
		},
	})
}

// RunNoGUI runs the sim without a GUI, in a headless / terminal mode.
func (ss *Sim) RunNoGUI() {
	ss.Init()

	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}
	if ss.Config.Log.SaveWeights {
		mpi.Printf("Saving final weights per run\n")
	}

	runName := ss.SetRunName()
	netName := ss.Net.Name
	cfg := &ss.Config.Log
	egomlx.OpenLogFiles(ss.Loops, ss.Stats, netName, runName, [][]string{cfg.Train, cfg.Test})

	mpi.Printf("Running %d Runs starting at %d: %s\n", ss.Config.Run.Runs, ss.Config.Run.Run, runName)
	ss.Loops.Loop(Train, Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.Runs)
	ss.Loops.Run(Train)

	egomlx.CloseLogFiles(ss.Loops, ss.Stats, Cycle)
}
