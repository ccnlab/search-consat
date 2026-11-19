// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflict

import (
	"cogentcore.org/core/core"
	"github.com/emer/emergent/v2/egui"
)

// EnvConfig has config params for environment
// note: only adding fields for key Env params that matter for both Network and Env
// other params are set via the Env map data mechanism.
type EnvConfig struct {
	// Env parameters: can set any field/subfield on Env struct,
	// using standard TOML formatting.
	Env map[string]any
}

// ParamConfig has config parameters related to sim params.
type ParamConfig struct {

	// NEncoder is the number of encoder layers (Encoder1, ...), in a stack
	// above the Input, each with one pool per input slot and pool-to-pool
	// (one-to-one) connections: Forward from the Input, and bidirectional
	// between encoder layers. 0 = none.
	NEncoder int `default:"2" min:"0"`

	// NHidden is the number of hidden layers (Hidden1, ...), fully connected
	// bidirectionally in a stack above the encoders (or the Input),
	// below the Output. 0 = none: the Output is connected to the last
	// encoder layer (or the Input).
	NHidden int `default:"1" min:"0"`

	// EncoderSize is the size of each encoder layer pool (square).
	EncoderSize int `default:"12"`

	// HiddenSize is the size of each hidden layer (square).
	HiddenSize int `default:"20"`

	// EncoderLateral adds lateral (self) connections within each pool of
	// each encoder layer (pool one-to-one).
	EncoderLateral bool

	// Lateral adds full lateral (self) connections within each hidden layer.
	HiddenLateral bool

	// OutputLateral adds full lateral (self) connections within the Output layer.
	OutputLateral bool `default:"true"`

	// Shortcut adds full Forward connections from the last encoder layer to
	// the Output, bypassing the hidden layers (class Shortcut). Only used
	// if there are encoder and hidden layers (otherwise the last encoder is
	// connected to the Output anyway).
	Shortcut bool `default:"true"`

	// EncoderBack adds full Back connections from the Output to the last
	// encoder layer (class EncoderBack, also BackPath), with Shortcut or
	// without. Only used if there are encoder and hidden layers.
	EncoderBack bool `default:"true"`

	// TestSynFail resamples the random synaptic failures (Com.PFail) after each
	// testing trial, without any learning or slow adaptation (Network.SynFail),
	// so that testing has new failures on each trial, as training does.
	// Otherwise the failures from the last training trial stay fixed
	// throughout testing. Turn this on if Com.PFail > 0.
	TestSynFail bool

	// Script is an interpreted script that is run to set parameters in Layer and Path
	// sheets, by default using the "Script" set name.
	Script string `new-window:"+" width:"100"`

	// Sheet is the extra params sheet name(s) to use (space separated
	// if multiple). Must be valid name as listed in compiled-in params
	// or loaded params.
	Sheet string

	// Tag is an extra tag to add to file names and logs saved from this run.
	Tag string

	// Note is additional info to describe the run params etc,
	// like a git commit message for the run.
	Note string

	// SaveAll will save a snapshot of all current param and config settings
	// in a directory named params_<datestamp> (or _good if Good is true),
	// then quit. Useful for comparing to later changes and seeing multiple
	// views of current params.
	SaveAll bool `nest:"+"`

	// Good is for SaveAll, save to params_good for a known good params state.
	// This can be done prior to making a new release after all tests are passing.
	// Add results to git to provide a full diff record of all params over level.
	Good bool `nest:"+"`
}

func (pc *ParamConfig) FieldWidget(field string) core.Value {
	return egui.ScriptFieldWidget(field)
}

// RunConfig has config parameters related to running the sim.
type RunConfig struct {

	// GPUDevice selects the gpu device to use.
	GPUDevice int

	// NData is the number of data-parallel items to process in parallel per trial.
	// Is significantly faster for both CPU and GPU. Results in an effective
	// mini-batch of learning.
	NData int `default:"16" min:"1"`

	// NThreads is the number of parallel threads for CPU computation;
	// 0 = use default.
	NThreads int `default:"0"`

	// Run is the _starting_ run number, which determines the random seed.
	// Runs counts up from there. Can do all runs in parallel by launching
	// separate jobs with each starting Run, Runs = 1.
	Run int `default:"0" flag:"run"`

	// Runs is the total number of runs to do when running Train, starting from Run.
	Runs int `default:"5" min:"1"`

	// Epochs is the total number of epochs per run.
	Epochs int `default:"50"`

	// Trials is the total number of trials per epoch.
	// Should be an even multiple of NData.
	Trials int `default:"512"`

	// ISICycles is the number of no-input inter-stimulus interval
	// cycles at the start of the trial.
	ISICycles int `default:"0"`

	// MinusCycles is the number of cycles in the minus phase per trial.
	MinusCycles int `default:"150"`

	// PlusCycles is the number of cycles in the plus phase per trial.
	PlusCycles int `default:"50"`

	// TestInterval is how often (in epochs) to run through all the test
	// items during training, in addition to the final test at the end of
	// each run. Can use 0 or -1 for only the final test.
	TestInterval int `default:"0"`

	// PCAInterval is how often (in epochs) to compute PCA on hidden
	// representations to measure variance.
	PCAInterval int `default:"10"`

	// StartWeights is the name of weights file to load at start of first run.
	StartWeights string
}

// Cycles returns the total number of cycles per trial: ISI + Minus + Plus.
func (rc *RunConfig) Cycles() int {
	return rc.ISICycles + rc.MinusCycles + rc.PlusCycles
}

// LogConfig has config parameters related to logging data.
type LogConfig struct {

	// SaveWeights will save final weights after each run.
	SaveWeights bool

	// Train has the list of Train mode levels to save log files for.
	Train []string `default:"['Expt', 'Run', 'Epoch']" nest:"+"`

	// Test has the list of Test mode levels to save log files for.
	Test []string `default:"['Epoch']" nest:"+"`
}

// Config has the overall Sim configuration options.
type Config struct {
	egui.BaseConfig

	// Env has environment related configuration options.
	Env EnvConfig `display:"add-fields"`

	// Params has parameter related configuration options.
	Params ParamConfig `display:"add-fields"`

	// Run has sim running related configuration options.
	Run RunConfig `display:"add-fields"`

	// Log has data logging related configuration options.
	Log LogConfig `display:"add-fields"`
}

func (cfg *Config) Defaults() {
	cfg.Name = "Conflict"
	cfg.Title = "Axon conflict"
	cfg.URL = "https://github.com/ccnlab/search-consat/blob/main/sims/conflict/README.md"
	cfg.Doc = "This tests what the network produces when its inputs conflict: it is trained on consistent inputs that all map to the same output, and tested with inputs from two groups that map to different outputs."
}
