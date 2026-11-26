// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflictff

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

	// NEncoder is the number of encoder layers (Encoder1, ...), in a
	// feedforward stack above the Input, each with one pool per input slot
	// and pool-to-pool (one-to-one) connections. 0 = none.
	NEncoder int `default:"0" min:"0"`

	// EncoderSize is the size of each encoder layer pool (square).
	EncoderSize int `default:"12"`

	// NHidden is the number of hidden layers (Hidden1, ...), in a fully
	// connected feedforward stack above the encoders (or the Input), below
	// the Output. 0 = none: the Output is connected to the last encoder
	// layer (or the Input).
	NHidden int `default:"2" min:"0"`

	// HiddenSize is the size of each hidden layer (square).
	HiddenSize int `default:"15"`

	// Shortcut adds full connections from the last encoder layer to the
	// Output, bypassing the hidden layers. Only used if there are encoder
	// and hidden layers.
	Shortcut bool

	// WeightDecay is the decoupled weight decay strength: if > 0, the
	// AdamW optimizer is used with this weight decay, which shrinks the
	// weights by LRate * WeightDecay on each step. 0 = Adam, no decay.
	WeightDecay float64

	// Dropout is the dropout probability for the hidden layer units, in
	// training only (on every cycle for the recurrent network). 0 = none.
	Dropout float64

	// TestDropout is the dropout probability for the hidden layer units
	// in testing only, with a new random pattern on every test trial, so
	// the choice for the same item can vary across its repetitions, as
	// with the test-time synaptic failures (PFail) of the axon model.
	// It is independent of Dropout (training). 0 = none.
	TestDropout float64

	// Noise is the standard deviation of the Gaussian noise added to the
	// net input of the hidden layer units, in training and testing.
	// 0 = none.
	Noise float64

	// OutNoise is the standard deviation of the Gaussian noise added to the
	// net input of the Output layer units, in training and testing, which
	// acts on the choice more directly than Noise on the hidden layers
	// (whose noise is averaged over many units). 0 = none.
	OutNoise float64

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

	// SearchN prints the total number of parameter search jobs in
	// [Search] (search.go), then quits. The simmer ParamSearch function
	// calls this to find how many jobs to launch (from the first line).
	// The following lines are the labels of the jobs, in order.
	SearchN bool

	// SearchAt runs the parameter search job at this index in [Search]
	// (search.go), in the [1..N] range (inclusive of N), so that a
	// non-zero value means to do a parameter search. The label of the
	// job is saved in job.label.
	SearchAt int
}

// FieldWidget returns a special GUI widget for given field name.
func (pc *ParamConfig) FieldWidget(field string) core.Value {
	return egui.ScriptFieldWidget(field)
}

// RunConfig has config parameters related to running the sim.
type RunConfig struct {

	// Run is the _starting_ run number, which determines the random seed.
	// Runs counts up from there. Can do all runs in parallel by launching
	// separate jobs with each starting Run, Runs = 1.
	Run int `default:"0"`

	// Runs is the total number of runs to do when running Train, starting from Run.
	Runs int `default:"5" min:"1"`

	// Epochs is the total number of epochs per run.
	Epochs int `default:"100"`

	// Trials is the total number of trials per epoch.
	// Should be an even multiple of NData.
	Trials int `default:"512"`

	// NData is the number of data-parallel items to process in parallel per trial.
	// This is the batch size of the feedforward pass: an effective mini-batch
	// of input patterns processed through the network in parallel.
	NData int `default:"16" min:"1"`

	// Cycles is the total number of cycles per trial.
	// For a feedforward backprop model this is a single pass per trial.
	Cycles int `default:"1" min:"1"`

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

	// Backend is the gomlx compute backend, in the format of the
	// GOMLX_BACKEND environment variable, which it sets if not empty:
	// e.g., "xla:cpu", "xla:cuda", or "go" (the pure-Go CPU backend).
	// The default is xla:cpu, which is faster than the GPU for these small
	// networks (on the h200). If empty, GOMLX_BACKEND is used if set, else
	// XLA, which uses the GPU if there is one (cuda), else the CPU.
	Backend string

	// Env has environment related configuration options.
	Env EnvConfig `display:"add-fields"`

	// Params has parameter related configuration options.
	Params ParamConfig `display:"add-fields"`

	// Run has sim running related configuration options.
	Run RunConfig `display:"add-fields"`

	// Log has data logging related configuration options.
	Log LogConfig `display:"add-fields"`
}

// Defaults sets default values for the config.
func (cfg *Config) Defaults() {
	cfg.Backend = "xla:cpu" // a default tag would stop at the ":"
	cfg.Name = "ConflictFF"
	cfg.Title = "EgoMLX feedforward conflict"
	cfg.URL = "https://github.com/ccnlab/search-consat/blob/main/sims/conflict_ff/README.md"
	cfg.Doc = "A feedforward backprop baseline for the conflict task: trained on consistent inputs, tested with inputs from two groups that map to different outputs."
}
