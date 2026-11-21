// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflictrnn

import (
	"cogentcore.org/core/core"
	"github.com/emer/egomlx/egomlx"
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
	// (one-to-one) connections: Forward from the Input, and Forward up and
	// Back down between encoder layers. 0 = none.
	NEncoder int `default:"0" min:"0"`

	// EncoderSize is the size of each encoder layer pool (square).
	EncoderSize int `default:"12"`

	// NHidden is the number of hidden layers (Hidden1, ...), in a fully
	// connected stack above the encoders (or the Input), below the Output,
	// with Forward paths up and Back paths down. 0 = none: the Output is
	// connected to the last encoder layer (or the Input).
	NHidden int `default:"2" min:"0"`

	// EncoderLateral adds lateral (self) connections within each pool of
	// each encoder layer (pool one-to-one).
	EncoderLateral bool

	// Shortcut adds full Forward connections from the last encoder layer to
	// the Output, bypassing the hidden layers. Only used if there are
	// encoder and hidden layers.
	Shortcut bool

	// EncoderBack adds full Back connections from the Output to the last
	// encoder layer. Only used if there are encoder and hidden layers.
	EncoderBack bool

	// HiddenFn is the activation function of the hidden layers.
	HiddenFn egomlx.ActFuns `default:"ReLU"`

	// NCycles is the number of cycles of recurrent settling per trial,
	// with the loss on the final cycle (backpropagation through time).
	NCycles int `default:"10" min:"1"`

	// Noise is the standard deviation of the Gaussian noise added to the
	// net input of the hidden layers on every cycle, in training and
	// testing: the only source of variability across repeated trials
	// (each trial starts from zero activity, with no carryover).
	Noise float64 `default:"0.2"`

	// Decay is the proportion of the final activity of the previous trial
	// that decays before the next trial: the first cycle starts from
	// (1 - Decay) times that activity, so the previous trial carries over
	// through the recurrent paths, as in the axon model. 1 = each trial
	// starts from zero (no carryover).
	Decay float64 `default:"1" min:"0" max:"1"`

	// Lateral adds full lateral (self) connections within each hidden layer.
	Lateral bool

	// OutputLateral adds full lateral (self) connections within the Output layer.
	OutputLateral bool

	// HiddenSize is the size of each hidden layer (square).
	HiddenSize int `default:"15"`

	// WeightDecay is the decoupled weight decay strength: if > 0, the
	// AdamW optimizer is used with this weight decay, which shrinks the
	// weights by LRate * WeightDecay on each step. 0 = Adam, no decay.
	WeightDecay float64

	// Dropout is the dropout probability for the hidden layer units, in
	// training only (on every cycle for the recurrent network). 0 = none.
	Dropout float64

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

// FieldWidget returns a special GUI widget for given field name.
func (pc *ParamConfig) FieldWidget(field string) core.Value {
	return egui.ScriptFieldWidget(field)
}

// RunConfig has config parameters related to running the sim.
type RunConfig struct {

	// Run is the _starting_ run number, which determines the random seed.
	// Runs counts up from there. Can do all runs in parallel by launching
	// separate jobs with each starting Run, Runs = 1.
	Run int `default:"0" flag:"run"`

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

	// RecordCycles reads back the activity of every layer on every cycle
	// of settling (egomlx Network.RecordCycles), for the test
	// within-trial competition stats (LeadSwaps, CommitCyc, ...; see
	// StatCompete) and the settling microscope (Test Microscope and
	// Test MicroscopeTrial tabs; see StatMicroscope), as in the axon model.
	RecordCycles bool `default:"true"`

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
	cfg.Name = "ConflictRNN"
	cfg.Title = "EgoMLX recurrent conflict"
	cfg.URL = "https://github.com/ccnlab/search-consat/blob/main/sims/conflict_rnn/README.md"
	cfg.Doc = "A recurrent backprop baseline for the conflict task: the same bidirectional network as the axon model, settling with noise, trained by backpropagation through time on consistent inputs, and tested with inputs from two groups."
}
