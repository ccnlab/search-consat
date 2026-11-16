// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package combsearchff

import (
	"cogentcore.org/core/core"
	"cogentcore.org/core/math32/vecint"
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

	// EncodeSize is the size of each per-slot Encode layer.
	EncodeSize vecint.Vector2i `default:"{'X':8,'Y':8}" nest:"+"`

	// HiddenSize is the size of the hidden layer.
	HiddenSize vecint.Vector2i `default:"{'X':12,'Y':12}" nest:"+"`

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

	// TestInterval is how often (in epochs) to run through all the test patterns,
	// in terms of training epochs. Can use 0 or -1 for no testing.
	TestInterval int `default:"10"`

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
	cfg.Name = "CombSearchFF"
	cfg.Title = "EgoMLX feedforward combinatorial search"
	cfg.URL = "https://github.com/ccnlab/search-consat/blob/main/sims/combsearch_ff/README.md"
	cfg.Doc = "A feedforward backprop baseline for the combsearch task: per-slot Input -> Encode -> Output columns, coordinated through a shared Hidden layer."
}
