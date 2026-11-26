// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflictrnn

import "github.com/emer/emergent/v2/params"

// Search is the parameter search: each job runs the sim with one of
// these values of one parameter, with the rest at their configured
// values (the base config, including config.toml and any -args given
// to the simmer search). Run the jobs with the simmer ParamSearch
// function, which launches one job per value with -search-at 1..N
// (see emergent params/README.md). List the jobs with:
//
//	./conflict_rnn/conflict_rnn -nogui -search-n
var Search = params.Searches[*Config]{
	// job 1: the base config, with the same args as the other jobs
	params.Baseline[*Config](),

	// From the conflict_ff searches: test dropout grades the choices only
	// by making the output noisy (at .8: In67 .88, MatchCenter 2.2, with
	// MatchCos .84 and SecondRatio .24, vs axon pfail .4: In67 .67,
	// MatchCenter 1.0, MatchCos .86, SecondRatio .07), and training
	// Dropout and WeightDecay make the choices more deterministic. So this
	// searches the carryover (for MinorPrevMinor), test dropout up to .9,
	// and the recurrent competition that could give axon's clean commitment
	// to one group (low SecondRatio) with proportional choices
	// (MatchCenter near 1).

	// carryover: the proportion of the previous trial's final activity
	// that decays before the next trial: 1 (no carryover) in the base
	// config. 0 and .5 were about the same, so the sensitive range is
	// likely between .5 and 1.
	{Sel: "Params.Decay", Set: func(cfg *Config, v float32) {
		cfg.Params.Decay = params.Float64(v)
	}, Vals: func() []float32 {
		return params.Range(0.5, 0.95, 0.05)
	}},

	// test-only dropout of the hidden units, new on every test trial,
	// like test-time PFail in axon: 0 in the base config. Up to .9,
	// because ff needed .8 to reach the axon intrusion rate.
	{Sel: "Params.TestDropout", Set: func(cfg *Config, v float32) {
		cfg.Params.TestDropout = params.Float64(v)
	}, Vals: func() []float32 {
		return params.Range(0.1, 0.9, 0.1)
	}},

	// Gaussian noise SD on the hidden net inputs, in training and testing:
	// .2 in the base config. A wide range, because the network learns to
	// override noise up to 4.
	{Sel: "Params.Noise", Set: func(cfg *Config, v float32) {
		cfg.Params.Noise = params.Float64(v)
	}, Vals: func() []float32 {
		return []float32{0, 0.1, 0.5, 1, 2, 4}
	}},

	// lateral connections within the Output layer (none in the base
	// config): competition among the output groups during settling
	{Sel: "Params.OutputLateral", Set: func(cfg *Config, v float32) {
		cfg.Params.OutputLateral = v > 0
	}, Vals: func() []float32 {
		return []float32{1}
	}},

	// lateral connections within the hidden layers (none in the base config)
	{Sel: "Params.Lateral", Set: func(cfg *Config, v float32) {
		cfg.Params.Lateral = v > 0
	}, Vals: func() []float32 {
		return []float32{1}
	}},

	// cycles of settling per trial: 10 in the base config
	{Sel: "Params.NCycles", Set: func(cfg *Config, v float32) {
		cfg.Params.NCycles = int(v)
	}, Vals: func() []float32 {
		return []float32{5, 20}
	}},
}

// Search1 is the first version of the search (43 jobs, not run), kept for
// the record: it also had training Dropout (.05-.5) and WeightDecay
// (.05-.3), which made the conflict_ff choices more deterministic, and
// TestDropout only up to .5.
var Search1 = params.Searches[*Config]{
	// job 1: the base config, with the same args as the other jobs
	params.Baseline[*Config](),

	// carryover: the proportion of the previous trial's final activity
	// that decays before the next trial: 1 (no carryover) in the base
	// config. 0 and .5 were about the same, so the sensitive range is
	// likely between .5 and 1.
	{Sel: "Params.Decay", Set: func(cfg *Config, v float32) {
		cfg.Params.Decay = params.Float64(v)
	}, Vals: func() []float32 {
		return params.Range(0.5, 0.95, 0.05)
	}},

	// Gaussian noise SD on the hidden net inputs, in training and testing:
	// .2 in the base config. A wide range, because the network learns to
	// override noise up to 4.
	{Sel: "Params.Noise", Set: func(cfg *Config, v float32) {
		cfg.Params.Noise = params.Float64(v)
	}, Vals: func() []float32 {
		return []float32{0, 0.1, 0.5, 1, 2, 4}
	}},

	// test-only dropout of the hidden units, new on every test trial,
	// like test-time PFail in axon: 0 in the base config
	{Sel: "Params.TestDropout", Set: func(cfg *Config, v float32) {
		cfg.Params.TestDropout = params.Float64(v)
	}, Vals: func() []float32 {
		return params.Range(0.05, 0.5, 0.05)
	}},

	// training-only dropout of the hidden units: 0 in the base config
	{Sel: "Params.Dropout", Set: func(cfg *Config, v float32) {
		cfg.Params.Dropout = params.Float64(v)
	}, Vals: func() []float32 {
		return params.Range(0.05, 0.5, 0.05)
	}},

	// AdamW weight decay: 0 (Adam) in the base config. It was unstable
	// with carryover (Decay .5) and ReLU hidden layers.
	{Sel: "Params.WeightDecay", Set: func(cfg *Config, v float32) {
		cfg.Params.WeightDecay = params.Float64(v)
	}, Vals: func() []float32 {
		return params.Range(0.05, 0.3, 0.05)
	}},
}
