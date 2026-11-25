// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflictff

import "github.com/emer/emergent/v2/params"

// Search is the parameter search: each job runs the sim with one of
// these values of one parameter, with the rest at their configured
// values (the base config, including config.toml and any -args given
// to the simmer search). Run the jobs with the simmer ParamSearch
// function, which launches one job per value with -search-at 1..N
// (see emergent params/README.md). List the jobs with:
//
//	./conflict_ff/conflict_ff -nogui -search-n
var Search = params.Searches[*Config]{
	// job 1: the base config, with the same args as the other jobs
	params.Baseline[*Config](),

	// training-only dropout of the hidden units: 0 in the base config
	{Sel: "Params.Dropout", Set: func(cfg *Config, v float32) {
		cfg.Params.Dropout = params.Float64(v)
	}, Vals: func() []float32 {
		return params.Range(0.05, 0.5, 0.05)
	}},

	// test-only dropout of the hidden units, new on every test trial,
	// like test-time PFail in axon: 0 in the base config
	{Sel: "Params.TestDropout", Set: func(cfg *Config, v float32) {
		cfg.Params.TestDropout = params.Float64(v)
	}, Vals: func() []float32 {
		return params.Range(0.05, 0.5, 0.05)
	}},

	// Gaussian noise SD on the hidden net inputs, in training and testing:
	// 0 in the base config. A wide range, because the conflict_rnn
	// network learns to override noise up to 4.
	{Sel: "Params.Noise", Set: func(cfg *Config, v float32) {
		cfg.Params.Noise = params.Float64(v)
	}, Vals: func() []float32 {
		return []float32{0.1, 0.2, 0.5, 1, 2, 4}
	}},

	// AdamW weight decay: 0 (Adam) in the base config
	{Sel: "Params.WeightDecay", Set: func(cfg *Config, v float32) {
		cfg.Params.WeightDecay = params.Float64(v)
	}, Vals: func() []float32 {
		return params.Range(0.05, 0.3, 0.05)
	}},
}
