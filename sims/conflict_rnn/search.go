// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflictrnn

import (
	"fmt"

	"github.com/emer/emergent/v2/params"
)

// Search is the parameter search: each job runs the sim with one of
// these values of one parameter, with the rest at their configured
// values (the base config, including config.toml and any -args given
// to the simmer search). Run the jobs with the simmer ParamSearch
// function, which launches one job per value with -search-at 1..N
// (see emergent params/README.md). List the jobs with:
//
//	./conflict_rnn/conflict_rnn -nogui -search-n
//
// After SearchPS1, where carryover (Decay) was the only way to grade the
// majority choice (by perseveration), and Noise and TestDropout only made
// the choices noisy, this searches Decay below .5, and Decay combined
// with Noise or TestDropout, to see whether the noise breaks up the
// perseveration into more independent graded choices. The combinations
// set two fields per job, with the fixed one in the label (Sel).
var Search = params.Searches[*Config]{
	// job 1: the base config, with the same args as the other jobs
	params.Baseline[*Config](),

	// carryover below .5 (In67 leveled off at about .82 from .65 to .5)
	{Sel: "Params.Decay", Set: func(cfg *Config, v float32) {
		cfg.Params.Decay = params.Float64(v)
	}, Vals: func() []float32 {
		return params.Range(0, 0.4, 0.1)
	}},

	// noise with carryover, where Decay .7 gave In67 .87, RepeatPrev 1.0
	decayWith(0.7, "Noise", func(cfg *Config, v float32) {
		cfg.Params.Noise = params.Float64(v)
	}, []float32{0.5, 1, 2, 4}),

	// noise with less carryover (Decay .8: In67 .97)
	decayWith(0.8, "Noise", func(cfg *Config, v float32) {
		cfg.Params.Noise = params.Float64(v)
	}, []float32{1, 2, 4}),

	// noise with more carryover (Decay .5: In67 .82, MinorPrevMinor .52)
	decayWith(0.5, "Noise", func(cfg *Config, v float32) {
		cfg.Params.Noise = params.Float64(v)
	}, []float32{1, 2, 4}),

	// test dropout with carryover: alone, it graded the choice only at
	// .8, with competition dynamics (LeadSwaps .68)
	decayWith(0.7, "TestDropout", func(cfg *Config, v float32) {
		cfg.Params.TestDropout = params.Float64(v)
	}, []float32{0.6, 0.7, 0.8}),

	decayWith(0.5, "TestDropout", func(cfg *Config, v float32) {
		cfg.Params.TestDropout = params.Float64(v)
	}, []float32{0.6, 0.7, 0.8}),

	// test dropout with high noise, without carryover
	{Sel: "Noise=2,TestDropout", Set: func(cfg *Config, v float32) {
		cfg.Params.Noise = 2
		cfg.Params.TestDropout = params.Float64(v)
	}, Vals: func() []float32 {
		return []float32{0.7, 0.8}
	}},
}

// decayWith returns a search over the values of another parameter, set
// by set, with Decay at the given value, labeled Decay=decay,name=value.
func decayWith(decay float32, name string, set func(cfg *Config, v float32), vals []float32) *params.Search[*Config] {
	return &params.Search[*Config]{Sel: fmt.Sprintf("Decay=%g,%s", decay, name), Set: func(cfg *Config, v float32) {
		cfg.Params.Decay = params.Float64(decay)
		set(cfg, v)
	}, Vals: func() []float32 { return vals }}
}

// SearchPS1 is the first search that was run (psearch1, 30 jobs:
// conflict_rnn ore00003-32, 10 runs each, Tanh hidden layers), kept for
// the record. Results (Test stats, vs the baseline: In67 1.0, In33 0,
// MatchCenter 2.94, TieConsist .91, RepeatPrev .52):
//   - Decay was the only parameter that graded the majority choice, with
//     clean outputs: In67 .99 at .85, .92 at .75, .84 at .65, .82 at .5
//     (MatchCenter 1.9), but through perseveration: RepeatPrev 1.0 at
//     .6-.85, MinorPrevMinor .4-.5 (vs In33 ~.17).
//   - TestDropout had no effect on the majority choice up to .7, In67 .95
//     at .8 (LeadSwaps .68, CommitCyc 42), and broke down at .9
//     (Intrusion .46, MatchCos .76).
//   - Noise (0-4) only changed the tie choices (TieConsist 1.0 at 0 to
//     .72 at 2-4); OutputLateral, Lateral and NCycles had no effect.
var SearchPS1 = params.Searches[*Config]{
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
