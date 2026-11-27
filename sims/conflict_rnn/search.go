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
// After SearchPS4, where TestNoise in both hidden layers graded the
// choice toward the target only at the cost of intrusions (the output
// pattern itself broke down), this puts the TestNoise in Hidden1 only, so
// that Hidden2 can clean it up, over a wider range (the noise is diluted
// by the cleanup), with Hidden2 only for comparison.
// The combinations set the fixed fields in each job, which are in the
// label (Sel).
//
// This was the fifth and last search that was run (psearch5, 14 jobs:
// conflict_rnn ore00104-117, 10 runs each). Results:
//   - TestNoise in Hidden1 only had about half the intrusions of
//     TestNoise in both hidden layers (SearchPS4) at the same In67 (e.g.,
//     In67 .77: Intrusion .17 vs about .32), with about twice the SD for
//     the same In67, but along the same kind of tradeoff curve: In67 .91,
//     Intrusion .015 at 4; .84, .07 at 5; .77, .17 at 6; .71, .36 at 8;
//     .65 (MatchCenter .89), .48 at 10 (P100 .43).
//   - TestNoise in Hidden2 only was worse than in both layers (In67 .82,
//     Intrusion .38 at 8): with no layer to clean it up, the noise
//     corrupts the output directly.
//   - At all levels, the minority choices were independent of the
//     previous choice (MinorPrev equal after minor, major, other;
//     RepeatPrev .50), with competition (LeadSwaps 1-3), and graded
//     (In83 > In67 > In50 > In33 > In17).
//
// Conclusion of the searches: in this network, noise is the only generic
// way found to grade the choice toward the axon target (In67 .67,
// MatchCenter 1.0) without perseveration, and each level of noise that
// grades the choice corrupts a matching proportion of the outputs;
// moving the noise away from the output only shifts that tradeoff.
var Search = params.Searches[*Config]{
	// job 1: the base config, with the same args as the other jobs
	params.Baseline[*Config](),

	testNoiseWith("TestNoiseLayers=Hidden1", func(cfg *Config) { cfg.Params.TestNoiseLayers = "Hidden1" }, []float32{2, 3, 4, 5, 6, 8, 10, 12}),
	testNoiseWith("TestNoiseLayers=Hidden2", func(cfg *Config) { cfg.Params.TestNoiseLayers = "Hidden2" }, []float32{2, 3, 4, 6, 8}),
}

// testNoiseWith returns a search over TestNoise values, with the fixed
// settings applied by fix, labeled fixed,TestNoise=value.
func testNoiseWith(fixed string, fix func(cfg *Config), vals []float32) *params.Search[*Config] {
	return &params.Search[*Config]{Sel: fixed + ",TestNoise", Set: func(cfg *Config, v float32) {
		fix(cfg)
		setTestNoise(cfg, v)
	}, Vals: func() []float32 { return vals }}
}

// SearchPS4 is the fourth search that was run (psearch4, 22 jobs:
// conflict_rnn ore00082-103, 10 runs each), kept for the record. Results:
//   - TestNoise (both hidden layers, Decay 1) from 2.25 to 3.5 graded the
//     valid choices smoothly (TestNoise 3: In83 .94, In67 .78, In17 .07,
//     MatchCenter 1.67), independent of the previous choice (MinorPrev
//     about equal after minor, major, other), with competition (LeadSwaps
//     2.5), but the intrusions rose along the same curve (.02 at 2.25,
//     .29 at 3, .52 at 3.5): the output pattern itself broke down (P100
//     .72 at 3).
//   - OutputLateral was slightly better (TestNoise 3: In67 .79, Intrusion
//     .26), Lateral worse; NCycles 5 or 20 made no difference; training
//     Noise 1 moved along the same curve; Decay .3 was slightly worse and
//     history dependent.
//
// It mapped TestNoise from 2 to 4, and tried to keep the intrusions down
// with laterals, NCycles, training Noise and Decay .3.
var SearchPS4 = params.Searches[*Config]{
	// job 1: the base config, with the same args as the other jobs
	params.Baseline[*Config](),

	// test-only noise between 2 and 4 (Decay 1, training Noise .2)
	{Sel: "Params.TestNoise", Set: func(cfg *Config, v float32) {
		cfg.Params.TestNoise = params.Float64(v)
	}, Vals: func() []float32 {
		return params.Range(2.25, 3.5, 0.25)
	}},

	// cleanup by lateral connections, which had no effect without noise
	testNoiseWith("OutputLateral=true", func(cfg *Config) { cfg.Params.OutputLateral = true }, []float32{2, 3, 4}),
	testNoiseWith("Lateral=true", func(cfg *Config) { cfg.Params.Lateral = true }, []float32{2, 3, 4}),

	// settling cycles: noise in each cycle, averaged over more cycles
	testNoiseWith("NCycles=20", func(cfg *Config) { cfg.Params.NCycles = 20 }, []float32{2, 3, 4}),
	testNoiseWith("NCycles=5", func(cfg *Config) { cfg.Params.NCycles = 5 }, []float32{2, 3}),

	// more training noise, to learn to override noise
	testNoiseWith("Noise=1", func(cfg *Config) { cfg.Params.Noise = 1 }, []float32{3, 4}),

	// some carryover (Decay .3 was not perseverative)
	decayWith(0.3, "TestNoise", setTestNoise, []float32{3, 4}),
}

// SearchPS3 is the third search that was run (psearch3, 24 jobs:
// conflict_rnn ore00058-81, 10 runs each), kept for the record. Results:
//   - TestNoise alone (Decay 1) had no effect on the choice up to 1 (In67
//     1.0, only TieConsist went down), and at 2 gave In67 .95, Intrusion
//     .005, with minority choices independent of the previous choice
//     (MinorPrevMinor .024 vs MinorPrevMajor .022, RepeatPrev .50) and
//     competition (LeadSwaps 1.1, CommitCyc 55 vs 16). At 4: In67 .68,
//     MatchCenter 1.08, but Intrusion .68.
//   - The training Noise made no difference (Noise 0 = Noise .2).
//   - With Decay .5 or .7, TestNoise made the choice LESS graded (Decay .5:
//     In67 .79 -> .96 from TestNoise .25 to 1), by breaking up the
//     perseveration (MinorPrevMinor .46 -> .07). At TestNoise 2, Decay
//     .3-1 all gave In67 .93-.95 and RepeatPrev .5: the noise overrides
//     the carryover.
//   - Clean reruns (without the training leak) of Decay .5 with
//     TestDropout .6: In67 .87, Intrusion .047, RepeatPrev .76,
//     MinorPrevMinor .17 vs .03 (still perseverative); TestDropout .8:
//     In67 .91, Intrusion .08.
//
// It searched TestNoise (noise in testing only), alone and with
// carryover (Decay), with the training Noise at 0, and with Decay .5,
// TestDropout .6, which was also rerun without the training leak.
var SearchPS3 = params.Searches[*Config]{

	// job 1: the base config, with the same args as the other jobs
	params.Baseline[*Config](),

	// test-only noise, on top of the training Noise (.2)
	{Sel: "Params.TestNoise", Set: func(cfg *Config, v float32) {
		cfg.Params.TestNoise = params.Float64(v)
	}, Vals: func() []float32 {
		return []float32{0.25, 0.5, 1, 2, 4}
	}},

	// with carryover: Decay .5 was the most graded (perseverative), .7
	// less so, and .3 less graded but not perseverative
	decayWith(0.5, "TestNoise", setTestNoise, []float32{0.25, 0.5, 1, 2}),
	decayWith(0.7, "TestNoise", setTestNoise, []float32{0.25, 0.5, 1, 2}),
	decayWith(0.3, "TestNoise", setTestNoise, []float32{0.5, 1, 2}),

	// without training noise, so that the network has learned to override
	// no noise at all
	{Sel: "Noise=0,TestNoise", Set: func(cfg *Config, v float32) {
		cfg.Params.Noise = 0
		cfg.Params.TestNoise = params.Float64(v)
	}, Vals: func() []float32 {
		return []float32{0.25, 0.5, 1}
	}},

	// with the best combination of SearchPS2
	{Sel: "Decay=0.5,TestDropout=0.6,TestNoise", Set: func(cfg *Config, v float32) {
		cfg.Params.Decay = 0.5
		cfg.Params.TestDropout = 0.6
		cfg.Params.TestNoise = params.Float64(v)
	}, Vals: func() []float32 {
		return []float32{0.5, 1}
	}},

	// reruns of SearchPS2 jobs: with carryover, the test dropout used to
	// leak into training (egomlx TrainTrial ran its forward pass with it,
	// and that activity was carried over to the next trial), so these are
	// now with the test dropout in testing only
	decayWith(0.5, "TestDropout", func(cfg *Config, v float32) {
		cfg.Params.TestDropout = params.Float64(v)
	}, []float32{0.6, 0.8}),
}

// setTestNoise sets the TestNoise, for [decayWith].
func setTestNoise(cfg *Config, v float32) {
	cfg.Params.TestNoise = params.Float64(v)
}

// decayWith returns a search over the values of another parameter, set
// by set, with Decay at the given value, labeled Decay=decay,name=value.
func decayWith(decay float32, name string, set func(cfg *Config, v float32), vals []float32) *params.Search[*Config] {
	return &params.Search[*Config]{Sel: fmt.Sprintf("Decay=%g,%s", decay, name), Set: func(cfg *Config, v float32) {
		cfg.Params.Decay = params.Float64(decay)
		set(cfg, v)
	}, Vals: func() []float32 { return vals }}
}

// SearchPS2 is the second search that was run (psearch2, 24 jobs:
// conflict_rnn ore00034-57, 10 runs each), kept for the record. Results:
//   - Decay 0-.4 (strong carryover) was less graded than .5-.65 (In67 .95
//     at 0, .875 at .3-.4) but not perseverative (MinorPrevMinor about the
//     minority rate), with competition (LeadSwaps .4-.8, CommitCyc 25-30).
//   - Noise (in training and testing) with Decay made the choices more
//     deterministic (Decay .7: In67 .87 -> 1.0 at Noise 2): the network
//     learns to override the noise, and the carryover with it.
//   - Decay .5 with TestDropout .6 broke up the perseveration (In67 .88,
//     Intrusion .016, RepeatPrev .74, MinorPrevMinor .13 vs In33 .12,
//     LeadSwaps .74); more TestDropout only added noise (intrusions).
//     But the Decay x TestDropout jobs trained with the test dropout in
//     the carried-over activity (egomlx TrainTrial ran its forward pass
//     with it, since fixed), so they are not test-only results.
//
// It searched Decay below .5, and Decay combined with Noise or
// TestDropout, after SearchPS1, where carryover (Decay) was the only way
// to grade the majority choice (by perseveration), to see whether the
// noise breaks up the perseveration into more independent graded choices.
var SearchPS2 = params.Searches[*Config]{
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
