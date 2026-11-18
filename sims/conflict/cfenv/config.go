// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfenv

import (
	"fmt"
	"math"

	"cogentcore.org/lab/patterns"
)

// Config has configuration parameters for the environment.
type Config struct {

	// NSlots is the number of input slots (k).
	// 6 is the best for testing graded dynamics.
	NSlots int `default:"6"`

	// NGroups is the number of groups (G), each with its own output pattern.
	// 5 with bar outputs results in sparse 20% activity, which is important for axon.
	NGroups int `default:"5"`

	// GroupN is the number of input patterns in each group (g),
	// all of which map to the group's output pattern.
	// Not particularly theory-relevant: just enough to have some variety.
	GroupN int `default:"4"`

	// Freqs are the relative frequencies of the GroupN input patterns
	// within each group, used for sampling the training inputs and
	// for weighting the evidence in the expected probabilities.
	// Empty means all equal.
	Freqs []float32

	// PoolSize is the size of each (square) input slot pool.
	PoolSize int

	// PctAct is the proportion of active units in each input pattern.
	PctAct float32

	// MaxOverlap is the maximum proportion of active units that any
	// two random patterns can share: the input patterns without Prototypes,
	// and the RandOutput patterns.
	MaxOverlap float32

	// Prototypes generates the input patterns of each group from a group
	// prototype, by flipping FlipPct of the active units, instead of making
	// all the input patterns independently random. This makes the mapping
	// from inputs to groups systematic: inputs of the same group are similar.
	Prototypes bool

	// ProtoMaxOverlap is the maximum proportion of active units that any
	// two group prototypes can share.
	ProtoMaxOverlap float32

	// FlipPct is the proportion of the active units of a prototype that are
	// turned off in each of its input patterns, with the same number of
	// inactive units turned on, so the activity level is the same.
	FlipPct float32

	// NPerOutput is the number of output units per group: the height of
	// the output layer, which is NGroups wide. Each group's output pattern
	// is a vertical bar of NPerOutput units in its column.
	NPerOutput int

	// RandOutput uses random output patterns (with the same number of
	// active units as the bars) instead of the vertical bars,
	// as a sanity check.
	RandOutput bool

	// ActThr is the threshold on output unit activity for summing the
	// activity of each group: low-level activity is ignored.
	ActThr float32

	// BlendThr is the maximum SecondRatio (the summed output activity of
	// the second input group over that of the top input group) for choosing
	// the top group. Above it is a Blend.
	BlendThr float32

	// IntrusionThr is the maximum OtherRatio (the summed output activity of
	// all the other, non-input groups over that of the top input group).
	// Above it is an Intrusion.
	IntrusionThr float32

	// TestPerLevel is the number of random test items for each number of
	// conflicting inputs. 0 = all of the items.
	TestPerLevel int

	// TestReps is the number of times the full test list is presented in
	// each test, in a new random order each time.
	TestReps int

	// Seed is the random seed for generating the patterns and the test
	// items, which are shared across all envs and runs.
	Seed int64
}

func (cfg *Config) Defaults() {
	cfg.NSlots = 6
	cfg.NGroups = 5
	cfg.GroupN = 4
	cfg.PoolSize = 7
	cfg.PctAct = 0.2
	cfg.MaxOverlap = 0.5
	cfg.Prototypes = true
	cfg.ProtoMaxOverlap = 0.2
	cfg.FlipPct = 0.2
	cfg.NPerOutput = 6
	cfg.ActThr = 0.1
	cfg.BlendThr = 0.75
	cfg.IntrusionThr = 0.75
	cfg.TestPerLevel = 200
	cfg.TestReps = 20
	cfg.Seed = 137
}

// PoolUnits returns the number of units in each input slot pool.
func (cfg *Config) PoolUnits() int { return cfg.PoolSize * cfg.PoolSize }

// NOn returns the number of active units per input pattern.
func (cfg *Config) NOn() int {
	return patterns.NFromPct(float64(cfg.PctAct), cfg.PoolUnits())
}

// MinDiff returns the minimum number of active units that must differ
// between any two patterns with non active units, to satisfy MaxOverlap.
func (cfg *Config) MinDiff(non int) int { return minDiff(non, cfg.MaxOverlap) }

// ProtoMinDiff returns the minimum number of active units that must differ
// between any two prototypes, to satisfy ProtoMaxOverlap.
func (cfg *Config) ProtoMinDiff() int { return minDiff(cfg.NOn(), cfg.ProtoMaxOverlap) }

// NFlip returns the number of active units turned off (and inactive
// units turned on) in each input pattern relative to its prototype.
func (cfg *Config) NFlip() int {
	return patterns.NFromPct(float64(cfg.FlipPct), cfg.NOn())
}

func minDiff(non int, maxOverlap float32) int {
	return non - int(math.Floor(float64(maxOverlap)*float64(non)))
}

// NInputs returns the total number of input patterns: NGroups * GroupN.
func (cfg *Config) NInputs() int { return cfg.NGroups * cfg.GroupN }

// Group returns the group of input pattern in.
func (cfg *Config) Group(in int) int { return in / cfg.GroupN }

// Freq returns the relative frequency of input pattern in.
func (cfg *Config) Freq(in int) float32 {
	if len(cfg.Freqs) == 0 {
		return 1
	}
	return cfg.Freqs[in%cfg.GroupN]
}

// NLevels returns the number of conflict levels: the number of
// inputs from the other group ranges from 0 to NSlots/2.
func (cfg *Config) NLevels() int { return cfg.NSlots/2 + 1 }

// Key returns a string that identifies the patterns and test items.
func (cfg *Config) Key() string {
	return fmt.Sprintf("k%d_G%d_g%d_f%v_p%d_a%g_o%g_pr%v_po%g_fp%g_n%d_r%v_t%d_sd%d", cfg.NSlots, cfg.NGroups, cfg.GroupN, cfg.Freqs, cfg.PoolSize, cfg.PctAct, cfg.MaxOverlap, cfg.Prototypes, cfg.ProtoMaxOverlap, cfg.FlipPct, cfg.NPerOutput, cfg.RandOutput, cfg.TestPerLevel, cfg.Seed)
}
