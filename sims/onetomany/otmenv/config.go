// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package otmenv

import (
	"fmt"
	"math"

	"cogentcore.org/lab/patterns"
)

// Config has configuration parameters for the environment.
type Config struct {

	// NInputs is the number of distinct input patterns (n).
	NInputs int

	// NOut is the number of output patterns for each input (o).
	// On each trial, one of them is chosen at random as the target.
	NOut int

	// PoolSize is the size of the (square) input and output layers.
	// Larger layers are needed to fit more patterns under the
	// PctAct and MaxOverlap constraints.
	PoolSize int

	// PctAct is the proportion of active units in each pattern.
	PctAct float32

	// MaxOverlap is the maximum proportion of active units that any
	// two patterns in the vocabulary can share.
	MaxOverlap float32

	// MatchThr is the minimum cosine between the output activity and
	// the closest output pattern for it to count as producing that
	// pattern. This must be high enough to reject blends of patterns:
	// an equal blend of two patterns that share 3 of 10 active units has
	// a cosine of 0.81 with each of them.
	MatchThr float32

	// Seed is the random seed for generating the vocabulary, which is
	// shared across all envs and runs.
	Seed int64
}

func (cfg *Config) Defaults() {
	cfg.NInputs = 32
	cfg.NOut = 2
	cfg.PoolSize = 7
	cfg.PctAct = 0.2
	cfg.MaxOverlap = 0.5
	cfg.MatchThr = 0.9
	cfg.Seed = 137
}

// PoolUnits returns the number of units in the input and output layers.
func (cfg *Config) PoolUnits() int { return cfg.PoolSize * cfg.PoolSize }

// NOn returns the number of active units per pattern.
func (cfg *Config) NOn() int {
	return patterns.NFromPct(float64(cfg.PctAct), cfg.PoolUnits())
}

// MinDiff returns the minimum number of active units that must differ
// between any two patterns, to satisfy MaxOverlap.
func (cfg *Config) MinDiff() int {
	non := cfg.NOn()
	return non - int(math.Floor(float64(cfg.MaxOverlap)*float64(non)))
}

// NOutputs returns the total number of output patterns: NInputs * NOut.
func (cfg *Config) NOutputs() int { return cfg.NInputs * cfg.NOut }

// NVocab returns the total number of patterns in the vocabulary:
// NInputs input patterns followed by NOutputs output patterns.
func (cfg *Config) NVocab() int { return cfg.NInputs + cfg.NOutputs() }

// OutputIndex returns the vocabulary index of output o of input in.
func (cfg *Config) OutputIndex(in, o int) int { return cfg.NInputs + in*cfg.NOut + o }

// Key returns a string that identifies the vocabulary for this config.
func (cfg *Config) Key() string {
	return fmt.Sprintf("n%d_o%d_p%d_a%g_o%g_sd%d", cfg.NInputs, cfg.NOut, cfg.PoolSize, cfg.PctAct, cfg.MaxOverlap, cfg.Seed)
}
