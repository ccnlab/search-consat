// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcombenv

import (
	"fmt"

	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/minmax"
	"github.com/emer/emergent/v2/popcode"
)

// Config has configuration parameters for the environment.
// Probabilities and values are discretized, so the full space of
// cue combinations is finite and can be split cleanly into training
// and testing items.
type Config struct {

	// NCues is the number of cue slots in the input. Each slot has
	// a separate Value and Prob input pool.
	NCues int

	// NActive is the range of the number of active cues per trial.
	// The min is 2 by default, to avoid the special case of a single cue,
	// which is always chosen regardless of its probability.
	NActive minmax.Int

	// ProbStep is the spacing of discrete probability values,
	// which range from ProbStep to 1 (0 is excluded).
	ProbStep float32

	// ValueStep is the spacing of discrete values, which range from -1 to 1.
	ValueStep float32

	// PopCodeUnits is the number of population code units per value.
	PopCodeUnits int

	// Sigma is the Gaussian bump width in normalized units.
	Sigma float32

	// SalientFull is the minimum value magnitude that is fully salient:
	// the salience output is a copy of the value output bump.
	SalientFull float32

	// SalientHalf is the minimum value magnitude that is half salient:
	// the half of the value bump toward 0 is cut off.
	// Smaller magnitudes have no salience output.
	SalientHalf float32

	// TestPct is the proportion of items (for each number of active cues)
	// held out for testing.
	TestPct float32

	// UniformN samples the number of active cues uniformly on each trial,
	// instead of in proportion to the number of items with that many cues
	// (which is dominated by the largest number).
	UniformN bool

	// MatchThr is the max distance between the decoded output value and
	// the nearest cue value to count as choosing that value; otherwise it is
	// a miss (e.g., a blend between cue values).
	MatchThr float32

	// NBins is the number of normalized probability bins for calibration stats.
	NBins int

	// ValuePop is the population code for values, set in Update.
	ValuePop popcode.OneD `display:"-"`

	// ProbPop is the population code for probabilities, set in Update.
	ProbPop popcode.OneD `display:"-"`
}

func (cfg *Config) Defaults() {
	cfg.NCues = 2
	cfg.NActive.Set(1, 2)
	cfg.ProbStep = 0.2
	cfg.ValueStep = 0.25
	cfg.PopCodeUnits = 22
	cfg.Sigma = 0.1
	cfg.SalientFull = 0.75
	cfg.SalientHalf = 0.5
	cfg.TestPct = 0.1
	cfg.UniformN = true
	cfg.MatchThr = 0.2
	cfg.NBins = 10
	cfg.Update()
}

// Update sets the population codes from the other parameters.
func (cfg *Config) Update() {
	cfg.ValuePop.Defaults()
	vmin := -1 - cfg.ValueStep
	vmax := 1 + cfg.ValueStep
	cfg.ValuePop.SetRange(vmin, vmax, cfg.Sigma)
	cfg.ProbPop.Defaults()
	pmin := float32(0)
	pmax := 1 + cfg.ProbStep
	cfg.ProbPop.SetRange(pmin, pmax, cfg.Sigma)
}

// NProbs is the number of discrete probability values.
func (cfg *Config) NProbs() int { return int(math32.Round(1 / cfg.ProbStep)) }

// NValues is the number of discrete values.
func (cfg *Config) NValues() int { return int(math32.Round(2/cfg.ValueStep)) + 1 }

// NTypes is the number of distinct cue types (probability x value).
func (cfg *Config) NTypes() int { return cfg.NProbs() * cfg.NValues() }

// Prob returns the probability for given discrete index.
func (cfg *Config) Prob(pi int) float32 { return float32(pi+1) * cfg.ProbStep }

// Value returns the value for given discrete index.
func (cfg *Config) Value(vi int) float32 { return -1 + float32(vi)*cfg.ValueStep }

// Salience returns the salience level for given value:
// 1 = full, 0.5 = half, 0 = none.
func (cfg *Config) Salience(v float32) float32 {
	av := math32.Abs(v)
	eps := 0.001 * cfg.ValueStep
	switch {
	case av >= cfg.SalientFull-eps:
		return 1
	case av >= cfg.SalientHalf-eps:
		return 0.5
	}
	return 0
}

// Key returns a string that identifies the item space for this config.
func (cfg *Config) Key() string {
	return fmt.Sprintf("np%d_nv%d_na%d-%d_tp%g", cfg.NProbs(), cfg.NValues(), cfg.NActive.Min, cfg.NActive.Max, cfg.TestPct)
}
