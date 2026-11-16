// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package csenv

import (
	"fmt"
	"math"

	"cogentcore.org/lab/patterns"
)

// Config has configuration parameters for the environment.
type Config struct {

	// NSlots is the number of input / output slots (pools).
	NSlots int

	// NInputs is the number of distinct input patterns (n).
	NInputs int

	// NGroups is the total number of groups (G).
	NGroups int

	// GroupsPer is the number of groups that each input belongs to (g).
	// This is the ambiguity of each input: it is consistent with GroupsPer
	// of the NGroups groups, and only the intersection across slots
	// identifies the target group(s).
	GroupsPer int

	// GroupN is the number of output patterns for each input in each
	// of its groups (g_n). If > 1, the network must choose one of them
	// on each slot; these choices are independent across slots.
	GroupN int

	// NValid is the number of valid groups shared by all the inputs in
	// an item (m = the size of the intersection of their groups).
	// Only items with exactly this many valid groups are used.
	// If 1, the target group is determined by the inputs. If > 1, the
	// target group is chosen at random on each trial, and the network
	// must commit to one group coherently across all slots.
	NValid int

	// PoolSize is the size of each (square) input and output pool.
	// Larger pools are needed to fit more patterns under the
	// PctAct and MaxOverlap constraints.
	PoolSize int

	// PctAct is the proportion of active units in each pattern.
	PctAct float32

	// MaxOverlap is the maximum proportion of active units that any
	// two patterns in the vocabulary can share.
	MaxOverlap float32

	// TestPct is the proportion of items held out for testing.
	TestPct float32

	// MatchThr is the minimum cosine between the output activity in a
	// slot and the closest output pattern for it to count as producing
	// that pattern; otherwise it is a miss.
	MatchThr float32

	// Seed is the random seed for generating the vocabulary, the group
	// memberships, and the train / test split, which are shared across
	// all envs and runs.
	Seed int64
}

func (cfg *Config) Defaults() {
	cfg.NSlots = 4
	cfg.NInputs = 32
	cfg.NGroups = 8
	cfg.GroupsPer = 3
	cfg.GroupN = 1
	cfg.NValid = 2
	cfg.PoolSize = 7
	cfg.PctAct = 0.2
	cfg.MaxOverlap = 0.5
	cfg.TestPct = 0.5
	cfg.MatchThr = 0.5
	cfg.Seed = 137
}

// PoolUnits returns the number of units per pool.
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

// NOutputs returns the number of output patterns: NInputs * GroupsPer * GroupN.
func (cfg *Config) NOutputs() int { return cfg.NInputs * cfg.GroupsPer * cfg.GroupN }

// NVocab returns the total number of patterns in the vocabulary:
// NInputs input patterns followed by NOutputs output patterns.
func (cfg *Config) NVocab() int { return cfg.NInputs + cfg.NOutputs() }

// Key returns a string that identifies the item space for this config.
func (cfg *Config) Key() string {
	return fmt.Sprintf("s%d_n%d_G%d_g%d_gn%d_m%d_p%d_a%g_o%g_tp%g_sd%d", cfg.NSlots, cfg.NInputs, cfg.NGroups, cfg.GroupsPer, cfg.GroupN, cfg.NValid, cfg.PoolSize, cfg.PctAct, cfg.MaxOverlap, cfg.TestPct, cfg.Seed)
}
