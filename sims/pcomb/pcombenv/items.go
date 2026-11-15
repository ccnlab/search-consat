// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcombenv

import (
	"fmt"
	"math/big"
	"strings"
	"sync"

	"cogentcore.org/core/math32"
	"cogentcore.org/lab/base/randx"
)

// Cue is one cue, with discrete probability and value indexes
// (see [Config.Prob], [Config.Value]).
type Cue struct {
	P, V int8
}

// Item is one item in the space of cue combinations: an unordered set
// (multiset) of cues, in sorted order. Items are assigned to random
// input slots when presented, so the item space does not include
// slot positions.
type Item struct {
	Cues []Cue
}

func (it *Item) String(cfg *Config) string {
	var b strings.Builder
	for i, c := range it.Cues {
		if i > 0 {
			b.WriteString(" ")
		}
		fmt.Fprintf(&b, "%.2g:%+.3g", cfg.Prob(int(c.P)), cfg.Value(int(c.V)))
	}
	return b.String()
}

// ItemSpace is the full list of items for a given config,
// split into training and testing items for each number of active cues.
// If [Config.DistinctTrain] is set, Train and Test only have items with
// distinct cue values, and the items with repeated values are in Agree.
type ItemSpace struct {
	// Items is the full ordered list of items.
	Items []Item

	// Train has the training item indexes, for each number of
	// active cues starting at NActive.Min.
	Train [][]int

	// Test has the testing item indexes, for each number of
	// active cues starting at NActive.Min.
	Test [][]int

	// Agree has the agreement item indexes (with repeated cue values),
	// for each number of active cues starting at NActive.Min, if
	// DistinctTrain is set. These are only used for testing.
	Agree [][]int
}

// HasRepeat returns true if any of the item cues have the same value.
func (it *Item) HasRepeat() bool {
	for i, c := range it.Cues {
		for _, o := range it.Cues[i+1:] {
			if o.V == c.V {
				return true
			}
		}
	}
	return false
}

var (
	spaces     = map[string]*ItemSpace{}
	spacesLock sync.Mutex
)

// Space returns the item space for this config, generating it on first use.
// It is shared across all envs with the same config.
func (cfg *Config) Space() *ItemSpace {
	spacesLock.Lock()
	defer spacesLock.Unlock()
	key := cfg.Key()
	if sp, ok := spaces[key]; ok {
		return sp
	}
	sp := cfg.MakeSpace()
	spaces[key] = sp
	fmt.Println(cfg.SpaceString())
	return sp
}

// NCombos returns the number of items with n active cues:
// multisets of size n from NTypes cue types = C(NTypes+n-1, n).
func (cfg *Config) NCombos(n int) int64 {
	return new(big.Int).Binomial(int64(cfg.NTypes()+n-1), int64(n)).Int64()
}

// MakeSpace generates the full item space, and splits it into
// training and testing items using a fixed random seed.
func (cfg *Config) MakeSpace() *ItemSpace {
	nt := cfg.NTypes()
	nv := cfg.NValues()
	sp := &ItemSpace{}
	total := int64(0)
	for n := cfg.NActive.Min; n <= cfg.NActive.Max; n++ {
		total += cfg.NCombos(n)
	}
	sp.Items = make([]Item, 0, total)
	var rnd randx.Rand
	randx.InitSysRand(&rnd, 137)
	for n := cfg.NActive.Min; n <= cfg.NActive.Max; n++ {
		st := len(sp.Items)
		// enumerate nondecreasing sequences of type indexes
		ts := make([]int, n)
		for {
			it := Item{Cues: make([]Cue, n)}
			for i, t := range ts {
				it.Cues[i] = Cue{P: int8(t / nv), V: int8(t % nv)}
			}
			sp.Items = append(sp.Items, it)
			i := n - 1
			for i >= 0 && ts[i] == nt-1 {
				i--
			}
			if i < 0 {
				break
			}
			ts[i]++
			for j := i + 1; j < n; j++ {
				ts[j] = ts[i]
			}
		}
		var split, agree []int
		for i := st; i < len(sp.Items); i++ {
			if cfg.DistinctTrain && sp.Items[i].HasRepeat() {
				agree = append(agree, i)
			} else {
				split = append(split, i)
			}
		}
		ni := len(split)
		perm := rnd.Perm(ni)
		ntest := int(math32.Round(cfg.TestPct * float32(ni)))
		test := make([]int, ntest)
		train := make([]int, ni-ntest)
		for i, p := range perm {
			if i < ntest {
				test[i] = split[p]
			} else {
				train[i-ntest] = split[p]
			}
		}
		sp.Train = append(sp.Train, train)
		sp.Test = append(sp.Test, test)
		sp.Agree = append(sp.Agree, agree)
	}
	return sp
}

// SpaceString returns a description of the item space sizes
// for each number of active cues.
func (cfg *Config) SpaceString() string {
	var b strings.Builder
	fmt.Fprintf(&b, "cue types: %d probs x %d values = %d\n", cfg.NProbs(), cfg.NValues(), cfg.NTypes())
	total := int64(0)
	for n := cfg.NActive.Min; n <= cfg.NActive.Max; n++ {
		nc := cfg.NCombos(n)
		total += nc
		nd := cfg.NDistinct(n)
		fmt.Fprintf(&b, "%d cues: %d items, %d with distinct values, %d with repeated values\n", n, nc, nd, nc-nd)
	}
	fmt.Fprintf(&b, "total: %d items\n", total)
	return b.String()
}

// NDistinct returns the number of items with n active cues that all
// have distinct values: C(NValues, n) * NProbs^n.
func (cfg *Config) NDistinct(n int) int64 {
	c := new(big.Int).Binomial(int64(cfg.NValues()), int64(n))
	return c.Mul(c, new(big.Int).Exp(big.NewInt(int64(cfg.NProbs())), big.NewInt(int64(n)), nil)).Int64()
}
