// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package csenv

import (
	"fmt"
	"slices"
	"strings"
	"sync"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/math32"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/patterns"
	"cogentcore.org/lab/tensor"
)

// Item is one item in the search space: an input in each slot,
// and the valid groups shared by all of the inputs.
type Item struct {
	// Inputs are the input indexes for each slot (all distinct).
	Inputs []int

	// Valid are the groups shared by all the inputs, in sorted order.
	Valid []int
}

func (it *Item) String() string {
	var b strings.Builder
	for i, in := range it.Inputs {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, "%d", in)
	}
	b.WriteString(" g:")
	for i, g := range it.Valid {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, "%d", g)
	}
	return b.String()
}

// Space has the vocabulary of patterns, the group memberships of each
// input, and the full list of items for a given config, split into
// training and testing items. It is shared across all envs.
type Space struct {
	// Vocab has the NVocab patterns, [NVocab, PoolSize, PoolSize]:
	// NInputs input patterns followed by NOutputs output patterns
	// (see [Space.OutputIndex]).
	Vocab *tensor.Float32

	// Groups are the groups for each input, in sorted order.
	Groups [][]int

	// Members are the inputs in each group, in sorted order.
	Members [][]int

	// Items are all the items with exactly NValid valid groups.
	Items []Item

	// Train are the training item indexes.
	Train []int

	// Test are the testing item indexes.
	Test []int

	// Coverage is the proportion of (slot, input, valid group)
	// combinations required by the test items that also occur in the
	// training items. If < 1, some test items require mappings in a slot
	// that were never trained there.
	Coverage float32
}

// GroupIndex returns the index of group g within the groups of
// input in, or -1 if the input is not in that group.
func (sp *Space) GroupIndex(in, g int) int {
	return slices.Index(sp.Groups[in], g)
}

// OutputIndex returns the vocabulary index of the output pattern
// for given input, group index within the input's groups
// (see [Space.GroupIndex]), and member within the group.
func (sp *Space) OutputIndex(cfg *Config, in, gi, mem int) int {
	return cfg.NInputs + (in*cfg.GroupsPer+gi)*cfg.GroupN + mem
}

// OutputSource returns the input, group (not the index) and member
// for given output index into the output patterns (0..NOutputs-1).
func (sp *Space) OutputSource(cfg *Config, oi int) (in, g, mem int) {
	mem = oi % cfg.GroupN
	oi /= cfg.GroupN
	in = oi / cfg.GroupsPer
	g = sp.Groups[in][oi%cfg.GroupsPer]
	return
}

var (
	spaces     = map[string]*Space{}
	spacesLock sync.Mutex
)

// Space returns the space for this config, generating it on first use.
// It is shared across all envs with the same config.
func (cfg *Config) Space() *Space {
	spacesLock.Lock()
	defer spacesLock.Unlock()
	key := cfg.Key()
	if sp, ok := spaces[key]; ok {
		return sp
	}
	sp := cfg.MakeSpace()
	spaces[key] = sp
	fmt.Println(cfg.SpaceString(sp))
	return sp
}

// MakeSpace generates the vocabulary, the group memberships, and the
// items, split into training and testing items, using [Config.Seed].
func (cfg *Config) MakeSpace() *Space {
	sp := cfg.makeItemSpace()
	ps := cfg.PoolSize
	sp.Vocab = tensor.NewFloat32(cfg.NVocab(), ps, ps)
	patterns.NewRand(cfg.Seed)
	errors.Log(patterns.PermutedBinaryMinDiff(sp.Vocab, cfg.NOn(), 1, 0, cfg.MinDiff()))
	return sp
}

// makeItemSpace makes the groups and items, without the vocabulary.
func (cfg *Config) makeItemSpace() *Space {
	sp := &Space{}
	var rnd randx.Rand
	randx.InitSysRand(&rnd, cfg.Seed)
	cfg.makeGroups(sp, rnd)
	cfg.makeItems(sp)

	ni := len(sp.Items)
	perm := rnd.Perm(ni)
	ntest := int(math32.Round(cfg.TestPct * float32(ni)))
	sp.Test = make([]int, ntest)
	sp.Train = make([]int, ni-ntest)
	for i, p := range perm {
		if i < ntest {
			sp.Test[i] = p
		} else {
			sp.Train[i-ntest] = p
		}
	}
	sp.Coverage = cfg.coverage(sp)
	return sp
}

// makeGroups assigns GroupsPer groups to each input, choosing among the
// groups with the fewest members so far, so that groups are balanced.
func (cfg *Config) makeGroups(sp *Space, rnd randx.Rand) {
	ng := cfg.NGroups
	sp.Groups = make([][]int, cfg.NInputs)
	sp.Members = make([][]int, ng)
	for in := range cfg.NInputs {
		for range cfg.GroupsPer {
			perm := rnd.Perm(ng)
			best := -1
			for _, g := range perm {
				if slices.Contains(sp.Groups[in], g) {
					continue
				}
				if best < 0 || len(sp.Members[g]) < len(sp.Members[best]) {
					best = g
				}
			}
			sp.Groups[in] = append(sp.Groups[in], best)
			sp.Members[best] = append(sp.Members[best], in)
		}
		slices.Sort(sp.Groups[in])
	}
}

// ValidGroups returns the groups shared by all the given inputs.
func (sp *Space) ValidGroups(inputs []int) []int {
	var valid []int
	for _, g := range sp.Groups[inputs[0]] {
		all := true
		for _, in := range inputs[1:] {
			if !slices.Contains(sp.Groups[in], g) {
				all = false
				break
			}
		}
		if all {
			valid = append(valid, g)
		}
	}
	return valid
}

// makeItems enumerates the ordered tuples of distinct inputs from the
// members of each group that have exactly NValid valid groups.
// Each tuple is generated from its lowest valid group only.
func (cfg *Config) makeItems(sp *Space) {
	ns := cfg.NSlots
	tuple := make([]int, ns)
	for g, mem := range sp.Members {
		var rec func(s int)
		rec = func(s int) {
			if s == ns {
				valid := sp.ValidGroups(tuple)
				if len(valid) == cfg.NValid && valid[0] == g {
					sp.Items = append(sp.Items, Item{Inputs: slices.Clone(tuple), Valid: valid})
				}
				return
			}
			for _, in := range mem {
				if slices.Contains(tuple[:s], in) {
					continue
				}
				tuple[s] = in
				rec(s + 1)
			}
		}
		rec(0)
	}
}

// coverage returns the proportion of (slot, input, valid group)
// combinations in the test items that occur in the training items.
func (cfg *Config) coverage(sp *Space) float32 {
	type key struct{ s, in, g int }
	trained := map[key]bool{}
	for _, ii := range sp.Train {
		it := &sp.Items[ii]
		for s, in := range it.Inputs {
			for _, g := range it.Valid {
				trained[key{s, in, g}] = true
			}
		}
	}
	needed := map[key]bool{}
	for _, ii := range sp.Test {
		it := &sp.Items[ii]
		for s, in := range it.Inputs {
			for _, g := range it.Valid {
				needed[key{s, in, g}] = true
			}
		}
	}
	if len(needed) == 0 {
		return 1
	}
	n := 0
	for k := range needed {
		if trained[k] {
			n++
		}
	}
	return float32(n) / float32(len(needed))
}

// SpaceString returns a description of the space.
func (cfg *Config) SpaceString(sp *Space) string {
	var b strings.Builder
	fmt.Fprintf(&b, "vocab: %d patterns (%d inputs + %d outputs), %dx%d pools, %d active, min diff %d\n", cfg.NVocab(), cfg.NInputs, cfg.NOutputs(), cfg.PoolSize, cfg.PoolSize, cfg.NOn(), cfg.MinDiff())
	mn, mx := len(sp.Members[0]), 0
	for _, m := range sp.Members {
		mn = min(mn, len(m))
		mx = max(mx, len(m))
	}
	fmt.Fprintf(&b, "groups: %d, %d per input, %d-%d members per group, %d outputs per input per group\n", cfg.NGroups, cfg.GroupsPer, mn, mx, cfg.GroupN)
	nall := 1
	for range cfg.NSlots {
		nall *= cfg.NInputs * cfg.GroupsPer * cfg.GroupN
	}
	fmt.Fprintf(&b, "slots: %d, output combinations: %d\n", cfg.NSlots, nall)
	fmt.Fprintf(&b, "items with %d valid groups: %d, train: %d, test: %d, test coverage: %.3g\n", cfg.NValid, len(sp.Items), len(sp.Train), len(sp.Test), sp.Coverage)
	return b.String()
}
