// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfenv

import (
	"fmt"
	"math"
	"slices"
	"sync"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/patterns"
	"cogentcore.org/lab/tensor"
)

// Item is one input combination.
type Item struct {

	// Inputs are the input patterns in each slot.
	Inputs []int

	// Groups are the groups in the inputs: the first group (A) is
	// in NSlots - NConflict slots, and the second (B, if any) in NConflict.
	Groups []int

	// Probs are the expected probabilities of choosing each of Groups,
	// in proportion to the summed frequency of its inputs.
	Probs []float32

	// NConflict is the number of inputs from the second group (B).
	NConflict int
}

// String returns the inputs as group:member per slot.
func (it *Item) String(cfg *Config) string {
	s := ""
	for i, in := range it.Inputs {
		if i > 0 {
			s += " "
		}
		s += fmt.Sprintf("%d:%d", cfg.Group(in), in%cfg.GroupN)
	}
	return s
}

// IsTie returns true if the item has two groups with equal expected
// probabilities (equal evidence): a tie that the choice has to break.
func (it *Item) IsTie() bool {
	return len(it.Probs) == 2 && math.Abs(float64(it.Probs[0]-it.Probs[1])) < 1e-6
}

// GroupIndex returns the index of group g in Groups, or -1 if not present.
func (it *Item) GroupIndex(g int) int { return slices.Index(it.Groups, g) }

// NewItem returns the item for the given inputs, with groups ordered by
// the number of inputs (most first).
func (cfg *Config) NewItem(inputs []int) *Item {
	it := &Item{Inputs: inputs}
	var wts []float32
	var ns []int
	tot := float32(0)
	for _, in := range inputs {
		g := cfg.Group(in)
		gi := it.GroupIndex(g)
		if gi < 0 {
			gi = len(it.Groups)
			it.Groups = append(it.Groups, g)
			wts = append(wts, 0)
			ns = append(ns, 0)
		}
		w := cfg.Freq(in)
		wts[gi] += w
		ns[gi]++
		tot += w
	}
	if len(ns) == 2 && ns[1] > ns[0] {
		it.Groups[0], it.Groups[1] = it.Groups[1], it.Groups[0]
		wts[0], wts[1] = wts[1], wts[0]
		ns[0], ns[1] = ns[1], ns[0]
	}
	for _, w := range wts {
		it.Probs = append(it.Probs, w/tot)
	}
	if len(ns) > 1 {
		it.NConflict = ns[1]
	}
	return it
}

// Space has the patterns and test items for a config.
type Space struct {

	// Inputs are the input patterns: [NInputs, PoolSize, PoolSize],
	// with GroupN consecutive patterns per group.
	Inputs *tensor.Float32

	// Protos are the group prototypes, if Prototypes: [NGroups, PoolSize, PoolSize].
	Protos *tensor.Float32

	// Outputs are the output patterns: [NGroups, NPerOutput, NGroups].
	Outputs *tensor.Float32

	// Test are the test items, ordered by NConflict.
	Test []*Item

	// Targets are the distinct expected probabilities in the test
	// items, as rounded percents, in decreasing order.
	Targets []int
}

var (
	spaces     = map[string]*Space{}
	spacesLock sync.Mutex
)

// Space returns the patterns and test items for this config,
// generating them on first use. It is shared across all envs with the
// same config.
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

// MakeSpace generates the patterns and test items using [Config.Seed].
func (cfg *Config) MakeSpace() *Space {
	sp := &Space{}
	ps := cfg.PoolSize
	sp.Inputs = tensor.NewFloat32(cfg.NInputs(), ps, ps)
	patterns.NewRand(cfg.Seed)
	if cfg.Prototypes {
		cfg.makeProtoInputs(sp)
	} else {
		errors.Log(patterns.PermutedBinaryMinDiff(sp.Inputs, cfg.NOn(), 1, 0, cfg.MinDiff(cfg.NOn())))
	}
	ng, npo := cfg.NGroups, cfg.NPerOutput
	sp.Outputs = tensor.NewFloat32(ng, npo, ng)
	if cfg.RandOutput {
		errors.Log(patterns.PermutedBinaryMinDiff(sp.Outputs, npo, 1, 0, cfg.MinDiff(npo)))
	} else {
		for g := range ng {
			for y := range npo {
				sp.Outputs.Set(1, g, y, g)
			}
		}
	}
	rnd := randx.NewSysRand(cfg.Seed)
	for c := range cfg.NLevels() {
		if cfg.TestPerLevel > 0 {
			sp.Test = append(sp.Test, cfg.sampleItems(c, cfg.TestPerLevel, rnd)...)
		} else {
			sp.Test = append(sp.Test, cfg.allItems(c)...)
		}
	}
	tg := map[int]bool{100: true}
	for _, it := range sp.Test {
		for _, p := range it.Probs {
			tg[Percent(p)] = true
		}
	}
	for p := range tg {
		sp.Targets = append(sp.Targets, p)
	}
	slices.Sort(sp.Targets)
	slices.Reverse(sp.Targets)
	return sp
}

// makeProtoInputs makes the input patterns from group prototypes,
// which differ by at least ProtoMinDiff active units: each input
// pattern is its prototype with NFlip active units turned off and NFlip
// inactive ones turned on. The input patterns of a group are distinct.
func (cfg *Config) makeProtoInputs(sp *Space) {
	ps, gn := cfg.PoolSize, cfg.GroupN
	sp.Protos = tensor.NewFloat32(cfg.NGroups, ps, ps)
	errors.Log(patterns.PermutedBinaryMinDiff(sp.Protos, cfg.NOn(), 1, 0, cfg.ProtoMinDiff()))
	nf := cfg.NFlip()
	for g := range cfg.NGroups {
		proto := sp.Protos.SubSpace(g).(*tensor.Float32).Values
		for m := range gn {
			pat := sp.Inputs.SubSpace(g*gn + m).(*tensor.Float32)
			for range 100 {
				copy(pat.Values, proto)
				patterns.FlipBits(pat, nf, nf, 1, 0)
				dup := false
				for j := range m {
					if slices.Equal(pat.Values, sp.Inputs.SubSpace(g*gn+j).(*tensor.Float32).Values) {
						dup = true
						break
					}
				}
				if !dup {
					break
				}
			}
		}
	}
}

// Percent returns probability p as a rounded percent.
func Percent(p float32) int { return int(math.Round(100 * float64(p))) }

// sampleItems returns n random items with c inputs from the second
// group, at random slots, with inputs chosen uniformly within groups.
func (cfg *Config) sampleItems(c, n int, rnd randx.Rand) []*Item {
	k, ng, gn := cfg.NSlots, cfg.NGroups, cfg.GroupN
	items := make([]*Item, n)
	for i := range n {
		a := rnd.Intn(ng)
		b := (a + 1 + rnd.Intn(ng-1)) % ng
		inputs := make([]int, k)
		for s := range k {
			inputs[s] = a*gn + rnd.Intn(gn)
		}
		for _, s := range rnd.Perm(k)[:c] {
			inputs[s] = b*gn + rnd.Intn(gn)
		}
		items[i] = cfg.NewItem(inputs)
	}
	return items
}

// allItems returns all the items with c inputs from the second group.
// If the two groups have the same number of inputs, the first group is
// always in slot 0, so that each combination appears once.
func (cfg *Config) allItems(c int) []*Item {
	k, ng, gn := cfg.NSlots, cfg.NGroups, cfg.GroupN
	var items []*Item
	var slotSets [][]int // the slots of the second group
	for _, set := range subsets(k, c) {
		if 2*c == k && slices.Contains(set, 0) {
			continue
		}
		slotSets = append(slotSets, set)
	}
	nmem := 1
	for range k {
		nmem *= gn
	}
	for a := range ng {
		for b := range ng {
			if (c == 0 && b != 0) || (c > 0 && b == a) {
				continue
			}
			for _, set := range slotSets {
				for m := range nmem {
					inputs := make([]int, k)
					for s := range k {
						g := a
						if slices.Contains(set, s) {
							g = b
						}
						inputs[s] = g*gn + m%gn
						m /= gn
					}
					items = append(items, cfg.NewItem(inputs))
				}
			}
		}
	}
	return items
}

// subsets returns all the subsets of size c of 0..n-1.
func subsets(n, c int) [][]int {
	if c == 0 {
		return [][]int{nil}
	}
	var res [][]int
	for i := c - 1; i < n; i++ {
		for _, s := range subsets(i, c-1) {
			res = append(res, append(s, i))
		}
	}
	return res
}

// SpaceString returns a description of the space.
func (cfg *Config) SpaceString(sp *Space) string {
	out := "bars"
	if cfg.RandOutput {
		out = "random"
	}
	inp := fmt.Sprintf("random, min diff %d", cfg.MinDiff(cfg.NOn()))
	if cfg.Prototypes {
		inp = fmt.Sprintf("from prototypes with min diff %d, %d flipped", cfg.ProtoMinDiff(), cfg.NFlip())
	}
	s := fmt.Sprintf("conflict: %d slots, %d groups x %d inputs (%dx%d, %d active, %s), %s output patterns %dx%d\n", cfg.NSlots, cfg.NGroups, cfg.GroupN, cfg.PoolSize, cfg.PoolSize, cfg.NOn(), inp, out, cfg.NPerOutput, cfg.NGroups)
	nc := make([]int, cfg.NLevels())
	for _, it := range sp.Test {
		nc[it.NConflict]++
	}
	s += fmt.Sprintf("test items per number of conflicting inputs: %v, x %d reps; target probabilities: %v", nc, cfg.TestReps, sp.Targets)
	return s
}
