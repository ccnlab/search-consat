// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// cfenv implements the conflict environment: NGroups groups of GroupN
// input patterns each, where all the inputs of a group map to the same
// output pattern. Training is only on consistent inputs, all from one
// group, in NSlots input slots. Testing introduces conflict, with some
// of the slots from a second group.
package cfenv

//go:generate core generate -add-types -add-funcs

import (
	"fmt"

	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/patterns"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/emergent/v2/env"
)

// CFEnv is the conflict environment.
//
// Training: each trial chooses a group at random, and fills each slot
// with one of its inputs, sampled according to Freqs.
//
// Testing: each test runs through the full list of test items
// ([Space.Test]) TestReps times, in a new random order each time.
// The items are divided among the NData parallel envs.
//
// States:
//   - Input: [1, NSlots, PoolSize, PoolSize]
//   - Output: [NPerOutput, NGroups]
type CFEnv struct {
	// Name of environment: Train or Test.
	Name string

	// Configuration options.
	Configs Config

	// IsTest is true for the testing env.
	IsTest bool

	// NData is the number of parallel envs, which divide the test items.
	NData int

	// Di is the index of this env among the NData envs.
	Di int

	// Trial counts trials within an epoch.
	Trial env.Counter

	// States are the named state tensors.
	States map[string]*tensor.Float32

	// Items are the test items for this env (every NData'th item).
	Items []int `display:"-"`

	// Order is the current test sequence: indexes into Items.
	Order []int `display:"-"`

	// Ctr is the current position in Order.
	Ctr int

	// Cur is the current item.
	Cur *Item

	// Group is the target group (the first group of Cur).
	Group int

	// PrevChoice is the group produced on the previous trial (-1 = none).
	PrevChoice int

	// Counts are the counts of each choice (index Choice + 1, so 0 = Blend)
	// for each test item (by index in [Space.Test]) in the current test,
	// across its repetitions, for [ItemConsistency].
	Counts map[int][]int `display:"-"`

	// Rand is the random number generator for the env.
	Rand randx.Rand `display:"-"`

	// RandSeed is the random seed multiplier for the run counter.
	// It is set to 173 if 0 at start for consistent results by default.
	RandSeed int64 `edit:"-"`

	// Root is the root tensorfs directory, where the patterns are saved
	// in Patterns (see [CFEnv.SavePatterns]), if set before Config.
	Root *tensorfs.Node `display:"-"`
}

func (ev *CFEnv) Label() string { return ev.Name }

func (ev *CFEnv) Defaults() {
	ev.Configs.Defaults()
}

// Space returns the shared patterns and test items.
func (ev *CFEnv) Space() *Space { return ev.Configs.Space() }

// Config configures the env, after Configs have been set.
func (ev *CFEnv) Config(trials, ndata, di int, test bool, rndseed int64) {
	cfg := &ev.Configs
	ev.IsTest = test
	ev.NData = ndata
	ev.Di = di
	ev.Trial.Max = trials
	ev.RandSeed = rndseed
	ps := cfg.PoolSize
	ev.States = make(map[string]*tensor.Float32)
	ev.States["Input"] = tensor.NewFloat32(1, cfg.NSlots, ps, ps)
	ev.States["Output"] = tensor.NewFloat32(cfg.NPerOutput, cfg.NGroups)
	if ev.Root != nil {
		ev.SavePatterns(ev.Root.Dir("Patterns"))
	}
	if test {
		ev.Items = nil
		for i := di; i < len(ev.Space().Test); i += ndata {
			ev.Items = append(ev.Items, i)
		}
		ev.Trial.Max = ev.TestTrials()
	}
}

// TestTrials returns the number of trials in a full test for each env:
// TestReps times the items per env, rounded up.
func (ev *CFEnv) TestTrials() int {
	nt := len(ev.Space().Test)
	return ev.Configs.TestReps * ((nt + ev.NData - 1) / ev.NData)
}

func (ev *CFEnv) Init(run int) {
	if ev.RandSeed == 0 {
		ev.RandSeed = 173
	}
	randx.InitSysRand(&ev.Rand, ev.RandSeed*(int64(run)+1))
	ev.Trial.Init()
	ev.PrevChoice = -1
	ev.Ctr = -1
	if !ev.IsTest {
		return
	}
	ev.Counts = make(map[int][]int)
	// each rep is a new permutation of the items, padded with random
	// items so that all the envs have the same number of trials.
	per := ev.TestTrials() / ev.Configs.TestReps
	ev.Order = ev.Order[:0]
	for range ev.Configs.TestReps {
		perm := ev.Rand.Perm(len(ev.Items))
		for len(perm) < per {
			perm = append(perm, ev.Rand.Intn(len(ev.Items)))
		}
		ev.Order = append(ev.Order, perm...)
	}
}

// SavePatterns saves the patterns in the given tensorfs directory:
// the input patterns of each group in Items/G<g> (GroupN rows each),
// the group prototypes (if Prototypes) in Prototypes/G<g>,
// and the output patterns in Outputs.
func (ev *CFEnv) SavePatterns(dir *tensorfs.Node) {
	cfg := &ev.Configs
	sp := ev.Space()
	var names []string
	var protoRows, itemRows []int
	for g := range cfg.NGroups {
		names = append(names, fmt.Sprintf("G%d", g))
		if g > 0 {
			protoRows = append(protoRows, g)
			itemRows = append(itemRows, g*cfg.GroupN)
		}
	}
	patterns.SplitRows(dir.Dir("Items"), sp.Inputs, names, itemRows...)
	if sp.Protos != nil {
		patterns.SplitRows(dir.Dir("Prototypes"), sp.Protos, names, protoRows...)
	}
	dir.Float32("Outputs", sp.Outputs.ShapeSizes()...).CopyFrom(sp.Outputs)
}

func (ev *CFEnv) State(el string) tensor.Values {
	return ev.States[el]
}

// String returns the current inputs as group:member per slot.
func (ev *CFEnv) String() string {
	if ev.Cur == nil {
		return ""
	}
	return fmt.Sprintf("%s c:%d", ev.Cur.String(&ev.Configs), ev.Cur.NConflict)
}

// Step does one trial: the next test item, or a new random training item.
func (ev *CFEnv) Step() bool {
	ev.Trial.Incr()
	if ev.IsTest {
		ev.Ctr = (ev.Ctr + 1) % len(ev.Order)
		ev.Cur = ev.Space().Test[ev.Items[ev.Order[ev.Ctr]]]
	} else {
		ev.Cur = ev.TrainItem()
	}
	ev.Group = ev.Cur.Groups[0]
	ev.Render()
	return true
}

// TrainItem returns a new random training item: a random group,
// with inputs from that group sampled according to Freqs.
func (ev *CFEnv) TrainItem() *Item {
	cfg := &ev.Configs
	g := ev.Rand.Intn(cfg.NGroups)
	inputs := make([]int, cfg.NSlots)
	var probs []float32
	if len(cfg.Freqs) > 0 {
		tot := float32(0)
		for _, f := range cfg.Freqs {
			tot += f
		}
		for _, f := range cfg.Freqs {
			probs = append(probs, f/tot)
		}
	}
	for s := range inputs {
		m := 0
		if probs != nil {
			m = randx.PChoose32(probs, ev.Rand)
		} else {
			m = ev.Rand.Intn(cfg.GroupN)
		}
		inputs[s] = g*cfg.GroupN + m
	}
	return cfg.NewItem(inputs)
}

// Render renders the current item into the states. The Output target is
// the output pattern of the first group, which is the only group in
// training.
func (ev *CFEnv) Render() {
	sp := ev.Space()
	nu := ev.Configs.PoolUnits()
	inp := ev.States["Input"].Values
	for s, in := range ev.Cur.Inputs {
		copy(inp[s*nu:(s+1)*nu], sp.Inputs.SubSpace(in).(*tensor.Float32).Values)
	}
	copy(ev.States["Output"].Values, sp.Outputs.SubSpace(ev.Group).(*tensor.Float32).Values)
}
