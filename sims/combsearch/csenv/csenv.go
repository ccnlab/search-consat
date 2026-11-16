// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// csenv implements the combinatorial search environment: each input
// pattern is ambiguous, belonging to several groups, each with its own
// output pattern(s), and the target is an output in each slot from a
// group shared by all of the inputs.
package csenv

//go:generate core generate -add-types -add-funcs

import (
	"fmt"
	"strings"

	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
)

// CSEnv is the combinatorial search environment.
// Each trial presents an item: a distinct input pattern in each slot.
// Each input belongs to GroupsPer of NGroups groups, and has GroupN
// output patterns for each of its groups. The valid groups are those
// shared by all the inputs (exactly NValid of them). The target group
// is one of the valid groups (chosen at random if NValid > 1), and the
// target in each slot is one of the input's output patterns for that
// group (chosen at random if GroupN > 1).
//
// States (slots along the outer X axis):
//   - Input: [1, NSlots, PoolSize, PoolSize]
//   - Output: [1, NSlots, PoolSize, PoolSize]
type CSEnv struct {
	// Name of environment: Train or Test.
	Name string

	// Test uses the testing items instead of the training items.
	Test bool

	// Configuration options.
	Configs Config

	// NData is the number of data-parallel envs.
	NData int

	// Di is the data-parallel index of this env.
	Di int

	// Trial counts trials within an epoch.
	Trial env.Counter

	// States are the named state tensors.
	States map[string]*tensor.Float32

	// Items are the item indexes for this env.
	Items []int

	// Order is the permuted order of Items, permuted after each pass.
	Order []int

	// Ctr is the current position in Order.
	Ctr int

	// Item is the current item index into the space.
	Item int

	// Group is the target group for the current trial.
	Group int

	// Outputs are the target output vocabulary indexes for each slot.
	Outputs []int

	// Rand is the random number generator for the env.
	Rand randx.Rand `display:"-"`

	// RunRandSeed is the random seed multiplier for the run counter.
	// It is set to 173 if 0 at start for consistent results by default.
	RunRandSeed int64 `edit:"-"`
}

func (ev *CSEnv) Label() string { return ev.Name }

func (ev *CSEnv) Defaults() {
	ev.Configs.Defaults()
}

// Config configures the env, after Configs have been set.
func (ev *CSEnv) Config(trials, ndata, di int, test bool, rndseed int64) {
	cfg := &ev.Configs
	ev.Trial.Max = trials
	ev.NData = ndata
	ev.Di = di
	ev.Test = test
	ev.RunRandSeed = rndseed
	ps := cfg.PoolSize
	ev.States = make(map[string]*tensor.Float32)
	ev.States["Input"] = tensor.NewFloat32(1, cfg.NSlots, ps, ps)
	ev.States["Output"] = tensor.NewFloat32(1, cfg.NSlots, ps, ps)
	ev.Outputs = make([]int, cfg.NSlots)
}

func (ev *CSEnv) Init(run int) {
	if ev.RunRandSeed == 0 {
		ev.RunRandSeed = 173
	}
	randx.InitSysRand(&ev.Rand, ev.RunRandSeed*(int64(run)+1))
	ev.Trial.Init()
	sp := ev.Space()
	its := sp.Train
	if ev.Test {
		its = sp.Test
		if ev.NData > 1 {
			neven := len(its) / ev.NData
			its = its[neven*ev.Di : neven*(ev.Di+1)]
		}
	}
	ev.Items = its
	ev.Order = ev.Rand.Perm(len(its))
	ev.Ctr = -1
}

// Space returns the shared item space.
func (ev *CSEnv) Space() *Space { return ev.Configs.Space() }

// CurItem returns the current item.
func (ev *CSEnv) CurItem() *Item { return &ev.Space().Items[ev.Item] }

func (ev *CSEnv) State(el string) tensor.Values {
	return ev.States[el]
}

// String returns the current item inputs and target group.
func (ev *CSEnv) String() string {
	it := ev.CurItem()
	var b strings.Builder
	for i, in := range it.Inputs {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, "%d", in)
	}
	fmt.Fprintf(&b, " g:%d", ev.Group)
	if len(it.Valid) > 1 {
		fmt.Fprintf(&b, " of %v", it.Valid)
	}
	return b.String()
}

// Step does one trial: selects the next item, the target group and
// output patterns, and renders the states.
func (ev *CSEnv) Step() bool {
	ev.Trial.Incr()
	ev.Ctr++
	if ev.Ctr >= len(ev.Order) {
		randx.PermuteInts(ev.Order, ev.Rand)
		ev.Ctr = 0
	}
	ev.SetItem(ev.Items[ev.Order[ev.Ctr]])
	ev.Render()
	return true
}

// SetItem sets the current item, and samples the target group
// and output patterns.
func (ev *CSEnv) SetItem(item int) {
	cfg := &ev.Configs
	sp := ev.Space()
	ev.Item = item
	it := &sp.Items[item]
	ev.Group = it.Valid[ev.Rand.Intn(len(it.Valid))]
	for s, in := range it.Inputs {
		gi := sp.GroupIndex(in, ev.Group)
		ev.Outputs[s] = sp.OutputIndex(cfg, in, gi, ev.Rand.Intn(cfg.GroupN))
	}
}

// Render renders the current item inputs and target outputs into the states.
func (ev *CSEnv) Render() {
	sp := ev.Space()
	it := ev.CurItem()
	inp := ev.States["Input"]
	out := ev.States["Output"]
	for s, in := range it.Inputs {
		copy(inp.SubSpace(0, s).(*tensor.Float32).Values, sp.Vocab.SubSpace(in).(*tensor.Float32).Values)
		copy(out.SubSpace(0, s).(*tensor.Float32).Values, sp.Vocab.SubSpace(ev.Outputs[s]).(*tensor.Float32).Values)
	}
}
