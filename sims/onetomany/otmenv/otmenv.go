// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// otmenv implements a one-to-many environment: each of NInputs input
// patterns has NOut output patterns, and the target on each trial is
// one of them, chosen at random.
package otmenv

//go:generate core generate -add-types -add-funcs

import (
	"fmt"

	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
)

// OTMEnv is the one-to-many environment.
// Each trial presents one of the NInputs input patterns, in a permuted
// order, with one of its NOut output patterns as the target,
// chosen at random.
//
// States:
//   - Input: [PoolSize, PoolSize]
//   - Output: [PoolSize, PoolSize]
type OTMEnv struct {
	// Name of environment: Train or Test.
	Name string

	// Configuration options.
	Configs Config

	// Trial counts trials within an epoch.
	Trial env.Counter

	// States are the named state tensors.
	States map[string]*tensor.Float32

	// Order is the permuted order of the inputs, permuted after each pass.
	Order []int

	// Ctr is the current position in Order.
	Ctr int

	// Input is the current input index.
	Input int

	// Out is the current target output, among the NOut outputs of Input.
	Out int

	// Rand is the random number generator for the env.
	Rand randx.Rand `display:"-"`

	// RandSeed is the random seed multiplier for the run counter.
	// It is set to 173 if 0 at start for consistent results by default.
	RandSeed int64 `edit:"-"`
}

func (ev *OTMEnv) Label() string { return ev.Name }

func (ev *OTMEnv) Defaults() {
	ev.Configs.Defaults()
}

// Config configures the env, after Configs have been set.
func (ev *OTMEnv) Config(trials int, rndseed int64) {
	cfg := &ev.Configs
	ev.Trial.Max = trials
	ev.RandSeed = rndseed
	ps := cfg.PoolSize
	ev.States = make(map[string]*tensor.Float32)
	ev.States["Input"] = tensor.NewFloat32(ps, ps)
	ev.States["Output"] = tensor.NewFloat32(ps, ps)
}

func (ev *OTMEnv) Init(run int) {
	if ev.RandSeed == 0 {
		ev.RandSeed = 173
	}
	randx.InitSysRand(&ev.Rand, ev.RandSeed*(int64(run)+1))
	ev.Trial.Init()
	ev.Order = ev.Rand.Perm(ev.Configs.NInputs)
	ev.Ctr = -1
}

// Vocab returns the shared vocabulary.
func (ev *OTMEnv) Vocab() *tensor.Float32 { return ev.Configs.Vocab() }

func (ev *OTMEnv) State(el string) tensor.Values {
	return ev.States[el]
}

// String returns the current input and target output.
func (ev *OTMEnv) String() string {
	return fmt.Sprintf("%d o:%d", ev.Input, ev.Out)
}

// Step does one trial: selects the next input and a random target
// output for it, and renders the states.
func (ev *OTMEnv) Step() bool {
	ev.Trial.Incr()
	ev.Ctr++
	if ev.Ctr >= len(ev.Order) {
		randx.PermuteInts(ev.Order, ev.Rand)
		ev.Ctr = 0
	}
	ev.Input = ev.Order[ev.Ctr]
	ev.Out = ev.Rand.Intn(ev.Configs.NOut)
	ev.Render()
	return true
}

// Render renders the current input and target output into the states.
func (ev *OTMEnv) Render() {
	vc := ev.Vocab()
	copy(ev.States["Input"].Values, vc.SubSpace(ev.Input).(*tensor.Float32).Values)
	copy(ev.States["Output"].Values, vc.SubSpace(ev.Configs.OutputIndex(ev.Input, ev.Out)).(*tensor.Float32).Values)
}
