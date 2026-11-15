// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// pcombenv implements the probabilistic combination environment:
// input cues each suggest a probability and value of an outcome,
// and the target output is the value of one cue, sampled according
// to the normalized probabilities across cues.
package pcombenv

//go:generate core generate -add-types -add-funcs

import (
	"fmt"
	"strings"

	"cogentcore.org/core/math32"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/popcode"
)

// PCombEnv is the probabilistic combination environment.
// Each trial presents a number of active cues (in [Config.NActive]),
// in random slots out of [Config.NCues], each with a probability
// and value, encoded as Gaussian bumps in the Prob and Value input pools.
// The Output target is a bump at the value of one cue, sampled with
// probability p_i / sum_j p_j. The Salience target is a copy of the
// Output bump for high magnitude values (positive and negative stay
// separate), the outer half of the bump for medium magnitudes,
// and nothing for low magnitudes.
//
// States:
//   - Value: [NCues, 1, 1, PopCodeUnits]
//   - Prob: [NCues, 1, 1, PopCodeUnits]
//   - Output: [1, PopCodeUnits]
//   - Salience: [1, PopCodeUnits]
type PCombEnv struct {
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

	// Items are the training or (distinct value) testing items for this env.
	Items ItemSet

	// AgreeItems are the agreement items (with repeated values) for testing,
	// used on AgreePct of test trials, if DistinctTrain is set.
	AgreeItems ItemSet

	// Item is the current item index into the item space.
	Item int

	// Agree is true if the current item has repeated cue values.
	Agree bool

	// Slots are the cues in each input slot for the current trial;
	// P = -1 for inactive slots.
	Slots []Cue

	// NormP are the normalized probabilities p_i / sum_j p_j for each slot.
	NormP []float32

	// Chosen is the slot of the sampled cue for the current trial.
	Chosen int

	// Rand is the random number generator for the env.
	Rand randx.Rand `display:"-"`

	// RunRandSeed is the random seed multiplier for the run counter.
	// It is set to 173 if 0 at start for consistent results by default.
	RunRandSeed int64 `edit:"-"`
}

func (ev *PCombEnv) Label() string { return ev.Name }

func (ev *PCombEnv) Defaults() {
	ev.Configs.Defaults()
}

// Config configures the env, after Configs have been set.
func (ev *PCombEnv) Config(trials, ndata, di int, test bool, rndseed int64) {
	cfg := &ev.Configs
	cfg.Update()
	ev.Trial.Max = trials
	ev.NData = ndata
	ev.Di = di
	ev.Test = test
	ev.RunRandSeed = rndseed
	npu := cfg.PopCodeUnits
	ev.States = make(map[string]*tensor.Float32)
	ev.States["Value"] = tensor.NewFloat32(cfg.NCues, 1, 1, npu)
	ev.States["Prob"] = tensor.NewFloat32(cfg.NCues, 1, 1, npu)
	ev.States["Output"] = tensor.NewFloat32(1, npu)
	ev.States["Salience"] = tensor.NewFloat32(1, npu)
	ev.Slots = make([]Cue, cfg.NCues)
	ev.NormP = make([]float32, cfg.NCues)
}

func (ev *PCombEnv) Init(run int) {
	cfg := &ev.Configs
	if ev.RunRandSeed == 0 {
		ev.RunRandSeed = 173
	}
	randx.InitSysRand(&ev.Rand, ev.RunRandSeed*(int64(run)+1))
	ev.Trial.Init()
	sp := cfg.Space()
	if ev.Test {
		ev.Items.Init(ev, sp.Test)
		ev.AgreeItems.Init(ev, sp.Agree)
	} else {
		ev.Items.Init(ev, sp.Train)
		ev.AgreeItems.Init(ev, nil)
	}
}

// ItemSet is a set of items with a permuted order for presenting them,
// organized by number of active cues.
type ItemSet struct {
	// Items are the item indexes, for each number of active cues
	// starting at NActive.Min. Test envs get non-overlapping subsets
	// of the test items for each Di.
	Items [][]int

	// Order is the permuted order of Items, permuted after each pass.
	Order [][]int

	// Ctr is the current position in Order for each number of active cues.
	Ctr []int
}

// Init initializes the item set from given source items.
func (is *ItemSet) Init(ev *PCombEnv, src [][]int) {
	nn := len(src)
	is.Items = make([][]int, nn)
	is.Order = make([][]int, nn)
	is.Ctr = make([]int, nn)
	for k, its := range src {
		if ev.Test && ev.NData > 1 {
			neven := len(its) / ev.NData
			its = its[neven*ev.Di : neven*(ev.Di+1)]
		}
		is.Items[k] = its
		is.Order[k] = ev.Rand.Perm(len(its))
		is.Ctr[k] = -1
	}
}

// Len returns the total number of items.
func (is *ItemSet) Len() int {
	n := 0
	for _, its := range is.Items {
		n += len(its)
	}
	return n
}

// Next returns the next item index.
func (is *ItemSet) Next(ev *PCombEnv) int {
	k := is.nextN(ev)
	is.Ctr[k]++
	if is.Ctr[k] >= len(is.Order[k]) {
		randx.PermuteInts(is.Order[k], ev.Rand)
		is.Ctr[k] = 0
	}
	return is.Items[k][is.Order[k][is.Ctr[k]]]
}

// nextN returns the index (number of active cues - NActive.Min)
// for the next item.
func (is *ItemSet) nextN(ev *PCombEnv) int {
	nn := len(is.Items)
	if ev.Configs.UniformN {
		for {
			k := ev.Rand.Intn(nn)
			if len(is.Items[k]) > 0 {
				return k
			}
		}
	}
	r := ev.Rand.Intn(is.Len())
	for k, its := range is.Items {
		if r < len(its) {
			return k
		}
		r -= len(its)
	}
	return nn - 1
}

func (ev *PCombEnv) State(el string) tensor.Values {
	return ev.States[el]
}

// String returns the current item, with the chosen cue marked by *.
func (ev *PCombEnv) String() string {
	cfg := &ev.Configs
	var b strings.Builder
	for i, c := range ev.Slots {
		if c.P < 0 {
			continue
		}
		if b.Len() > 0 {
			b.WriteString(" ")
		}
		if i == ev.Chosen {
			b.WriteString("*")
		}
		fmt.Fprintf(&b, "%.2g:%+.3g", cfg.Prob(int(c.P)), cfg.Value(int(c.V)))
	}
	return b.String()
}

// Step does one trial: selects the next item, assigns its cues
// to random slots, samples the chosen cue, and renders the states.
func (ev *PCombEnv) Step() bool {
	ev.Trial.Incr()
	if ev.AgreeItems.Len() > 0 && ev.Rand.Float32() < ev.Configs.AgreePct {
		ev.Item = ev.AgreeItems.Next(ev)
	} else {
		ev.Item = ev.Items.Next(ev)
	}
	ev.SetItem(&ev.Configs.Space().Items[ev.Item])
	ev.Render()
	return true
}

// SetItem assigns the item cues to random slots, computes the normalized
// probabilities, and samples the chosen cue.
func (ev *PCombEnv) SetItem(it *Item) {
	cfg := &ev.Configs
	for i := range ev.Slots {
		ev.Slots[i] = Cue{P: -1, V: -1}
		ev.NormP[i] = 0
	}
	ev.Agree = it.HasRepeat()
	perm := ev.Rand.Perm(cfg.NCues)
	sum := float32(0)
	for i, c := range it.Cues {
		ev.Slots[perm[i]] = c
		sum += cfg.Prob(int(c.P))
	}
	for i, c := range ev.Slots {
		if c.P >= 0 {
			ev.NormP[i] = cfg.Prob(int(c.P)) / sum
		}
	}
	r := ev.Rand.Float32()
	ev.Chosen = -1
	for i, p := range ev.NormP {
		if p == 0 {
			continue
		}
		ev.Chosen = i
		if r < p {
			break
		}
		r -= p
	}
}

// Render renders the current slots and chosen cue into the states.
func (ev *PCombEnv) Render() {
	cfg := &ev.Configs
	vals := ev.States["Value"]
	probs := ev.States["Prob"]
	vals.SetZeros()
	probs.SetZeros()
	npu := cfg.PopCodeUnits
	for i, c := range ev.Slots {
		if c.P < 0 {
			continue
		}
		vs := vals.SubSpace(i, 0).(*tensor.Float32)
		cfg.ValuePop.Encode(&vs.Values, cfg.Value(int(c.V)), npu, popcode.Set)
		ps := probs.SubSpace(i, 0).(*tensor.Float32)
		cfg.ProbPop.Encode(&ps.Values, cfg.Prob(int(c.P)), npu, popcode.Set)
	}
	v := ev.ChosenValue()
	out := ev.States["Output"]
	cfg.ValuePop.Encode(&out.Values, v, npu, popcode.Set)
	ev.SaliencePattern(ev.States["Salience"].Values, v)
}

// SaliencePattern renders the salience pattern for given value into pat,
// which must have PopCodeUnits: a copy of the value bump for full salience,
// the outer half of the bump (away from 0) for half salience, else zeros.
func (ev *PCombEnv) SaliencePattern(pat []float32, v float32) {
	cfg := &ev.Configs
	npu := cfg.PopCodeUnits
	sal := cfg.Salience(v)
	if sal == 0 {
		clear(pat)
		return
	}
	cfg.ValuePop.Encode(&pat, v, npu, popcode.Set)
	if sal == 1 {
		return
	}
	pc := &cfg.ValuePop
	incr := (pc.Max - pc.Min) / float32(npu-1)
	for i := range pat {
		trg := pc.Min + incr*float32(i)
		if (v > 0 && trg < v) || (v < 0 && trg > v) {
			pat[i] = 0
		}
	}
}

// ChosenValue returns the value of the chosen cue for the current trial.
func (ev *PCombEnv) ChosenValue() float32 {
	return ev.Configs.Value(int(ev.Slots[ev.Chosen].V))
}

// ValueProbs returns the distinct values among the active cues, and
// the total normalized probability for each value (summed over cues
// with the same value), which is the target probability of producing it.
func (ev *PCombEnv) ValueProbs() (vals, probs []float32) {
	vals, probs, _ = ev.ValueCounts()
	return
}

// ValueCounts returns the distinct values among the active cues, the total
// normalized probability for each value (summed over cues with the same
// value), and the number of cues with each value.
func (ev *PCombEnv) ValueCounts() (vals, probs []float32, counts []int) {
	cfg := &ev.Configs
	for i, c := range ev.Slots {
		if c.P < 0 {
			continue
		}
		v := cfg.Value(int(c.V))
		found := false
		for j, ov := range vals {
			if ov == v {
				probs[j] += ev.NormP[i]
				counts[j]++
				found = true
				break
			}
		}
		if !found {
			vals = append(vals, v)
			probs = append(probs, ev.NormP[i])
			counts = append(counts, 1)
		}
	}
	return
}

// Decode decodes value from a pattern of activation
// as the activation-weighted-average of the unit's preferred
// tuning values.
// must have 2 or more values in pattern pat.
func DecodePeak(pc *popcode.OneD, pat []float32, width int) float32 {
	n := len(pat)
	if n < 2 {
		return 0
	}
	rng := pc.Max - pc.Min
	incr := rng / float32(n-1)
	mx := float32(0)
	mxi := 0
	for i := range pat {
		sum := float32(0)
		ns := 0
		for j := -width; j < width; j++ {
			if i+j < 0 || i+j >= n {
				continue
			}
			act := pat[i]
			if act < pc.Thr {
				act = 0
			}
			sum += act
			ns++
		}
		if ns > 0 {
			sum /= float32(ns)
		}
		if sum > mx {
			mx = sum
			mxi = i
		}
	}
	trg := pc.Min + incr*float32(mxi)
	return trg
}

// Decode returns the value decoded from the given output pattern,
// which must have PopCodeUnits.
func (ev *PCombEnv) Decode(pat []float32) float32 {
	return DecodePeak(&ev.Configs.ValuePop, pat, 1)
}

// Match returns the index into vals (from [PCombEnv.ValueProbs]) of the
// value nearest to the given decoded output value, and the distance to it.
// The output is a miss if dist > [Config.MatchThr].
func (ev *PCombEnv) Match(vals []float32, out float32) (idx int, dist float32) {
	dist = math32.Inf(1)
	for i, v := range vals {
		d := math32.Abs(out - v)
		if d < dist {
			dist = d
			idx = i
		}
	}
	return
}

// Bin returns the calibration bin for given normalized probability.
func (ev *PCombEnv) Bin(p float32) int {
	nb := ev.Configs.NBins
	return min(int(p*float32(nb)), nb-1)
}
