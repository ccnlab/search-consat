// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfenv

import (
	"fmt"
	"math"

	"cogentcore.org/core/math32"
	"cogentcore.org/lab/tensor"
)

// TrialStatNames are the names of the fixed per-trial stats,
// in the order returned by [TrialStats.Values]. The probability
// stats for each target probability follow (see [CFEnv.TargetNames]).
var TrialStatNames = []string{"Blend", "Intrusion", "SecondRatio", "OtherRatio", "MatchCos", "Slot0", "RepeatPrev", "IntrusionPrev", "MinorPrevMinor", "MinorPrevMajor", "MinorPrevOther"}

// TrialStatDocs are the descriptions of the trial stats.
var TrialStatDocs = map[string]string{
	"Blend":          "1 if the two input groups are too close: SecondRatio is above BlendThr (and it is not an Intrusion), or there is no output activity.",
	"Intrusion":      "1 if the activity of the groups that are not in the inputs is too high: OtherRatio is above IntrusionThr.",
	"SecondRatio":    "On conflict trials: the summed (thresholded) output activity of the second input group, divided by that of the top input group (the one with more activity).",
	"OtherRatio":     "The summed (thresholded) output activity of all the groups that are not in the inputs, divided by that of the top input group, capped at 1.",
	"MatchCos":       "Cosine between the output activity and the closest output pattern.",
	"Slot0":          "On tied conflict trials (two input groups with equal evidence) with a choice of one of them: 1 if it is the group in slot 0. 0.5 = no slot bias.",
	"RepeatPrev":     "On tied conflict trials (two input groups with equal evidence) with a choice, where the previous trial's choice is one of the input groups: 1 if the choice repeats the previous one. 0.5 = no carryover bias.",
	"IntrusionPrev":  "On Intrusion trials where the previous trial's choice is a group that is not in the inputs: 1 if the intrusion is that previous group. 1 / (NGroups - 2) = no relation to the previous trial (for 2 input groups).",
	"MinorPrevMinor": "On unequal conflict trials (two input groups with different evidence) with a choice of one of them, where the previous trial's choice is the minority group (less evidence): 1 if the choice is the minority group. Compare with MinorPrevOther: higher = carryover of the previous choice drives minority choices.",
	"MinorPrevMajor": "On unequal conflict trials with a choice of one of the input groups, where the previous trial's choice is the majority group (more evidence): 1 if the choice is the minority group.",
	"MinorPrevOther": "On unequal conflict trials with a choice of one of the input groups, where the previous trial's choice is a group that is not in the inputs: 1 if the choice is the minority group. The baseline for the minority choice without carryover to either input group.",
	"Consist":        "Test only: the consistency of the choice for the same conflict item across its repetitions: the proportion of repetitions with its most common choice (Blend counts as a choice), averaged over the conflict items. 1 = the same choice every time.",
	"TieConsist":     "Test only: Consist for the tied items (equal evidence for the two groups): 1 = the same choice every time, 0.5 = a coin flip.",
	"SlotRange":      "Test only: the range (max - min) across slots of the Slot<n> stats: how much the choice on ties depends on which slots the groups are in. 0 = no slot bias.",
	"AbsDiff":        "Mean across target probabilities of the absolute difference between the measured probability (P<target>) and the target.",
	"InAbsDiff":      "AbsDiff for the input-only probabilities (In<target>): mean absolute difference from the targets, among only the choices of an input group.",
}

// SlotName returns the stat name for the choice on ties of the group
// in the given slot, e.g., Slot2.
func SlotName(slot int) string { return fmt.Sprintf("Slot%d", slot) }

// SlotDoc returns the doc for the stat of a slot.
func SlotDoc(slot int) string {
	return fmt.Sprintf("On tied conflict trials (two input groups with equal evidence) with a choice of one of them: 1 if it is the group in slot %d. 0.5 = no bias for or against this slot.", slot)
}

// SlotNames returns the stat names for the slots after slot 0
// (Slot0 is one of the fixed [TrialStatNames]).
func (ev *CFEnv) SlotNames() []string {
	var nms []string
	for s := 1; s < ev.Configs.NSlots; s++ {
		nms = append(nms, SlotName(s))
	}
	return nms
}

// TargetName returns the stat name for a target probability as a
// rounded percent, e.g., P50.
func TargetName(pct int) string { return fmt.Sprintf("P%d", pct) }

// InTargetName returns the stat name for a target probability among
// only the choices of an input group, e.g., In50.
func InTargetName(pct int) string { return fmt.Sprintf("In%d", pct) }

// InTargetDoc returns the doc for the input-only stat of a target probability.
func InTargetDoc(pct int) string {
	return fmt.Sprintf("Probability of choosing an input group whose expected probability is %d%%, among only the trials where the choice is one of the input groups: Blends and Intrusions are left out. Separates the shape of the choice curve from the rate of blends and intrusions.", pct)
}

// TargetDoc returns the doc for the stat of a target probability.
func TargetDoc(pct int) string {
	return fmt.Sprintf("Probability of choosing an input group whose expected probability is %d%%. Blends and intrusions count as not choosing it.", pct)
}

// TargetNames returns the stat names for the target probabilities
// in the test items.
func (ev *CFEnv) TargetNames() []string {
	var nms []string
	for _, p := range ev.Space().Targets {
		nms = append(nms, TargetName(p))
	}
	return nms
}

// InTargetNames returns the input-only stat names for the target
// probabilities in the test items (see [InTargetName]).
func (ev *CFEnv) InTargetNames() []string {
	var nms []string
	for _, p := range ev.Space().Targets {
		nms = append(nms, InTargetName(p))
	}
	return nms
}

// TrialStats has the stats for one trial.
type TrialStats struct {
	Blend, Intrusion, SecondRatio, OtherRatio, MatchCos, Slot0, RepeatPrev, IntrusionPrev float64

	// MinorPrevMinor, MinorPrevMajor, and MinorPrevOther are whether the
	// choice is the minority group on unequal conflict trials, split by
	// the previous trial's choice: the minority group, the majority group,
	// or a group not in the inputs (see [TrialStatDocs]).
	MinorPrevMinor, MinorPrevMajor, MinorPrevOther float64

	// Probs are the per-trial probability stats for each target in
	// [Space.Targets]: the mean over the input groups with that expected
	// probability of whether that group was chosen (NaN if none).
	Probs []float64

	// InProbs are like Probs, but only for trials where the choice is one
	// of the input groups (all NaN for a Blend or Intrusion), so their
	// mean over trials is the probability among only those choices.
	InProbs []float64

	// Slots are the slot stats for each slot: on tied conflict trials
	// with a choice of one of the input groups, 1 if the choice is the
	// group in that slot (NaN otherwise). Slots[0] = Slot0.
	Slots []float64

	// GroupActs are the summed thresholded output activities of each group.
	GroupActs []float32

	// Choice is the chosen group: the top input group, or the most
	// active other group for an Intrusion (-1 = Blend).
	Choice int
}

// Values returns the fixed stats in the order of [TrialStatNames].
func (ts *TrialStats) Values() []float64 {
	return []float64{ts.Blend, ts.Intrusion, ts.SecondRatio, ts.OtherRatio, ts.MatchCos, ts.Slot0, ts.RepeatPrev, ts.IntrusionPrev, ts.MinorPrevMinor, ts.MinorPrevMajor, ts.MinorPrevOther}
}

// Cosine returns the cosine between a and b (0 if either is all zeros).
func Cosine(a, b []float32) float32 {
	var ab, aa, bb float32
	for i, av := range a {
		bv := b[i]
		ab += av * bv
		aa += av * av
		bb += bv * bv
	}
	if aa == 0 || bb == 0 {
		return 0
	}
	return ab / math32.Sqrt(aa*bb)
}

// Match returns the group whose output pattern is closest to the given
// output activity, and its cosine.
func (ev *CFEnv) Match(act []float32) (g int, cos float32) {
	sp := ev.Space()
	cos = -1
	for o := range ev.Configs.NGroups {
		c := Cosine(act, sp.Outputs.SubSpace(o).(*tensor.Float32).Values)
		if c > cos {
			cos = c
			g = o
		}
	}
	return
}

// GroupActs sets the summed output activity of each group over the
// units of its output pattern, counting only activity above ActThr.
// Units shared by the patterns of several groups count for each of them.
func (ev *CFEnv) GroupActs(out []float32, acts []float32) []float32 {
	cfg := &ev.Configs
	sp := ev.Space()
	if len(acts) != cfg.NGroups {
		acts = make([]float32, cfg.NGroups)
	}
	for g := range cfg.NGroups {
		sum := float32(0)
		for i, p := range sp.Outputs.SubSpace(g).(*tensor.Float32).Values {
			if p > 0 && out[i] > cfg.ActThr {
				sum += out[i]
			}
		}
		acts[g] = sum
	}
	return acts
}

// TrialStats computes the stats for the current trial from the given
// output layer activity, and updates PrevChoice.
func (ev *CFEnv) TrialStats(out []float32, ts *TrialStats) {
	cfg := &ev.Configs
	sp := ev.Space()
	it := ev.Cur
	_, cos := ev.Match(out)
	ts.MatchCos = float64(cos)
	ts.GroupActs = ev.GroupActs(out, ts.GroupActs)
	top, second := -1, -1 // input groups by activity
	other, otherG := float32(0), -1
	otherMax := float32(-1)
	for g, a := range ts.GroupActs {
		if it.GroupIndex(g) < 0 {
			other += a
			if a > otherMax {
				otherMax, otherG = a, g
			}
			continue
		}
		switch {
		case top < 0 || a > ts.GroupActs[top]:
			second, top = top, g
		case second < 0 || a > ts.GroupActs[second]:
			second = g
		}
	}
	topAct := ts.GroupActs[top]
	ts.SecondRatio, ts.OtherRatio = math.NaN(), math.NaN()
	if topAct > 0 {
		ts.OtherRatio = math.Min(float64(other/topAct), 1)
		if second >= 0 {
			ts.SecondRatio = float64(ts.GroupActs[second] / topAct)
		}
	} else if other > 0 {
		ts.OtherRatio = 1
	}
	ts.Blend, ts.Intrusion = 0, 0
	ts.Choice = top
	switch {
	case other > 0 && ts.OtherRatio > float64(cfg.IntrusionThr):
		ts.Intrusion = 1
		ts.Choice = otherG
	case topAct == 0 || (second >= 0 && ts.SecondRatio > float64(cfg.BlendThr)):
		ts.Blend = 1
		ts.Choice = -1
	}
	gi := it.GroupIndex(ts.Choice)
	if len(ts.Probs) != len(sp.Targets) {
		ts.Probs = make([]float64, len(sp.Targets))
		ts.InProbs = make([]float64, len(sp.Targets))
	}
	for ti, tp := range sp.Targets {
		sum, n := 0.0, 0
		for i, p := range it.Probs {
			if Percent(p) != tp {
				continue
			}
			n++
			if i == gi {
				sum++
			}
		}
		ts.Probs[ti], ts.InProbs[ti] = math.NaN(), math.NaN()
		if n > 0 {
			ts.Probs[ti] = sum / float64(n)
			if gi >= 0 {
				ts.InProbs[ti] = ts.Probs[ti]
			}
		}
	}
	if len(ts.Slots) != cfg.NSlots {
		ts.Slots = make([]float64, cfg.NSlots)
	}
	ts.RepeatPrev = math.NaN()
	tie := it.IsTie() && gi >= 0
	for s, in := range it.Inputs {
		ts.Slots[s] = math.NaN()
		if tie {
			ts.Slots[s] = 0
			if ts.Choice == cfg.Group(in) {
				ts.Slots[s] = 1
			}
		}
	}
	ts.Slot0 = ts.Slots[0]
	if tie {
		if it.GroupIndex(ev.PrevChoice) >= 0 {
			ts.RepeatPrev = 0
			if ts.Choice == ev.PrevChoice {
				ts.RepeatPrev = 1
			}
		}
	}
	ts.IntrusionPrev = math.NaN()
	if ts.Intrusion == 1 && ev.PrevChoice >= 0 && it.GroupIndex(ev.PrevChoice) < 0 {
		ts.IntrusionPrev = 0
		if ts.Choice == ev.PrevChoice {
			ts.IntrusionPrev = 1
		}
	}
	ts.minorPrev(it, gi, ev.PrevChoice)
	ev.PrevChoice = ts.Choice
	if ev.IsTest && ev.Counts != nil && ev.Ctr >= 0 {
		idx := ev.Items[ev.Order[ev.Ctr]]
		c := ev.Counts[idx]
		if c == nil {
			c = make([]int, cfg.NGroups+1)
			ev.Counts[idx] = c
		}
		c[ts.Choice+1]++
	}
}

// minorPrev sets the MinorPrev* stats for item it, with the index gi of
// the choice in its groups, and the previous trial's choice prev.
func (ts *TrialStats) minorPrev(it *Item, gi, prev int) {
	ts.MinorPrevMinor, ts.MinorPrevMajor, ts.MinorPrevOther = math.NaN(), math.NaN(), math.NaN()
	if len(it.Probs) != 2 || it.IsTie() || gi < 0 || prev < 0 {
		return
	}
	minor := 1
	if it.Probs[0] < it.Probs[1] {
		minor = 0
	}
	v := 0.0
	if gi == minor {
		v = 1
	}
	switch it.GroupIndex(prev) {
	case minor:
		ts.MinorPrevMinor = v
	case 1 - minor:
		ts.MinorPrevMajor = v
	default:
		ts.MinorPrevOther = v
	}
}

// ItemConsistency returns the consistency of the choices for the same
// test item across its repetitions in the current test, combining the
// counts of the given test envs: for each item, the proportion of its
// repetitions that produced its most common choice (with Blend as one of
// the choices), averaged over the conflict items (consist), and over only
// the tied items (ties). 1 = the same choice every time (deterministic
// for each item); for ties, 0.5 = a coin flip between the two groups.
// NaN if there are no such items.
func ItemConsistency(envs ...*CFEnv) (consist, ties float64) {
	if len(envs) == 0 {
		return math.NaN(), math.NaN()
	}
	sp := envs[0].Space()
	counts := map[int][]int{}
	for _, ev := range envs {
		for idx, c := range ev.Counts {
			if counts[idx] == nil {
				counts[idx] = make([]int, len(c))
			}
			for i, n := range c {
				counts[idx][i] += n
			}
		}
	}
	var sum, tsum float64
	var n, tn int
	for idx, c := range counts {
		it := sp.Test[idx]
		if it.NConflict == 0 {
			continue
		}
		tot, mx := 0, 0
		for _, k := range c {
			tot += k
			mx = max(mx, k)
		}
		if tot == 0 {
			continue
		}
		p := float64(mx) / float64(tot)
		sum += p
		n++
		if it.IsTie() {
			tsum += p
			tn++
		}
	}
	consist, ties = math.NaN(), math.NaN()
	if n > 0 {
		consist = sum / float64(n)
	}
	if tn > 0 {
		ties = tsum / float64(tn)
	}
	return
}

// MatchNames are the names of the probability-matching shape stats,
// computed above the Trial level from the input-only probabilities
// (In<target>) by [MatchSlope], for the target pairs from [MatchPairs].
var MatchNames = []string{"MatchCenter", "MatchExtreme"}

func init() {
	TrialStatDocs["MatchCenter"] = "Test only: the slope of the input-only choice curve for the complementary target pair closest to 50% (e.g., (In67 - In33) / .34): 1 = probability matching, above 1 = sharper than matching (toward winner-take-all), below 1 = flatter."
	TrialStatDocs["MatchExtreme"] = "Test only: the slope of the input-only choice curve for the complementary target pair farthest from 50%, short of 100/0 (e.g., (In83 - In17) / .66): 1 = probability matching, above 1 = sharper than matching, below 1 = flatter."
}

// MatchPairs returns the higher target of the complementary target
// pairs (p, 100-p, with 50 < p < 100, both in targets) closest to and
// farthest from 50, for the MatchCenter and MatchExtreme stats
// (0 if there is no such pair).
func MatchPairs(targets []int) (center, extreme int) {
	has := map[int]bool{}
	for _, p := range targets {
		has[p] = true
	}
	for _, p := range targets {
		if p <= 50 || p >= 100 || !has[100-p] {
			continue
		}
		if center == 0 || p < center {
			center = p
		}
		if p > extreme {
			extreme = p
		}
	}
	return
}

// MatchSlope returns the slope of the choice curve between the
// probabilities hi and lo of choosing the groups with the complementary
// targets pct and 100-pct: (hi - lo) / ((2 pct - 100) / 100). 1 = matching.
// It returns NaN if pct is 0 (no pair).
func MatchSlope(hi, lo float64, pct int) float64 {
	if pct == 0 {
		return math.NaN()
	}
	return (hi - lo) / (float64(2*pct-100) / 100)
}
