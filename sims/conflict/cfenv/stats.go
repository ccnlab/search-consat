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
var TrialStatNames = []string{"Blend", "Intrusion", "SecondRatio", "OtherRatio", "MatchCos", "Slot0", "RepeatPrev"}

// TrialStatDocs are the descriptions of the trial stats.
var TrialStatDocs = map[string]string{
	"Blend":       "1 if the two input groups are too close: SecondRatio is above BlendThr (and it is not an Intrusion), or there is no output activity.",
	"Intrusion":   "1 if the activity of the groups that are not in the inputs is too high: OtherRatio is above IntrusionThr.",
	"SecondRatio": "On conflict trials: the summed (thresholded) output activity of the second input group, divided by that of the top input group (the one with more activity).",
	"OtherRatio":  "The summed (thresholded) output activity of all the groups that are not in the inputs, divided by that of the top input group, capped at 1.",
	"MatchCos":    "Cosine between the output activity and the closest output pattern.",
	"Slot0":       "On tied conflict trials (two input groups with equal evidence) with a choice of one of them: 1 if it is the group in slot 0. 0.5 = no slot bias.",
	"RepeatPrev":  "On tied conflict trials (two input groups with equal evidence) with a choice, where the previous trial's choice is one of the input groups: 1 if the choice repeats the previous one. 0.5 = no carryover bias.",
	"AbsDiff":     "Mean across target probabilities of the absolute difference between the measured probability (P<target>) and the target.",
}

// TargetName returns the stat name for a target probability as a
// rounded percent, e.g., P50.
func TargetName(pct int) string { return fmt.Sprintf("P%d", pct) }

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

// TrialStats has the stats for one trial.
type TrialStats struct {
	Blend, Intrusion, SecondRatio, OtherRatio, MatchCos, Slot0, RepeatPrev float64

	// Probs are the per-trial probability stats for each target in
	// [Space.Targets]: the mean over the input groups with that expected
	// probability of whether that group was chosen (NaN if none).
	Probs []float64

	// GroupActs are the summed thresholded output activities of each group.
	GroupActs []float32

	// Choice is the chosen group: the top input group, or the most
	// active other group for an Intrusion (-1 = Blend).
	Choice int
}

// Values returns the fixed stats in the order of [TrialStatNames].
func (ts *TrialStats) Values() []float64 {
	return []float64{ts.Blend, ts.Intrusion, ts.SecondRatio, ts.OtherRatio, ts.MatchCos, ts.Slot0, ts.RepeatPrev}
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
		ts.Probs[ti] = math.NaN()
		if n > 0 {
			ts.Probs[ti] = sum / float64(n)
		}
	}
	ts.Slot0, ts.RepeatPrev = math.NaN(), math.NaN()
	if it.IsTie() && gi >= 0 {
		ts.Slot0 = 0
		if ts.Choice == cfg.Group(it.Inputs[0]) {
			ts.Slot0 = 1
		}
		if it.GroupIndex(ev.PrevChoice) >= 0 {
			ts.RepeatPrev = 0
			if ts.Choice == ev.PrevChoice {
				ts.RepeatPrev = 1
			}
		}
	}
	ev.PrevChoice = ts.Choice
}
