// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package otmenv

import (
	"math"

	"cogentcore.org/core/math32"
	"cogentcore.org/lab/tensor"
)

// TrialStatNames are the names of the per-trial stats,
// in the order returned by [TrialStats.Values].
var TrialStatNames = []string{"Valid", "Blend", "Intrusion", "SecondRatio", "OtherRatio", "MatchCos"}

// TrialStatDocs are the descriptions of the trial stats.
var TrialStatDocs = map[string]string{
	"Valid":       "1 if the output is a choice of one of the input's outputs: not a Blend or an Intrusion.",
	"Blend":       "1 if the input's top two outputs are too close: SecondRatio is above BlendThr (and it is not an Intrusion), or there is no output activity.",
	"Intrusion":   "1 if the activity outside of the input's outputs is too high: OtherRatio is above IntrusionThr.",
	"SecondRatio": "The mean (thresholded) output activity of the input's second output divided by that of its top output, over the units that belong to only one of the input's outputs (NaN for NOut = 1).",
	"OtherRatio":  "The summed (thresholded) output activity outside all of the input's outputs, divided by that of the top output (over all of its units), capped at 1.",
	"MatchCos":    "Cosine between the output activity and the closest output pattern, as a raw diagnostic.",
}

// TrialStats has the stats for one trial.
type TrialStats struct {
	Valid, Blend, Intrusion, SecondRatio, OtherRatio, MatchCos float64

	// Out is the chosen output of the input (-1 if not Valid).
	Out int
}

// Values returns the stats in the order of [TrialStatNames].
func (ts *TrialStats) Values() []float64 {
	return []float64{ts.Valid, ts.Blend, ts.Intrusion, ts.SecondRatio, ts.OtherRatio, ts.MatchCos}
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

// Match returns the index (0..NOutputs-1) of the output pattern closest
// to the given activity, and its cosine.
func (ev *OTMEnv) Match(act []float32) (oi int, cos float32) {
	cfg := &ev.Configs
	vc := ev.Vocab()
	cos = -1
	for o := range cfg.NOutputs() {
		c := Cosine(act, vc.SubSpace(cfg.NInputs+o).(*tensor.Float32).Values)
		if c > cos {
			cos = c
			oi = o
		}
	}
	return
}

// OutputActs returns the output activity above ActThr for each of the
// input's outputs: summed over all of its units (full), and the mean over
// only the units that are not in any other output of the input (excl:
// the mean, so that outputs with different numbers of exclusive units are
// comparable). It also returns the summed activity of the units outside
// all of the input's outputs (other).
func (ev *OTMEnv) OutputActs(out []float32) (full, excl []float32, other float32) {
	cfg := &ev.Configs
	vc := ev.Vocab()
	no := cfg.NOut
	pats := make([][]float32, no)
	for o := range no {
		pats[o] = vc.SubSpace(cfg.OutputIndex(ev.Input, o)).(*tensor.Float32).Values
	}
	full = make([]float32, no)
	excl = make([]float32, no)
	nexcl := make([]int, no)
	for i, a := range out {
		n, last := 0, -1
		for o, p := range pats {
			if p[i] > 0 {
				n++
				last = o
			}
		}
		if n == 1 {
			nexcl[last]++
		}
		if a <= cfg.ActThr {
			continue
		}
		switch n {
		case 0:
			other += a
		case 1:
			excl[last] += a
		}
		for o, p := range pats {
			if p[i] > 0 {
				full[o] += a
			}
		}
	}
	for o := range no {
		if nexcl[o] > 0 {
			excl[o] /= float32(nexcl[o])
		}
	}
	return
}

// TrialStats computes the stats for the current trial from the given
// output layer activity.
func (ev *OTMEnv) TrialStats(out []float32, ts *TrialStats) {
	cfg := &ev.Configs
	_, cos := ev.Match(out)
	ts.MatchCos = float64(cos)
	full, excl, other := ev.OutputActs(out)
	top, second := 0, -1
	for o := 1; o < cfg.NOut; o++ {
		if excl[o] > excl[top] {
			second, top = top, o
		} else if second < 0 || excl[o] > excl[second] {
			second = o
		}
	}
	ts.SecondRatio, ts.OtherRatio = math.NaN(), math.NaN()
	if full[top] > 0 {
		ts.OtherRatio = math.Min(float64(other/full[top]), 1)
	} else if other > 0 {
		ts.OtherRatio = 1
	}
	if second >= 0 {
		switch {
		case excl[top] > 0:
			ts.SecondRatio = float64(excl[second] / excl[top])
		case full[top] > 0:
			ts.SecondRatio = 1 // only shared units active
		}
	}
	ts.Valid, ts.Blend, ts.Intrusion = 0, 0, 0
	ts.Out = -1
	switch {
	case other > 0 && ts.OtherRatio > float64(cfg.IntrusionThr):
		ts.Intrusion = 1
	case full[top] == 0 || (second >= 0 && ts.SecondRatio > float64(cfg.BlendThr)):
		ts.Blend = 1
	default:
		ts.Valid = 1
		ts.Out = top
	}
}
