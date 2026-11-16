// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package otmenv

import (
	"cogentcore.org/core/math32"
	"cogentcore.org/lab/tensor"
)

// TrialStatNames are the names of the per-trial stats,
// in the order returned by [TrialStats.Values].
var TrialStatNames = []string{"Valid", "MatchCos", "Margin"}

// TrialStatDocs are the descriptions of the trial stats.
var TrialStatDocs = map[string]string{
	"Valid":    "1 if the output matches one of the output patterns for the input: the closest output pattern has a cosine of at least MatchThr, and belongs to the input.",
	"MatchCos": "Cosine between the output activity and the closest output pattern.",
	"Margin":   "Cosine to the closest output pattern minus the cosine to the next closest one: high for a single clean pattern, near 0 for a blend of two patterns.",
}

// TrialStats has the stats for one trial.
type TrialStats struct {
	Valid, MatchCos, Margin float64

	// Out is the output of the input that is matched (-1 if not Valid).
	Out int
}

// Values returns the stats in the order of [TrialStatNames].
func (ts *TrialStats) Values() []float64 {
	return []float64{ts.Valid, ts.MatchCos, ts.Margin}
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
// to the given activity, its cosine, and the cosine of the next closest
// output pattern.
func (ev *OTMEnv) Match(act []float32) (oi int, cos, cos2 float32) {
	cfg := &ev.Configs
	vc := ev.Vocab()
	cos, cos2 = -1, -1
	for o := range cfg.NOutputs() {
		c := Cosine(act, vc.SubSpace(cfg.NInputs+o).(*tensor.Float32).Values)
		if c > cos {
			cos2 = cos
			cos = c
			oi = o
		} else if c > cos2 {
			cos2 = c
		}
	}
	return
}

// TrialStats computes the stats for the current trial from the given
// output layer activity.
func (ev *OTMEnv) TrialStats(out []float32, ts *TrialStats) {
	cfg := &ev.Configs
	oi, cos, cos2 := ev.Match(out)
	ts.MatchCos = float64(cos)
	ts.Margin = float64(cos - cos2)
	ts.Valid = 0
	ts.Out = -1
	if cos >= cfg.MatchThr && oi/cfg.NOut == ev.Input {
		ts.Valid = 1
		ts.Out = oi % cfg.NOut
	}
}
