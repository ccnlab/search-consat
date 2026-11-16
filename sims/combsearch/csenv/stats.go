// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package csenv

import (
	"math"
	"slices"

	"cogentcore.org/core/base/slicesx"
	"cogentcore.org/core/math32"
	"cogentcore.org/lab/tensor"
)

// TrialStatNames are the names of the per-trial stats,
// in the order returned by [TrialStats.Values].
var TrialStatNames = []string{"Miss", "Valid", "NSame", "Coherent", "MatchCos", "Margin"}

// TrialStatDocs are the descriptions of the trial stats.
var TrialStatDocs = map[string]string{
	"Miss":     "Proportion of slots whose output does not match any of the output patterns for its input: the closest output pattern has a cosine below MatchThr, or belongs to a different input.",
	"Valid":    "Proportion of slots whose output matches an output pattern for its input in one of the valid groups (shared by all the inputs).",
	"NSame":    "number of slots with the same group.",
	"Coherent": "1 if every slot matches an output pattern for its input, all from the same group: the search target. (Such a group is necessarily valid.)",
	"MatchCos": "Mean cosine across slots between the output activity and the closest output pattern.",
	"Margin":   "Mean across slots of the cosine to the closest output pattern minus the cosine to the next closest one: high for a single clean pattern, near 0 for a blend of two patterns.",
}

// TrialStats has the stats for one trial.
type TrialStats struct {
	Miss, Valid, NSame, Coherent, MatchCos, Margin float64

	// Groups are the groups of the matched output in each slot (-1 = miss).
	Groups []int

	// GroupCount is number of each group active
	GroupCount []int
}

// Values returns the stats in the order of [TrialStatNames].
func (ts *TrialStats) Values() []float64 {
	return []float64{ts.Miss, ts.Valid, ts.NSame, ts.Coherent, ts.MatchCos, ts.Margin}
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
// to the given slot activity, its cosine, and the cosine of the next
// closest output pattern.
func (ev *CSEnv) Match(act []float32) (oi int, cos, cos2 float32) {
	cfg := &ev.Configs
	sp := ev.Space()
	cos, cos2 = -1, -1
	for o := range cfg.NOutputs() {
		c := Cosine(act, sp.Vocab.SubSpace(cfg.NInputs+o).(*tensor.Float32).Values)
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
// output layer activity, which has NSlots pools of PoolUnits each,
// in slot order.
func (ev *CSEnv) TrialStats(out []float32, ts *TrialStats) {
	cfg := &ev.Configs
	sp := ev.Space()
	it := ev.CurItem()
	ns := cfg.NSlots
	nu := cfg.PoolUnits()
	ts.Groups = slicesx.SetLength(ts.Groups, ns)
	ts.GroupCount = slicesx.SetLength(ts.GroupCount, cfg.NGroups)
	nmiss, nvalid := 0, 0
	sumCos, sumMargin := float32(0), float32(0)
	for s, in := range it.Inputs {
		oi, cos, cos2 := ev.Match(out[s*nu : (s+1)*nu])
		sumCos += cos
		sumMargin += cos - cos2
		oin, g, _ := sp.OutputSource(cfg, oi)
		if cos < cfg.MatchThr || !sp.OutputFor(cfg, in, oin, g) {
			ts.Groups[s] = -1
			nmiss++
			continue
		}
		ts.Groups[s] = g
		if slices.Contains(it.Valid, g) {
			nvalid++
		}
	}
	ts.Miss = float64(nmiss) / float64(ns)
	ts.Valid = float64(nvalid) / float64(ns)
	ts.MatchCos = float64(sumCos) / float64(ns)
	ts.Margin = float64(sumMargin) / float64(ns)
	ts.Coherent = math.NaN()
	ts.NSame = math.NaN()
	if nmiss > 0 {
		return
	}
	for g := range ts.GroupCount {
		ts.GroupCount[g] = 0
	}
	for _, g := range ts.Groups {
		ts.GroupCount[g]++
	}
	mx := slices.Max(ts.GroupCount)
	ts.NSame = float64(mx)
	ts.Coherent = 0
	if mx == ns {
		ts.Coherent = 1
	}
}
