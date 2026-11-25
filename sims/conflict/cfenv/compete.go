// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfenv

import (
	"maps"
	"math"

	"cogentcore.org/lab/tensor"
)

// CompeteNames are the names of the within-trial competition stats,
// in the order returned by [CompeteStats.Values].
var CompeteNames = []string{"LeadSwaps", "CommitCyc", "LoserFrac", "AltCorr", "Agree50", "Agree100", "LeadSwapsMinor", "LeadSwapsMajor", "CommitCycMinor", "CommitCycMajor"}

// CompeteSkip is the number of cycles at the start of the minus phase
// that AltCorr leaves out, so that the common rise of the two groups
// at stimulus onset does not count as co-activation.
const CompeteSkip = 25

func init() {
	docs := map[string]string{
		"LeadSwaps":      "Test only: on conflict trials with a choice of one of the input groups, the number of times the leading input group (the one with the most summed output activity, sampled every few cycles of the minus phase) changes. A perforated attractor swaps the lead repeatedly on close calls; a smooth one swaps at most once.",
		"CommitCyc":      "Test only: on conflict trials with a choice of one of the input groups, the minus phase cycle of the last change of the leading input group (or of the first lead, if it never changes): when the network commits.",
		"LoserFrac":      "Test only: on conflict trials with a choice of one of the input groups, the share of the losing input group in the summed output activity of the two input groups over the second half of the minus phase: the residual minority signal.",
		"AltCorr":        "Test only: on conflict trials with a choice of one of the input groups, the correlation between the cycle-to-cycle changes of the two input groups' summed output activity, after the onset: negative = they alternate in time (multiplexing), positive = they rise and fall together.",
		"Agree50":        "Test only: on conflict trials with a choice of one of the input groups, 1 if the leading input group at minus phase cycle 50 is the final choice (0 if it is the other group or there is no lead yet).",
		"Agree100":       "Test only: Agree50 at minus phase cycle 100.",
		"LeadSwapsMinor": "Test only: LeadSwaps on unequal conflict trials where the choice is the minority group (less evidence). More swaps than LeadSwapsMajor = minority choices come from late reversals, not from an early lead.",
		"LeadSwapsMajor": "Test only: LeadSwaps on unequal conflict trials where the choice is the majority group (more evidence).",
		"CommitCycMinor": "Test only: CommitCyc on unequal conflict trials where the choice is the minority group.",
		"CommitCycMajor": "Test only: CommitCyc on unequal conflict trials where the choice is the majority group.",
	}
	maps.Copy(TrialStatDocs, docs)
}

// CompeteStats has the within-trial competition stats for one trial
// (see [CompeteNames] and [TrialStatDocs]). All are NaN unless the
// trial is a conflict trial with a choice of one of the input groups.
type CompeteStats struct {
	LeadSwaps, CommitCyc, LoserFrac, AltCorr, Agree50, Agree100 float64

	LeadSwapsMinor, LeadSwapsMajor, CommitCycMinor, CommitCycMajor float64
}

// Values returns the stats in the order of [CompeteNames].
func (cs *CompeteStats) Values() []float64 {
	return []float64{cs.LeadSwaps, cs.CommitCyc, cs.LoserFrac, cs.AltCorr, cs.Agree50, cs.Agree100, cs.LeadSwapsMinor, cs.LeadSwapsMajor, cs.CommitCycMinor, cs.CommitCycMajor}
}

// GroupSums returns the summed values of the output units of each group,
// without a threshold (as for a spiking variable like CaM).
func (ev *CFEnv) GroupSums(out []float32, sums []float32) []float32 {
	cfg := &ev.Configs
	sp := ev.Space()
	if len(sums) != cfg.NGroups {
		sums = make([]float32, cfg.NGroups)
	}
	for g := range cfg.NGroups {
		sum := float32(0)
		for i, p := range sp.Outputs.SubSpace(g).(*tensor.Float32).Values {
			if p > 0 {
				sum += out[i]
			}
		}
		sums[g] = sum
	}
	return sums
}

// Compete computes the within-trial competition stats for the current
// item, from drive[s][g]: the summed output activity of group g at
// sample s, where sample s is at the end of minus phase cycle
// (s+1)*interval, and choice: the chosen group.
func (ev *CFEnv) Compete(drive [][]float32, interval int, choice int, cs *CompeteStats) {
	nan := math.NaN()
	*cs = CompeteStats{nan, nan, nan, nan, nan, nan, nan, nan, nan, nan}
	it := ev.Cur
	if it == nil || len(it.Groups) != 2 || len(drive) == 0 {
		return
	}
	ci := it.GroupIndex(choice)
	if ci < 0 {
		return
	}
	win, lose := it.Groups[ci], it.Groups[1-ci]
	ga, gb := it.Groups[0], it.Groups[1]
	leader := -1
	swaps, commit := 0, -1
	lead50, lead100 := -1, -1
	for s, d := range drive {
		cyc := (s + 1) * interval
		switch {
		case d[ga] > d[gb]:
			if leader != ga {
				if leader >= 0 {
					swaps++
				}
				leader, commit = ga, cyc
			}
		case d[gb] > d[ga]:
			if leader != gb {
				if leader >= 0 {
					swaps++
				}
				leader, commit = gb, cyc
			}
		}
		if cyc <= 50 {
			lead50 = leader
		}
		if cyc <= 100 {
			lead100 = leader
		}
	}
	if commit < 0 { // no activity
		return
	}
	cs.LeadSwaps = float64(swaps)
	cs.CommitCyc = float64(commit)
	agree := func(l int) float64 {
		if l == win {
			return 1
		}
		return 0
	}
	cs.Agree50, cs.Agree100 = agree(lead50), agree(lead100)

	var sw, sl float64
	for _, d := range drive[len(drive)/2:] {
		sw += float64(d[win])
		sl += float64(d[lose])
	}
	if sw+sl > 0 {
		cs.LoserFrac = sl / (sw + sl)
	}

	s0 := CompeteSkip / interval
	var da, db []float64
	for s := max(s0, 1); s < len(drive); s++ {
		da = append(da, float64(drive[s][ga]-drive[s-1][ga]))
		db = append(db, float64(drive[s][gb]-drive[s-1][gb]))
	}
	cs.AltCorr = correlation(da, db)

	if !it.IsTie() {
		if ci == 1 { // Groups are ordered by evidence: 1 is the minority
			cs.LeadSwapsMinor, cs.CommitCycMinor = cs.LeadSwaps, cs.CommitCyc
		} else {
			cs.LeadSwapsMajor, cs.CommitCycMajor = cs.LeadSwaps, cs.CommitCyc
		}
	}
}

// correlation returns the Pearson correlation of a and b,
// or NaN if there are fewer than 2 values or either has no variance.
func correlation(a, b []float64) float64 {
	n := len(a)
	if n < 2 {
		return math.NaN()
	}
	var ma, mb float64
	for i := range n {
		ma += a[i]
		mb += b[i]
	}
	ma /= float64(n)
	mb /= float64(n)
	var sab, saa, sbb float64
	for i := range n {
		x, y := a[i]-ma, b[i]-mb
		sab += x * y
		saa += x * x
		sbb += y * y
	}
	if saa == 0 || sbb == 0 {
		return math.NaN()
	}
	return sab / math.Sqrt(saa*sbb)
}
