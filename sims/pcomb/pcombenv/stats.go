// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcombenv

import (
	"math"
)

// TrialStatNames are the names of the scalar per-trial stats in
// [TrialStats], in order (see [TrialStats.Values]).
var TrialStatNames = []string{"Value", "CueErr", "Miss", "ChoseP", "PGain", "OutPErr", "SalErr", "AgreeP", "AgreeBoost"}

// TrialStatDocs are the docs for each stat, including the epoch-level CalibErr.
var TrialStatDocs = map[string]string{
	"Value":      "Decoded Output value on the current trial.",
	"CueErr":     "Distance between the decoded Output value and the nearest active cue value.",
	"Miss":       "Proportion of trials where the decoded Output value is not within MatchThr of any cue value (e.g., a blend between cue values).",
	"ChoseP":     "Normalized probability of the cue value chosen by the network, for items with distinct cue values (excluding misses). Compare to the probabilities of the available values to see over- or under-weighting.",
	"PGain":      "ChoseP minus 1 / number of cue values, for items with distinct cue values (excluding misses): the gain over choosing a value regardless of the probabilities, which is 0 on average for any choice that ignores the probabilities.",
	"OutPErr":    "Total variation distance between the target probabilities and the Output activity at each cue value, normalized over the cue values, for items with distinct cue values: measures whether the output represents the probabilities, regardless of the decoded peak.",
	"SalErr":     "Mean absolute difference between the Salience layer activity and the salience pattern for the value chosen by the network (excluding misses).",
	"AgreeP":     "For agreement items (with repeated cue values), the summed normalized probability of the repeated values (excluding misses): the frequency of choosing them for calibrated sampling.",
	"AgreeBoost": "For agreement items (with repeated cue values), 1 if the network chose a repeated value, minus AgreeP (excluding misses): positive values mean agreeing cues are favored beyond their summed probability (superlinear).",
	"CalibErr":   "Calibration error, for items with distinct cue values: mean absolute difference between the frequency of choosing values in each normalized probability bin and the mean normalized probability in that bin, weighted by the number of values per bin.",
}

// CalibStatNames are the per-trial calibration counts in [TrialStats],
// with one cell per normalized probability bin, for items with distinct
// cue values (NaN for agreement items): CalN = number of values in the
// bin, CalC = 1 if the chosen value is in the bin, CalP = sum of the
// values' probabilities.
var CalibStatNames = []string{"CalN", "CalC", "CalP"}

// TrialStats are the sampling stats for one trial, computed by
// [PCombEnv.TrialStats] from the network Output and Salience activity.
// Stats that do not apply to the current trial are NaN.
type TrialStats struct {
	Value, CueErr, Miss, ChoseP, PGain, OutPErr, SalErr, AgreeP, AgreeBoost float64

	// CalN, CalC, CalP are the calibration counts per bin
	// (see [CalibStatNames]).
	CalN, CalC, CalP []float64

	salPat []float32
}

// Values returns the scalar stat values in the order of [TrialStatNames].
func (ts *TrialStats) Values() []float64 {
	return []float64{ts.Value, ts.CueErr, ts.Miss, ts.ChoseP, ts.PGain, ts.OutPErr, ts.SalErr, ts.AgreeP, ts.AgreeBoost}
}

// TrialStats computes the stats for the current trial from given
// Output and Salience layer activity (PopCodeUnits each).
func (ev *PCombEnv) TrialStats(out, sal []float32, ts *TrialStats) {
	cfg := &ev.Configs
	nan := math.NaN()
	nb := cfg.NBins
	if len(ts.CalN) != nb {
		ts.CalN = make([]float64, nb)
		ts.CalC = make([]float64, nb)
		ts.CalP = make([]float64, nb)
	}
	vals, probs, counts := ev.ValueCounts()
	valDec := ev.Decode(out)
	idx, dist := ev.Match(vals, valDec)
	miss := dist > cfg.MatchThr
	ts.Value = float64(valDec)
	ts.CueErr = float64(dist)
	ts.Miss = 0
	if miss {
		ts.Miss = 1
	}
	ts.ChoseP, ts.PGain, ts.OutPErr, ts.SalErr, ts.AgreeP, ts.AgreeBoost = nan, nan, nan, nan, nan, nan
	if !miss {
		if len(ts.salPat) != len(sal) {
			ts.salPat = make([]float32, len(sal))
		}
		ev.SaliencePattern(ts.salPat, vals[idx])
		se := 0.0
		for i, a := range sal {
			se += math.Abs(float64(a - ts.salPat[i]))
		}
		ts.SalErr = se / float64(len(sal))
	}
	if ev.Agree {
		for i := range ts.CalN {
			ts.CalN[i], ts.CalC[i], ts.CalP[i] = nan, nan, nan
		}
		if !miss {
			ap := 0.0
			for j, p := range probs {
				if counts[j] > 1 {
					ap += float64(p)
				}
			}
			ts.AgreeP = ap
			ts.AgreeBoost = -ap
			if counts[idx] > 1 {
				ts.AgreeBoost += 1
			}
		}
		return
	}
	if !miss {
		ts.ChoseP = float64(probs[idx])
		ts.PGain = ts.ChoseP - 1/float64(len(vals))
	}
	ts.OutPErr = ev.OutPErr(out, vals, probs)
	clear(ts.CalN)
	clear(ts.CalC)
	clear(ts.CalP)
	for i, p := range probs {
		b := ev.Bin(p)
		ts.CalN[b]++
		ts.CalP[b] += float64(p)
		if !miss && i == idx {
			ts.CalC[b]++
		}
	}
}

// OutPErr returns the total variation distance between the given target
// probabilities and the output activity at the unit nearest to each value,
// normalized over the values. Returns NaN if there is no output activity.
func (ev *PCombEnv) OutPErr(out []float32, vals, probs []float32) float64 {
	pc := &ev.Configs.ValuePop
	n := len(out)
	incr := (pc.Max - pc.Min) / float32(n-1)
	q := make([]float64, len(vals))
	sum := 0.0
	for j, v := range vals {
		ui := min(max(int(math.Round(float64((v-pc.Min)/incr))), 0), n-1)
		q[j] = float64(max(out[ui], 0))
		sum += q[j]
	}
	if sum == 0 {
		return math.NaN()
	}
	tv := 0.0
	for j, p := range probs {
		tv += math.Abs(q[j]/sum - float64(p))
	}
	return 0.5 * tv
}

// CalibStats computes the epoch-level calibration stats from the mean
// per-trial calibration counts for each bin: calib = choice frequency,
// calibP = mean normalized probability, calibN = mean number of values
// per trial, and the calibration error weighted by calibN.
func CalibStats(mn, mc, mp []float64) (calib, calibP, calibN []float64, calibErr float64) {
	nb := len(mn)
	calib = make([]float64, nb)
	calibP = make([]float64, nb)
	calibN = make([]float64, nb)
	errSum, nSum := 0.0, 0.0
	for b := range nb {
		n := mn[b]
		calibN[b] = n
		if n == 0 || math.IsNaN(n) {
			calib[b], calibP[b] = math.NaN(), math.NaN()
			continue
		}
		calib[b] = mc[b] / n
		calibP[b] = mp[b] / n
		errSum += n * math.Abs(calib[b]-calibP[b])
		nSum += n
	}
	calibErr = errSum / nSum
	return
}
