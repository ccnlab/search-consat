// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcomb

import (
	"math"

	"cogentcore.org/core/base/metadata"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/stats"
	"github.com/ccnlab/search-consat/sims/pcomb/pcombenv"
	"github.com/emer/axon/v2/axon"
)

// samplingStats are the scalar sampling stats computed per trial.
var samplingStats = []string{"Value", "CueErr", "Miss", "ChoseP", "SalErr"}

var samplingDocs = map[string]string{
	"Value":    "Decoded value on the current trial.",
	"CueErr":   "Distance between the decoded Output value (minus phase) and the nearest active cue value.",
	"Miss":     "Proportion of trials where the decoded Output value is not within MatchThr of any cue value (e.g., a blend between cue values).",
	"ChoseP":   "Normalized probability of the cue value chosen by the network (excluding misses), summed over all cues with that value, because the output cannot distinguish them. Compare to the probabilities of the available values to see over- or under-weighting.",
	"SalErr":   "Mean absolute difference between the Salience layer activity (minus phase) and the salience pattern for the value chosen by the network (excluding misses).",
	"CalibErr": "Calibration error: mean absolute difference between the frequency of choosing values in each normalized probability bin and the mean normalized probability in that bin, weighted by the number of values per bin. Each distinct cue value counts once per trial, with its normalized probability summed over all cues with that value.",
}

// calibStats are per-trial calibration counts, with one cell per
// normalized probability bin, accumulated over the epoch in
// Current/<mode>/Calib/Trial (not Stats, because they are vectors that
// cannot be plotted as such). Each distinct cue value counts once,
// binned by its normalized probability summed over all cues with that
// value: CalN = number of distinct values in the bin, CalC = 1 if the
// chosen value is in the bin, CalP = sum of the values' probabilities.
var calibStats = []string{"CalN", "CalC", "CalP"}

// layerValues returns the given variable for all neurons in the layer for di.
func layerValues(ly *axon.Layer, vr axon.NeuronVars, di int, vals []float32) []float32 {
	idx := &ly.Params.Indexes
	nn := int(idx.NNeurons)
	if len(vals) != nn {
		vals = make([]float32, nn)
	}
	for lni := range nn {
		vals[lni] = axon.Neurons.Value3D(int(idx.NeurSt)+lni, di, int(vr))
	}
	return vals
}

// StatSampling returns the stats function for the sampling performance
// of the network, based on the decoded Output value relative to the cue
// values, and the calibration of choice frequencies relative to the
// normalized probabilities. At the end of each Epoch,
// Current/<mode>/Calib/Epoch has Calib = the choice frequency for each
// normalized probability bin, CalibP = the mean normalized probability
// in each bin, and CalibN = the mean number of distinct values per trial
// in each bin, as [NBins] vectors for the most recent epoch.
// These are vectors, so they are kept out of the plotted Stats levels;
// the scalar CalibErr summary is recorded in Stats.
func (ss *Sim) StatSampling() func(mode Modes, level Levels, start bool) {
	var outVals, salVals, salPat []float32
	return func(mode Modes, level Levels, start bool) {
		if level <= Cycle {
			return
		}
		ev0 := ss.Envs.ByModeDi(Train, 0).(*pcombenv.PCombEnv)
		nbins := ev0.Configs.NBins
		modeDir := ss.Stats.Dir(mode.String())
		curModeDir := ss.Current.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		subDir := modeDir.Dir((level - 1).String())
		calDir := curModeDir.Dir("Calib").Dir(level.String())
		calSubDir := curModeDir.Dir("Calib").Dir((level - 1).String())
		scalars := samplingStats
		if level > Trial {
			scalars = append(scalars[:len(scalars):len(scalars)], "CalibErr")
		}
		if start {
			for _, name := range scalars {
				tsr := levelDir.Float64(name)
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) {
					s.Range.SetMin(0).SetMax(1)
					s.On = (name != "SalErr" && name != "Value")
				})
				metadata.SetDoc(tsr, samplingDocs[name])
			}
			if level == Trial {
				for _, name := range calibStats {
					calDir.Float64(name, 0, nbins).SetNumRows(0)
				}
			}
			return
		}
		switch level {
		case Trial:
			ndata := int(ss.Net.Context().NData)
			outLy := ss.Net.LayerByName("Output")
			salLy := ss.Net.LayerByName("Salience")
			cn := make([]float64, nbins)
			cc := make([]float64, nbins)
			cp := make([]float64, nbins)
			for di := range ndata {
				ev := ss.Envs.ByModeDi(mode, di).(*pcombenv.PCombEnv)
				outVals = layerValues(outLy, axon.ActM, di, outVals)
				salVals = layerValues(salLy, axon.ActM, di, salVals)
				vals, probs := ev.ValueProbs()
				valDec := ev.Decode(outVals)
				idx, dist := ev.Match(vals, valDec)
				miss := dist > ev.Configs.MatchThr
				choseP, salErr := math.NaN(), math.NaN()
				if !miss {
					choseP = float64(probs[idx])
					if len(salPat) != len(salVals) {
						salPat = make([]float32, len(salVals))
					}
					ev.SaliencePattern(salPat, vals[idx])
					se := 0.0
					for i, a := range salVals {
						se += math.Abs(float64(a - salPat[i]))
					}
					salErr = se / float64(len(salVals))
				}
				missF := 0.0
				if miss {
					missF = 1
				}
				for _, name := range samplingStats {
					var stat float64
					switch name {
					case "Value":
						stat = float64(valDec)
					case "CueErr":
						stat = float64(dist)
					case "Miss":
						stat = missF
					case "ChoseP":
						stat = choseP
					case "SalErr":
						stat = salErr
					}
					curModeDir.Float64(name, ndata).SetFloat1D(stat, di)
					levelDir.Float64(name).AppendRowFloat(stat)
				}
				clear(cn)
				clear(cc)
				clear(cp)
				for i, p := range probs {
					b := ev.Bin(p)
					cn[b]++
					cp[b] += float64(p)
					if !miss && i == idx {
						cc[b]++
					}
				}
				calDir.Float64("CalN").AppendRowFloat(cn...)
				calDir.Float64("CalC").AppendRowFloat(cc...)
				calDir.Float64("CalP").AppendRowFloat(cp...)
			}
		case Epoch:
			for _, name := range samplingStats {
				levelDir.Float64(name).AppendRowFloat(stats.StatMean.Call(subDir.Value(name)).Float1D(0))
			}
			mn := stats.StatMean.Call(calSubDir.Value("CalN"))
			mc := stats.StatMean.Call(calSubDir.Value("CalC"))
			mp := stats.StatMean.Call(calSubDir.Value("CalP"))
			calib := make([]float64, nbins)
			calibP := make([]float64, nbins)
			calibN := make([]float64, nbins)
			errSum, nSum := 0.0, 0.0
			for b := range nbins {
				n := mn.Float1D(b)
				calibN[b] = n
				if n == 0 {
					calib[b], calibP[b] = math.NaN(), math.NaN()
					continue
				}
				calib[b] = mc.Float1D(b) / n
				calibP[b] = mp.Float1D(b) / n
				errSum += n * math.Abs(calib[b]-calibP[b])
				nSum += n
			}
			for name, vals := range map[string][]float64{"Calib": calib, "CalibP": calibP, "CalibN": calibN} {
				tsr := calDir.Float64(name, nbins)
				for b, v := range vals {
					tsr.SetFloat1D(v, b)
				}
			}
			levelDir.Float64("CalibErr").AppendRowFloat(errSum / nSum)
		case Run:
			for _, name := range scalars {
				levelDir.Float64(name).AppendRowFloat(stats.StatFinal.Call(subDir.Value(name)).Float1D(0))
			}
		default:
			for _, name := range scalars {
				levelDir.Float64(name).AppendRowFloat(stats.StatMean.Call(subDir.Value(name)).Float1D(0))
			}
		}
	}
}
