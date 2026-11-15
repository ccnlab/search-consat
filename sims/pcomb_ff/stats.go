// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcombff

import (
	"cogentcore.org/core/base/metadata"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/tensor"
	"github.com/ccnlab/search-consat/sims/pcomb/pcombenv"
	"github.com/emer/egomlx/egomlx"
)

// layerValues returns the given variable for all units in the layer for di.
func layerValues(ly *egomlx.Layer, varNm string, di int, vals []float32) []float32 {
	vi, err := egomlx.NeuronVarIndexByName(varNm)
	if err != nil {
		return vals
	}
	nn := ly.Shape.Len()
	if len(vals) != nn {
		vals = make([]float32, nn)
	}
	for ui := range nn {
		vals[ui] = ly.UnitValue1D(vi, ui, di)
	}
	return vals
}

// StatSampling returns the stats function for the sampling performance
// of the network, based on the decoded Output value relative to the cue
// values, computed by [pcombenv.PCombEnv.TrialStats] from the Output and
// Salience Act values (see [pcombenv.TrialStatDocs]).
// ChoseP, PGain, OutPErr and the calibration stats are only computed for
// items with distinct cue values, and AgreeP and AgreeBoost only for
// agreement items with repeated values (test only, by default).
// The per-trial calibration counts are accumulated over the epoch in
// Current/<mode>/Calib/Trial, and at the end of each Epoch,
// Current/<mode>/Calib/Epoch has Calib = the choice frequency for each
// normalized probability bin, CalibP = the mean normalized probability
// in each bin, and CalibN = the mean number of values per trial in each
// bin, as [NBins] vectors for the most recent epoch. These are vectors,
// so they are kept out of the plotted Stats levels; the scalar CalibErr
// summary is recorded in Stats.
func (ss *Sim) StatSampling() func(mode Modes, level Levels, start bool) {
	var outVals, salVals []float32
	var ts pcombenv.TrialStats
	names := pcombenv.TrialStatNames
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
		scalars := names
		if level > Trial {
			scalars = append(scalars[:len(scalars):len(scalars)], "CalibErr")
		}
		if start {
			for _, name := range scalars {
				tsr := levelDir.Float64(name)
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) {
					s.Range.SetMin(0).SetMax(1)
					switch name {
					case "Value", "SalErr", "AgreeP":
						s.On = false
					case "PGain", "AgreeBoost":
						s.On = true
						s.Range.SetMin(-0.2).SetMax(0.2)
					default:
						s.On = true
					}
				})
				metadata.SetDoc(tsr, pcombenv.TrialStatDocs[name])
			}
			if level == Trial {
				for _, name := range pcombenv.CalibStatNames {
					calDir.Float64(name, 0, nbins).SetNumRows(0)
				}
			}
			return
		}
		switch level {
		case Trial:
			ndata := ss.Net.NData
			outLy := ss.Net.LayerByName("Output")
			salLy := ss.Net.LayerByName("Salience")
			for di := range ndata {
				ev := ss.Envs.ByModeDi(mode, di).(*pcombenv.PCombEnv)
				outVals = layerValues(outLy, "Act", di, outVals)
				salVals = layerValues(salLy, "Act", di, salVals)
				ev.TrialStats(outVals, salVals, &ts)
				for i, stat := range ts.Values() {
					name := names[i]
					curModeDir.Float64(name, ndata).SetFloat1D(stat, di)
					levelDir.Float64(name).AppendRowFloat(stat)
				}
				calDir.Float64("CalN").AppendRowFloat(ts.CalN...)
				calDir.Float64("CalC").AppendRowFloat(ts.CalC...)
				calDir.Float64("CalP").AppendRowFloat(ts.CalP...)
			}
		case Epoch:
			for _, name := range names {
				levelDir.Float64(name).AppendRowFloat(stats.StatMean.Call(subDir.Value(name)).Float1D(0))
			}
			mean := func(name string) []float64 {
				return tensor.AsFloat64(stats.StatMean.Call(calSubDir.Value(name))).Values
			}
			calib, calibP, calibN, calibErr := pcombenv.CalibStats(mean("CalN"), mean("CalC"), mean("CalP"))
			for name, vals := range map[string][]float64{"Calib": calib, "CalibP": calibP, "CalibN": calibN} {
				tsr := calDir.Float64(name, nbins)
				for b, v := range vals {
					tsr.SetFloat1D(v, b)
				}
			}
			levelDir.Float64("CalibErr").AppendRowFloat(calibErr)
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
