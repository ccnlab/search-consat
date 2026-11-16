// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package combsearch

import (
	"cogentcore.org/core/base/metadata"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/stats"
	"github.com/ccnlab/search-consat/sims/combsearch/csenv"
	"github.com/emer/axon/v2/axon"
)

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

// StatSearch returns the stats function for the search performance
// of the network, computed by [csenv.CSEnv.TrialStats] from the
// Output minus phase activity (ActM) in each slot pool, matched against
// the output patterns (see [csenv.TrialStatDocs]).
func (ss *Sim) StatSearch() func(mode Modes, level Levels, start bool) {
	var outVals []float32
	var ts csenv.TrialStats
	names := csenv.TrialStatNames
	return func(mode Modes, level Levels, start bool) {
		if level <= Cycle {
			return
		}
		modeDir := ss.Stats.Dir(mode.String())
		curModeDir := ss.Current.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		subDir := modeDir.Dir((level - 1).String())
		if start {
			for _, name := range names {
				tsr := levelDir.Float64(name)
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) {
					s.Range.SetMin(0).SetMax(1)
					switch name {
					case "MatchCos":
						s.On = false
					default:
						s.On = true
					}
				})
				metadata.SetDoc(tsr, csenv.TrialStatDocs[name])
			}
			return
		}
		switch level {
		case Trial:
			ndata := int(ss.Net.Context().NData)
			outLy := ss.Net.LayerByName("Output")
			for di := range ndata {
				ev := ss.Envs.ByModeDi(mode, di).(*csenv.CSEnv)
				outVals = layerValues(outLy, axon.ActM, di, outVals)
				ev.TrialStats(outVals, &ts)
				for i, stat := range ts.Values() {
					name := names[i]
					curModeDir.Float64(name, ndata).SetFloat1D(stat, di)
					levelDir.Float64(name).AppendRowFloat(stat)
				}
			}
		case Run:
			for _, name := range names {
				levelDir.Float64(name).AppendRowFloat(stats.StatFinal.Call(subDir.Value(name)).Float1D(0))
			}
		default:
			for _, name := range names {
				levelDir.Float64(name).AppendRowFloat(stats.StatMean.Call(subDir.Value(name)).Float1D(0))
			}
		}
	}
}
