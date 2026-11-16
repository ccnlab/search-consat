// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package combsearchff

import (
	"fmt"

	"cogentcore.org/core/base/metadata"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/stats"
	"github.com/ccnlab/search-consat/sims/combsearch/csenv"
	"github.com/emer/egomlx/egomlx"
)

// layerValues appends the given variable for all units in the layer for di.
func layerValues(ly *egomlx.Layer, vi, di int, vals []float32) []float32 {
	for ui := range ly.Shape.Len() {
		vals = append(vals, ly.UnitValue1D(vi, ui, di))
	}
	return vals
}

// StatSearch returns the stats function for the search performance
// of the network, computed by [csenv.CSEnv.TrialStats] from the
// Act values of the Output<s> layers, in slot order, matched against
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
			vi, err := egomlx.NeuronVarIndexByName("Act")
			if err != nil {
				return
			}
			ndata := ss.Net.NData
			for di := range ndata {
				ev := ss.Envs.ByModeDi(mode, di).(*csenv.CSEnv)
				outVals = outVals[:0]
				for s := range ev.Configs.NSlots {
					outVals = layerValues(ss.Net.LayerByName(fmt.Sprintf("Output%d", s)), vi, di, outVals)
				}
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
