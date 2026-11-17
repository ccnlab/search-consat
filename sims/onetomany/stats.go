// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package onetomany

import (
	"math"

	"cogentcore.org/core/base/metadata"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/histogram"
	"cogentcore.org/lab/stats/stats"
	"github.com/ccnlab/search-consat/sims/onetomany/otmenv"
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

// StatOutput returns the stats function for the output of the network,
// computed by [otmenv.OTMEnv.TrialStats] from the Output minus phase
// activity (ActM), matched against the output patterns
// (see [otmenv.TrialStatDocs]).
func (ss *Sim) StatOutput() func(mode Modes, level Levels, start bool) {
	var outVals []float32
	var ts otmenv.TrialStats
	names := otmenv.TrialStatNames
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
					case "Valid", "Blend", "Intrusion":
						s.On = true
					}
				})
				metadata.SetDoc(tsr, otmenv.TrialStatDocs[name])
			}
			return
		}
		switch level {
		case Trial:
			ndata := int(ss.Net.Context().NData)
			outLy := ss.Net.LayerByName("Output")
			for di := range ndata {
				ev := ss.Envs.ByModeDi(mode, di).(*otmenv.OTMEnv)
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

// StatSecondHist returns a stats function that computes the histogram of
// the per-trial SecondRatio values, in nBins bins from 0 to 1, as the
// proportion of trials (excluding NaN) in each bin, in
// Stats/<Mode>/SecondRatioHist. The counts are accumulated over all the
// epochs (and tests) from training epoch Epochs / 2 on, in each run, because
// the distribution is highly variable from one epoch to the next. Before
// that, it shows the last epoch. It shows whether a mean SecondRatio
// reflects a mix of clear choices and blends (bimodal), or a consistent
// partial preference.
func (ss *Sim) StatSecondHist(nBins int) func(mode Modes, level Levels, start bool) {
	var vals, hist []float64
	type accum struct {
		run    int
		counts []float64
		n      int
	}
	accums := map[Modes]*accum{}
	return func(mode Modes, level Levels, start bool) {
		if level != Epoch {
			return
		}
		dir := ss.Stats.Dir(mode.String()).Dir("SecondRatioHist")
		bins := dir.Float64("Value", nBins)
		props := dir.Float64("Prop", nBins)
		if start {
			plot.SetFirstStyler(bins, func(s *plot.Style) {
				s.Role = plot.X
				s.Plotter = "Bar"
			})
			plot.SetFirstStyler(props, func(s *plot.Style) {
				s.On = true
				s.Role = plot.Y
				s.Plotter = "Bar"
				s.Range.SetMin(0).SetMax(1)
			})
			metadata.SetDoc(props, "Proportion of trials with SecondRatio in each bin, accumulated from training epoch Epochs / 2 on (before that, the last epoch).")
			for i := range nBins {
				bins.SetFloat1D(float64(i)/float64(nBins), i)
			}
			return
		}
		trl := ss.Stats.Dir(mode.String()).Dir("Trial").Float64("SecondRatio")
		vals = vals[:0]
		for _, v := range trl.Values {
			if !math.IsNaN(v) {
				vals = append(vals, v)
			}
		}
		histogram.F64(&hist, vals, nBins, 0, 1)
		counts, n := hist, len(vals)
		run := ss.Loops.Loop(Train, Run).Counter.Cur
		ac := accums[mode]
		if ac == nil || ac.run != run {
			ac = &accum{run: run, counts: make([]float64, nBins)}
			accums[mode] = ac
		}
		if ss.Loops.Loop(Train, Epoch).Counter.Cur >= ss.Config.Run.Epochs/2 {
			for i, c := range hist {
				ac.counts[i] += c
			}
			ac.n += len(vals)
			counts, n = ac.counts, ac.n
		}
		for i, c := range counts {
			p := math.NaN()
			if n > 0 {
				p = c / float64(n)
			}
			props.SetFloat1D(p, i)
		}
	}
}
