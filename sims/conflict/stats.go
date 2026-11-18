// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflict

import (
	"fmt"
	"math"
	"strconv"

	"cogentcore.org/core/base/metadata"
	"cogentcore.org/core/core"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/histogram"
	"cogentcore.org/lab/stats/metric"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorcore"
	"github.com/ccnlab/search-consat/sims/conflict/cfenv"
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

// targetNames returns the probability stat names for the test targets.
func (ss *Sim) targetNames() []string {
	return ss.Envs.ByModeDi(Test, 0).(*cfenv.CFEnv).TargetNames()
}

// StatOutput returns the stats function for the output of the network,
// computed by [cfenv.CFEnv.TrialStats] from the Output minus phase
// activity (ActM), matched against the group output patterns
// (see [cfenv.TrialStatDocs]). The probability stats (P100, P50, ...)
// are averaged over trials, and AbsDiff is computed from them above the
// Trial level.
func (ss *Sim) StatOutput() func(mode Modes, level Levels, start bool) {
	var outVals []float32
	var ts cfenv.TrialStats
	fixed := cfenv.TrialStatNames
	targs := ss.targetNames()
	names := append(append([]string{}, fixed...), targs...)
	names = append(names, "AbsDiff")
	pcts := make([]float64, len(targs))
	for i, nm := range targs {
		p, _ := strconv.Atoi(nm[1:])
		pcts[i] = float64(p) / 100
	}
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
					if mode == Test {
						s.Plot.Plotter = "Bar"
						s.Plot.XAxis.Column = "TrainEpoch"
					}
					switch name {
					case "P100", "AbsDiff", "Blend", "Intrusion":
						s.On = true
					}
					if mode == Test && name[0] == 'P' {
						s.On = true
					}
				})
				if doc, ok := cfenv.TrialStatDocs[name]; ok {
					metadata.SetDoc(tsr, doc)
				} else if name[0] == 'P' {
					p, _ := strconv.Atoi(name[1:])
					metadata.SetDoc(tsr, cfenv.TargetDoc(p))
				}
			}
			return
		}
		switch level {
		case Trial:
			ndata := int(ss.Net.Context().NData)
			outLy := ss.Net.LayerByName("Output")
			for di := range ndata {
				ev := ss.Envs.ByModeDi(mode, di).(*cfenv.CFEnv)
				outVals = layerValues(outLy, axon.ActM, di, outVals)
				ev.TrialStats(outVals, &ts)
				vals := append(ts.Values(), ts.Probs...)
				vals = append(vals, math.NaN())
				for i, stat := range vals {
					name := names[i]
					curModeDir.Float64(name, ndata).SetFloat1D(stat, di)
					levelDir.Float64(name).AppendRowFloat(stat)
				}
			}
		default:
			sf := stats.StatMean
			if level == Run {
				sf = stats.StatFinal
			}
			for _, name := range names {
				if name == "AbsDiff" && level != Run {
					continue
				}
				levelDir.Float64(name).AppendRowFloat(sf.Call(subDir.Value(name)).Float1D(0))
			}
			if level == Run {
				return
			}
			ad, n := 0.0, 0
			for i, nm := range targs {
				p := levelDir.Float64(nm).Float1D(-1)
				if math.IsNaN(p) {
					continue
				}
				ad += math.Abs(p - pcts[i])
				n++
			}
			if n > 0 {
				ad /= float64(n)
			} else {
				ad = math.NaN()
			}
			levelDir.Float64("AbsDiff").AppendRowFloat(ad)
		}
	}
}

// StatTestCopy returns a stats function that copies the latest Test Epoch
// values of the given stats into the Train Epoch and Run levels, as
// Test_<name>. Each epoch has the values from the latest test in the
// current run (NaN before the first one), so they plot as a step line.
// The Run value is from the final test at the end of the run.
func (ss *Sim) StatTestCopy(names ...string) func(mode Modes, level Levels, start bool) {
	return func(mode Modes, level Levels, start bool) {
		if mode != Train || (level != Epoch && level != Run) {
			return
		}
		levelDir := ss.Stats.Dir(mode.String()).Dir(level.String())
		testDir := ss.Stats.Dir("Test/Epoch")
		tested := ss.TestCount > 0
		for _, snm := range names {
			name := "Test_" + snm
			tsr := levelDir.Float64(name)
			if start {
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) {
					s.Range.SetMin(0).SetMax(1)
					s.On = true
				})
				continue
			}
			v := math.NaN()
			if src := testDir.Node(snm); tested && src != nil {
				if st := testDir.Float64(snm); st.DimSize(0) > 0 {
					v = st.Float1D(-1)
				}
			}
			tsr.AppendRowFloat(v)
		}
	}
}

// PatternDistMatrix computes the distance matrix between all of the input
// patterns, in Stats/Patterns/DistMatrix, using 1 - cosine: for binary
// patterns with the same number of active units, that is 1 minus the
// proportion of active units they share. The patterns are in group order.
func (ss *Sim) PatternDistMatrix() *tensor.Float64 {
	sp := ss.Envs.ByModeDi(Train, 0).(*cfenv.CFEnv).Space()
	dm := ss.Stats.Dir("Patterns").Float64("DistMatrix")
	metric.MatrixOut(metric.InvCosine, sp.Inputs, dm)
	metadata.SetDoc(dm, "Distance (1 - cosine) between all of the input patterns, in group order.")
	tensorcore.AddGridStylerTo(dm, func(s *tensorcore.GridStyle) {
		s.TopZero = true
		s.Range.SetMin(0).SetMax(1)
		s.ColorMap = core.ColorMapName("Viridis")
		s.GridFill = 1
		s.DimExtra = 0.15
	})
	return dm
}

// PatternDistGrid shows the input pattern distance matrix in a grid tab,
// labeled by group.
func (ss *Sim) PatternDistGrid() {
	cfg := &ss.Envs.ByModeDi(Train, 0).(*cfenv.CFEnv).Configs
	labels := make([]string, cfg.NInputs())
	for g := range cfg.NGroups {
		labels[g*cfg.GroupN] = fmt.Sprintf("G%d", g)
	}
	tbs := ss.GUI.Tabs.AsLab()
	_, idx := tbs.CurrentTab()
	tg := tbs.TensorGrid("Pattern Distances", ss.Stats.Dir("Patterns").Float64("DistMatrix"))
	tg.RowLabels = labels
	tg.ColumnLabels = labels
	tbs.SelectTabIndex(idx)
}

// StatSecondHist returns a stats function that computes the histogram of
// the per-trial SecondRatio values, in nBins bins from 0 to 1, as the
// proportion of conflict trials (excluding NaN) in each bin, in
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
			metadata.SetDoc(props, "Proportion of conflict trials with SecondRatio in each bin, accumulated from training epoch Epochs / 2 on (before that, the last epoch).")
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

// StatConsist returns a stats function that computes the consistency of
// the choices for the same test item across its repetitions, at the end
// of each test, from all the parallel test envs ([cfenv.ItemConsistency]):
// Consist over the conflict items, and TieConsist over the tied items.
func (ss *Sim) StatConsist() func(mode Modes, level Levels, start bool) {
	names := []string{"Consist", "TieConsist"}
	return func(mode Modes, level Levels, start bool) {
		if mode != Test || level != Epoch {
			return
		}
		levelDir := ss.Stats.Dir("Test/Epoch")
		if start {
			for _, name := range names {
				tsr := levelDir.Float64(name)
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) {
					s.Range.SetMin(0).SetMax(1)
					s.On = true
				})
				metadata.SetDoc(tsr, cfenv.TrialStatDocs[name])
			}
			return
		}
		var envs []*cfenv.CFEnv
		for di := range ss.Config.Run.NData {
			envs = append(envs, ss.Envs.ByModeDi(Test, di).(*cfenv.CFEnv))
		}
		c, t := cfenv.ItemConsistency(envs...)
		levelDir.Float64("Consist").AppendRowFloat(c)
		levelDir.Float64("TieConsist").AppendRowFloat(t)
	}
}
