// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflict

import (
	"fmt"
	"math"
	"strconv"
	"strings"

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

// inTargetNames returns the input-only probability stat names for the
// test targets (In100, In50, ...).
func (ss *Sim) inTargetNames() []string {
	return ss.Envs.ByModeDi(Test, 0).(*cfenv.CFEnv).InTargetNames()
}

// slotNames returns the slot stat names after Slot0 (Slot1, ...).
func (ss *Sim) slotNames() []string {
	return ss.Envs.ByModeDi(Test, 0).(*cfenv.CFEnv).SlotNames()
}

// StatOutput returns the stats function for the output of the network,
// computed by [cfenv.CFEnv.TrialStats] from the Output minus phase
// activity (ActM), matched against the group output patterns
// (see [cfenv.TrialStatDocs]). The probability stats (P100, P50, ...)
// are averaged over trials, and AbsDiff is computed from them above the
// Trial level. The input-only probability stats (In100, In50, ...) and
// InAbsDiff are the same, among only the trials where the choice is one
// of the input groups (leaving out Blends and Intrusions). The slot
// stats (Slot0, Slot1, ...) are for the choice on ties of the group in
// each slot, and SlotRange is their range above the Trial level.
func (ss *Sim) StatOutput() func(mode Modes, level Levels, start bool) {
	var outVals []float32
	var ts cfenv.TrialStats
	fixed := cfenv.TrialStatNames
	targs := ss.targetNames()
	names := append(append([]string{}, fixed...), targs...)
	names = append(names, "AbsDiff")
	inTargs := ss.inTargetNames()
	names = append(names, inTargs...)
	names = append(names, "InAbsDiff")
	slotNms := append([]string{"Slot0"}, ss.slotNames()...)
	names = append(names, slotNms[1:]...)
	names = append(names, "SlotRange")
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
					if mode == Test && (name[0] == 'P' || strings.HasPrefix(name, "In")) {
						s.On = true
					}
				})
				if doc, ok := cfenv.TrialStatDocs[name]; ok {
					metadata.SetDoc(tsr, doc)
				} else if name[0] == 'P' {
					p, _ := strconv.Atoi(name[1:])
					metadata.SetDoc(tsr, cfenv.TargetDoc(p))
				} else if strings.HasPrefix(name, "In") {
					p, _ := strconv.Atoi(name[2:])
					metadata.SetDoc(tsr, cfenv.InTargetDoc(p))
				} else if strings.HasPrefix(name, "Slot") {
					sl, _ := strconv.Atoi(name[4:])
					metadata.SetDoc(tsr, cfenv.SlotDoc(sl))
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
				vals = append(vals, ts.InProbs...)
				vals = append(vals, math.NaN())
				vals = append(vals, ts.Slots[1:]...)
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
				if (name == "AbsDiff" || name == "InAbsDiff" || name == "SlotRange") && level != Run {
					continue
				}
				levelDir.Float64(name).AppendRowFloat(sf.Call(subDir.Value(name)).Float1D(0))
			}
			if level == Run {
				return
			}
			absDiff := func(nms []string) float64 {
				ad, n := 0.0, 0
				for i, nm := range nms {
					p := levelDir.Float64(nm).Float1D(-1)
					if math.IsNaN(p) {
						continue
					}
					ad += math.Abs(p - pcts[i])
					n++
				}
				if n == 0 {
					return math.NaN()
				}
				return ad / float64(n)
			}
			levelDir.Float64("AbsDiff").AppendRowFloat(absDiff(targs))
			levelDir.Float64("InAbsDiff").AppendRowFloat(absDiff(inTargs))
			mn, mx := math.Inf(1), math.Inf(-1)
			for _, nm := range slotNms {
				v := levelDir.Float64(nm).Float1D(-1)
				if !math.IsNaN(v) {
					mn, mx = min(mn, v), max(mx, v)
				}
			}
			rng := math.NaN()
			if mx >= mn {
				rng = mx - mn
			}
			levelDir.Float64("SlotRange").AppendRowFloat(rng)
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

// silentNames returns the names of the StatSilent stats.
func (ss *Sim) silentNames() []string {
	var names []string
	for _, ly := range ss.Net.Layers {
		if ly.Type == axon.SuperLayer || ly.Type == axon.TargetLayer {
			names = append(names, ly.Name+"_Silent")
		}
	}
	return names
}

// StatSilent returns the stats function for whether each of the Encoder,
// Hidden and Output layers is silent on a trial: 1 if no unit has minus
// phase activity (ActM) above the env ActThr, as <Layer>_Silent. This
// distinguishes a failure to propagate activity through the network
// (e.g., a silent hidden layer on conflict trials, producing no output,
// which counts as a Blend) from a real blend of two active groups.
// Averaged over trials above the Trial level.
func (ss *Sim) StatSilent() func(mode Modes, level Levels, start bool) {
	var vals []float32
	names := ss.silentNames()
	var lays []*axon.Layer
	for _, nm := range names {
		lays = append(lays, ss.Net.LayerByName(strings.TrimSuffix(nm, "_Silent")))
	}
	thr := ss.Envs.ByModeDi(Test, 0).(*cfenv.CFEnv).Configs.ActThr
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
						s.On = true
					}
				})
				metadata.SetDoc(tsr, "1 if no unit in the layer has minus phase activity above ActThr on the trial (averaged over trials above the Trial level): a failure to propagate activity. A silent Output counts as a Blend.")
			}
			return
		}
		if level == Trial {
			ndata := int(ss.Net.Context().NData)
			for li, ly := range lays {
				name := names[li]
				for di := range ndata {
					vals = layerValues(ly, axon.ActM, di, vals)
					silent := 1.0
					for _, v := range vals {
						if v > thr {
							silent = 0
							break
						}
					}
					curModeDir.Float64(name, ndata).SetFloat1D(silent, di)
					levelDir.Float64(name).AppendRowFloat(silent)
				}
			}
			return
		}
		sf := stats.StatMean
		if level == Run {
			sf = stats.StatFinal
		}
		for _, name := range names {
			levelDir.Float64(name).AppendRowFloat(sf.Call(subDir.Value(name)).Float1D(0))
		}
	}
}

// actGeNames returns the names of the StatTestActGe stats for the given
// layers, the same as the training [axon.StatLayerActGe] stats.
func actGeNames(layers []string) []string {
	var names []string
	for _, lnm := range layers {
		for _, st := range []string{"ActMAvg", "ActMMax", "MaxGeM"} {
			names = append(names, lnm+"_"+st)
		}
	}
	return names
}

// StatTestActGe returns the stats function for the same layer activity
// and conductance stats as [axon.StatLayerActGe] in training (ActMAvg,
// ActMMax, MaxGeM), for testing: each is the mean over the test trials
// at the Epoch level (where the training ActMAvg uses the layer's long
// running average, which only reflects training).
func (ss *Sim) StatTestActGe(layers ...string) func(mode Modes, level Levels, start bool) {
	names := actGeNames(layers)
	docs := []string{
		"Minus phase average activity in the layer, averaged over test trials.",
		"Minus phase maximum activity in the layer, averaged over test trials. If a layer is not very active, activity will not propagate efficiently to other layers.",
		"Minus phase maximum Ge excitatory conductance in the layer, averaged over test trials. In general this should be above 1, but not greater than 3 or so.",
	}
	return func(mode Modes, level Levels, start bool) {
		if mode != Test || level <= Cycle {
			return
		}
		modeDir := ss.Stats.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		if start {
			for i, name := range names {
				tsr := levelDir.Float64(name)
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) { s.Range.SetMin(0) })
				metadata.SetDoc(tsr, docs[i%3])
			}
			return
		}
		if level == Trial {
			ndata := int(ss.Net.Context().NData)
			for li, lnm := range layers {
				lpi := ss.Net.LayerByName(lnm).Params.PoolIndex(0)
				for di := range ndata {
					levelDir.Float64(names[3*li]).AppendRowFloat(float64(axon.PoolAvgMax(axon.AMAct, axon.AMMinus, axon.Avg, lpi, uint32(di))))
					levelDir.Float64(names[3*li+1]).AppendRowFloat(float64(axon.PoolAvgMax(axon.AMAct, axon.AMMinus, axon.Max, lpi, uint32(di))))
					levelDir.Float64(names[3*li+2]).AppendRowFloat(float64(axon.PoolAvgMax(axon.AMGeInt, axon.AMMinus, axon.Max, lpi, uint32(di))))
				}
			}
			return
		}
		subDir := modeDir.Dir((level - 1).String())
		for _, name := range names {
			levelDir.Float64(name).AppendRow(stats.StatMean.Call(subDir.Value(name)))
		}
	}
}
