// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflictff

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
	"github.com/emer/egomlx/egomlx"
)

// layerValues returns the given variable for all neurons in the layer for di.
func layerValues(ly *egomlx.Layer, vi, di int, vals []float32) []float32 {
	vals = vals[:0]
	for ui := range ly.Shape.Len() {
		vals = append(vals, ly.UnitValue1D(vi, ui, di))
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
// activity (Act), matched against the group output patterns
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
	names = append(names, cfenv.MatchNames...)
	slotNms := append([]string{"Slot0"}, ss.slotNames()...)
	names = append(names, slotNms[1:]...)
	names = append(names, "SlotRange")
	matchCenter, matchExtreme := cfenv.MatchPairs(ss.Envs.ByModeDi(Test, 0).(*cfenv.CFEnv).Space().Targets)
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
					case "MatchCenter", "MatchExtreme":
						s.Range.SetMax(2)
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
			ndata := ss.Net.NData
			vi, _ := egomlx.NeuronVarIndexByName("Act")
			outLy := ss.Net.LayerByName("Output")
			for di := range ndata {
				ev := ss.Envs.ByModeDi(mode, di).(*cfenv.CFEnv)
				outVals = layerValues(outLy, vi, di, outVals)
				ev.TrialStats(outVals, &ts)
				vals := append(ts.Values(), ts.Probs...)
				vals = append(vals, math.NaN())
				vals = append(vals, ts.InProbs...)
				vals = append(vals, math.NaN(), math.NaN(), math.NaN())
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
				if (name == "AbsDiff" || name == "InAbsDiff" || name == "MatchCenter" || name == "MatchExtreme" || name == "SlotRange") && level != Run {
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
			slope := func(pct int) float64 {
				if pct == 0 {
					return math.NaN()
				}
				hi := levelDir.Float64(cfenv.InTargetName(pct)).Float1D(-1)
				lo := levelDir.Float64(cfenv.InTargetName(100 - pct)).Float1D(-1)
				return cfenv.MatchSlope(hi, lo, pct)
			}
			levelDir.Float64("MatchCenter").AppendRowFloat(slope(matchCenter))
			levelDir.Float64("MatchExtreme").AppendRowFloat(slope(matchExtreme))
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

// carryNames returns the names of the StatCarry stats: <Layer>_CarryEnd
// for every layer, including Input, whose carryover should be ~0 (its
// state is clamped), as a check on the baseline.
func (ss *Sim) carryNames() []string {
	var names []string
	for _, ly := range ss.Net.Layers {
		names = append(names, ly.Name+"_CarryEnd")
	}
	return names
}

// StatCarry returns the stats function for the carryover of each layer's
// state from the previous trial, as <Layer>_CarryEnd: the cosine between
// the layer's activity (Act) on the current trial and on the previous
// trial of the same data-parallel stream, minus the mean cosine with the
// previous activity of the other streams. The other streams run
// independent random sequences through the same network, so this
// baseline has the same expected input overlap and shared
// representational structure, without the carryover: 0 = no carryover
// beyond that expected from the inputs. This matches the axon and rnn
// models' _CarryEnd stats. The feedforward network has no state that
// carries over between trials, so this is a control: it should be ~0.
// NaN for the first trial of a run, and with NData < 2.
// Averaged over trials above the Trial level.
func (ss *Sim) StatCarry() func(mode Modes, level Levels, start bool) {
	lays := ss.Net.Layers
	names := ss.carryNames()
	vi, _ := egomlx.NeuronVarIndexByName("Act")
	var prev [][][]float32 // [layer][di] previous Act
	var cur []float32
	return func(mode Modes, level Levels, start bool) {
		if level <= Cycle {
			return
		}
		modeDir := ss.Stats.Dir(mode.String())
		curModeDir := ss.Current.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		subDir := modeDir.Dir((level - 1).String())
		if start {
			if mode == Train && level == Run {
				prev = nil // new weights and state
			}
			for _, name := range names {
				tsr := levelDir.Float64(name)
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) {
					s.On = false
				})
				metadata.SetDoc(tsr, "Carryover of the layer's state from the previous trial: cosine of the activity (Act) with that of the previous trial in the same data-parallel stream, minus the mean cosine with the previous activity of the other streams (the baseline for the input overlap). 0 = no carryover.")
			}
			return
		}
		if level == Trial {
			ndata := ss.Net.NData
			if len(prev) != len(lays) || len(prev[0]) != ndata {
				prev = make([][][]float32, len(lays))
				for li := range prev {
					prev[li] = make([][]float32, ndata)
				}
			}
			for li, ly := range lays {
				for di := range ndata {
					cur = layerValues(ly, vi, di, cur)
					v := carryover(cur, prev[li], di)
					curModeDir.Float64(names[li], ndata).SetFloat1D(v, di)
					levelDir.Float64(names[li]).AppendRowFloat(v)
				}
				for di := range ndata {
					prev[li][di] = layerValues(ly, vi, di, prev[li][di])
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

// carryover returns the cosine of cur with prev[di] minus the mean cosine
// of cur with the other prev values (see [Sim.StatCarry]).
// NaN if any prev is missing or there is only one.
func carryover(cur []float32, prev [][]float32, di int) float64 {
	if len(prev) < 2 {
		return math.NaN()
	}
	var base float64
	for d, p := range prev {
		if len(p) != len(cur) {
			return math.NaN()
		}
		if d != di {
			base += float64(cfenv.Cosine(cur, p))
		}
	}
	base /= float64(len(prev) - 1)
	return float64(cfenv.Cosine(cur, prev[di])) - base
}
