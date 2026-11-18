// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfenv

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorfs"
)

func countOn(pv []float32) int {
	n := 0
	for _, v := range pv {
		if v > 0 {
			n++
		}
	}
	return n
}

// overlap returns the number of active units shared by a and b.
func overlap(a, b []float32) int {
	ov := 0
	for k, v := range a {
		if v > 0 && b[k] > 0 {
			ov++
		}
	}
	return ov
}

// checkPatterns checks that all the patterns have non active units and
// returns the max overlap between them.
func checkPatterns(t *testing.T, pats *tensor.Float32, non int) int {
	n := pats.DimSize(0)
	worst := 0
	for i := range n {
		pv := pattern(pats, i)
		if c := countOn(pv); c != non {
			t.Fatalf("pattern %d: %d on, want %d", i, c, non)
		}
		for j := i + 1; j < n; j++ {
			worst = max(worst, overlap(pv, pattern(pats, j)))
		}
	}
	return worst
}

// TestPatterns checks the random input and output patterns.
func TestPatterns(t *testing.T) {
	var cfg Config
	cfg.Defaults()
	cfg.Prototypes = false
	cfg.RandOutput = true
	sp := cfg.MakeSpace()
	for _, c := range []struct {
		name string
		pats *tensor.Float32
		non  int
	}{{"inputs", sp.Inputs, cfg.NOn()}, {"outputs", sp.Outputs, cfg.NPerOutput}} {
		worst := checkPatterns(t, c.pats, c.non)
		allowed := c.non - cfg.MinDiff(c.non)
		t.Logf("random %s: %d patterns, %d on, max overlap %d (allowed %d)", c.name, c.pats.DimSize(0), c.non, worst, allowed)
		if worst > allowed {
			t.Errorf("max overlap %d > %d", worst, allowed)
		}
	}
	cfg.RandOutput = false
	sp = cfg.MakeSpace()
	for g := range cfg.NGroups {
		if countOn(pattern(sp.Outputs, g)) != cfg.NPerOutput {
			t.Errorf("bar %d", g)
		}
	}
}

// TestProtoPatterns checks the input patterns generated from prototypes:
// each differs from its prototype by exactly NFlip active units, the inputs
// of a group are distinct, and the prototypes satisfy ProtoMaxOverlap.
func TestProtoPatterns(t *testing.T) {
	for _, gn := range []int{4, 8} {
		var cfg Config
		cfg.Defaults()
		cfg.GroupN = gn
		sp := cfg.MakeSpace()
		non, nf := cfg.NOn(), cfg.NFlip()
		pworst := checkPatterns(t, sp.Protos, non)
		if pworst > non-cfg.ProtoMinDiff() {
			t.Errorf("prototype overlap %d > %d", pworst, non-cfg.ProtoMinDiff())
		}
		checkPatterns(t, sp.Inputs, non)
		within, between := []int{non, 0}, []int{non, 0} // min, max overlaps
		for i := range cfg.NInputs() {
			pv := pattern(sp.Inputs, i)
			g := cfg.Group(i)
			if ov := overlap(pv, pattern(sp.Protos, g)); ov != non-nf {
				t.Fatalf("input %d: overlap with prototype %d, want %d", i, ov, non-nf)
			}
			for j := i + 1; j < cfg.NInputs(); j++ {
				ov := overlap(pv, pattern(sp.Inputs, j))
				r := between
				if cfg.Group(j) == g {
					r = within
					if ov == non {
						t.Fatalf("inputs %d and %d are the same", i, j)
					}
				}
				r[0], r[1] = min(r[0], ov), max(r[1], ov)
			}
		}
		t.Logf("g=%d: %d on, %d flipped; prototype max overlap %d (allowed %d); input overlap within group %d-%d, between groups %d-%d", gn, non, nf, pworst, non-cfg.ProtoMinDiff(), within[0], within[1], between[0], between[1])
		if between[1] >= within[0] {
			t.Errorf("groups not separated: between max %d >= within min %d", between[1], within[0])
		}
	}
}

func TestItems(t *testing.T) {
	for _, k := range []int{2, 3, 4} {
		for _, per := range []int{0, 100} {
			var cfg Config
			cfg.Defaults()
			cfg.NSlots = k
			cfg.GroupN = 4
			cfg.TestPerLevel = per
			sp := cfg.MakeSpace()
			t.Log(cfg.SpaceString(sp))
			seen := map[string]bool{}
			nc := make([]int, cfg.NLevels())
			for _, it := range sp.Test {
				nc[it.NConflict]++
				key := fmt.Sprint(it.Inputs)
				if per == 0 && seen[key] {
					t.Fatalf("duplicate item: %v", it.Inputs)
				}
				seen[key] = true
				nb := 0
				for _, in := range it.Inputs {
					if cfg.Group(in) != it.Groups[0] {
						nb++
					}
				}
				if nb != it.NConflict {
					t.Fatalf("bad NConflict: %v", it)
				}
				if want := float32(k-nb) / float32(k); math.Abs(float64(it.Probs[0]-want)) > 1e-6 {
					t.Fatalf("bad Probs: %v", it)
				}
			}
			if per == 0 { // count all combinations
				G, g := cfg.NGroups, cfg.GroupN
				gk := int(math.Pow(float64(g), float64(k)))
				for c := range cfg.NLevels() {
					want := G * gk
					if c > 0 {
						want = G * (G - 1) * len(subsets(k, c)) * gk
						if 2*c == k {
							want /= 2
						}
					}
					if nc[c] != want {
						t.Errorf("k=%d c=%d: %d items, want %d", k, c, nc[c], want)
					}
				}
			}
		}
	}
}

func TestFreqs(t *testing.T) {
	var cfg Config
	cfg.Defaults()
	cfg.NSlots = 2
	cfg.GroupN = 2
	cfg.Freqs = []float32{3, 1}
	it := cfg.NewItem([]int{0, 3}) // group 0 member 0 (3), group 1 member 1 (1)
	if it.Probs[0] != 0.75 || it.Probs[1] != 0.25 {
		t.Errorf("bad freq probs: %v", it)
	}
	ev := &CFEnv{Configs: cfg}
	ev.Config(100, 1, 0, false, 0)
	ev.Init(0)
	n0 := 0
	for range 4000 {
		it := ev.TrainItem()
		if it.NConflict != 0 {
			t.Fatalf("conflict in training: %v", it)
		}
		for _, in := range it.Inputs {
			if in%2 == 0 {
				n0++
			}
		}
	}
	if p := float64(n0) / 8000; math.Abs(p-0.75) > 0.03 {
		t.Errorf("member 0 freq %g, want 0.75", p)
	}
}

// TestSequence checks that the parallel test envs together present
// every item TestReps times.
func TestSequence(t *testing.T) {
	var cfg Config
	cfg.Defaults()
	cfg.TestPerLevel = 50
	cfg.TestReps = 3
	ndata := 16
	sp := cfg.Space()
	counts := make([]int, len(sp.Test))
	for di := range ndata {
		ev := &CFEnv{Configs: cfg}
		ev.Config(0, ndata, di, true, 181+int64(di)*181)
		ev.Init(0)
		if ev.Trial.Max != ev.TestTrials() {
			t.Fatalf("trials %d != %d", ev.Trial.Max, ev.TestTrials())
		}
		for range ev.Trial.Max {
			ev.Step()
			counts[slices.Index(sp.Test, ev.Cur)]++
		}
	}
	for i, c := range counts {
		if c < cfg.TestReps {
			t.Fatalf("item %d presented %d times", i, c)
		}
	}
}

// TestStats checks the stats for clean and partial choices, blends,
// intrusions, and low-level activity below ActThr.
func TestStats(t *testing.T) {
	var cfg Config
	cfg.Defaults()
	cfg.NSlots = 2
	ev := &CFEnv{Configs: cfg}
	ev.Config(0, 1, 0, true, 0)
	ev.Init(0)
	sp := ev.Space()
	if !slices.Equal(ev.TargetNames(), []string{"P100", "P50"}) {
		t.Fatalf("targets: %v", ev.TargetNames())
	}
	// mix returns the sum of the group patterns with the given activities,
	// plus low-level activity everywhere.
	mix := func(acts map[int]float32) []float32 {
		out := make([]float32, sp.Outputs.SubSpace(0).Len())
		for i := range out {
			out[i] = 0.05
		}
		for g, a := range acts {
			for i, v := range sp.Outputs.SubSpace(g).(*tensor.Float32).Values {
				if v > 0 {
					out[i] = a
				}
			}
		}
		return out
	}
	var ts TrialStats
	nslot0 := 0.0
	nconf := 0
	for range 500 {
		ev.Step()
		it := ev.Cur
		a := it.Groups[0]
		other := 0
		for it.GroupIndex(other) >= 0 {
			other++
		}
		ev.TrialStats(mix(map[int]float32{a: 0.9}), &ts)
		if ts.Blend != 0 || ts.Intrusion != 0 || ts.Choice != a || ts.OtherRatio != 0 || ts.MatchCos < 0.95 {
			t.Fatalf("clean choice: %s: %v", ev.String(), ts.Values())
		}
		// other group below the intrusion threshold: still a choice
		ob := 0.9 * (cfg.IntrusionThr - 0.1)
		ev.TrialStats(mix(map[int]float32{a: 0.9, other: ob}), &ts)
		if ts.Intrusion != 0 || ts.Choice != a || math.Abs(ts.OtherRatio-float64(ob/0.9)) > 1e-5 {
			t.Fatalf("other below threshold: %s: %v", ev.String(), ts.Values())
		}
		// other group above the intrusion threshold
		ev.TrialStats(mix(map[int]float32{a: 0.6, other: 0.6 * (cfg.IntrusionThr + 0.1)}), &ts)
		if ts.Intrusion != 1 || ts.Choice != other {
			t.Fatalf("intrusion: %s: %v", ev.String(), ts.Values())
		}
		ev.TrialStats(mix(map[int]float32{a: 0.9}), &ts) // reset PrevChoice to a
		if it.NConflict == 0 {
			if ts.Probs[0] != 1 || !math.IsNaN(ts.Probs[1]) || !math.IsNaN(ts.Slot0) || !math.IsNaN(ts.SecondRatio) {
				t.Fatalf("consistent: %s: %v %v", ev.String(), ts.Values(), ts.Probs)
			}
			continue
		}
		b := it.Groups[1]
		if ts.Probs[1] != 0.5 || !math.IsNaN(ts.Probs[0]) || math.IsNaN(ts.Slot0) || ts.SecondRatio != 0 {
			t.Fatalf("conflict: %s: %v %v", ev.String(), ts.Values(), ts.Probs)
		}
		nslot0 += ts.Slot0
		nconf++
		// partial activity of the second group, above the blend ratio: B is top
		ev.TrialStats(mix(map[int]float32{a: 0.35, b: 0.8}), &ts)
		if ts.Blend != 0 || ts.Choice != b || math.Abs(ts.SecondRatio-0.35/0.8) > 1e-5 || ts.RepeatPrev != 0 {
			t.Fatalf("partial: %s: %v", ev.String(), ts.Values())
		}
		// below the blend ratio
		ev.TrialStats(mix(map[int]float32{a: 0.7, b: 0.6}), &ts)
		if ts.Blend != 1 || ts.Choice != -1 || ts.Probs[1] != 0 || !math.IsNaN(ts.RepeatPrev) {
			t.Fatalf("blend: %s: %v %v", ev.String(), ts.Values(), ts.Probs)
		}
		// no activity above threshold
		ev.TrialStats(mix(nil), &ts)
		if ts.Blend != 1 || ts.Intrusion != 0 || !math.IsNaN(ts.OtherRatio) {
			t.Fatalf("no activity: %v", ts.Values())
		}
	}
	// with equal counts (2 slots), the first group is the one in slot 0.
	t.Logf("slot0 for choosing group A: %.3f", nslot0/float64(nconf))
}

// pattern returns the values of pattern i in pats.
func pattern(pats *tensor.Float32, i int) []float32 {
	return pats.SubSpace(i).(*tensor.Float32).Values
}

// TestRender checks that the rendered Input and Output states are exactly
// the patterns of the current item, and that training items are consistent.
func TestRender(t *testing.T) {
	for _, test := range []bool{false, true} {
		var cfg Config
		cfg.Defaults()
		ev := &CFEnv{Configs: cfg}
		ev.Config(100, 4, 1, test, 77)
		ev.Init(0)
		sp := ev.Space()
		nu := cfg.PoolUnits()
		inp, out := ev.States["Input"], ev.States["Output"]
		if !slices.Equal(inp.Shape().Sizes, []int{1, cfg.NSlots, cfg.PoolSize, cfg.PoolSize}) {
			t.Fatalf("input shape: %v", inp.Shape().Sizes)
		}
		if !slices.Equal(out.Shape().Sizes, []int{cfg.NPerOutput, cfg.NGroups}) {
			t.Fatalf("output shape: %v", out.Shape().Sizes)
		}
		for range 1000 {
			ev.Step()
			it := ev.Cur
			if ev.Group != it.Groups[0] {
				t.Fatalf("target group %d != %d", ev.Group, it.Groups[0])
			}
			for s, in := range it.Inputs {
				if !slices.Equal(inp.Values[s*nu:(s+1)*nu], pattern(sp.Inputs, in)) {
					t.Fatalf("slot %d is not input %d: %s", s, in, ev.String())
				}
				if !test && cfg.Group(in) != ev.Group {
					t.Fatalf("training input %d not in group %d", in, ev.Group)
				}
			}
			// the output is the bar in the group's column
			for y := range cfg.NPerOutput {
				for x := range cfg.NGroups {
					want := 0.0
					if x == ev.Group {
						want = 1
					}
					if out.Float(y, x) != want {
						t.Fatalf("output %d,%d = %g for group %d", y, x, out.Float(y, x), ev.Group)
					}
				}
			}
			if !slices.Equal(out.Values, pattern(sp.Outputs, ev.Group)) {
				t.Fatalf("output is not the group pattern")
			}
		}
	}
}

// TestTrainMapping checks that each input pattern always appears with the
// same target over many training trials, that all inputs and groups are
// sampled about equally, and that every slot gets every input.
func TestTrainMapping(t *testing.T) {
	var cfg Config
	cfg.Defaults()
	ev := &CFEnv{Configs: cfg}
	ev.Config(100, 1, 0, false, 73)
	ev.Init(0)
	ni := cfg.NInputs()
	target := make([]int, ni)
	for i := range target {
		target[i] = -1
	}
	counts := make([][]int, cfg.NSlots)
	for s := range counts {
		counts[s] = make([]int, ni)
	}
	gcounts := make([]int, cfg.NGroups)
	n := 20000
	for range n {
		ev.Step()
		gcounts[ev.Group]++
		for s, in := range ev.Cur.Inputs {
			counts[s][in]++
			if target[in] >= 0 && target[in] != ev.Group {
				t.Fatalf("input %d has targets %d and %d", in, target[in], ev.Group)
			}
			target[in] = ev.Group
		}
	}
	for g, c := range gcounts {
		if math.Abs(float64(c)/float64(n)-1/float64(cfg.NGroups)) > 0.02 {
			t.Errorf("group %d sampled %d of %d", g, c, n)
		}
	}
	for s := range counts {
		for in, c := range counts[s] {
			if math.Abs(float64(c)/float64(n)-1/float64(ni)) > 0.01 {
				t.Errorf("slot %d input %d sampled %d of %d", s, in, c, n)
			}
		}
	}
	t.Logf("group counts: %v", gcounts)
}

// TestLearnable trains a linear classifier (delta rule on the rendered
// Input state, one output unit per group) on the training env, and checks
// that it gets all of the consistent test items right. This confirms that
// the rendered inputs and targets are a simple, learnable mapping.
func TestLearnable(t *testing.T) {
	var cfg Config
	cfg.Defaults()
	trn := &CFEnv{Configs: cfg}
	trn.Config(100, 1, 0, false, 73)
	trn.Init(0)
	tst := &CFEnv{Configs: cfg}
	tst.Config(0, 1, 0, true, 181)
	tst.Init(0)
	sp := trn.Space()
	nin := trn.States["Input"].Len()
	ng := cfg.NGroups
	wts := make([]float32, nin*ng)
	outs := func(ev *CFEnv) []float32 {
		o := make([]float32, ng)
		for g := range ng {
			for i, v := range ev.States["Input"].Values {
				o[g] += v * wts[g*nin+i]
			}
		}
		return o
	}
	lrate := float32(0.02)
	epochs := 0
	for epoch := range 50 {
		errs := 0
		for range 500 {
			trn.Step()
			o := outs(trn)
			// target from the rendered Output state: the active column
			if g, _ := trn.Match(trn.States["Output"].Values); g != trn.Group {
				t.Fatalf("output state group %d != %d", g, trn.Group)
			}
			for g := range ng {
				tg := float32(0)
				if g == trn.Group {
					tg = 1
				}
				d := lrate * (tg - o[g])
				for i, v := range trn.States["Input"].Values {
					wts[g*nin+i] += d * v
				}
			}
			if argmax(o) != trn.Group {
				errs++
			}
		}
		epochs = epoch + 1
		if errs == 0 {
			break
		}
	}
	nc := 0
	for range len(sp.Test) {
		tst.Step()
		if tst.Cur.NConflict > 0 {
			continue
		}
		nc++
		if g := argmax(outs(tst)); g != tst.Group {
			t.Fatalf("consistent test item %s: got group %d", tst.String(), g)
		}
	}
	t.Logf("linear classifier learned the training mapping in %d epochs of 500 trials; %d consistent test items correct", epochs, nc)
}

func argmax(v []float32) int {
	mi := 0
	for i, x := range v {
		if x > v[mi] {
			mi = i
		}
	}
	return mi
}

// TestSavePatterns checks that the patterns are saved in tensorfs.
func TestSavePatterns(t *testing.T) {
	root, _ := tensorfs.NewDir("Root")
	var cfg Config
	cfg.Defaults()
	ev := &CFEnv{Configs: cfg, Root: root}
	ev.Config(100, 1, 0, false, 0)
	sp := ev.Space()
	pats := root.Dir("Patterns")
	for g := range cfg.NGroups {
		items := pats.Dir("Items").Float32(fmt.Sprintf("G%d", g))
		if !slices.Equal(items.ShapeSizes(), []int{cfg.GroupN, cfg.PoolSize, cfg.PoolSize}) {
			t.Fatalf("items shape: %v", items.ShapeSizes())
		}
		for m := range cfg.GroupN {
			if !slices.Equal(pattern(items, m), pattern(sp.Inputs, g*cfg.GroupN+m)) {
				t.Fatalf("item G%d %d != input", g, m)
			}
		}
		proto := pats.Dir("Prototypes").Float32(fmt.Sprintf("G%d", g))
		if !slices.Equal(pattern(proto, 0), pattern(sp.Protos, g)) {
			t.Fatalf("prototype G%d", g)
		}
	}
	if !slices.Equal(pats.Float32("Outputs").Values, sp.Outputs.Values) {
		t.Fatalf("outputs")
	}
}

// TestTies checks that Slot0 and RepeatPrev are only computed on tied
// conflict trials (equal evidence for the two input groups), for 4 slots,
// and that IsTie uses the frequency-weighted evidence.
func TestTies(t *testing.T) {
	var cfg Config
	cfg.Defaults()
	cfg.NSlots = 4
	cfg.TestPerLevel = 50
	ev := &CFEnv{Configs: cfg}
	ev.Config(0, 1, 0, true, 0)
	ev.Init(0)
	sp := ev.Space()
	if !slices.Equal(ev.TargetNames(), []string{"P100", "P75", "P50", "P25"}) {
		t.Fatalf("targets: %v", ev.TargetNames())
	}
	var ts TrialStats
	nties := 0
	for range len(sp.Test) {
		ev.Step()
		it := ev.Cur
		ev.TrialStats(pattern(sp.Outputs, it.Groups[0]), &ts)
		tie := it.NConflict == 2
		if it.IsTie() != tie {
			t.Fatalf("IsTie %v for %s", it.IsTie(), ev.String())
		}
		if tie {
			nties++
			if math.IsNaN(ts.Slot0) {
				t.Fatalf("tie without Slot0: %s", ev.String())
			}
		} else if !math.IsNaN(ts.Slot0) || !math.IsNaN(ts.RepeatPrev) {
			t.Fatalf("non-tie with Slot0 / RepeatPrev: %s: %v", ev.String(), ts.Values())
		}
	}
	if nties == 0 {
		t.Fatalf("no ties")
	}
	// with frequencies, 2:2 is not a tie unless the summed frequencies are equal
	cfg.Freqs = []float32{3, 1, 1, 1}
	if cfg.NewItem([]int{0, 0, 5, 5}).IsTie() { // group 0 member 0 (3+3) vs group 1 member 1 (1+1)
		t.Errorf("unequal frequencies counted as a tie")
	}
	if !cfg.NewItem([]int{1, 2, 5, 6}).IsTie() { // all frequency 1
		t.Errorf("equal frequencies not a tie")
	}
}

// TestItemConsistency checks the per-item choice consistency across the
// test repetitions, split across parallel envs: 1 for a deterministic
// choice for each item, and about 0.59 for coin flips on 20 repetitions.
func TestItemConsistency(t *testing.T) {
	var cfg Config
	cfg.Defaults()
	cfg.NSlots = 2
	cfg.TestPerLevel = 40
	cfg.TestReps = 20
	ndata := 4
	run := func(choose func(ev *CFEnv, rnd *rand.Rand) int) (float64, float64) {
		rnd := rand.New(rand.NewPCG(1, 2))
		var envs []*CFEnv
		for di := range ndata {
			ev := &CFEnv{Configs: cfg}
			ev.Config(0, ndata, di, true, 181+int64(di))
			ev.Init(0)
			sp := ev.Space()
			var ts TrialStats
			for range ev.Trial.Max {
				ev.Step()
				ev.TrialStats(pattern(sp.Outputs, choose(ev, rnd)), &ts)
			}
			envs = append(envs, ev)
		}
		return ItemConsistency(envs...)
	}
	// deterministic: the group in slot 0
	c, ties := run(func(ev *CFEnv, rnd *rand.Rand) int { return cfg.Group(ev.Cur.Inputs[0]) })
	if c != 1 || ties != 1 {
		t.Errorf("deterministic: consist %g, ties %g", c, ties)
	}
	// coin flip between the two groups
	c, ties = run(func(ev *CFEnv, rnd *rand.Rand) int { return ev.Cur.Groups[rnd.IntN(len(ev.Cur.Groups))] })
	t.Logf("coin flip: consist %.3f, ties %.3f", c, ties)
	if math.Abs(ties-0.59) > 0.05 {
		t.Errorf("coin flip ties consistency %g, want about 0.59", ties)
	}
}
