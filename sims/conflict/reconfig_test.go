// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflict

import (
	"math"
	"strings"
	"testing"

	"cogentcore.org/lab/tensor"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/egui"
)

// reconfigTestConfig returns a small config with the given number of
// data-parallel streams and hidden layers.
func reconfigTestConfig(gpu bool, ndata, nhidden int) *Config {
	cfg, _ := egui.NewConfig[Config]()
	cfg.GPU = gpu
	cfg.GUI = false
	cfg.Run.NData = ndata
	cfg.Run.Trials = 8
	cfg.Run.Epochs = 2
	cfg.Run.Runs = 1
	cfg.Run.TestInterval = 1
	cfg.Params.NHidden = nhidden
	cfg.Env.Env = map[string]any{"TestPerLevel": 4, "TestReps": 1}
	return cfg
}

// epochStats returns the numeric Train and Test Epoch stats, except the
// timing ones.
func epochStats(ss *Sim) map[string][]float64 {
	st := map[string][]float64{}
	for _, mode := range []string{"Train", "Test"} {
		dir := ss.Stats.Dir(mode + "/Epoch")
		for _, nd := range dir.NodesFunc(nil) {
			if strings.Contains(nd.Name(), "MSec") || nd.Tensor == nil || nd.Tensor.IsString() {
				continue
			}
			st[mode+"/"+nd.Name()] = tensor.AsFloat64(nd.Tensor).Values
		}
	}
	return st
}

// TestReConfig checks that ReConfig with a changed config (fewer
// data-parallel streams and hidden layers) gives the same network and
// results as a sim configured from the start with that config, on the
// CPU and the GPU (where the GPU buffers are resized), and that nothing
// of the old network is left in the network or the stats.
func TestReConfig(t *testing.T) {
	for _, gpu := range []bool{false, true} {
		ss := &Sim{}
		ss.SetConfig(reconfigTestConfig(gpu, 4, 2))
		ss.ConfigSim()
		ss.Loops.Run(Train)
		if ss.Net.LayerByName("Hidden2") == nil {
			t.Fatalf("GPU %v: no Hidden2 in the first network", gpu)
		}

		ss.SetConfig(reconfigTestConfig(gpu, 2, 1))
		ss.ReConfig()
		ss.Loops.Run(Train)
		reSyns := ss.Net.NetIxs().NSyns
		reStats := epochStats(ss)
		if ss.Net.LayerByName("Hidden2") != nil {
			t.Errorf("GPU %v: Hidden2 is still found after ReConfig", gpu)
		}
		for name := range reStats {
			if strings.Contains(name, "Hidden2") {
				t.Errorf("GPU %v: old stat %s after ReConfig", gpu, name)
			}
		}
		if n := ss.Loops.Loop(Train, Trial).Counter.Inc; n != 2 {
			t.Errorf("GPU %v: Trial increment %d after ReConfig, want NData 2", gpu, n)
		}

		fr := &Sim{}
		fr.SetConfig(reconfigTestConfig(gpu, 2, 1))
		fr.ConfigSim()
		fr.Loops.Run(Train)
		frStats := epochStats(fr)
		if fr.Net.NetIxs().NSyns != reSyns {
			t.Errorf("GPU %v: NSyns %d after ReConfig != %d fresh", gpu, reSyns, fr.Net.NetIxs().NSyns)
		}
		if len(reStats) != len(frStats) {
			t.Errorf("GPU %v: %d stats after ReConfig != %d fresh", gpu, len(reStats), len(frStats))
		}
		ndiff := 0
		for name, fv := range frStats {
			rv, ok := reStats[name]
			if !ok {
				t.Errorf("GPU %v: stat %s missing after ReConfig", gpu, name)
				continue
			}
			if len(rv) != len(fv) {
				t.Errorf("GPU %v: %s has %d rows after ReConfig != %d fresh", gpu, name, len(rv), len(fv))
				continue
			}
			for i := range fv {
				if fv[i] != rv[i] && !(math.IsNaN(fv[i]) && math.IsNaN(rv[i])) {
					ndiff++
					if ndiff <= 10 {
						t.Errorf("GPU %v: %s[%d] = %g after ReConfig != %g fresh", gpu, name, i, rv[i], fv[i])
					}
				}
			}
		}
		t.Logf("GPU %v: %d epoch stats compared, %d values differ; Test_P100 %v", gpu, len(frStats), ndiff, frStats["Train/Test_P100"])
		if gpu {
			axon.GPURelease()
			axon.UseGPU = false
		}
	}
}
