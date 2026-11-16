// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package otmenv

import (
	"fmt"
	"sync"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/lab/patterns"
	"cogentcore.org/lab/tensor"
)

var (
	vocabs     = map[string]*tensor.Float32{}
	vocabsLock sync.Mutex
)

// Vocab returns the vocabulary for this config, generating it on first
// use: [NVocab, PoolSize, PoolSize], with NInputs input patterns followed
// by NOut output patterns for each input (see [Config.OutputIndex]).
// It is shared across all envs with the same config.
func (cfg *Config) Vocab() *tensor.Float32 {
	vocabsLock.Lock()
	defer vocabsLock.Unlock()
	key := cfg.Key()
	if vc, ok := vocabs[key]; ok {
		return vc
	}
	vc := cfg.MakeVocab()
	vocabs[key] = vc
	fmt.Printf("vocab: %d patterns (%d inputs + %d outputs, %d per input), %dx%d, %d active, min diff %d\n", cfg.NVocab(), cfg.NInputs, cfg.NOutputs(), cfg.NOut, cfg.PoolSize, cfg.PoolSize, cfg.NOn(), cfg.MinDiff())
	return vc
}

// MakeVocab generates the vocabulary using [Config.Seed],
// with [patterns.PermutedBinaryMinDiff].
func (cfg *Config) MakeVocab() *tensor.Float32 {
	ps := cfg.PoolSize
	vc := tensor.NewFloat32(cfg.NVocab(), ps, ps)
	patterns.NewRand(cfg.Seed)
	errors.Log(patterns.PermutedBinaryMinDiff(vc, cfg.NOn(), 1, 0, cfg.MinDiff()))
	return vc
}
