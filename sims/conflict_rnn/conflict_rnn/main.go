// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	conflictrnn "github.com/ccnlab/search-consat/sims/conflict_rnn"
	"github.com/emer/emergent/v2/egui"
)

// main runs the conflict_rnn sim.
func main() { egui.Run[conflictrnn.Sim, conflictrnn.Config]() }
