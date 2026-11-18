// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	conflictff "github.com/ccnlab/search-consat/sims/conflict_ff"
	"github.com/emer/emergent/v2/egui"
)

// main runs the conflict_ff sim.
func main() { egui.Run[conflictff.Sim, conflictff.Config]() }
