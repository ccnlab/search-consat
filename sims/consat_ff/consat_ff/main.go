// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	consatff "github.com/ccnlab/search-consat/sims/consat_ff"
	"github.com/emer/emergent/v2/egui"
)

// main runs the consat_ff sim.
func main() { egui.Run[consatff.Sim, consatff.Config]() }
