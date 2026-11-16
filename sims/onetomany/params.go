// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package onetomany

import "github.com/emer/axon/v2/axon"

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
// These are the combsearch params, with the slot pool inhibition as layer
// inhibition, because the layers here are 2D, equivalent to one slot.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "generic params for all layers",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.AdaptRate = 0.1
				ly.Inhib.ActAvg.AdaptMax = 0.05
				ly.Acts.Noise.On.SetBool(false)
				ly.Acts.Noise.Ge = 0.01
				ly.Acts.Noise.Gi = 0.01
				// ly.Acts.Decay.GBuffs.SetBool(true) // this is important for non-PFC layers
			}},
		{Sel: ".Input", Doc: "one sparse pattern: 20% activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.1
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.0
			}},
		{Sel: ".Encode", Doc: "encoding layers",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.04
				ly.Inhib.ActAvg.Offset = 0.01
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.0
			}},
		{Sel: "#Output", Doc: "one sparse pattern: 20% activity",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Clamp.Ge = 1.0
				ly.Inhib.ActAvg.Nominal = 0.1
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.0
			}},
	},
	"Test": {},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "std",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.005 // .005 > .01
				pt.SWts.Init.SPct = 1
				pt.SWts.Adapt.LRate = 0.0001
				pt.Learn.DWt.SubMean = 1
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2
			}},
	},
	"Test": {},
}
