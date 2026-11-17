// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflict

import "github.com/emer/axon/v2/axon"

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "generic params for all layers",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.AdaptRate = 0.1
				ly.Inhib.ActAvg.AdaptMax = 0.05
				ly.Acts.Noise.On.SetBool(false)
				ly.Acts.Noise.Ge = 0.001
				ly.Acts.Noise.Gi = 0.001
				ly.Acts.Decay.GBuffs.SetBool(true) // this is important for non-PFC layers
			}},
		{Sel: ".Input", Doc: "one sparse pattern per slot pool: 20% activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.08
				ly.Inhib.Layer.On.SetBool(false)
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.Gi = 0.95
				ly.Inhib.Pool.Gi = 0.95
			}},
		{Sel: ".Hidden", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.02
				ly.Inhib.ActAvg.Offset = 0.03
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.1
			}},
		{Sel: "#Output", Doc: "one bar of NGroups: 20% activity",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Clamp.Ge = 0.8
				ly.Inhib.ActAvg.Nominal = 0.1
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.05
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
				pt.Learn.LRate.Base = 0.005
				pt.SWts.Init.SPct = 1
				pt.SWts.Adapt.LRate = 0.0001
				pt.Learn.DWt.SubMean = 1
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2
			}},
		// {Sel: ".LateralPath", Doc: "lateral connections within the hidden layers and Output (if Lateral, OutputLateral)",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Rel = 0.2
		// 	}},
	},
	"Test": {},
}
