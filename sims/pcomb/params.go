// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcomb

import "github.com/emer/axon/v2/axon"

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "generic params for all layers",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.AdaptRate = 0.1 // 0.1 default > 0.05?
				ly.Inhib.ActAvg.AdaptMax = 0.05 // 0.05 > 0.01
				ly.Acts.Noise.On.SetBool(false) // not neccessary
				ly.Acts.Noise.Ge = 0.001
				ly.Acts.Noise.Gi = 0.001
				// ly.Acts.Decay.GBuffs.SetBool(true) // this is important for non-PFC layers
			}},
		{Sel: ".Input", Doc: "Value and Prob cue inputs: about 6 of 22 units active per active cue, 2-4 of 4 cues active",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.08 // 0.08 > 0.05 for long-term gemax
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.Gi = 0.95
				ly.Inhib.Pool.Gi = 0.95
			}},
		{Sel: ".Encode", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.03
				ly.Inhib.ActAvg.Offset = 0.03
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.1
			}},
		{Sel: "#Hidden", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.02
				ly.Inhib.ActAvg.Offset = 0.03
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.1
			}},
		{Sel: "#Output", Doc: "one value bump: about 6 of 22 units",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Clamp.Ge = 1.2
				ly.Inhib.ActAvg.Nominal = 0.05
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.05 // 1.1 or 1.05 about same
			}},
		{Sel: "#Salience", Doc: "salience bump, or nothing for low magnitude values",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Clamp.Ge = 1.2
				ly.Inhib.ActAvg.Nominal = 0.05
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.05 // 1.1 or 1.05 about same
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
				pt.Learn.LRate.Base = 0.005 // 0.005 >= 0.01 (deteriorates?) > 0.002 (too slow)
				pt.SWts.Init.SPct = 1
				pt.SWts.Adapt.LRate = 0.0001
				pt.Learn.DWt.SubMean = 1
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2 // 0.2 maybe > 0.3, 0.1?
			}},
	},
	"Test": {},
}
