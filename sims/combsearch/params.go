// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package combsearch

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
				ly.Acts.Noise.Ge = 0.01
				ly.Acts.Noise.Gi = 0.01
				// ly.Acts.Decay.GBuffs.SetBool(true) // this is important for non-PFC layers
			}},
		{Sel: ".Input", Doc: "one sparse pattern per slot pool: 20% activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.05
				ly.Inhib.Layer.On.SetBool(false)
				ly.Inhib.Pool.On.SetBool(true)
				// ly.Inhib.Layer.Gi = 1.0
				ly.Inhib.Pool.Gi = 1.0
			}},
		{Sel: ".Encode", Doc: "per-slot encoding pools, coordinated through Hidden",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.02
				ly.Inhib.ActAvg.Offset = 0.03
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.Layer.On.SetBool(false)
				ly.Inhib.Pool.On.SetBool(true)
				// ly.Inhib.Layer.Gi = 1.0
				ly.Inhib.Pool.Gi = 1.0
			}},
		{Sel: "#Hidden", Doc: "coordinates the group choice across slots",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.02
				ly.Inhib.ActAvg.Offset = 0.03
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.0
			}},
		{Sel: "#Output", Doc: "one sparse pattern per slot pool: 20% activity",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Clamp.Ge = 0.8
				ly.Inhib.ActAvg.Nominal = 0.1
				ly.Inhib.Layer.On.SetBool(false)
				ly.Inhib.Pool.On.SetBool(true)
				// ly.Inhib.Layer.Gi = 1.0
				ly.Inhib.Pool.Gi = 0.95
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
		// {Sel: ".LateralPath", Doc: "self-connections among encode layers",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Rel = 0.2
		// 	}},
		{Sel: "#HiddenToEncode", Doc: "back-path from hidden -- needs to be stronger",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2
			}},
		{Sel: "#HiddenToOutput", Doc: "makes encoding worse..",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2
			}},
		{Sel: "#EncodeToOutput", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 1.0
			}},
	},
	"Test": {},
}
