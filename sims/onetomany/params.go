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
				// ly.Inhib.ActAvg.AdaptMax = 0.05 // 0.01 default
				ly.Inhib.Layer.FB = 0.5 // 1.0 default; ra25x: 0.5 DEFINNITELY better.
				// ly.Acts.NMDA.MgC = 1.2   // not beneficial
				// ly.Acts.VGCC.Ge = 0 / not beneficial
				// ly.Learn.TrgAvgAct.SynScaleRate = 0.0002 // 0.005 default; is better here..
				ly.Acts.Noise.On.SetBool(false)
				ly.Acts.Noise.Ge = 0.01
				ly.Acts.Noise.Gi = 0.01
				// ly.Acts.Decay.GBuffs.SetBool(true) // this is important for non-PFC layers
			}},
		{Sel: ".Input", Doc: "one sparse pattern: 20% activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.1
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 0.95 // 0.9 == 0.95
				// ly.Acts.Decay.Act = 1 // no diff
				// ly.Acts.Decay.Glong = 1
			}},
		{Sel: ".Encode", Doc: "encoding layers",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.04
				ly.Inhib.ActAvg.Offset = 0.01
				// ly.Inhib.ActAvg.AdaptGi.SetBool(true) // worse with, small dif
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.05
				// ly.Inhib.Layer.FB = 1 // 1.0 default; ra25x: 0.5
			}},
		{Sel: "#Output", Doc: "one sparse pattern: 20% activity",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Clamp.Ge = 0.8
				// ly.Acts.Spikes.Tr = 1             // better without.
				ly.Inhib.ActAvg.Nominal = 0.1
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.1
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
				pt.Learn.LRate.Base = 0.01 // 0.01 > 0.02, .005
				pt.SWts.Init.SPct = 1
				pt.SWts.Adapt.LRate = 0.0001
				pt.Learn.DWt.SubMean = 1 // better long-lasting perf
				// pt.Learn.DWt.LearnThr = 0.1
				// pt.Com.PFail = 0.01
				// pt.Com.PFailSWt.SetBool(true)
			}},
		{Sel: ".ForwardPath", Doc: "std",
			Set: func(pt *axon.PathParams) {
				// pt.Com.PFail = 0.05
				// pt.Com.PFailSWt.SetBool(true)
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2 // 0.3 == 0.2 but maybe less variance at 0.3
			}},
		// {Sel: ".LateralPath", Doc: "lateral connections within the Output (if OutputLateral)",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Rel = 0.3 // 0.3 == 0.2 but maybe less variance at 0.3
		// 	}},
		{Sel: ".ToOutput", Doc: "from the last Encode layer to the Output",
			Set: func(pt *axon.PathParams) {
				// pt.Learn.LRate.Base = 0.005 // .005 > .01
				pt.PathScale.Abs = 1
				pt.Com.PFail = 0
				// pt.Learn.DWt.CaPScale = 0.95 // not helpful
			}},
	},
	"Test": {},
}
