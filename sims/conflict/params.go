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
				ly.Learn.TrgAvgAct.SynScaleRate = 0.0002 // 0.005 default; 0.0002 better!
				ly.Acts.Noise.On.SetBool(false)
				ly.Acts.Noise.Ge = 0.1 // can handle up to .1!
				ly.Acts.Noise.Gi = 0.1
				// ly.Acts.Decay.GBuffs.SetBool(true) // no diff?
				ly.Acts.Decay.Act = 0.2   // 0.2 default, 1.0, 1.0 doesn't learn!
				ly.Acts.Decay.Glong = 0.6 // 0.6 default
			}},
		{Sel: ".Input", Doc: "one sparse pattern per slot pool: 20% activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.05
				ly.Inhib.Layer.On.SetBool(false)
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.Gi = 0.95
				ly.Inhib.Pool.Gi = 0.95
				// ly.Acts.Decay.Act = 1 // this is subtly beneficial
				// ly.Acts.Decay.Glong = 1
			}},
		{Sel: ".Encoder", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.02 // .03 > 0.4
				ly.Inhib.ActAvg.Offset = 0.05  // .05 > 0.6
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.Layer.On.SetBool(true) // true is more standard and gives more competition
				ly.Inhib.Layer.Gi = 1.2         // 1.2 > 1.05 for gradedness
				ly.Inhib.Layer.FB = 1           // 1.0 default; ra25x: 0.5
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Pool.Gi = 1.05
			}},
		{Sel: ".Hidden", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.03 // .03 > 0.4
				ly.Inhib.ActAvg.Offset = 0.05  // .05 > 0.6
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.1 // 1.1
				ly.Inhib.Layer.FB = 1   // 1.0 default; ra25x: 0.5
			}},
		{Sel: "#Output", Doc: "one bar of NGroups: 20% activity",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Clamp.Ge = 0.8
				ly.Inhib.ActAvg.Nominal = 0.2
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.1
				ly.Acts.Decay.Act = 1
				ly.Acts.Decay.Glong = 1
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
				pt.PathScale.Abs = 1.0      // critical to reset to default b/c not otherwise done!
				pt.Learn.LRate.Base = 0.005 // 0.005 def -- faster worse interference
				pt.SWts.Adapt.LRate = 0.0001
				pt.Learn.DWt.SubMean = 1
				pt.SWts.Init.SPct = 1.0 // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
				// NOTE: MUST turn on Config.Params.TestSynFail for this to affect testing:
				pt.Com.PFail = 0.4            // 0.4 most linear
				pt.Com.PFailSWt.SetBool(true) // swt is essential!
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2 // 0.2 is general default, works fine with PFail
			}},
		{Sel: ".LateralPath", Doc: "lateral connections within the hidden layers and Output (if Lateral, OutputLateral)",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2
			}},
		// {Sel: ".Shortcut", Doc: "shortcut connections from Encoder to output",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Rel = 0.2
		// 	}},
	},
	"Test": {
		{Sel: "Path", Doc: "test manip",
			Set: func(pt *axon.PathParams) {
				// pt.PathScale.Abs = 0.8 // 0.8 simulates effects of pfail
				// pt.Com.PFail = 0.4
				// pt.Com.PFailSWt.SetBool(false)
			}},
	},
}
