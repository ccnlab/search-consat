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
				// ly.Inhib.Layer.FB = 0.5 // 1.0 default; ra25x: 0.5
				// ly.Acts.NMDA.MgC = 1.2                   // 1.4 default; ra25x: 1.2
				ly.Learn.TrgAvgAct.SynScaleRate = 0.0002 // 0.005 default; 0.0002 better!
				ly.Acts.Noise.On.SetBool(false)
				ly.Acts.Noise.Ge = 0.001
				ly.Acts.Noise.Gi = 0.001
				// ly.Acts.Decay.GBuffs.SetBool(true) // no diff?
			}},
		{Sel: ".Input", Doc: "one sparse pattern per slot pool: 20% activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.05
				ly.Inhib.Layer.On.SetBool(false)
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.Gi = 0.95
				ly.Inhib.Pool.Gi = 0.95
				ly.Acts.Decay.Act = 1 // this is subtly beneficial
				ly.Acts.Decay.Glong = 1
			}},
		{Sel: ".Encoder", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.02 // .06 in ra25x
				ly.Inhib.ActAvg.Offset = 0.03  // .03 best
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.05
				ly.Inhib.Layer.FB = 1 // 1.0 default; ra25x: 0.5
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Pool.Gi = 1.05
			}},
		{Sel: ".Hidden", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.02 // .06 in ra25x
				ly.Inhib.ActAvg.Offset = 0.03  // .03 best
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.1
				ly.Inhib.Layer.FB = 1 // 1.0 default; ra25x: 0.5
			}},
		{Sel: "#Output", Doc: "one bar of NGroups: 20% activity",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Clamp.Ge = 0.8
				ly.Inhib.ActAvg.Nominal = 0.15
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1.05
				// ly.Acts.Spikes.Tr = 1             // 1 is new minimum.. > 3
				// ly.Learn.RLRate.On.SetBool(true)  // beneficial for trace
				// ly.Learn.RLRate.SigmoidMin = 0.05 // sigmoid derivative actually useful here!
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
				pt.SWts.Adapt.LRate = 0.0001
				pt.Learn.DWt.SubMean = 1
				pt.SWts.Init.SPct = 0.5 // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
				pt.Learn.DWt.LearnThr = .1
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2
			}},
		{Sel: ".LateralPath", Doc: "lateral connections within the hidden layers and Output (if Lateral, OutputLateral)",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2
			}},
	},
	"Test": {},
}
