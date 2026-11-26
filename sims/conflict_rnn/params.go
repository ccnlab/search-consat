// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conflictrnn

import (
	"github.com/emer/egomlx/egomlx"
)

// NetworkParams sets the minimal non-default global network params,
// controlling the loss function and optimizer for the backward pass.
// Base is always applied, and others can be optionally selected to apply on top of that.
var NetworkParams = egomlx.NetworkSheets{
	"Base": {
		{Sel: "Network", Doc: "BCE loss for the 0-1 bar targets, Adam optimizer",
			Set: func(np *egomlx.NetworkParams) {
				np.Loss = egomlx.BCE
				np.Optim = egomlx.Adam
				np.LRate = 0.001
			}},
	},
}

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = egomlx.LayerSheets{
	"Base": {
		{Sel: ".Hidden", Doc: "hidden layers use Tanh activation, bounded for stable recurrent settling (overridden by HiddenFn, and Noise is set, from the config in ApplyParams)",
			Set: func(ly *egomlx.LayerParams) {
				ly.Act.Fn = egomlx.Tanh
			}},
		{Sel: ".Target", Doc: "the Output layer uses a Sigmoid, giving bounded 0-1 output activations for BCE",
			Set: func(ly *egomlx.LayerParams) {
				ly.Act.Fn = egomlx.Sigmoid
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = egomlx.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "all forward paths: bias on, Glorot weight init, learning on",
			Set: func(pt *egomlx.PathParams) {
				pt.UseBias = true
				pt.WtInit.Init = egomlx.Glorot
				pt.WtInit.Scale = 1
				pt.Learn = true
			}},
	},
}
