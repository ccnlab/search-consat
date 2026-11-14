# consat_ff: feedforward baseline

A feedforward backprop version of the [consat](../consat) model, built with [egomlx](https://github.com/emer/egomlx) (gomlx). It uses the same environment, [consat/consatenv](../consat/consatenv): partial / ambiguous line segments rendered over a 5x5 grid of 5x5 hypercolumns.

In the axon `consat` model, the Full line pattern is reconstructed on the Input layer via top-down projections from higher layers. That is not possible in a feedforward network, so here `Full` is the top output Target layer:

```
Input (5x5x5x5) -> Hidden (20x20, ReLU) -> Full (5x5x5x5, Sigmoid, Target)
```

All connections are Full; BCE loss with the Adam optimizer (see `params.go`).

Train trials use partial line inputs (`Partial = true`); Test trials present only the maximally ambiguous central points of each hypercolumn (`Partial = false`).

## Stats

Same as `consat`, computed on the Full layer Act vs. Targ:

* `CorSim`: correlation of output with the target pattern.
* `OnWasOff`, `OffWasOn`: proportion of target-on units that were off, and target-off units that were on (threshold `Log.OnOffThr`).
* `Test_*` copies of these in the Train epoch log.
* `Hidden_ActAvg/ActMax`, `Full_ActAvg/ActMax` (feedforward analog of axon's ActGe stats).
* `Hidden_PCA_*` every `Run.PCAInterval` epochs.
* `Train/RunAll` accumulates Run stats across runs.

The plus-phase (`*P`) stats, GiMult, and learning-timing stats in `consat` are axon-specific and have no feedforward analog.

## Running

From `consat_ff/`: `go run .` for the GUI, or `go run . -nogui -runs 1 -epochs 30` for a headless run writing `.tsv` logs.

Note: `NData` defaults to 10 as in `consat`: the env has only 70 lines, and `consatenv` item slicing overruns for larger NData (e.g., 16).
