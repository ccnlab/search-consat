# pcomb_ff: feedforward baseline

A feedforward backprop version of the [pcomb](../pcomb) model, built with [egomlx](https://github.com/emer/egomlx) (gomlx). It uses the same environment, [pcomb/pcombenv](../pcomb/pcombenv), with the same defaults, train/test split and `[Env.Env]` config overrides (see the pcomb README "First pass" section for the task details).

The network replicates the axon pcomb network, without the top-down back pathways:

```
Value (4x1x1x22) -> ValueEnc (15x15, ReLU) --\
                                              Hidden (15x15, ReLU) -> Output (1x22, Sigmoid, Target)
Prob  (4x1x1x22) -> ProbEnc  (15x15, ReLU) --/                     -> Salience (1x22, Sigmoid, Target)
```

All connections are full, with BCE loss (summed over both target layers) and the Adam optimizer, learning rate 0.001 (see `params.go`). The encoding and hidden layer sizes are `Params.EncodeSize` and `Params.HiddenSize`.

## Stats

The same stats as pcomb, computed by the shared `PCombEnv.TrialStats` in [pcombenv/stats.go](../pcomb/pcombenv/stats.go) on the `Output` and `Salience` Act values instead of the axon minus phase. See the pcomb README Statistics section for the definitions and reference values. The main ones are `Miss`, `PGain` (the gain from using the probabilities, 0 for any choice that ignores them), `OutPErr`, `CalibErr`, and `AgreeBoost` (on the agreement test items), with the test versions copied into the train epoch log. The calibration vectors are in `Current/<mode>/Calib`, as in pcomb.

Also: `*_ActAvg/ActMax` for the hidden and target layers, `*_PCA_*` for the hidden layers every `Run.PCAInterval` epochs, and `Train/RunAll` across runs. The axon GiMult and learning timing stats have no feedforward analog.

Note that the feedforward network is deterministic: a given input always produces the same output. Trained on sampled targets, it learns the expected target, which is ideally a blend of the cue value bumps weighted by their probabilities, and the decoded `Value` is the peak of that blend. So `OutPErr` is the best measure of whether it uses the probabilities: 0.08 for a perfect probability-weighted blend, vs. 0.19 for a uniform blend that ignores them. A probability-weighted blend would also give a `PGain` of about 0.12.

## Running

From `pcomb_ff/`: `go run ./pcomb_ff` for the GUI, or `go run ./pcomb_ff -gui=false -runs 1 -epochs 20` for a headless run writing `.tsv` logs.

In a 20-epoch run (default config, about 10 seconds), with training on distinct-value items only (`DistinctTrain`), test `Miss` gets down to about 2%, but `PGain` stays at 0 (train and test) and `OutPErr` at about 0.22, the same as a uniform blend. So even with the probability pathway present, the network does not use the probabilities at all. Before `DistinctTrain`, its `ChoseP` of 0.41 came entirely from the cue count shortcut on agreement items. Test `AgreeBoost` is about 0.2-0.3, which, without any use of the probabilities, reflects the peak of the blend landing on the repeated values.
