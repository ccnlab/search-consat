# conflict_ff

This is a feedforward backprop baseline for the [conflict](../conflict/README.md) task, built with egomlx (gomlx). It uses the same `cfenv` environment, item space, and stats as the axon model: see the conflict README for the task and the stats.

## Network

```
Output   [NPerOutput, NGroups]   Target (sigmoid)
   ↑
Hidden2  [15, 15]                (ReLU)
   ↑
Hidden1  [15, 15]                (ReLU)
   ↑
Input    [1, k, 7, 7]
```

- `NHidden` (default 2) sets the number of hidden layers, and `HiddenSize` their size.
- `NEncoder` (default 0) adds encoder layers below the hidden layers (`Encoder1`, ...), as in the axon model: one pool per input slot, of size `EncoderSize`, with pool-to-pool (one-to-one) connections, so each slot is encoded separately. With both encoder and hidden layers, `Shortcut` adds full connections from the last encoder layer to the Output. Either `NEncoder` or `NHidden` can be 0.
- Loss and optimizer: BCE loss, Adam with LR 0.001. Weights: Glorot init, with biases.
- The network is deterministic: each test item always produces the same output, so the `TestReps` repetitions are identical, and there is no carryover between trials (`RepeatPrev` is at chance). `<Layer>_CarryEnd` (every layer, Train and Test logs, copied as `Test_*`; off in the plots) is the same carryover stat as in the axon and rnn models: the cosine of each layer's activity with its activity on the previous trial of the same data-parallel stream, minus the mean cosine with the other streams' previous activity (the baseline for the input overlap). Here it is a control, and should be ~0. It needs `NData` > 1.

Training runs for `Epochs` (default 100), with a final test at the end of each run, and optional tests every `TestInterval` epochs.

## Results

1 run, 30 epochs of 512 trials, default params. Training is perfect by epoch 5.

| k | P100 | P83 | P75 | P67 | P50 | P33 | P25 | P17 | Blend |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1.0 | | 1.0 | | 0.35 | | 0 | | 0.10 |
| 6 | 1.0 | 1.0 | | 1.0 | 0.38 | 0 | | 0 | 0.06 |

- The network follows a strict majority: the group with more inputs always wins.
- About 25–30% of the ties (equal evidence for two groups) are blends. The rest go to one group, with a fixed choice for each item (`Slot0` is 0.61 for k = 4: a per-item preference, not randomness).
- So majority rule alone does not distinguish the axon model: a saturating sigmoid output trained on consistent inputs already turns unequal evidence into a clean winner.
- `Consist` and `TieConsist` are 1: being deterministic, the network always makes the same choice for the same item.

### Weight decay and dropout

k = 6, 1 run each, 30 epochs. `WeightDecay` > 0 uses the AdamW optimizer, and `Dropout` applies to the hidden layer units in training only.

| | P83 | P67 | P50 | P33 | Blend | Consist | TieConsist |
|---|---|---|---|---|---|---|---|
| none | 1.0 | 1.0 | 0.38 | 0 | 0.06 | 1.0 | 1.0 |
| WeightDecay 0.01 | 1.0 | 1.0 | 0.37 | 0 | 0.06 | 1.0 | 1.0 |
| WeightDecay 0.1 | 1.0 | 1.0 | 0.38 | 0 | 0.06 | 1.0 | 1.0 |
| Dropout 0.2 | 1.0 | 1.0 | 0.36 | 0 | 0.07 | 1.0 | 1.0 |
| Dropout 0.5 | 1.0 | 1.0 | 0.41 | 0 | 0.05 | 1.0 | 1.0 |

Neither changes anything: the network is still a strict majority rule, and deterministic.
