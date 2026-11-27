# conflict_rnn

This is a recurrent backprop baseline for the [conflict](../conflict/README.md) task, built with egomlx (gomlx). It uses the same `cfenv` environment, item space, and stats as the axon model: see the conflict README for the task and the stats. See [conflict_ff](../conflict_ff/README.md) for the feedforward version.

## Network

The same layers as conflict_ff, with the bidirectional connectivity of the axon model:

```
Output   [NPerOutput, NGroups]   Target (sigmoid)
   ↕
Hidden2  [15, 15]                (tanh)
   ↕
Hidden1  [15, 15]                (tanh)
   ↑
Input    [1, k, 7, 7]
```

- Forward paths go up the stack, and Back paths come down it (except to the Input). `Lateral` adds lateral connections within each hidden layer, and `OutputLateral` within the Output. These full lateral connections include each unit's connection to itself.
- `NEncoder` (default 0) adds encoder layers below the hidden layers (`Encoder1`, ...), as in the axon model: one pool per input slot, of size `EncoderSize`, with pool-to-pool (one-to-one) Forward and Back connections, so each slot is encoded separately. `EncoderLateral` adds lateral connections within each pool. With both encoder and hidden layers, `Shortcut` adds full Forward connections from the last encoder layer to the Output, and `EncoderBack` full Back connections from the Output to it. Either `NEncoder` or `NHidden` can be 0.
- The network settles for `NCycles` (default 10) cycles, using egomlx recurrent settling: on each cycle, Forward paths use the sender's current-cycle activity, and Back and Lateral paths its previous-cycle activity (zero on the first cycle). The weights are shared across cycles, and the loss is on the final cycle, so learning is backpropagation through time.
- `Noise` (default 0.2) is the standard deviation of Gaussian noise added to the net input of the hidden layers on every cycle, in training and testing. With the default `Decay` of 1, it is the only source of variability across repetitions of the same item: each trial starts from zero activity, with no carryover between trials.

- `TestNoise` (default 0) is the standard deviation of additional Gaussian noise added to the hidden net inputs on every cycle in testing only, as `TestDropout` is to `Dropout`. The network learns to override the training `Noise` (more `Noise` makes the choices more deterministic), but it does not learn to override this. It is independent of `Noise`: the total SD in testing is sqrt(`Noise`² + `TestNoise`²).
- `HiddenFn` (default Tanh) is the activation function of the hidden layers. Tanh activity is bounded, which keeps the recurrent settling stable. ReLU activity is unbounded, and the Back and Lateral loops can amplify it: the first 10-run baseline on the cluster (ReLU, 151 epochs) had hidden activity growing to 40–55 in the healthy runs, and in run 3 it grew to about 600 in the first epoch and millions by the second, until the loss became NaN at epoch 59. ReLU learned the task as well as Tanh in short runs (MatchCos 1.0 by epoch 15), so the results below with ReLU are still valid, but runs with it can blow up.
- `Decay` (default 1) is the proportion of the previous trial's final activity that decays before the next trial: the first cycle starts from (1 − `Decay`) times that activity, so the previous trial carries over through the recurrent paths, as in the axon model. 1 = each trial starts from zero. Gradients do not flow back into the previous trial.
- Loss and optimizer: BCE loss, Adam with LR 0.001. Weights: Glorot init, with biases.

## Compute backend

`Backend` (`-backend` on the command line, e.g., in the simmer `Args`) selects the gomlx compute backend, as the `GOMLX_BACKEND` environment variable does: `xla:cuda` (the GPU), `xla:cpu`, or `go` (the pure-Go CPU backend). The default is `xla:cpu`, which is faster than the GPU for these small networks: on the h200, conflict_ff training took a median 0.48 ms/trial on `xla:cpu`, 0.66 on `xla:cuda`, and 1.4 on `go`. An empty `Backend` uses `GOMLX_BACKEND` if set, else XLA, which uses the GPU if there is one. The backend in use is printed at the start of a `-nogui` run (in the simmer `job.0.out`).

## Parameter search

`Search` in `search.go` lists the parameter search: a baseline job with the base config (`Decay` 1, `Noise` .2; job 1), and then one job per value of one parameter, with everything else at the base config. Build the sim, list the jobs with `./conflict_rnn/conflict_rnn -nogui -search-n`, and run them with the simmer `ParamSearch` button, whose `Args` set the base config of all the jobs (e.g., `-decay 0.5` to search the others with carryover), including the baseline. See the emergent `params` README for how the search works.

The first search that was run, kept as `SearchPS1` (30 jobs, 10 runs each), found that the Tanh network is a near-perfect majority detector (In67 1.0, In33 0, no intrusions). `Decay` was the only parameter that graded the majority choice with clean outputs (In67 .92 at .75, .84 at .65, about .82 at .5, MatchCenter 1.9), but by perseveration: RepeatPrev was 1.0 from .6 to .85, and MinorPrevMinor .4 to .5. `TestDropout` changed the majority choice only at .8 (In67 .95, with competition: LeadSwaps .68) and broke down at .9 (Intrusion .46). `Noise` (0 to 4) changed only the tie choices, and `OutputLateral`, `Lateral` and `NCycles` had no effect.

The second search, kept as `SearchPS2` (24 jobs), found that strong carryover (`Decay` 0 to .4) is less graded (In67 .95 at 0, .875 at .3 to .4) but not perseverative, with competition during settling (LeadSwaps .4 to .8); that `Noise` (in training and testing) with `Decay` makes the choices more deterministic, as the network learns to override it; and that `Decay` .5 with `TestDropout` .6 breaks up the perseveration (In67 .88, Intrusion .016, MinorPrevMinor about the minority rate). Its `Decay` × `TestDropout` jobs trained with the test dropout leaking into the carried-over activity (egomlx `TrainTrial` ran its forward pass with the test-only perturbations; fixed since), so they are not test-only results.

The current `Search` (24 jobs) covers `TestNoise` (test-only noise, which the network has not learned to override): alone (.25 to 4), with `Decay` .5, .7 and .3, with the training `Noise` at 0, and with `Decay` .5 and `TestDropout` .6, plus clean reruns of `Decay` .5 with `TestDropout` .6 and .8. Each combination job sets all of its fields, which are in its label (e.g., `Decay=0.5,TestNoise=1`). The first version, with training `Dropout` and `WeightDecay` (not run), is kept as `Search1`.

`config_job.toml` has the run settings of the simmer jobs: 10 runs of 151 epochs, testing every 10 epochs.

## Results

1 run, 30 epochs of 512 trials, default params. Training is perfect by epoch 5.

| k | P100 | P83 | P75 | P67 | P50 | P33 | P25 | P17 | Blend |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1.0 | | 1.0 | | 0.49 | | 0 | | 0.005 |
| 6 | 1.0 | 1.0 | | 0.996 | 0.49 | 0.004 | | 0 | 0.005 |

- Like the feedforward network, it follows a strict majority.
- Unlike the feedforward network, the settling resolves almost all of the ties cleanly: very few blends, and a `SecondRatio` of about 0.02 (the losing group is nearly silent).
- The axon model is more graded at k = 6: P67 of about 0.75–0.87 and P33 of about 0.015 (with `Noise` 0.01, with and without output laterals). The amount of noise here has not been explored: more noise might also make the choices more graded.

### Noise sweep

k = 6, 1 run each, 30 epochs:

| Noise | P83 | P67 | P50 | P33 | Blend | SecondRatio |
|---|---|---|---|---|---|---|
| 0 | 1.0 | 0.95 | 0.43 | 0.015 | 0.035 | 0.08 |
| 0.2 | 1.0 | 0.996 | 0.49 | 0.004 | 0.005 | 0.02 |
| 0.5 | 1.0 | 1.0 | 0.49 | 0 | 0.007 | 0.03 |
| 1 | 1.0 | 0.98 | 0.47 | 0.02 | 0.016 | 0.05 |
| 2 | 1.0 | 0.95 | 0.47 | 0.04 | 0.006 | 0.03 |
| 4 | 1.0 | 0.97 | 0.42 | 0.03 | 0.04 | 0.11 |

- The noise has little effect on the probabilities, even at 4, and training is perfect at every level. Because the noise is present in training, the network can learn net inputs large enough to override it, so it does not act like a softmax temperature on the choice.
- Even without noise, P67 is 0.95 and P33 0.015, like the axon model's P33. With no noise the network is deterministic, so these are specific items that always go to the minority group, not stochastic choices. The P stats do not distinguish between item-specific preferences and stochastic choice: that needs the consistency of the choice across the repetitions of each item.

### Choice consistency

`Consist` and `TieConsist` (see the conflict README) measure how consistent the choice for the same item is across its 20 repetitions: 1 = the same choice every time, about 0.59 on ties = a coin flip. k = 6, 1 run each, 30 epochs:

| Noise | P67 | P33 | Consist | TieConsist |
|---|---|---|---|---|
| 0 | 0.95 | 0.015 | 1.0 | 1.0 |
| 0.2 | 0.99 | 0.005 | 0.98 | 0.95 |
| 2 | 0.96 | 0.039 | 0.96 | 0.89 |

With noise, the choices do vary somewhat across the repetitions of the same item, but even with a lot of noise, the ties are mostly decided by the item (0.89), far from a coin flip. The conflict_ff network is 1.0 on both.

### Carryover

`Decay` < 1 carries the previous trial's activity over into the next one. k = 6, 1 run each, 30 epochs, compared to the axon model (`Noise` 0.01):

| | P67 | P33 | Intrusion | RepeatPrev | Consist | TieConsist |
|---|---|---|---|---|---|---|
| Decay 1 (no carryover), Noise 0 | 0.95 | 0.015 | 0.011 | 0.54 | 1.0 | 1.0 |
| Decay 0.5, Noise 0 | 0.93 | 0.063 | 0.001 | 0.86 | 0.92 | 0.82 |
| Decay 0 (full carryover), Noise 0 | 0.92 | 0.076 | 0.001 | 0.93 | 0.90 | 0.77 |
| Decay 0.5, Noise 0.5 | 0.94 | 0.047 | 0.001 | 0.92 | 0.87 | 0.67 |
| axon, lateral | 0.87 | 0.014 | 0.10 | 0.90 | 0.77 | 0.43 |
| axon, no lateral | 0.76 | 0.031 | 0.14 | 0.89 | 0.72 | 0.41 |

- With carryover, even without noise, ties are broken by the previous trial: the choice repeats the previous one 86–93% of the time, like the axon model's 0.90. So the choice for the same item varies across its repetitions, because of the different trials that came before it.
- Training is still perfect: the network learns to override the carryover when all the inputs agree, and it only decides close calls.
- Unlike the axon model, there are essentially no intrusions: the axon model produces the previous trial's group 10–14% of the time even when it is not one of the inputs. The axon model's choices on ties are also more variable (partly because the intrusions are a third outcome), and its P67 is lower.

`<Layer>_CarryEnd` (every layer, in the Train and Test logs, copied to the Train log as `Test_*`; off in the plots) measures how much of each layer's state carries over, as in the axon model: the cosine between the layer's final activity (`Act`) and its final activity on the previous trial of the same data-parallel stream, minus the mean cosine with the previous activity of the other streams. The other streams run independent random item sequences through the same weights, so the baseline has the same input overlap without the carryover: 0 = none. There is no `_CarryMid` (the per-cycle activity below could provide it). The stat is computed for any `Decay`, so `Decay` 1 is the no-carryover control, and `Input_CarryEnd` (clamped) should be ~0. It needs `NData` > 1.

### Settling dynamics

`Run.RecordCycles` (on by default, with `NCycles` > 1) reads back the activity of every layer on every settling cycle (egomlx `Network.RecordCycles`), for the same test stats as the axon model (see the conflict README):

- The within-trial competition stats (`LeadSwaps`, `CommitCyc`, `LoserFrac`, `AltCorr`, `Agree50`, `Agree100`, and the Minor/Major splits), from the summed Output `Act` of each input group on each cycle.
- The settling microscope (`Test Microscope` and `Test MicroscopeTrial` tabs): the cosine of each layer's activity on each cycle with its final activity, and of the Output with the chosen and the other group's patterns (`OutWin`, `OutLose`). `Test MicroscopeTrial` is the current trial of data-parallel stream 0, titled with the trial name and its choice (also in its `TrialName` column), and `Test Microscope` is the average over the latest test.

The settling cycles are mapped onto the 150-cycle axon minus phase, at 150 / `NCycles` cycles each (15 for 10 cycles), so `CommitCyc`, `Agree50`, `Agree100` and the microscope's `Cycle` axis are on the axon time scale. With only `NCycles` samples (vs 30 in the axon model), `LeadSwaps` can count fewer swaps; more cycles (see Tanh and longer settling) give a finer view. The Output is a sigmoid, so its group sums include the low activity of all the units, not just those above a threshold.

The rate-code prediction of the perforated-attractor idea: the lead is set early and never swaps (`LeadSwaps` about 0), `LoserFrac` goes to 0, and the microscope approaches the final state smoothly.

### Weight decay and dropout

k = 6, 1 run each, 30 epochs, `Noise` 0.2 unless noted. `WeightDecay` > 0 uses the AdamW optimizer, and `Dropout` applies to the hidden layer units in training only, on every cycle.

| | P83 | P67 | P50 | P33 | Blend | SecondRatio | RepeatPrev | Consist | TieConsist |
|---|---|---|---|---|---|---|---|---|---|
| none | 1.0 | 0.996 | 0.49 | 0.004 | 0.005 | 0.02 | 0.53 | 0.98 | 0.95 |
| WeightDecay 0.01 | 1.0 | 1.0 | 0.48 | 0 | 0.009 | 0.03 | 0.56 | 0.99 | 0.97 |
| WeightDecay 0.1 | 1.0 | 1.0 | 0.48 | 0 | 0.004 | 0.03 | 0.48 | 0.98 | 0.95 |
| WeightDecay 0.1, Noise 1 | 1.0 | 0.96 | 0.48 | 0.033 | 0.009 | 0.04 | 0.54 | 0.96 | 0.91 |
| Dropout 0.2 | 1.0 | 0.998 | 0.50 | 0.002 | 0.001 | 0.002 | 0.57 | 0.99 | 0.96 |
| Dropout 0.5 | 1.0 | 1.0 | 0.50 | 0 | 0 | 0.001 | 0.53 | 0.98 | 0.95 |
| Dropout 0.2, Decay 0.5 | 1.0 | 0.98 | 0.48 | 0.015 | 0.013 | 0.05 | 0.89 | 0.90 | 0.73 |

- Weight decay does not let the noise act as a temperature: even with WeightDecay 0.1 and Noise 1, the choice is close to a strict majority, and mostly determined by the item.
- Dropout makes the choices even cleaner (`SecondRatio` 0.001–0.002), without making them more variable: dropout is only in training, so testing is as deterministic as before (apart from the noise).
- With carryover (`Decay` 0.5), dropout gives the same history-dependent tie breaking as carryover alone.
- WeightDecay 0.1 with `Decay` 0.5 was unstable: the hidden ReLU activity grew over training (mean about 3), and blew up at epoch 21 (NaN loss, then no activity). ReLU activity is unbounded, and carryover can feed it back across trials.

### Tanh and longer settling

The most graded settings (full carryover: `Decay` 0, `Noise` 0), with ReLU or tanh hidden layers (`HiddenFn`), and 10 or 20 cycles of settling. k = 6, 1 run each, 30 epochs:

| | P67 | P33 | SecondRatio | RepeatPrev | Consist | TieConsist |
|---|---|---|---|---|---|---|
| ReLU, 10 cycles | 0.92 | 0.076 | 0.029 | 0.93 | 0.90 | 0.77 |
| Tanh, 10 cycles | 0.93 | 0.068 | 0.007 | 0.92 | 0.89 | 0.72 |
| ReLU, 20 cycles | 0.93 | 0.060 | 0.019 | 0.73 | 0.94 | 0.88 |
| Tanh, 20 cycles | 0.85 | 0.116 | 0.027 | 0.68 | 0.96 | 0.93 |
| axon (output laterals) | 0.87 | 0.014 | 0.12 | 0.90 | 0.77 | 0.43 |

- Tanh is about the same as ReLU with 10 cycles, with slightly cleaner and slightly more variable choices on ties. Its activity is bounded, unlike ReLU, which blew up with weight decay and carryover.
- With 20 cycles, the carryover has less effect (`RepeatPrev` 0.68–0.73), and the choices are more determined by the item (`TieConsist` 0.88–0.93): the previous trial only enters through the first cycle, so with longer settling, the current inputs dominate more.
- Tanh with 20 cycles has a P67 (0.85) like the axon model's, but a much higher P33 (0.116 vs 0.014): the curve is flatter, not steeper, and the minority choices are specific items (`Consist` 0.96). The axon model combines a steep curve (P33 near 0) with choices that vary across repetitions, which none of these settings do.

The runs above have no lateral connections. With `OutputLateral` (tanh, `Decay` 0, `Noise` 0):

| | P67 | P33 | SecondRatio | RepeatPrev | Consist | TieConsist |
|---|---|---|---|---|---|---|
| 10 cycles | 0.93 | 0.068 | 0.007 | 0.92 | 0.89 | 0.72 |
| 10 cycles, output laterals | 0.94 | 0.064 | 0.037 | 0.69 | 0.96 | 0.91 |
| 20 cycles | 0.85 | 0.116 | 0.027 | 0.68 | 0.96 | 0.93 |
| 20 cycles, output laterals | 0.87 | 0.118 | 0.004 | 0.95 | 0.88 | 0.77 |

With 5 runs per condition (mean ± standard deviation across runs):

| | P67 | P33 | RepeatPrev | Consist | TieConsist |
|---|---|---|---|---|---|
| 10 cycles | 0.96 ± 0.02 | 0.037 ± 0.02 | 0.91 ± 0.03 | 0.92 | 0.80 ± 0.04 |
| 10 cycles, output laterals | 0.96 ± 0.02 | 0.035 ± 0.02 | 0.79 ± 0.07 | 0.95 | 0.88 ± 0.03 |
| 20 cycles | 0.96 ± 0.05 | 0.036 ± 0.04 | 0.61 ± 0.04 | 0.99 | 0.97 ± 0.02 |
| 20 cycles, output laterals | 0.84 ± 0.05 | 0.10 ± 0.02 | 0.93 ± 0.04 | 0.86 | 0.73 ± 0.03 |

- The differences between the conditions are consistent across runs.
- Without laterals, 20 cycles of settling washes out the carryover (`RepeatPrev` 0.61 in every run), and the choices are almost completely determined by the item (0.97).
- With output laterals, the carryover persists through 20 cycles (`RepeatPrev` 0.93): the laterals keep the previous trial's winner active in the output as an attractor, instead of letting it fade. This is also the condition with the flattest choice curve (P67 0.84, P33 0.10).
- The single-run P67 of 0.85 for tanh with 20 cycles in the previous table was an outlier: the mean over 5 runs is 0.96.
- None of the conditions has the axon model's combination of a steep curve (P33 about 0.01) with highly variable choices (`TieConsist` about 0.43).

### Network size

The most graded settings above (tanh, 20 cycles, output laterals, `Decay` 0, `Noise` 0), with `HiddenSize` 15 (15×15 = 225 units per hidden layer) and 21 (441 units, about twice as many). k = 6, 5 runs each, 30 epochs (mean ± standard deviation across runs):

| HiddenSize | P67 | P50 | P33 | P17 | RepeatPrev | Consist | TieConsist | Blend |
|---|---|---|---|---|---|---|---|---|
| 15 | 0.84 ± 0.06 | 0.46 | 0.10 ± 0.03 | 0.019 | 0.93 ± 0.05 | 0.86 | 0.73 ± 0.03 | 0.001 |
| 21 | 0.92 ± 0.10 | 0.44 | 0.06 ± 0.07 | 0.005 | 0.76 ± 0.12 | 0.93 | 0.85 ± 0.10 | 0.014 |

- The larger network is closer to a strict majority rule (higher P67, lower P33). It is less affected by the carryover (`RepeatPrev` 0.76), and its choices on ties are more determined by the item (`TieConsist` 0.85).
- It varies much more across runs:
  - Three of the five runs are nearly a strict majority rule (P67 about 0.98, P33 about 0.01).
  - One run is like the smaller network (P67 0.77, P33 0.17, `RepeatPrev` 0.88).
  - In the most extreme run (`RepeatPrev` 0.58, `TieConsist` 0.99), P50 is only 0.32, with 4% blends: on ties, it often does not cleanly produce either input group.
- So the graded, history-dependent choices of the smaller network partly reflect its limited capacity. With more units, the network separates the evidence more sharply, which overrides the carryover held in the output attractor. Scale trades the graded curve for more deterministic choices, which is the opposite of the axon model's combination of a steep curve with variable choices.
