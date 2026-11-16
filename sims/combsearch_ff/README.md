# combsearch_ff

This is a feedforward backprop baseline for the [combsearch](../combsearch/README.md) task, built with egomlx (gomlx). It uses the same `csenv` environment, with the same seeds, item space and stats. See the combsearch README for the task and the stats.

## Network

egomlx only has full (dense) pathways, so each slot is its own column of layers. This is equivalent to the `PoolOneToOne` pathways between slot pools in the axon network:

```
Output0   Output1   Output2      (Target, sigmoid)
   ↑    ↖    ↑    ↖    ↑    ↖
Encode0   Encode1   Encode2  →  Hidden   (ReLU)
   ↑         ↑         ↑
Input0    Input1    Input2
```

- Each `Input<s>` projects only to its own `Encode<s>`.
- All the Encode layers project to a shared `Hidden` layer.
- Each `Output<s>` receives from its own `Encode<s>` and from `Hidden`.
- The axon network's Encode ↔ Hidden loop becomes Encode → Hidden → Output here.
- Slot s of the env `Input` and `Output` states is applied to the `Input<s>` and `Output<s>` layers.
- The stats concatenate the `Output<s>` activities in slot order.

Other settings:

- Layer sizes: `EncodeSize` (8×8 per slot) and `HiddenSize` (12×12).
- Loss and optimizer: BCE loss summed over the Output layers, Adam with LR 0.001.
- Weights: Glorot init, with biases.

## Results

Settings: 3 slots, n = 24 (the earlier defaults), G = 8, g = 3, 1 run, 100 epochs of 512 trials, about 30 seconds.

**m = 1** (the target group is determined by the inputs):

| epoch | train Valid | train Solved | test Valid | test Solved |
|---|---|---|---|---|
| 10 | 0.53 | 0.17 | 0.50 | 0.15 |
| 50 | 0.94 | 0.86 | 0.87 | 0.66 |
| 99 | 1.00 | 0.99 | 0.95 | 0.88 |

- The ff net learns the task and generalizes well to novel input combinations. Test `Solved` is still rising at epoch 100.
- This is expected. With a single shared-group constraint, Hidden can find the group by voting: it sums the group evidence across slots and broadcasts the winner back. That takes one feedforward pass.

**m = 2** (the target group is chosen at random from the two valid groups on each trial):

| epoch | train Valid | train Solved | test Valid | test Solved |
|---|---|---|---|---|
| 20 | 0.99 | 0.52 | 0.75 | 0.16 |
| 99 | 1.00 | 0.65 | 0.76 | 0.18 |

- Every slot produces an output from one of the two valid groups (train `Valid` = 1).
- On training items, the slots agree on the same group only 65% of the time. On test items, `Solved` (0.18) is below the 0.245 that independent per-slot choices among the valid groups would give.
- A deterministic network can't commit to one group across slots when the target is split between two. This is the condition that should separate settling from feedforward processing.

There were only 99 training items for m = 2 with n = 24, which is why the default is now n = 32 (366 training items).

### 4 slots

Same settings with `NSlots = 4`. The m = 2 run uses n = 32, because n = 24 has only 72 training items. The n = 32 rows are the current defaults.

| n | m | train Valid | train Solved | test Valid | test Solved |
|---|---|---|---|---|---|
| 24 | 1 | 1.00 | 0.99 | 1.00 | 0.99 |
| 32 | 1 | 0.98 | 0.92 | 0.98 | 0.94 |
| 32 | 2 | 1.00 | 0.32 | 0.98 | 0.19 |

- **m = 1:** a 4th slot doesn't make it harder. The fourth input constrains the group more, and there are many more training items (11952 for n = 24). Test `Solved` matches train, so the generalization is complete. n = 32 is slower (0.94 at epoch 100, still rising).
- **m = 2:** the chance of all 4 slots agreeing by independent choice falls to 2·(1/2)⁴ = 0.125. The ff net is only a little above that on test items (0.19), and flat from epoch 30 on. Even on training items it reaches only 0.32.
- So adding slots makes the coordination problem much harder for feedforward without making the determined case harder. The gap between m = 1 and m = 2 widens from 0.88 vs 0.18 to 0.94 vs 0.19, with chance dropping from 0.25 to 0.125.
