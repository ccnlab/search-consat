# Probabilistic combinations: PComb

This project is used for the paper on Search and the unreasonable effectiveness of stochastic gradient descent in both weight and activation space. The goal of this project is to test the idea that bidirectionally-connected spiking networks are particularly effective at this kind of gradient-descent in activation space.

To test this hypothesis, we need comparison networks, including standard feedforward and recurrent neural networks, including transformers. The comparison against feedforward nets is particularly challenging because they have to have separate output layers, and thus they can just use backprop to integrate all the separate information that might otherwise need to be integrated over the settling process in a bidirectionally-connected network (see the topdown sim for an example).

Thus, while there is theoretically a major advantage of bidirectional connectivity such that each sensory domain remains more "modular" and self-contained, and yet responsive to what is happening in other areas, this is very difficult to demonstrate in any given simple benchmark test.

Here are the principles from the search paper, which motivate the design of this test:

* The ultimate function of the neocortex is to integrate massive amounts of sensory input data and reduce it down to a decision of what to do in the current situation, taking into account all of the internal state and prior learning. This is fundamentally a massive **compression** process, reducing a very high-dimensional input space into a **low dimensional internal latent space**. This represents an important additional constraint on the inference search process, which also favors the properties of the bidirectional neural network search dynamics, as compared to the feedforward loop unrolling. Specifically, the bidirectional network tends to settle into discrete attractor states, whereas the loop unrolling process through feedforward layers does not have this property. This is also likely a factor in the tendency of LLM models to confabulate.

* Compared to rate-coded networks, spiking dynamics allow graded, **probabilistic** information to be represented throughout the network (not just in restricted layers, such as the variational auto-encoder), and overall are much better at avoiding getting stuck in bad local minima during the gradient-descent process.

* Pervasive bidirectional connectivity allows detailed constraints in deep, modality-specific processing areas to effectively constrain more abstract interpretations generated in higher-level integrative areas, which contrasts with the loop-unrolling behavior of deep feedforward networks, which only support unidirectional flow through more abstract, higher-level areas. This may contribute to the tendency of LLM models to confabulate, for example. (Note: this is hard to test in a small, focused sim!)

Thus, this analysis suggests that the specific properties of the mammalian neocortex could be particularly well-suited for this inferential online search process, for finding sensible low-dimensional interpretations of the environment, that then provide a suitable basis for acting in the world. Given that the motor action space is necessarily low-dimensional, the ability to rapidly compress the high-dimensional environmental state into appropriate low-dimensional interpretations likely has significant survival benefits ([[@SridharLiGorbonosEtAl21]]).

## Task structure

The task represents an abstract decision process, with input **cues** that each suggest the **probability** and **value** of a relevant outcome under consideration, with each cue having a separate value and probability input slot. Values range from -1 to 1 (aversive vs. appetitive). As an initial simplification, each slot represents its quantity using a Gaussian bump with a fixed $\sigma$ standard deviation, and the target output will also be similar such Gaussian bumps. Therefore, we are not capturing differences in variance of values or probability distributions at the individual cue level -- that instead will be captured across the distribution of available cues.

The output is a net estimate of the expected value of the outcome, weighted by the probability associated with each cue, critically with the values activated **stochastically** according to their probabilities. The probability distribution determines a sampling distribution across the output space.

The different cues contribute to the outcome as if they could be the only deciding factor, normalized by the probabilities of all the cues. Thus, for example, if you had two active cues, each with $p_i = 1$, with $v_0 = -1$ and $v_1 = 1$, then the output should activate -1 50% of the time and +1 the other 50% of the time.

The overall expected value is given by:
$$
E[v] = \frac{\sum_i p_i v_i}{\sum_i p_i}
$$

(the all-zero $p_i$ case is excluded in our environment, but would result in a 0 value output).

The input generation process specifies a number of active cues within some range (1-n) and the probability and value quantities are generated according to uniform distributions with specified ranges, where p specifically excludes 0 (typically starting at .2, pending exploration -- don't want to get bogged down with a lot of low-probability events). The target output value on any given trial is generated according to the normalized probability distribution across cues:

$$
P(v_i) = \frac{p_i}{\sum_j p_j}
$$

The statistics tracking performance measure how close the decoded output value is to one of the cue input values (abs diff) and the accumulated probability distribution for that value relative to its normalized target. Specifically, we can accumulate binned probabilities in 0.1 bins (0.9-1, 0.8-0.9, etc), with the normalized frequency that a value of that probability is generated by the model, and then report this distribution and the average abs deviation from the middle value in each bin. We expect the model to tend to underweight low-probabilities, and over-weight higher ones, so it will be interesting to see what the distribution looks like.

Even with additional noise injected into a feedforward network, it should not in general be able to produce reasonable stochastic sampling behavior. A key question is whether introducing more hidden layers, to simulate unrolling over time, will facilitate this ability. Likewise, do the attentional mechanisms in a transformer help in this regard? An RNN should fare better, but the prediction is that it will not do as good of sampling as the spiking network.

## First pass

The first pass uses a small, coarse version of the task, implemented in [pcombenv](pcombenv). Defaults are in `pcombenv.Config.Defaults`, and any field can be set from the `[Env.Env]` section of a local `config.toml`.

### Discretized item space

Probabilities and values are discretized so the full space of cue combinations is finite, and can be split cleanly into training and testing items:

* Probabilities: 0.2 to 1 in steps of 0.2 (`ProbStep`), excluding 0: 5 values.
* Values: -1 to 1 in steps of 0.25 (`ValueStep`): 9 values.
* Cue types: 5 x 9 = 45 (probability, value) combinations.
* Number of active cues: 2 to 4 (`NActive`), out of 4 cue slots (`NCues`; 6 slots is a bit harder but also learns eventually). The minimum of 2 avoids the special case of a single cue, which is always chosen regardless of its probability.

An item is an unordered set of cues (with repetition allowed: several cues can have the same type), so the number of items with $n$ cues is the number of multisets of size $n$ from 45 types, $\binom{45+n-1}{n}$. The cues are assigned to random slots on each trial, so slot positions are not part of the item space. The item counts for different ranges of active cues:

| Active cues | Items |
|---|---|
| 2 | 1,035 |
| 3 | 16,215 |
| 4 | 194,580 |
| **2-4 (default)** | **211,830** |
| 2-3 | 17,250 |
| 2-5 | 2,118,714 |
| 2-6 | 18,009,414 |
| 2-8 | 886,322,664 |

Items include agreement cases, where several cues have the same value (e.g., 0.2 / -0.75 and 0.6 / -0.75). In training, these provide a shortcut: the number of cues with a value goes along with its summed probability, so a network can do well without using the probabilities at all. Choosing the most repeated value, ignoring the probabilities entirely, gives a `ChoseP` of 0.42 over the full item space, which is higher than calibrated sampling (0.40). Agreement items in training would also confound the test for a superlinear agreement effect.

So by default (`DistinctTrain`), training only uses items where all the cue values are distinct, and the agreement items are only used for testing:

| Active cues | Distinct values | Repeated values (agreement) |
|---|---|---|
| 2 | 900 | 135 |
| 3 | 10,500 | 5,715 |
| 4 | 78,750 | 115,830 |

Half of the distinct-value items for each number of cues (`TestPct` = 0.5) are held out for testing, using a fixed random seed. Test trials use agreement items half of the time (`AgreePct`), and held-out distinct items otherwise. Each data-parallel test env gets a non-overlapping subset of the test items. The number of active cues on each trial is sampled in proportion to the number of items with that many cues, so 4-cue items make up about 87% of training trials. Setting `UniformN` instead samples the number of cues uniformly, and then an item with that many cues.

### Input and output layers

Each quantity is encoded as a Gaussian bump over 22 units (`PopCodeUnits`), with $\sigma$ = 0.1 of the range (`Sigma`), and one extra step of range on either end so the end values are not truncated. For values (range -1.25 to 1.25), $\sigma$ is 0.25, one value step. For probabilities (range 0 to 1.2), it is 0.12, a bit over half a probability step. Either way, about 6 of 22 units are above 0.1 activity, under the roughly 30% max activity that the axon inhibition handles well.

* `Value`: [4, 1, 1, 22]: one pool per cue slot.
* `Prob`: [4, 1, 1, 22]: one pool per cue slot, aligned with `Value`.
* `Output` target: [1, 22]: a bump at the value of the cue sampled with probability $p_i / \sum_j p_j$.
* `Salience` target: [1, 22], using the same value units as `Output`, so positive and negative salience are separate:
    + $|v| \geq 0.75$ (`SalientFull`): a copy of the `Output` bump (6 units).
    + $|v| = 0.5$ (`SalientHalf`): the `Output` bump with the half toward 0 cut off (3 units).
    + $|v| \leq 0.25$: no salience activity.

Inactive cue slots are all zeros in both input layers.

### Network

`Value` and `Prob` each project to their own 15x15 encoding layer (`ValueEnc`, `ProbEnc`), which are both bidirectionally connected with a 15x15 `Hidden` layer. `Hidden` is bidirectionally connected with the `Output` and `Salience` target layers. All connections are full, and back pathways have a relative weight scale of 0.2, otherwise the network hallucinates.

The main params: the learning rate is 0.01. The encoding and `Hidden` layers have low nominal activity (0.03 and 0.02) with adaptive inhibition. The `Output` and `Salience` targets have layer inhibition with Gi 1.05.

In a 65-epoch run (`Trials` 512, `NData` 16), before the `DistinctTrain` change and the probability-sensitive stats below, training mostly levels off by epoch 25-30, with about 15% misses on both training and held-out test items, and test `CalibErr` about 0.08 (see [results](results)). `ChoseP` levels off around 0.36, against 0.33 for uniform choice over the distinct values and 0.40 for calibrated sampling. However, as noted above, the cue count shortcut can do this well without the probabilities, and indeed removing the probability pathway did not hurt performance on these stats.

### Statistics

The stats are computed by `PCombEnv.TrialStats` in [pcombenv/stats.go](pcombenv/stats.go) from the `Output` and `Salience` activity (minus phase in axon), so they are the same for [pcomb_ff](../pcomb_ff). At the trial level, for each data-parallel item:

* `Value`: the decoded `Output` value.
* `CueErr`: distance between the decoded `Output` value and the nearest cue value.
* `Miss`: 1 if the decoded value is not within 0.2 (`MatchThr`) of any cue value, e.g., a blend between cue values.
* `SalErr`: mean absolute difference between the `Salience` activity and the salience pattern for the value chosen by the network (excluding misses).

For items with distinct cue values (all training items, and the distinct test items):

* `ChoseP`: the normalized probability of the cue value chosen by the network (excluding misses).
* `PGain`: `ChoseP` minus 1 / number of values: the gain over choosing a value regardless of the probabilities. Because the probabilities are independent of the values, this is 0 on average for any choice that ignores the probabilities, which makes it the main test of whether the probabilities are used.
* `OutPErr`: the total variation distance between the target probabilities and the `Output` activity at the unit nearest each cue value, normalized over the values. This measures whether the output represents the probabilities regardless of where its peak is, which is mainly relevant for a blended output, as in the feedforward model. Adjacent value bumps overlap, so even a perfect blend is not 0, and a single bump scores worse than a blend.
* Calibration, using 10 bins of normalized probability (`NBins`). On each trial, each cue value counts once, in the bin for its normalized probability, and counts as chosen if it is the value the network produced. At the epoch level:
    + `Calib`: the frequency of choosing values in each bin.
    + `CalibP`: the mean normalized probability of values in each bin (the target frequency for a calibrated model, not the bin midpoint, because values are not evenly distributed within bins).
    + `CalibN`: the mean number of values per trial in each bin, which varies widely across bins.
    + `CalibErr`: the mean absolute difference between `Calib` and `CalibP`, weighted by `CalibN`.

For agreement items (test only, by default):

* `AgreeP`: the summed normalized probability of the repeated values (excluding misses), which is how often a calibrated model chooses one of them.
* `AgreeBoost`: 1 if the network chose a repeated value, minus `AgreeP` (excluding misses). Positive values mean agreeing cues are favored beyond their summed probability (superlinear). However, any winner-take-all sharpening also produces a positive value, so this needs to be compared with the general sharpening on distinct items (`PGain` above calibrated).

Reference values for these stats on the test items, computed by `TestBaselines` in [pcombenv_test.go](pcombenv/pcombenv_test.go) by applying `TrialStats` to synthetic outputs:

| Output strategy | Uses probs | `ChoseP` | `PGain` | `OutPErr` | `AgreeBoost` |
|---|---|---|---|---|---|
| Uniform choice over values | no | 0.26 | 0.00 | 0.60 | -0.16 |
| Most repeated value | no | 0.26 | 0.00 | 0.60 | 0.43 |
| Uniform blend of value bumps | no | 0.26 | 0.00 | 0.19 | -0.16 |
| Calibrated sampling | yes | 0.31 | 0.05 | 0.55 | 0.00 |
| Probability-weighted blend | yes | 0.39 | 0.12 | 0.08 | 0.25 |
| Most probable value | yes | 0.39 | 0.13 | 0.48 | 0.28 |

`Calib`, `CalibP` and `CalibN` are vectors over bins, so they are kept out of the plotted `Stats` levels. They are stored for the most recent epoch in `Current/<mode>/Calib/Epoch`, with the per-trial counts accumulated in `Current/<mode>/Calib/Trial`. Only the scalar `CalibErr` is logged and plotted.

The test epoch `CueErr`, `Miss`, `PGain`, `OutPErr`, `CalibErr` and `AgreeBoost` are also recorded in the train epoch log.

## old notes

TODO: go through these.

The Hopfield & Tank (1985) (HT85) model relies on single units with stereotyped wiring.

An actual Axon model requires distributed representations and collaborative interactions among sparsely-active populations of neurons. It is nothing like the HT85 paradigm. In particular, the normal inhibitory dynamics tend to restrict the impact of lateral connections gaining strength in a specific area and getting over threshold. In the current version, noisy weights are determining the entire dynamic of what gets active. This needs to be trained weights somehow. But how?

One alternative is training a network with an input of city-dots and output of some kind of ordering representation, but then it turns the problem into one of pattern recognition instead of CS. However, given all the interdependencies among the elements, and the high dimensional combinatorial space, perhaps this is still a good test. It seems like the right way to go next. Can pre-generate larger space if necessary.

Key points:

* standard visual representation of TSP does not encode the relevant data. distance needs to be explicitly represented. This was done by fiat in the flat TSP model, but the visual dots one simply does not have it.

* metric information like distance is not really of the essence for constraint satisfaction -- requires either "more is more" activation, pop codes, or well-tuned weights, etc. Usual issues. 

* simplest case would be binary input features, where the key property is that the overall *configuration* makes a difference -- it is about the *relationship* among the different input features. For example, figure-ground illusions, where suspicious configuration of elements "adds up" to a perceived figure.

* key challenge: need some automatic, algorithmic way of generating a sufficient number of these problems! what is the actual algorithmic recipe for how this works? some way of translating TSP into this? some kind of maze with different gates? but then it becomes a spatial problem again! need to have everything be simple and explicit.

What about the 3SAT CNF satisfiability problem?? Binary, seems good! Except: binary is terrible for the sparse activation constraint. And indeed, this seems fatal in the implemented network. Note: implemented 3SAT using 3 variables that were repeated across clauses, so minimal number of vars, and thus the full CNF condition could be expressed only in terms of the negation operators, which avoids binding issues etc: the variables are constant and implicit. Interestingly, it requires 8 clauses before unsatisfiable cases emerge, and initially they are a tiny fraction. The summary on SO (https://cstheory.stackexchange.com/questions/2168/how-many-instances-of-3-sat-are-satisfiable) suggests that the ratio of clauses / variables defines a critical point (at a value of 4.2) where the space transitions from predominantly satisfiable to predominantly unsatisfiable. This accords well: 8/3 = 2.6667 is first emergence of any unsat, and by 10/3 (3.333) it is growing as a proportion steadily. The 4.2 ratio suggests that between 12 and 13 (12.6) is the transition point, but the combinatorial explosion at that point is already pushing the limits.

Next step is NAry CNF: OR = MAX, AND = MIN. Really the key thing about all these problems is that it needs to depend on the configuration of things, and the MIN and MAX clearly have that character. So, start with just computing MIN and MAX operators and see how that goes.

CNF is too easy: the MIN and MAX operators are easily computed with weights that generalize very well. This is too much pattern recognition and not enough constraint satisfaction.

Per https://en.wikipedia.org/wiki/Constraint_satisfaction_problem -- generic version of CSP is just variables with some kind of arbitrary constraint matricies. So, just need to come up with a bunch of generic variables and then some random constraints, and then have the thing learn this. Don't need any actual semantics. Just need to have relationships and sufficient numbers of constraints to balance things out -- randomly generate constraints until finding one that produces balanced results. Relationships can just be > = <.

Key thing: need the output to be somehow more than binary! Ok, so could define different *classes* according to these constraints, and the problem is to classify. So you have N different sets of constraints, and you need to find the one that fits the best -- that seems good!!

## TODO

* Hierarchical hidden -- 2x2 tiled with wrap -- needs more local
* minus-plus in same layer with soft clamp.

