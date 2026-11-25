# Perforated attractor: literature and proposed stats

Notes from 2026-11-20, after jobs 16–25.

## Hypothesis

In a rate-code attractor, the winning state fills the competition continuously: the losing ("minority") alternatives are suppressed smoothly and monotonically, so the outcome is set by the initial conditions (inputs plus carryover) and comes out near winner-take-all, and nearly the same on every trial. In the spiking model, discrete spikes leave temporal gaps in the winner's drive and in the inhibition. Minority signals can percolate through those gaps (the "perforations"), so the losing alternatives keep influencing the settling and sometimes win, giving graded, probability-matching-like choices. Synaptic failure perforates the attractor further, and introduces a different kind of noise from additive activation noise. NMDA and GABAB, which stabilize activity over time, should work against the perforation.

Evidence so far:
- **Synaptic failure .4 (job 25):** In67 .67 with half the test intrusions of the failure-.1 runs at the same In67. 67 vs 33 is matched exactly, and 83/17 are sharpened.
- **rnn additive noise:** even large noise (SD 4) leaves P67 ≥ .95 (conflict_rnn README, Noise sweep). Additive noise doesn't make a rate attractor graded.
- **Output decay 1 (job 24):** barely changes Output carryover (`Output_CarryEnd` .41 vs .44). The previous choice is rebuilt from the Hidden layer's carried-over state.

## Literature

I found no paper that states the "perforated attractor" idea: that discrete spiking leaves gaps that let minority alternatives survive, and so produces graded choice where a rate attractor gives winner-take-all. The term doesn't appear in the attractor literature. The closest lines of work:

1. **Binary units stabilize mixture states; analog units don't.**
   - In Hopfield networks, mixture states (blends of several stored patterns) are metastable at nonzero temperature, i.e. with noise (Amit, Gutfreund & Sompolinsky 1985).
   - Roudi & Treves argue that symmetric mixture states are almost never stable in *analog* networks, and call them an artifact of binary models.
   - Read the other way, that is close to the claim here: discrete (binary or spiking) units plus noise support blends of attractors that continuous units collapse. As far as I found, nobody has connected this to spiking versus rate models, or to choice probabilities.
   - Sources: [AGS 1985](https://gwern.net/doc/ai/nn/1985-amit.pdf), [Roudi & Treves](https://pdf.arxiv.org/pdf/cond-mat/0302029)
2. **Neural sampling.**
   - Stochastic spiking networks sample from a distribution, which gives choices in proportion to probability (Buesing, Bill, Nessler & Maass 2011).
   - Spiking soft winner-take-all circuits sample posteriors over hidden causes (Nessler et al. 2013).
   - Even deterministic spiking networks can generate the variability that sampling needs (Jordan et al., [arXiv 1710.04931](https://arxiv.org/abs/1710.04931); Dold et al. 2019, [arXiv 1809.08045](https://arxiv.org/abs/1809.08045)). This is close to "the discreteness of spikes is itself the perforation."
   - Mixing problem: Gibbs-like sampling gets stuck in deep attractors. Short-term synaptic plasticity in spiking networks improves mixing (Leng et al. 2018, Sci Rep 8:10651, [arXiv 1709.08166](https://arxiv.org/abs/1709.08166)). That parallels the carryover/intrusion problem here: a deep attractor that one level of decay can't break.
   - Sources: [Buesing et al. 2011](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1002211), [Nessler et al. 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3636028/)
3. **Synaptic unreliability as the noise source.**
   - Probabilistic synapses alone produce Poisson-like spiking over a wide range of rates. Other jitter sources (random delays, spike-generation jitter) can't be amplified enough by the recurrent network to do the same. The paper suggests probabilistic synapses let networks sample neuronal states (Moreno-Bote 2014). This supports the idea that synaptic failure is a different kind of noise from additive noise.
   - Synaptic sampling machines: random synaptic masks (blank-out, DropConnect-like) drive Monte Carlo sampling, with both spiking and discrete-time units (Neftci et al. 2016).
   - Sources: [Moreno-Bote 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4102400), [Neftci et al. 2016](https://www.frontiersin.org/articles/10.3389/fnins.2016.00241/text)
4. **Spiking decision attractors.**
   - Wang 2002 is the classic biophysical spiking model of binary choice. NMDA-mediated slow recurrent excitation integrates the inputs and forms categorical choices, with graded choice probabilities from noise.
   - Wong & Wang's reduction to a rate model needs explicit noise added back to reproduce the choice probabilities.
   - Noise-driven switching between attractors is a well-known alternative to adaptation-driven switching, in both mean-field and spiking versions (Moreno-Bote, Rinzel & Rubin 2007, bistable perception).
   - Relevant to the planned NMDA/GABAB manipulations: slow NMDA reverberation is what makes these models integrate and commit.
   - Sources: [Wang 2002](https://cenl.ucsd.edu/Jclub/Wang02.pdf), [Moreno-Bote et al. 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2702529)
5. **Neural evidence that minority representations persist: multiplexing.**
   - With two simultaneous stimuli, single neurons alternate between the responses to each, across or within trials. Trial averaging hides this. The fluctuations can predict which stimulus is reported (Caruso et al. 2018, Nat Commun; inferior colliculus and IT face patches).
   - V1 does the same, but only when the stimuli form separate objects. Neuron pairs are positively correlated when they prefer the same object and negatively when they prefer different ones (Jun et al. 2022, eLife).
   - This is the biological signature of the minority opinion staying represented over time, and the alternation stat proposed below (stat 1) tests for it directly.
   - Sources: [Caruso et al. 2018](https://scholars.duke.edu/publication/1330993), [Jun et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9708082)
6. **Carryover and repetition biases.**
   - Decaying activity from the previous decision in a competitive attractor network causes choice hysteresis, mostly on difficult trials and at short intervals. Changing the decay in the model, or with tDCS over dlPFC, shifts it in the predicted direction (Bonaiuto et al. 2016, eLife 5:e20047).
   - Persistent and activity-silent (short-term plasticity) traces both contribute to serial biases in PFC (Barbosa et al. 2020).
   - These parallel the intrusion and MinorPrev results, and the job 24 finding that the Output carryover is rebuilt from the Hidden layer.
   - Sources: [Bonaiuto et al. 2016](https://elifesciences.org/articles/20047), [Barbosa et al. 2020](https://link.springer.com/article/10.1038/s41593-020-0644-4)
7. **Mapping rate to spiking.** Kim, Li & Sejnowski 2019 map trained rate RNNs one-to-one onto spiking networks. They aim for equivalent task performance, so they don't look for behavioral differences like this one. ([PNAS 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6842655))

**Possible novelty:** I found prior work showing that spiking variability or synaptic noise can make a network sample. I found none showing the specific contrast here: the same architecture and task, a rate attractor that stays winner-take-all under additive noise while the spiking one is graded, explained by temporal gaps in the attractor. Item 1 (binary versus analog mixture states) and Moreno-Bote 2014 (synaptic noise is amplified by recurrence; other jitter isn't) are the closest theoretical anchors.

## Proposed stats

They are ordered by how directly they test the perforation idea against the rnn.

### 1. Within-trial competition dynamics (axon; rnn needs per-cycle readout)

*rnn: implemented (2026-11-20) from the new egomlx `Network.RecordCycles` per-cycle readout (`Layer.CycleActs`), with the settling cycles mapped onto the 150-cycle axon minus phase (15 cycles each for `NCycles` 10).* *Implemented in axon (2026-11-20), with stat 2's `EarlyAgree` as `Agree50` and `Agree100`: the summed Output `CaM` of each input group is sampled every `Run.CompeteInterval` (5) minus phase cycles (`cfenv.CFEnv.Compete`, `Sim.StatCompete`). `CaM` (tau 5) stands in for the 10-cycle spike bins, since axon has no per-neuron spike counter. `AltCorr` uses the cycle-to-cycle changes after the first 25 cycles, so the common rise at onset doesn't count as co-activation. The Minor/Major splits are for `LeadSwaps` and `CommitCyc`; the per-item splits can be done offline from the Test Trial log.*

On each two-group test trial, sum the spikes of each Output group's units in 10-cycle bins over the minus phase. From those bins:
- **`LeadSwaps`:** how many times the leading group changes. A perforated attractor should swap the lead repeatedly on close calls. A rate attractor should swap at most once.
- **`CommitCyc`:** the last cycle at which the lead changes, i.e. when the network commits.
- **`LoserFrac`:** the losing group's share of the spikes in the second half of the minus phase. This is the residual minority signal.
- **`AltCorr`:** the correlation of the two groups' binned spike counts. Negative means the two alternate in time, the Caruso/Jun multiplexing signature.

Split each by item probability (In67, In50 trials) and by choice (majority versus minority choice). The prediction is that minority choices come from late swaps, not from an early lead set by the carryover.

The rnn equivalent needs egomlx to read back the activity of every cycle, not just the last one. That is an optional output of all the cycles' Act from `buildModelFn`, which is a modest change. Then the same stats apply. The prediction is that the rnn commits early, with about 0 swaps and `LoserFrac` going to 0.

### 2. Early versus late choice prediction

Does the state early in the trial predict the final choice?
- **`EarlyAgree`:** whether the leading group at cycle 50, and at cycle 100, is the final choice. Report it as a function of cycle.
- **`CarryPred`:** among trials where the previous choice is one of the item's groups, how often the final choice is the previous one. MinorPrev already covers this partly.

If the choice is mostly set by the carryover, early agreement is high. If the perforations let later fluctuations decide, it is lower. This separates the two sources of gradedness seen in job 25 versus jobs 20–24.

### 3. Probability-matching shape (all three sims, cheap)

*Implemented in all three sims (2026-11-20): `MatchCenter` and `MatchExtreme`, test only, copied into the Train log as `Test_*`.*

- **`MatchCenter`** = (In67 − In33) / .34, and **`MatchExtreme`** = (In83 − In17) / .66. 1 = matching, above 1 = sharper than matching, below 1 = flatter.
- Job 25 is about 1.0 / 1.4. The other jobs are about .5 / 1.1.

This turns the shape difference into two numbers to compare across noise types and models.

### 4. Is it the spikes or the synaptic noise? (the key control)

- **rnn with test-time synaptic dropout (DropConnect).** egomlx `Dropout` applies only during training, and only per unit. A per-synapse dropout used at test too is the rate-model equivalent of `TestSynFail`, as in synaptic sampling machines.
  - If the rnn becomes graded with low intrusions, the effect is the multiplicative synaptic noise.
  - If it stays winner-take-all, the effect needs spikes.
  - This needs an egomlx option, e.g. `Act.DropConnect` plus a `TestDrop` flag.
- **axon with activation noise matched in variance** (the comparison you planned), against synaptic failure at the same effective noise.
- **Optionally, axon in rate-code mode, if it can be made to run:** it would isolate the spiking itself.

### 5. Hidden-layer minority trace (axon and rnn)

On trials where the majority group wins, how much of the minority group's Hidden representation is still active?
- Keep running prototypes of the Hidden `ActM` for each group from the unanimous (P100) trials.
- **`MinorTrace`** = cos(Hidden, minority prototype) − cos(Hidden, prototype of a group absent from the item).
- Compare it between choice outcomes and between models. A perforated attractor should keep a larger trace, and a larger trace should predict a minority choice on the next similar trial.

### 6. Attractor depth versus NMDA/GABAB (for the planned manipulations)

Track `LeadSwaps`, `CommitCyc`, Intrusion and the `_CarryEnd` stats as NMDA and GABAB are scaled. The prediction is that stronger slow currents mean fewer swaps, earlier commitment, more carryover, and flatter-than-matching choices, and weaker ones the reverse.

### 7. Settling microscope (axon, implemented)

*Implemented 2026-11-20 (`Run.Microscope`, `Sim.StatMicroscope`), after O'Reilly, Wyatte, Herd et al. 2013:* the cosine of each layer's `Act` at each 5-cycle sample with its final state, and of the Output with the chosen (`OutWin`) and other (`OutLose`) group patterns, per trial (`Test MicroscopeTrial`) and averaged over a test, split by minority versus majority choice. It is a continuous view of stat 1. A perforated attractor should show `OutLose` staying up, or rising and falling, late into the minus phase on minority-choice trials, and a non-monotonic approach to the final state. A rate attractor should show a smooth monotonic approach. The rnn version is implemented too, from the egomlx per-cycle readout (see stat 1).

## Results: synaptic failure manipulations (jobs 25–35, 2026-11-23)

All jobs have the same network: 6 slots, 4 per group, enc1 hid1, Output decay 1, lr .005, 150 epochs, tested every 10 epochs. The table shows the mean over test epochs 120–150 for both runs. Job 31 has only run 0, through epoch 130. The two runs agree well in every job. The competition stats only exist from job 31 on.

| Job | Training failures | Test failures | In83 | In67 | Intrusion | Output_CarryEnd | MinorPrevMinor | LeadSwapsMinor | CommitCycMinor / Major | Hidden1_MaxGeM |
|---|---|---|---|---|---|---|---|---|---|---|
| 26 | none | .4, SWt | .99 | .83 | .08 | .22 | .34 | – | – | 2.49 |
| 25, 32 | .4, SWt | .4, SWt | .98 | .68 | .11 | .27 | .43 | 1.63 | 71 / 38 | 3.12 |
| 27 | .4, SWt | one frozen pattern\* | .93 | .65 | .18 | .32 | .49 | – | – | 3.25 |
| 28 | .4, SWt | none | .79 | .54 | .33 | .45 | .66 | – | – | 3.68 |
| 31 | .4, SWt | none, PathScale.Abs .8 | .84 | .57 | .24 | .40 | .65 | .51 | 33 / 34 | 2.57 |
| 34 | .4, SWt | .4, no SWt | 1.00 | .91 | .01 | .15 | .14 | 3.22 | 108 / 40 | 2.10 |
| 33 | .4, no SWt | .4, no SWt | 1.00 | .97 | .01 | .12 | .06 | 4.78 | 128 / 32 | 2.74 |
| 35 | .3, SWt | .3, SWt | .94 | .63 | .18 | .33 | .51 | .86 | 51 / 37 | 3.10 |

"SWt" means `PFailSWt`: each synapse fails with probability PFail × (1 − SWt), so synapses with a strong structural weight rarely fail. \*Job 27 kept one failure pattern for the whole test, because failures were not resampled with test PFail = 0. Job 32 is a rerun of 25 with the new stats, and its values match 25 to three decimals.

**One trained network under four test conditions (28, 31, 32, 34).**
- These jobs share the same training: PFail .4 with SWt.
- From no failures, to uniform scaling, to SWt-weighted failures, to failures that ignore SWt, the network moves steadily away from carryover and toward majority rule. Carryover (`Output_CarryEnd`) goes .45 → .40 → .27 → .15, minority-after-minority choices (`MinorPrevMinor`) .66 → .65 → .43 → .14, and In67 .54 → .57 → .68 → .91.
- How the network was trained matters much less: 34 versus 33 is In67 .91 versus .97.
- 35, with fewer failures (.3), lands between 28 and 32.

**It is not the overall gain (31).**
- Scaling every pathway to .8 at test (`PathScale.Abs`), with no failures, lowers peak drive below that of test-time failures. Hidden1_MaxGeM is 2.57 in 31 versus 3.12 in 32.
- Yet its behavior stays almost at 28's.
- The likely reason is that FFFB inhibition scales with excitation and largely cancels a uniform gain change: Hidden1_ActMAvg barely changes.

**It is not the weak synapses coming back (34).**
- The first explanation was that the network learned to work without its weak, low-SWt synapses, and that restoring them at test added diffuse cross-talk between groups.
- 34 rules this out. It restores the weak synapses on average and still removes nearly all carryover.

**Revised interpretation: random failures break the specific connections that carry the previous trial.**
- Counting the evidence is a sum over many input synapses, so it is robust to a random mask.
- Carryover depends on the specific recurrent connections of the previous trial's attractor (Hidden↔Output) to reinstate that pattern. A random mask removes a share of exactly those connections, and inhibition cannot compensate for a change in which connections are present.
- The more the failures hit strong synapses, the more carryover is lost. SWt-weighted failures spare the strong synapses and keep part of it; failures that ignore SWt do not.
- Job 27 suggests a single fixed mask does most of this: it is close to 25, though with somewhat more intrusions and carryover, so resampling the mask every trial is not required.

**Two kinds of minority choice.**
- **Little disruption (31, 35):** minority choices settle as early as majority choices (cycle 33–51), with few lead swaps, and they follow previous minority choices. The bias left by the previous trial decides them before the stimulus is processed.
- **Heavy disruption (34, 33):** the few minority choices that remain come from late reversals: 3–5 lead swaps, settling around cycle 108–128 of the 150-cycle minus phase.
- **In between (32):** it is between the two on every measure, and it is also the closest to matching (In67 .68, `MatchCenter` 1.06).
- Majority choices settle around cycle 32–40 in every condition.
- `AltCorr` stays below .05 throughout: no cycle-by-cycle alternation, only single switches of the lead.

**Over training, matching emerges together with carryover.**
- With SWt failures (28, 32, 35), test choices are majority rule early in training: In67 is about 1.0 at epochs 20–40.
- After that, In67 falls while carryover, minority-after-minority choices and intrusions all rise together. This begins just after peak drive (Hidden1_MaxGeM is highest around epochs 30–40).
- With failures that ignore SWt (33), there is only a faint version of this: In67 drops from .99 to .97 while carryover rises from .08 to .12.

**Implications.**
- The matching-like curve with SWt failures is largely choice history: carryover through attractor connections that strengthen with training.
- Within-trial competition (late reversals) supplies the minority choices that remain when carryover is disrupted.
- A model that matches through within-trial sampling alone would need minority choices with late commits and little dependence on the previous trial. None of the conditions so far does both at a matching level of In67.
- Possible next controls:
  - Test-time scaling of each synapse by its expected survival, 1 − PFail × (1 − SWt), with no failures: a deterministic version of the SWt-weighted mask.
  - Carryover reduced directly (more decay between trials) under SWt failures, to see how much of In67 survives without it.

## Suggested order

1. Stat 3: a few lines in each sim.
2. Stat 1: axon now; the rnn side after the egomlx per-cycle readout.
3. Test-time DropConnect in egomlx (4): the decisive control.
4. Stats 2 and 5 if 1 and 4 leave the mechanism ambiguous.
