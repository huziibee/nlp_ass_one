# A/B/C Mechanism Discrimination — Original Omniglot Extended G10

**15 October 2026. All statements here describe actually executed validation experiments in GitHub Actions run 38719244340, original saved Base/Extended checkpoints and original HDF5 features.** Source `docs/experiments/abc_mechanism/`. All novel A/B/C interventions used 128 validation episodes per role and the primary original initialization only. No G10 final test was reopened.

## Scientific question and main answer

**A — Original circuit degradation:** Not supported in the simple sense that original Layer 2 matching heads cease attending to the right contextual label. Recursive G10 head 1 still has +0.892 G10 first-label matching margin on a fresh val-relabel probe (clean +0.860); it also has large margins on val-classes. Yet its test-time ablation becomes less consequential for first-number accuracy. Hence continued matching attention is distinguished from continuing *causal necessity*. For contextual q2 generation, valid probability mass degrades despite numerical matching remaining intact; mechanism of q2 failure cannot be inferred from head 1 alone.

**B — Retrieval redistributed to different heads:** Possible in the weak form of less dependence on the original top three heads: G10 1st-label accuracy loss under fixed G0 top-three removal is **51.56 percentage points in clean vs 12.50 in recursive** on the identical 128 val-relabel episodes. **However**, knocking out the five *other* heads changes recursive 1st-label accuracy by **-0.78 pp** (a slight improvement), compared with **-0.78 or +3.91 pp** elsewhere, so these five heads as a *group* are not individually required under this intervention. Full eight-head knockout reduces recursive 1st-label accuracy by **71.88 pp**. The best additional head selected from recursive G10 TRAIN based on individually largest accuracy drop (head 5) had **0 pp** standalone effect on val-relabel; adding it to the old G0 top three changes drop from **12.50 pp** to **72.66 pp**. This is a strong *interaction / redundancy* signal but does not locate a unique head/path that computes retrieval. Group knockout may create severe out-of-distribution hidden states. Do not claim proof of exact algorithmic relocation. Clean-only also reorganizes relative to G0.

**C — Label-prediction shortcut not following newly assigned context labels:** Strongly disfavored for simple fixed-label, location-only or class-ID-only shortcuts. On 128 val-relabel examples recursive G10 gets **99.22% (127/128)** accuracy after swapping support numeric labels while keeping images/query unchanged; its outputs change on 98.44% of examples. On 128 held-out-class episodes, swap-label accuracy is **97.66% (125/128)**, showing dependence on per-episode assigned labels even for unseen classes. Opposite support query flip produces **98.44%** on relabel and **99.22%** on class split; this condition is easier because the swapped query is an exact support exemplar. Reversing support order while keeping pair labels matched preserves 99.22% first-label accuracy on relabel. These tests exclude simple memorized class-label maps and positional label guessing. They do **not** exclude a different attention-independent, context-sensitive computational algorithm. Numerically correct labels coexist with invalid next-symbol generation.

## Primary 128 episode val-relabel quantitative matrix

| Measure | Clean G10 | Recursive G10 |
|---|---:|---:|
| First numeric-label accuracy, natural | 100.00% | 99.22% |
| Layer 2 G0-selected head-1 first-label matching margin | +0.860 | +0.892 |
| First-label accuracy loss after G0 Head 1 knockout | 4.69 pp | -0.78 pp |
| First-label loss after knocking out G0 Heads 1/0/7 | 51.56 pp | 12.50 pp |
| First-label loss after knocking out remaining five heads | 3.91 pp | -0.78 pp |
| First-label loss after all 8 L2 heads knockout | 93.75 pp | 71.88 pp |
| First-label loss after G0 1/0/7 + recursive-train-selected new head 5 | N/A under clean selection (different candidate) | 72.66 pp |
| Counterfactual swapped-support-label accuracy | 100.00% | 99.22% |
| Counterfactual opposite-support-query accuracy | 100.00% | 98.44% |
| Probability mass on context-valid q2 | 96.78% | 77.95% |

Results use common held-out validation examples and original models; they are conditional on one pair of models, not an across-seed mechanistic certainty.

## Evidential decision matrix

| Hypothesis | Prediction | Observed | Conclusion |
|---|---|---|---|
| A strong form: original retrieval heads stop matching | Original head-1 margin collapses | Recursive G10 remains +0.892 | **Disfavored** for numerical-label matching |
| A weaker form: original heads become causally unnecessary | Fixed heads have smaller ablation effects | Old top3 knockout loss drops 51.56 -> 12.50 pp | **Supported** as decreased original-set causal necessity; cause unresolved |
| B redistribution to other heads | Individually or jointly ablated alternative heads significantly impair remaining accuracy | Remaining-five knockout alone causes no first-label loss; adding head 5 to old top3 produces severe drop | **Suggestive head interaction, not proven relocation** |
| C context-free shortcut | Incorrect after support-label reassignment | 99.22% label-swapped accuracy | **Strongly disfavored** |
| C positional shortcut | Reversing support order causes big loss | 99.22% accuracy preserved | **Strongly disfavored** |
| C alternative *context-sensitive* computation | Could still achieve correct counterfactual labels | No path-level tracing performed | **Still possible** |

## Limitations critical for a 100% scientific write-up

1. Test-time knockout effects are causal to head output contribution but large simultaneous ablations can change hidden-state distributions unnaturally. There is no source-to-target activation patching or mediation identification.
2. Heads are reinitialized/retrained across generations. Head *indices* are not aligned latent algorithmic components; fixed G0 head comparisons diagnose dependency on a predetermined basis, not tracked neurons.
3. Single selected G10 model seed and 128 examples per role. Independent G10 *behavioral* replicates exist, not all the new counterfactual/causal mechanism claims.
4. Teacher-forced first/second labels are different targets from *free-running sampled* q2 contextual validity; probability mass is not exactly the same as sampled hit rate. Use the correct metric labels.
5. At the current G0->G10 comparison we cannot causally say data corruption *caused* head redistribution, because multiple training changes and independently initialized models coexist. Repair interventions establish training-target causality for symbol validity but not head mediation.
6. For held-out new classes, global q2 ID accuracy is undefined because the Extended symbol head covers 50 training classes.

## Recommended publishable next discriminator

**Do cross-checkpoint path-level activation patching with controlled representation matching**, then repeat on at least two additional independent model seeds. Specifically, localize the residual stream at pre-head/post-head positions for val-relabel episodes, patch the *same model's* clean/counterfactual labels within fixed weights to avoid cross-model gauge mismatch, and quantify restoration of correct logit margins with head-specific attention-output patching. Pair with labels swapped and support order permuted. A permutation of head indices between independent models is not a valid neuron alignment by itself. Add a dropout/zero-activation matched control. A clean gain when compensating heads are patched plus failure when their paths are removed would favor B over C.

## Assignment and paper implication

Proposed wording: **"Although original G0-selected matching-head ablations have a smaller effect at recursive G10, the same heads still attend to the correct contextual labels; the model remains sensitive to counterfactual relabeling and support order, excluding trivial memorized or positional label shortcuts. Nonadditive joint ablations suggest changed computational dependencies rather than wholesale loss of retrieval. Path-level mediation and seed replication are needed to distinguish circuit redistribution from a different context-dependent algorithm."**

Result files: `docs/experiments/abc_mechanism/abc-counterfactual-clean_only/clean_only.json` and `abc-counterfactual-recursive_replacement/recursive_replacement.json`.
