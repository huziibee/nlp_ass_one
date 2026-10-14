# Extended correctness audit and original-feature G10 follow-up

**Source task:** the Wits Project 1 brief (pp. 3-5); original Singh et al. pinned `icl-dynamics` submodule. This document describes what the *implemented* Extended model supports, not a claim of scientific equivalence to every possible three-token sequence model.

## Audit status

**PASS:** [trained-checkpoint Extended source/causal audit](https://github.com/huziibee/nlp_ass_one/actions/runs/38640576945).

The audit loaded **the actual training-selected Extended G0 checkpoint**, seed 4054, learning rate `3e-4`, trained with 300,000 original Omniglot-feature episodes. It verified:

1. Original two-way one-shot contextual symbol/number task and held-out label-pair, exemplar, class partitions.
2. Three distinct output distributions: first numeric label (5 classes), next symbol (unmasked 50 TRAINING Omniglot class IDs), second numeric label (5 classes).
3. Oracle clean targets: query's contextual number; independently chosen support symbol; the number corresponding to that support symbol.
4. Exact correspondence between sampled class IDs and source feature vectors, including generated next-symbol prototypes in the **training** partition.
5. First Extended logits equal the original Singh five-token Base logits before any future generated inputs.
6. No future-token leakage into first or second predictions under causal attention.
7. Nonzero finite supervised gradients through the source numeric classifier and added symbol classifier.
8. Actual autoregressive output ranges and meaningful held-out numeric-label predictions without pretending unseen class identities are in the 50-class symbol-head vocabulary.
9. No original 100 final-test classes accessed during the development audit.

**Important non-equivalences:** This task extends the original published Base classifier. Singh et al. did **not** provide a canonical next-symbol generation head. We chose to generate support symbols uniformly for clean Extended continuations: this is an openly declared **experimental convention**, not an asserted property of Singh et al.'s original training distribution or the lecturer's exact intended extended distribution.

### Definition of this feedback experiment

Each model is freshly initialized and trained on original, clean Omniglot context prefixes. The synthetic data consist of ***target continuations only***, generated with the previous model (or frozen G0), then used as new supervised targets. This is **not** fully generative resampling of input episodes and **not** the growing-corpus accumulation condition discussed in the broader literature.

Training controls: clean-only, previous-model replacement, 50/50 real anchor, frozen-G0 synthetic. Class split: 50 training / 1,473 validation / 100 final test, with the original exemplar/label-pair axis partitions.

## Completed G10 evidence

[Base G10 successful run](https://github.com/huziibee/nlp_ass_one/actions/runs/38632979396) and [Extended G10 successful run](https://github.com/huziibee/nlp_ass_one/actions/runs/38633208767), both four-condition/single-seed runs.

[Complete verified eight-condition source-linked validation trajectories, seven SVG figures, master CSV](https://github.com/huziibee/nlp_ass_one/actions/runs/38640835888).

- Base fully-recursive G10 held-out label-pair accuracy: **92.38%** versus **99.41%** clean; validation-class accuracy **94.14%** versus **97.27%** clean.
- Extended fully-recursive G10 held-out label-pair first-number accuracy: **100%**, essentially unchanged from clean; teacher-forced second-number accuracy: **99.61%**.
- On 512 held-out-label-pair episodes, Extended freely sampled contextual-symbol validity: **82.81%** fully recursive versus **96.48%** clean. Overall valid symbol **plus correct second number**: **81.45%** recursive versus **95.90%** clean.
- A separate, larger 1024-free-sample/condition [source audit](https://github.com/huziibee/nlp_ass_one/actions/runs/38640996978) showed Extended G10 **79.49%** contextual-symbol validity recursive versus **96.78%** clean. Differences from the earlier figure reflect different fixed evaluation episodes/draws.
- The generated global symbol marginal entropy at this 1024-sample audit changed from **3.852 nats at G0** to **3.790 nats at recursive G10**, versus maximal log(50)=3.912. Thus the much larger contextual-validity change does not automatically show extreme *global-vocabulary mode collapse*. The evidence currently favours **context-dependent continuation invalidity**, not disappearance of most symbol IDs.

## Mechanistic caution

Original-source [Base G0/G3/G6/G10 optogenetic interventions](https://github.com/huziibee/nlp_ass_one/actions/runs/38641237829) showed that the previously identified G0 head-3 query matching margin deteriorates substantially even for newly initialized **clean-only** G10 models. In one 128-episode validation condition it fell from **0.856 at G0** to **0.035** in clean G10 and **-0.011** in fully recursive G10, while clean validation accuracy remained high. A **fixed original head index** is not equivalent to the entire learned algorithm; models can reorganise heads.

The [actual seven-token Extended circuit and causal head-ablation workflow](https://github.com/huziibee/nlp_ass_one/actions/runs/38641720122) provides corresponding Extended evidence. Head interventions are **test-time causal perturbations**; they cannot alone prove that one training-data corruption caused circuit changes.

### Critical evaluation constraints

- One primary model seed: reported percentages are descriptive, not multi-seed population estimates.
- Base descendants each trained on 1M episodes, Extended descendants on 300k; a stronger between-task sensitivity claim needs matched training exposure.
- Next-symbol **global class ID accuracy is not defined for never-seen character IDs** in the 50-class symbol head. Only relevant numeric-label targets are evaluated for new validation classes.
- Extended generated-symbol diagnostics on held-out alternative exemplars have distinct generalisation difficulty and must not be conflated with held-out label pairs.
- G3 original final-test classes have already been scored on a separate frozen evaluation workflow. The later G10 outcomes are **validation-only follow-up results**, not a new untouched test confirmation.

## Current workstreams

1. Run independently initialized Extended G0-G10 clean-versus-recursive replications ([workflow](https://github.com/huziibee/nlp_ass_one/actions/runs/38641472801)); do not discard weaker G0 seeds solely because they miss the initial G0 validation gate.
2. Compare actual first and second seven-token Extended attention mechanisms and causal head ablations across G0/G3/G6/G10.
3. Apply matched G9-generated target corrections at G10 to distinguish invalid symbols, contextual second-label inconsistency and first-label noise ([workflow](https://github.com/huziibee/nlp_ass_one/actions/runs/38641963903)).
4. Treat all these follow-up interventions as **exploratory**. Reproduce a selected causal result on new seeds before stating a general claim.

Scientific question emerging from the actual results: **Can recursive training decrease contextual continuation validity while preserving induction-based numeric-label copying?** This is more precise than claiming that the induction-head circuit necessarily collapses.
