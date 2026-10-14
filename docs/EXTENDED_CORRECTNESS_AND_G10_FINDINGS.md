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

1. **RUNNING** independently initialized Extended G0-G10 clean-versus-recursive replications ([workflow](https://github.com/huziibee/nlp_ass_one/actions/runs/38641472801)); do not discard weaker G0 seeds solely because they miss the initial G0 validation gate.
2. **COMPLETED** actual first and second seven-token Extended attention mechanisms and causal head ablations across G0/G3/G6/G10 ([successful workflow](https://github.com/huziibee/nlp_ass_one/actions/runs/38641720122)). Both first and second numeric labels stay accurate even as G0-selected head ablation effects change. Candidate head 1 was identified using training-only G0 cases.
3. **COMPLETED** matched G9-generated target correction retraining ([successful workflow](https://github.com/huziibee/nlp_ass_one/actions/runs/38641963903)); all six conditions use identical G9 source, 300k episodes, initialization, training order, optimizer and validation. On 512 validation episodes:
   - raw G10: 82.81% context-valid q2, 81.45% valid+correct second label (exact original G10 reproduction);
   - first-label correction: 82.42% / 81.45%;
   - context-valid second-label correction: 81.64% / 81.05%;
   - **invalid-q2-plus-corresponding-label correction:** **97.66% / 95.70%**;
   - invalid-q2 and all second-number correction: 97.46% / 96.88%;
   - all three labels/symbol corrections: 97.85% / 97.07%.
   In the 300k G9-generated examples sampled for G10, 61,128 next symbols (20.38%) were absent from their context, compared with 5,363 wrong second numbers under a valid symbol (1.79%) and 5,295 wrong first numbers (1.77%). These error categories may overlap. Repairs of invalid q2 necessarily also correct the associated second number; the effect cannot be attributed to the q2 identity alone. This is **one matched model seed**, not general proof.
4. **PREPARED, WAITING FOR REPLICATION G10 ARTIFACTS:** [independent-seed repair verification workflow](../.github/workflows/extended-g10-repair-independent.yml), with G10 raw-versus-invalid-Q2 repair paired by seed 5054/6054. Treat initial effect as **exploratory** until these tests finish.
5. **RUNNING:** independent Base G0-G10 clean-versus-recursive replication at two new G0 seeds ([workflow](https://github.com/huziibee/nlp_ass_one/actions/runs/38642335173)). A [three-seed paired G10 analysis workflow](../.github/workflows/g10-independent-seed-comparison.yml) is prepared and unit-tested, awaiting completed artifacts.

Scientific question emerging from the actual results: **Can recursive training decrease contextual continuation validity while preserving induction-based numeric-label copying?** This is more precise than claiming that the induction-head circuit necessarily collapses.

## 15 October independent-repair replication update (validated results)

The independent-seed Extended full-budget G9-to-G10 matched correction workflow [38648489063](https://github.com/huziibee/nlp_ass_one/actions/runs/38648489063) **finished successfully**. This supersedes the earlier statement that independent repair is waiting; it does not change any previous primary-seed findings.

Two independently initialized models were trained using both raw G9-generated targets and the **invalid q2 plus corresponding second-number correction**, with identical per-seed initialization, training sample exposure (300,000 each), optimizer, and validation episodes. The same protected G3 final test was *not* reopened.

| Model seed | Raw q2 contextual validity | Joint-repaired validity | Paired change | Raw valid q2 plus correct second label | Repaired joint outcome |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5054 | 77.93% | 97.85% | +19.92 pp | 53.91% | 75.98% |
| 6054 | 80.27% | 97.46% | +17.19 pp | 72.66% | 89.65% |

The average within-seed contextual validity improvement across these **two model seeds** is **+18.55 percentage points**; the average valid-symbol-and-correct-number improvement is **+19.53 percentage points**. These are descriptive seed-paired statistics, not a population-level estimate or a fresh held-out-test confirmation. The intervention necessarily changes the next symbol **and its associated label**, so it still does **not** identify a pure symbol-identity-only causal effect. The baseline joint-outcome variation across seeds is substantial and should be reported rather than hidden. Full raw seed metrics and paired contrasts are permanently committed to [the independent repair archive](experiments/independent_repairs/).

The next scientific gate remains completion of independent **Base** G10 seeds and cross-task three-seed aggregation; followed by consistent scientific interpretation, genuine data accumulation decision, report and mandatory official ethics/contributions submission material. The G100 small-budget side experiment is deliberately outside the primary scientific argument.
