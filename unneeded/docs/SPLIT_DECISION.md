# Frozen dataset split decision

**Decision (14 October 2026): retain the original Singh et al. main-run split.** No 70/15/15 or 80/10/10 configuration is authorised for the active experiment. This is a *class-disjoint, episode-generating few-shot task*, not a conventional collection of i.i.d. fixed rows, so percentage-based episode splits would be misleading and can leak exemplar identities.

## Comparison and reasoning

| Candidate | Advantage | Problem for this assignment | Decision |
| --- | --- | --- | --- |
| 70/15/15 random episodes | Familiar and simple | Repeated Omniglot classes/exemplars can appear in every set; not equivalent to class-disjoint generalisation, and changes paper protocol | Reject |
| 80/10/10 random episodes | More episodes allocated to fitting | Same episode/class leakage problem; fewer independent development/test conditions | Reject |
| Repartition classes 70/15/15 or 80/10/10 | Could be class-disjoint | Changes training class diversity, which itself changes the induction-learning dynamics under study; weakens Singh reproduction | Reject for core study |
| **Original Singh partition** | Published baseline comparability, preexisting tested code, class-level holdouts and separate extrapolation probes | Only 50 training classes; uneven partition sizes, so clearly report per-condition metrics and sampling uncertainty | **Keep** |

## Single canonical split

Read source of truth from `scripts/reference_split.py` and `docs/data-audit/reference_split_manifest.json`; do not hand-author alternative manifests.

| Axis | Train | Development/validation | Sealed final test |
| --- | --- | --- | --- |
| Omniglot HDF5 class indices | 0..49 (50 classes) | 50..1522 (1,473 classes) | 1523..1622 (100 classes) |
| Exemplar index per class | 0 (one exemplar) | 1..4 (four alternative exemplars) | No additional exemplar partition |
| Unordered label pairs from 5 labels | 8 pairs | 2 pairs | No additional label-pair partition |

**Train episodes:** training classes + training exemplar + training label pairs.

**Development tracks:** held-out label pairs on training classes; held-out exemplars on training classes; held-out validation classes with training exemplars/pairs. All hyperparameters, early stops, head selections and causal hypotheses must be frozen using development evidence only.

**Once-only final:** 100 test classes using exemplar 0 and training-pair sampling after all choices are frozen. Extended output-symbol metrics on unseen global class IDs are not defined for a 50-class vocabulary; report only valid outcomes under the chosen Extended output contract. Do not invent zero-shot symbol-class accuracy.

**Paper-specific qualification:** the authors' periodic `fsl_test_class` evaluator accessed their test classes. Our course-safe wrapper substitutes 1,473 validation classes for that periodic evaluator; the 100 classes stay sealed. This is the *only* intentional split/evaluation difference from the published training command.

## Practical statistical protections

- Report number of independent model seeds, not just episode count.
- Report results separately for held-out pairs, exemplars and classes (their sample spaces differ).
- Sample validation episodes only from the declared split; tests guard against final-test usage.
- Never mix the previously evaluated private symbolic study's final scores into a newly sealed confirmatory test.
- If science ultimately motivates an alternate training-class diversity intervention, it is an explicitly labelled *ablation*, not another data split or unreported tuning decision.

**Implementation audit (14 October 2026):** The existing active runners, source-checker and tests already use the original partition. Repository search found no code implementing 70/15/15 or 80/10/10. No unrelated files or historical results were deleted merely to make the repository appear cleaner.
