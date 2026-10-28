# Frozen core-model evaluation protocol (before independent test)

**Status:** Training and validation-only implementation is complete. The independent 100-class final test remains unopened until all jobs listed below finish successfully. This protocol was authored *before* any new final-test model scores are generated.

## Data and model decisions

- **Only split:** original Singh et al. 50 train, 1,473 reserved validation, 100 sealed final test, original exemplar partition 1 train / 4 validation, 8/2/0 label-pair split.
- **Original G0 Base:** source JAX/Equinox, paper's configuration, one million sequences, original checkpoint in Actions run 38612082174.
- **Base G1–G3:** freshly initialised, 1,000,000 sequences per generation; seed 34001, 32 per batch, original source LR 1e-5. Four prespecified arms: clean-only, unmasked synthetic target replacement from G-1, mixed half-real/half-G-1, and stationary G0-generated targets. Endpoints are G3 from Actions run 38627953800.
- **Extended G0:** original same two-layer causal attention-only model and Omniglot ResNet18 features; five-way first label, **50-way unrestricted next-symbol training-vocabulary head**, five-way second label. Three-token teacher-forcing training, three-pass free generation. The clean next symbol is uniform among two supports, so oracle next label is defined. The selected G0 LR is 0.0003 for 300,000 examples, seed 4054, from two validation-only LR candidates in run 38630071147 (chosen before opening test).
- **Extended G1–G3:** freshly initialised with the **same fixed initial weight seed 4054 for every generation and arm**, while data randomness uses seed 6001 and fresh generation-specific streams; 300,000 sequences per generation, batch 32, LR 0.0003, four synthetic-target arms. **Why the correction was necessary (validation-only):** the exploratory first run 38630546270 varied initialisation by generation and its clean G3 model unexpectedly scored 50.98% on held-out label pairs; holding G3 data fixed while using source G0 initialisation recovered 99.80%/100.00% held-out-label-pair first/second accuracy (successful targeted run 38631485036). That prior run is explicitly labelled initialization-confounded. Reproduction with the frozen common initialization is run **38631919983**; only these new checkpoints will be used for any final Extended comparisons.
- **Sampling:** categorical with temperature 1.0, no q2 validity mask; synthetic outputs are the *target continuations*, not fully generated input prefixes. Exactly half-clean equal-sized mixing is NOT literal accumulation of every historical corpus.

## Evaluation procedure and measures

- Development panels are 512 episodes from held-out label pairs, exemplars, and 1,473 reserved validation classes; use the same fixed evaluator seed across conditions.
- **Final test** on **100 held-out HDF5 classes**, one independent set of **4,096** two-way, one-shot episodes, seed **20261027**, original exemplar 0 and training label pairs, exactly identical evaluation rows across all model conditions.
- For Base G0 and four G3 Base arms: query-label accuracy and full 5-label cross entropy.
- For Extended G0 and four G3 Extended arms: first query-label accuracy/cross entropy, and **teacher-forced second label** accuracy/cross entropy given a clean next symbol from the two contextual characters.
- **Crucial restriction:** The Extended next-symbol head has only 50 TRAINING-class logits. It cannot classify 100 never-seen test class IDs; this target's zero-shot test accuracy is **undefined**, not 0% and not a comparable metric. The Extended next-symbol diversity, valid mass, and free-generation consistency are evaluated on VALIDATION train-symbol scenarios, not OOD global IDs.
- Report each generation/arm using its exact saved train and validation metrics, and the chosen endpoint above. The final test is *not* a new head/hyperparameter-selection source. Individual sample accuracies do not replace independent model-seed replication.

## Test-opening gate

Do NOT run `scripts/evaluate_sealed.py --confirm-final-frozen` until **both** full Base run 38627953800 and fixed-initialization full Extended run 38631919983 have finished successfully in all four arms, and the evaluator validation-only smoke run 38630702001 has passed. The only supported purpose afterward is a single preregistered frozen comparison. Once final scores exist, new model changes are explicitly exploratory and require a separate independent final-test reserve to claim confirmation.

Code and implementation checks can be completed before this gate. Do not confuse passing a model smoke test with completion of its full experiment.
