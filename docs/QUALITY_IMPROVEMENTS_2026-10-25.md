# Quality improvement and reproducibility evidence (25 October 2026)

This note documents **new verification and read-only reporting**, not new training outcomes. Original study code, measured checkpoints and logged metrics have not been altered to change the scientific narrative.

## What the lecturer actually asks for

Sections **1.3–1.4** of the seven-page Wits NLP Project 1 brief explicitly request the original Singh symbol–label task, the Base one-number output, an Extended three-token continuation (query number, next symbol, corresponding number), recursive synthetic-data model training, performance trends and interpretability. Further reasonable model-collapse adjustments are permitted. The brief **does not mandate recursively synthesizing complete input contexts**. Existing target-only feedback is valid as a clearly disclosed modelling choice, not equivalent to fully generated corpora.

### 1. Original methodology controls were already executed

The project already contains two relevant follow-up studies. Instead of inventing alternative results, `scripts/audit_methodology.py` verifies their frozen recorded data, generates human-readable comparison tables, and rejects inconsistent metadata, incomplete G1–G10 histories and forbidden final-test flags.

| Historical control | Original n | Clean at G10 | Recursive/synthetic at G10 | Effect |
| --- | ---: | ---: | ---: | ---: |
| Base with truly 300k G0 and 300k descendants: held-out label-pair accuracy | 1 model seed | 76.95% | 49.41% | +27.54 percentage points |
| Extended growing-real-plus-synthetic history: generated-next-symbol contextual validity | 1 model seed | 87.11% | 83.20% | +3.91 percentage points |

For both rows the effect is **clean minus synthetic**, using `val_relabel`. These are single-seed *sensitivity controls*, not confirmatory estimates. The first matches Base and Extended **example-exposure budgets**, but not their targets, task difficulty or architecture. The second accumulates previous generated **targets on real input prefixes**; it does not generate complete synthetic input contexts. The original G10 cumulative pool contains 22,528 episodes, with 100,000 gradient example exposures per generation.

Read the source JSON at `docs/experiments/base_full_budget_300k/summary.json` and `docs/experiments/cumulative_corpus/summary.json`. Primary five-seed evidence remains separate at `docs/experiments/g10_five_seed/`.

### 2. Full original 20 checkpoint-backed G10 endpoints independently reaggregated

The original five-seed results were rebuilt from the 20 distinct recorded G10 model artifacts downloaded from six original GitHub Actions training runs. `scripts/aggregate_five_seeds_g10.py` rechecks G10 checkpoint presence, data/model seed, training exposures, evaluation roles and final-test metadata. The new workflow compares all eight metric endpoints, **including per-seed paired effects**, with the committed result JSON within numerical tolerance.

- [Successful artifact-based reaggregation run #40191322513](https://github.com/huziibee/nlp_ass_one/actions/runs/40191322513)
- [Recomputed CSV/JSON evidence artifact](https://github.com/huziibee/nlp_ass_one/actions/runs/40191322513/artifacts/12598555641)

This is an independent **reprocessing of original pretrained checkpoints**, not a fresh retraining of all G0–G10 chains. Fresh ten-generation, five-seed reruns were not completed in this audit and must not be claimed.

### 3. Safer portable checkpoint restoration and metadata provenance

`scripts/checkpoint_registry.py` provides authenticated GitHub CLI restoration, verifies pinned run/artifact identities and split/G0 training budgets, rejects final-test flags, refuses to overwrite any existing original checkpoint directory, and writes a SHA-256 lock of **locally downloaded bytes**. Tests reject altered split, altered test flag and attempts to overwrite. The lock is not presented as an externally published digest, and GitHub Actions artifact retention remains an external dependency. Original tagged releases may provide a manual fallback.

Student commands:

```bash
python run.py restore-checkpoints
python run.py verify-checkpoints
python run.py methodology
python run.py results
python run.py tests
```

`restore-checkpoints` requires authenticated `gh`. Running the other commands uses only appropriate files; `verify-checkpoints` requires the original G0 files to be present.

### 4. Code quality and validation

- Added negative tests for budget equality, ten-generation completeness, synthetic block history, and false final-test contamination.
- Added negative tests for checkpoint split/metadata, local digest output, GitHub CLI prerequisite and overwrite protection.
- Expanded the clean-room test workflow to compile **all** active `src/`, `scripts/`, `tests/` and `run.py` files and to exercise both original methodology-control reports.
- Existing fresh model-backed Base and Extended one-generation smoke jobs, original G0 data/causal checks and the broader 51-test suite remain the independent functional evidence.
- No model-training semantics, source split or preserved experimental result files were changed solely to improve a claimed grade.

## Remaining scientific constraints that code alone cannot erase

1. The original final-test partition was opened at G3, so it cannot retroactively become a never-seen G10 holdout. G10 analyses are validation-only. Obtaining an independent holdout requires **genuinely new data or a prospectively locked new protocol**, not relabeling current validation data.
2. Five new full G0–G10 training chains per task require additional substantive compute. The 20 original G10 model artifacts have been reaggregated, but no freshly trained duplicate of every chain is claimed.
3. The primary study retains its unequal Base (1 million) and Extended (300k) budgets. The documented one-seed exposure-parity analysis is a caveated sensitivity control, not a retroactive rewrite of the primary experiment.
4. Some older research scripts use compact formatting and import-path manipulation; their historical traceability is important. Refactoring them before submission would require a new regression check to establish numerical equivalence.
5. Assignment marks and research merit depend on valid interpretations and ethical disclosures, not simply a green CI badge.

**Conclusion:** These additions address the earlier documentation/reproducibility weaknesses and provide stronger, reproducible supporting evidence. They do not justify claiming new experimental work or a guaranteed perfect score.
