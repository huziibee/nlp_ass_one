# Project state — original audit and later 14 October update


## Later verified update — after G10 experiments

The sections below preserve the original source-audit history; they are **not** a current to-do list. Subsequent GitHub Actions completed [Base G10](https://github.com/huziibee/nlp_ass_one/actions/runs/38632979396), [Extended G10](https://github.com/huziibee/nlp_ass_one/actions/runs/38633208767), [eight-arm validation consolidation](https://github.com/huziibee/nlp_ass_one/actions/runs/38640835888), [trained Extended correctness checks](https://github.com/huziibee/nlp_ass_one/actions/runs/38640576945), [Base optogenetic head interventions](https://github.com/huziibee/nlp_ass_one/actions/runs/38641237829), [Extended seven-token head interventions](https://github.com/huziibee/nlp_ass_one/actions/runs/38641720122), and [causal G9-to-G10 target repairs](https://github.com/huziibee/nlp_ass_one/actions/runs/38641963903). New multi-seed checks were launched and must be verified on completion. Original G3 protected test classes were already scored once; G10 analyses remain validation-only. [Read current findings](EXTENDED_CORRECTNESS_AND_G10_FINDINGS.md).

**Active project:** https://github.com/huziibee/nlp_ass_one  
**Assignment:** Wits COMS4054A/COMS7062A/COMS7066A NLP Project 1, 27 October 2026 17:00.  
**Scope of this status:** only `nlp_ass_one` and its pinned original Singh et al. submodule.

## Verified successes — real author dataset

- **Source dataset:** Git submodule `external/icl-dynamics/`, original paper commit `85b895c...`; dataset Git blob SHA-1 `9eb588c516b641f5f25ac22227865a7ce5ab0b2f`.
- **HDF5 identity and contents:** verified SHA and byte size; `resnet18/224/feat` shape **1,623 classes × 5 exemplars × 512 float32 coordinates**.
- **Original paper main-run splits:** 50 train / 1,473 reserve-val / 100 test **classes**; exemplar IDs 0 for train, 1–4 for val; 8 train / 2 val / 0 test distinct unordered contextual label pairs.
- **Source-level caution:** original paper's `fsl_test_class` evaluator runs periodically, and the 1,473-class reserve-val pool is not among its four main named evaluators. Our course-safe protocol **does not monitor the 100 test classes** while tuning.
- **Original five figures:** stored in `docs/data-audit/` from actual HDF5, including same-vs-different cosine and candidate-matching baselines. These were exploratory across the HDF5, prior to the new protected-test protocol.
- **NEW split-aware diagnostics (train and val only):** Fig 06 comparisons and Fig 07 candidate difficulty curve. At validation: average same-character cosine **0.889**, different-character **0.772**; expected 2-way nearest-cosine matching **92.95%** (against one random distractor), falling to **16.94%** against 1,472 distractors. This is **not** transformer ICL.
- **Source-checking tests:** **23/23 passed** in [the data CI workflow](https://github.com/huziibee/nlp_ass_one/actions/runs/38610808506). Tests cover the original run's flags, train/val/test partition disjointness, oracle episodic relabeling, withheld-test guard, exact candidate-set probability math, and controlled target repairs.
- **Original authors' Base JAX training stack:** successful **four-update smoke (32 sequences)** and **500-update development pilot (16,000 sequences)**, CPU, Python 3.10/JAX 0.4.26, with validation-only evaluators. [Smoke workflow](https://github.com/huziibee/nlp_ass_one/actions/runs/38611038193). This confirms module compatibility and a successful forward/backward/update loop. The 16k pilot trained from 20.31% to 50.39% accuracy on training classes but only 21.48% to 35.55% on held-out label pairs: **not yet reliable ICL**. A source-derived 1,000,000-sequence Base run [completed](https://github.com/huziibee/nlp_ass_one/actions/runs/38612082174), with 97.46% reserved validation-class accuracy and 99.22% held-out-label-pair accuracy; see [measurement details](model-audit/README.md).
- **Target-intervention contracts:** `src/target_repairs.py` implements raw continuation, first-label repair, valid-second-label repair, label-where-defined repair, and joint repair. `tests/test_target_repairs.py` verifies causal invariants on **artificial toy examples**, not model-generated data.
- **New dataset-only experiment assembly:** `src/recursive_dataset.py` and `tests/test_recursive_dataset.py` prepare Base/Extended clean targets and provenance-tracked target replacement/fixed-size half-real mixing controls, rejecting held-out training classes and label pairs. No trained synthetic generator is yet wired into this component.
- **One active Jupyter notebook:** `notebooks/01_explore_omniglot.ipynb`, annotated with split-aware figures, source-derived label pairs, and plans for future attention-map/ablation integration. The notebook now loads only training/validation class features by default. Notebook 2 was removed as unnecessary.

## What can be concluded

1. **Data-only matching is usually easy in the two-candidate task** despite nontrivial errors when the candidate pool grows.
2. **Same-character pairs are more similar on average than different-character pairs** in both train and validation partitions; the distributions overlap.
3. **The original transformer training code is executable** in a pinned compatible JAX environment and can consume the verified HDF5.
4. **Our synthetic-target transformations are deterministic and obey contextual oracle invariants** on the test fixtures.

**Now established with limited-sample evidence:** a trained Base G0 with strong validation ICL-like accuracy, and validation-only causal interventions on candidate previous-token/induction heads (see [mechanistic audit](mechanistic-audit/README.md)). **Still not established:** multi-seed robustness, full unique circuit reconstruction, either Base or Extended recursive model collapse, training-data corruption causes induction failure, or clean-vs-synthetic matched retraining. The final test remains sealed.

## What we implemented and its role

| Component | File | State |
| --- | --- | --- |
| Source and HDF5 integrity | `scripts/check_data.py` | PASS |
| Original data visualisation | `scripts/inspect_data.py`, Notebook 1 | PASS |
| Original run train/val/test design | `scripts/reference_split.py`, `docs/TRAIN_VAL_TEST_PROTOCOL.md` | PASS |
| Contextual 2-way oracle episode sampler | `scripts/episode_probe.py` | Unit-tested |
| Candidate-difficulty and comparison visualisation | `scripts/split_diagnostics.py` | PASS, measured |
| Original Base forward/backward/update integration | `scripts/paper_base_smoke.py` | PASS (4 updates) |
| Original Base G0 full reproduction | `scripts/paper_base_full.py`, `docs/model-audit/` | PASS (1M sequences, validation-only) |
| Synthetic continuation repair | `src/target_repairs.py` | Unit-tested on toy continuation arrays |
| Base/Extended target preparation and matched-source controls | `src/recursive_dataset.py` | Unit-tested data assembly; not a trainer |
| Matched Base and Extended **training** across generations | Not yet implemented | REQUIRED |
| Trained Base head ablation, retain-only, oracle-match | `scripts/base_mechanistic.py`, `docs/mechanistic-audit/` | MEASURED (single seed, 256 validation episodes per track); Extended/generational comparisons pending |
| Independent final test and two-page submission | Not yet done | REQUIRED |

## Immediate priorities

1. **Choose a precise Extended output representation** before implementing `q2`: the original Base code produces five *label* logits, not generated *symbol* tokens. See [Extended output contract](EXTENDED_OUTPUT_CONTRACT.md). Avoid a class-ID prediction head that cannot meaningfully generate unseen test character classes.
2. **Replicate the completed Base G0** on independent seeds and carry its evaluation/circuit controls into the recursive experiment. Preserve authors' architecture, task distribution, provenance and checkpoint format. Never monitor final 100 test classes.
3. **Implement and validate Extended G0**, including three target positions and teacher-forced versus free-running evaluation. Require sufficient validation ICL and valid next-symbol behaviour before recursion.
4. **Build full generation-to-generation Base and Extended recursive training** with matched clean and fixed-source controls, seed pairs, data hashes and loss/accuracy curves. Define whether training uses full replacement or real+synthetic accumulation before running it.
5. **Hook the unit-tested repair functions into a matched retraining runner** on actual model samples. No synthetic intervention can be claimed causally effective until trained models on those arms are compared and independently evaluated.
6. **Mechanistic analysis and writing** are additional future stages. The user has chosen to finalise the current development path up to implementing matched experiments before designing everything after that.

## Links

- [Figure 2 and Figure 5 split-aware results](data-audit/SPLIT_AWARE_ANALYSIS.md)
- [Machine-readable numbers](data-audit/split_diagnostics.json)
- [Exact source-derived splits](data-audit/reference_split_manifest.json)
- [Next-stage plan](NEXT_STAGE_PLAN.md)
- [Wits task interpretation](ASSIGNMENT_COMPLIANCE.md)
- [Frozen original split decision](SPLIT_DECISION.md)
- [Complete assignment and submission checklist](ASSIGNMENT_REQUIREMENTS.md)
