# NLP Assignment 1 — reproducibility and code audit (25 October 2026)

**Scope:** The coding component of the Wits COMS4054/COMS7062/COMS7066 Project 1, *Interpretability of Model Collapse*.

**Commit tested:** The integrated `main` research source with the documented Python 3.10 setup, and the `verify-readme-cleanroom-20261025` branch which adds checkpoint-backed CI. All demonstrations below use the original pinned upstream Omniglot feature data and, where needed, original model checkpoints.

## Evidence table

| ID | Evidence type | GitHub Actions run | Observed result |
| --- | --- | --- | --- |
| V1 | Fresh README dependency, data, tests and results | [40189906960](https://github.com/huziibee/nlp_ass_one/actions/runs/40189906960) | **Success**; 41/41 unit tests; 192 committed evidence files inventoried across 17 experiment families; four principal reported effects checked |
| V2 | Fresh checkpoint-backed Base and Extended training smoke | [40190022936](https://github.com/huziibee/nlp_ass_one/actions/runs/40190022936) | **Success**; original Base and Extended G0 downloaded; both four-update smoke training commands completed; output `model.eqx` / `metrics.json` validated |
| V3 | Independent checkpoint-backed smoke plus causality audit | [40190022877](https://github.com/huziibee/nlp_ass_one/actions/runs/40190022877) | **Success**; original Base/Extended checkpoint integrity, `verify_extended_causality.py`, Base smoke, Extended smoke, metadata validation, artifacts |
| H1 | Historical original Base G0 | [38612082174](https://github.com/huziibee/nlp_ass_one/actions/runs/38612082174) | Success |
| H2 | Historical Extended G0 LR pilots | [38630071147](https://github.com/huziibee/nlp_ass_one/actions/runs/38630071147) | 2 successful jobs |
| H3 | Historical Base G4–G10 continuation (four arms) | [38632979396](https://github.com/huziibee/nlp_ass_one/actions/runs/38632979396) | 4 successful jobs |
| H4 | Historical Extended G4–G10 continuation (four arms) | [38633208767](https://github.com/huziibee/nlp_ass_one/actions/runs/38633208767) | 4 successful jobs |
| H5 | Historical independent G10 aggregate | [38648539187](https://github.com/huziibee/nlp_ass_one/actions/runs/38648539187) | Success |
| H6 | Historical two additional Extended G0/G10 independent seeds | [38788302762](https://github.com/huziibee/nlp_ass_one/actions/runs/38788302762) | 6 successful jobs |

### Details verified in raw logs

The V1 log records `Ran 41 tests in 0.771s` followed by `OK`, plus `Indexed 192 original evidence files in 17 families`. The results step wrote the Markdown/CSV summary and the evidence index/manifest, then uploaded `student-original-results-report`.

V2 log records `BASE_RECURSION_FINISHED` and `EXTENDED_RECURSION_FINISHED`, followed by `VERIFIED_STUDENT_SMOKE=smoke_base`, `VERIFIED_STUDENT_SMOKE=smoke_extended`, and an uploaded `student-smoke-checkpoints-and-metrics` ZIP. V3 independently passed `scripts/verify_extended_causality.py` and both smoke commands.

The short Base smoke reported approximately 96.9% held-out label accuracy for its loaded **pretrained** G0 on 32 examples, but the four-step new G1 models are intentionally not trained long enough to support a scientific accuracy claim. Smoke success means the pipeline executes, not that four steps reproduced G10 collapse.

The resulting *original*, pre-existing five-seed summaries are **read** by `python run.py results`: Base G10 clean-minus-recursive label accuracy +9.22 percentage points; Extended G10 sampled next-symbol validity +19.73 points; G10 invalid-target repair gain in valid probability mass +21.17 points; and logit-scale context preference gain +5.94.

## Mapping to explicit lecturer coding requirements

| Brief requirement | Source path / original evidence | Assessment |
| --- | --- | --- |
| Transformer implementing in-context symbol-label task / induction heads | Pinned `external/icl-dynamics/`; Base `scripts/paper_base_full.py`; mechanistic head interventions | Addressed; baseline training and head investigations executed |
| Base model-collapse pipeline | `scripts/base_recursive.py`; G0–G10 historical arms / metric summaries | Implemented and historically executed |
| Extended three-output model-collapse pipeline | `src/extended_omniglot.py`, `src/extended_learner.py`, `scripts/extended_recursive.py` | Implemented; causal and fresh 4-step execution confirmed |
| Training/validation/final-test protocol | `scripts/reference_split.py`, `scripts/evaluate_sealed.py`; frozen 50/1473/100 classes | Implemented; historical final test already opened at G3, so later G10 results are validation only |
| Train/validation trends and appropriate metrics | G1–G10 CSV/JSON/SVG under `docs/experiments/`; `scripts/aggregate_five_seeds_g10.py` | Evidence retained; descriptive five-seed paired comparisons |
| Interpretability / mechanistic analysis | `scripts/q2_context_activation_patch_pilot.py`, ablation and repair scripts; original logged outcomes | Historical evidence retained; causal intervention is not proof of unique mediation |
| Hyperparameter search / justify model decisions | Recorded Extended G0 learning-rate comparison and multiple controls | Addressed, subject to methodological evaluation |
| `requirements.txt`, `README.txt`, full source | Present on `main`; V1/V2 fresh execution | Verified for documented commands |

## Code-quality review

**Strengths:** The code uses a pinned upstream source, checks dataset identity and split, keeps final-test sampling out of normal train/validation code, stores provenance and seeded metrics, supplies controlled experimental arms, verifies complete expected checkpoints, uses defensive aggregation guards against missing seeds and pseudoreplication, and offers student-run commands and CI.

**Areas for improvement, not proven defects:** The research scripts use compact formatting, some bare `assert` statements, import-path manipulation, and historical GitHub Actions artifact IDs rather than a portable checkpoint registry. Some input-dependent randomness in the Extended embedding code deserves scrutiny if embedding dropout is enabled in an alternative configuration. The code is appropriate as an academic research artifact, but should not be called production-hardened or formally verified.

## Limits of this verification

1. **Not a newly repeated G0–G10 five-seed training run.** Historical G10 execution is supported by the original successful Actions records and persisted measurement artifacts. New execution covers **four optimizer steps** for Base and Extended; this cannot prove bit-for-bit reproduction of historical training.
2. **Not a newly independent final evaluation.** The original 100 held-out test classes were used at G3, and V1–V3 deliberately avoid reopening them. G10 research interpretations must remain validation-scoped.
3. **The scientific design is a target-only synthetic-feedback loop.** The input contexts remain real Omniglot feature data; it is not full autoregressive generation of every future input token.
4. **Base and Extended have unequal full-generation example budgets** (1,000,000 versus 300,000), limiting simple comparative interpretations.
5. **The brief rewards methodological decisions and report quality** rather than mere execution; passing tests does not guarantee full marks, publication acceptance or that every scientific inference is correct.

**Overall:** Coding requirements are implemented, and the public-facing documentation/test/report and both checkpoint-backed smoke interfaces have been shown to work. It is scientifically and ethically accurate to describe the implementation as **strong and technically verified at smoke-test level**, not as a fully independent reproduction of all recorded experiments.
