# Current project state — 14 October 2026

**Active workspace:** https://github.com/huziibee/nlp_ass_one (public)  
**Historical study:** https://github.com/huziibee/nlp_ass_1 (private, sealed, read-only for our purposes)  
**Deadline:** 27 October 2026, 17:00 (Wits NLP Project 1)

## What is already available — verified GitHub presence
1. A pinned, public upstream Git submodule `external/icl-dynamics` with the original 16,638,016-byte Omniglot ResNet18 feature file, training sampler, JAX/Equinox attention-only model, learning code, head-ablation/clamping machinery and original paper figures notebooks. The dataset's matching Git blob SHA was checked against its duplicate in the private archive.
2. The private `nlp_ass_1` repository contains Phase 1–6 of a **completed symbolic analogue** of the assignment: Base label-only and Extended triple-output tasks, clean/frozen/recursive G0–G10 experiments, checkpoint files, validation and final test artifacts, mechanistic studies, five-seed replications, and audit reports.
3. The previous project's two-page final abstract, contribution statement, NeurIPS ethics declaration and Faculty AI ethics declaration were **reported missing** by its own final audit.
4. This fresh repository contains only research preparation: data access, exploration scripts/notebook, source notes and next-study protocol guidance. **No new experiment has been run here.**

## What the historical report says (not freshly re-executed)
The archive describes a 50-discrete-symbol, five-label, two-layer transformer trained on episodically randomised mappings. Its final 5-seed G10 comparison (Recursive minus Fixed Generation-0 Synthetic) reports:
- Base first-value accuracy: **−0.30 percentage points**, inconsistent direction across seeds.
- Extended first-value accuracy: **−7.71 percentage points**, negative 5/5 seeds.
- Extended second-value accuracy: **−17.59 percentage points**, negative 5/5 seeds.
- Imposing a correct Layer-2 induction match recovers **+21.25 percentage points** in second-value accuracy in a held-out mechanistic subset.

These figures are *previously reported results*, not independently rerun findings in this repository. Its audit says its experiment-artifact checks passed, but fresh environment-specific test execution was not confirmed.

## What the previous study does NOT prove
- That symbol/label **structural corruption in the generated training data causally produces** induction-matching failures. The report explicitly identifies this missing mediation experiment.
- That induction heads disappear: its reported intervention indicates that much of the copying pathway remains effective.
- That the effect generalises from 50 learned symbols to **actual Omniglot ResNet18 features**, or to natural-language LLMs.
- That the completed symbolic setup is unquestionably accepted as the **exact** Singh et al. task in the assessment.
- That replacement of clean targets with synthetics is the same experimental intervention as accumulating real+synthetic examples.

## What remains
| Stage | Status | Gate |
| --- | --- | --- |
| Verify data provenance and expose original upstream code | DONE — pinned submodule and metadata | Clone with --recurse-submodules |
| Interactive original data exploration | READY to run locally | Independently read HDF5 and save exploratory plots |
| Audit old code for validity and portability | REVIEW NEEDED | Run checks; record deviations and reproduction failures |
| Confirm assignment compliance with lecturer | UNRESOLVED | Clarify exact task and how synthetic data must be added |
| Specify Option B structural-corruption intervention | PROPOSED | Freeze intervention definitions, hypotheses, controls, outcomes |
| Assign coding-model workstreams | WAITING FOR AGREEMENT | Agree file/experiment boundaries first |
| Conduct new valid tests and independent replications | NOT STARTED HERE | Predeclared new held-out design, matched seeds/compute |
| Complete course deliverables | NOT COMPLETE | Two-page abstract, ethics, contribution, code, README.txt |

**Next practical step:** Run `python scripts/check_data.py` and the notebook; inspect `docs/OPTION_B_PLAN.md` and agree on the small number of decisive experiments before asking the coding model to implement them.
