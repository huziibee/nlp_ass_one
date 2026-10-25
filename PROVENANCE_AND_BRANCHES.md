# Provenance, repository branches and retained evidence

**Active code branch: `main`.** Core executable source, the original numerical result summaries and their supporting per-seed CSV/JSON are intended to be inspectable without switching branches. The assigned research topic is target-only recursive model training on the original Singh/Omniglot task.

## What is original evidence?

- `docs/experiments/g10_five_seed/`: five independent model-seed comparisons across Base and Extended G10; **validation only**.
- `docs/experiments/all_seed_generations/`: G1–G10 per-seed onset, trajectory and role-level comparisons.
- `docs/experiments/q2_activation_patching/`: context-dependent attention activation intervention, five model pairs.
- `docs/experiments/q2_repair_mediation/`: five G9-target-corrected G10 model comparisons with raw recursive controls.
- `docs/experiments/archived_original_runs/`: original logged machine-readable observations, SVG plots and recorded conditions.
- `docs/experiments/conditional_symbol_distribution/`: separately measured conditional output distributions on the clean and recursive experiments.
- `docs/experiments/abc_mechanism/`: additional head counterfactual audit.

All are committed *as measured*. `python run.py results` parses the source summaries and produces new **derived display-only** CSV and Markdown; it does not train models, change original classifications, or impersonate a new run.

## What GitHub Actions still holds

Historical full checkpoint artifacts are generally much larger than the committed result summaries. These runs were inspected through the connected GitHub account on 25 October 2026, with these examples still reporting non-expired artifacts at inspection time:

| Purpose | Original run ID | Example available artifact |
| --- | --- | --- |
| Base G0 complete | 38612082174 | `base-paper-1m-validation-only` |
| Extended G0 complete | 38630071147 | `extended-omniglot-lr3e4-300k` |
| Five-seed extension (seeds 7054 and 8054) | 38788302762 | `extended-five-seed-g10-7054-recursive_replacement` |
| Independent 3-seed comparison | 38648539187 | `independently-confirmed-three-seed-g10-tables` |
| Base 300k matched-budget check | 38656720477 | `base-fully300k-g10-clean_only` |

Full run pages are `https://github.com/huziibee/nlp_ass_one/actions/runs/RUN_ID`. Workflow artifacts expire according to repository retention policies; do not claim current availability without rechecking. Large model weights are not duplicated as text files on `main`.

Preserved tagged releases:

- [Original Omniglot checkpoints](https://github.com/huziibee/nlp_ass_one/releases/tag/nlp-assignment1-original-omniglot-checkpoints-20261015)
- [Mechanistic follow-up checkpoints](https://github.com/huziibee/nlp_ass_one/releases/tag/nlp-assignment1-mechanistic-followups-20261015)

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for exact extraction/loader path expectations.

## Branch policy

- **`main`**: student-facing repository, runnable research source, tests, required README and dependency files, published/underlying measured evidence.
- **`research-archive-20261022`**: pre-cleanup historical research-branch snapshot, preserving the previous layout.
- **`research-archive-complete-20261025`**: comprehensive pre-cleanup snapshot incorporating the additional conditional distribution and ABC results.
- **Older experimental branches**: remain useful for the original stage-by-stage commit ancestry; their distinct results are consolidated into `main`. Avoid force-pushing or deleting those original branches merely for cosmetic cleanliness.

Archive material and obsolete progress plans are **not** the current source of truth. The old `docs/PROJECT_STATUS.md`, `docs/NEXT_STAGE_PLAN.md` and numbered checklist statuses describe earlier 14–15 October work, not the final measured paper.

Not every script can be removed from `main`: analysis, graph generation, target repair, and checkpoint verification are executable parts of the claimed research. Moving all of them out would break the lecturer's requirement to submit the complete implementation. Move *archival-only* scripts and old roadmaps to the archive instead.

## Meaningful scientific caveats

1. **Only targets are recursive.** Real Omniglot contexts are reused; the project does not simulate an internet-scale corpus of synthetic prompts.
2. **Held-out testing is limited.** The 100-class test was accessed at G3 and provides no untouched independent confirmation of later G10 results.
3. **Different training budgets.** Base descendants use 1,000,000 examples and Extended descendants 300,000; effect sizes must not be compared as if exposure were identical.
4. **Restricted symbol vocabulary.** Next-symbol IDs are restricted to the 50 training identities even though numeric-label retrieval can be evaluated on unseen classes.
5. **Patching is conditional.** A head's causal contribution to the computed output is not proof that synthetic target corruption isolated that specific head as the sole learning mechanism.

## Submission artifact status

A local extended-abstract PDF may include appendices, but a PDF attached to a chat is **not automatically in Git history**. This repository contains source/metrics and a complete guide; the final PDF must be separately included in the actual university upload. The official NeurIPS 2024 questionnaire and Faculty of Science AI Ethics declaration must be completed as prescribed in the assignment brief.
