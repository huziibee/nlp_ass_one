# NLP Assignment 1 — October 15 overnight repository handoff

## Verified release and G10 scientific aggregation completed (later 15 October update)

**FULL CHECKPOINTS ARE NOW PERSISTENT.** After fixing the original numerical .eqx checkpoint filename detection and a find/head SIGPIPE, the archival job [38651168336](https://github.com/huziibee/nlp_ass_one/actions/runs/38651168336) **completed successfully** and published the [versioned model checkpoint release](https://github.com/huziibee/nlp_ass_one/releases/tag/nlp-assignment1-original-omniglot-checkpoints-20261015). It has original Base/Extended G0 and G10, independent Base/Extended G10 replicates, original and independently corrected G10, SHA256SUMS, README. This is an explicit completed and independently verified release, not a planned future backup. Together with the 117 source-linked CSV/JSON/SVG results committed to Git, the original experiments are recoverable.

**THREE-SEED G10 STATISTICS COMPLETED:** [38648539187](https://github.com/huziibee/nlp_ass_one/actions/runs/38648539187), committed to `docs/experiments/g10_three_seed`. Base 3-seed paired clean minus recursive held-out-label accuracy **+8.53 percentage points** [3.67,13.39] indicative t-95% interval; Extended contextual next-symbol validity **+17.06 pp** [9.63,24.48]. Other Extended first-label differences vary widely by model seed and should not be claimed as universally stable.

**SOURCE-BASED OBSERVED ONSET AUDIT COMPLETED:** [38651210291](https://github.com/huziibee/nlp_ass_one/actions/runs/38651210291); `docs/experiments/behavioral_onset`. Earliest post-hoc 5 pp clean-recursive divergence: Extended q2 validity G2, Base label accuracy G7 in **one primary run**, not independently established change points. G1 Extended validity gap 4.49 pp. The project's 48-step roadmap is now **39 checked / 9 open**; of those nine, report and submission preparation remain the high priority.

**Still waiting:** low-assignment-value G100 short-budget run [38648425960](https://github.com/huziibee/nlp_ass_one/actions/runs/38648425960) remains in progress at last observed check; do not elevate it to a full-budget inference.

**Submission:** A locally compiled two-page CCN-template LaTeX manuscript draft is prepared for user review. It has not been pushed as repo source at this time. The group must verify author contributions and obtain the exact Moodle NeurIPS 2024 questionnaire and Faculty AI Ethics form; placeholders **cannot** be submitted as if they satisfy mandatory requirements.


## Verified live update: two independent repair seeds succeeded

**Run [38648489063](https://github.com/huziibee/nlp_ass_one/actions/runs/38648489063) COMPLETED/SUCCESS**, including result aggregation and a commit to `docs/experiments/independent_repairs/`. In seed 5054, free-symbol contextual validity improved **77.93% -> 97.85%** with invalid-q2+associated-label target correction; in seed 6054, **80.27% -> 97.46%**. Across these two seeds, paired mean improvement **+18.55 percentage points**. Joint validity/label correctness improvement **+19.53 pp** mean. This is a descriptive replication of a *joint symbol-plus-label correction*, not evidence that q2 symbol identity alone is the unique cause. Both seeds are retained regardless of their G0/G10 baseline quality.

**Checklist now 33/48 checked** (36 and 41 additionally completed). **Permanent measured archive succeeded:** [38648663762](https://github.com/huziibee/nlp_ass_one/actions/runs/38648663762), **117 source-linked nonbinary files** with SHA-256 manifest in `docs/experiments/archived_original_runs/`. A separate checkpoint-release job [38648902189](https://github.com/huziibee/nlp_ass_one/actions/runs/38648902189) has launched and awaits the independent Base job; do not assume the GitHub Release exists before that job succeeds.


**Canonical current research source:** `extended-correctness-and-g10-analysis` (staged behind draft PR #3, which is stacked on draft PR #2). **Default `main` is stale** and reports missing G10 experiments even though they have completed on branches. This new branch `assignment-roadmap-g100-20261015` is based on the research branch and adds the marked roadmap plus jobs and evidence archiving. Do not treat the new branch as fully reviewed or merge it blindly.

## Where earlier work actually stopped

The original Omniglot ResNet18 50-class two-way, one-shot Base and Extended source-derived models were trained with 50 train / 1473 reserved validation / 100 original test classes. Base G0 and Extended G0 trained and passed causal correctness checks. Four conditions for each model (clean-only, recursively replaced model-produced targets, 50/50 real-anchor, frozen G0 generator) were trained through G10, with full eight-arm validation trajectories, seven-token Extended circuits, Base head interventions and six matched G9->G10 target repairs. **Do not rerun this primary G0–G10 experiment.**

Original measured G10 examples: Base held-out label-pair accuracy recursive 92.38% versus clean 99.41%; Extended contextual sampled next-symbol validity 82.81% recursive versus 96.48% clean on 512 validation episodes, while the first numeric label retains high accuracy; initial matched invalid-q2 plus corresponding-second-label training target correction improved 82.81% to 97.66% (primary seed only). The head-index matching margin also deteriorated on some clean models: do not interpret head-index movement as proof that an entire learned induction circuit collapsed.

The protected 100-class G3 test was already opened once. **G10 and G100 analysis must remain validation-only**, not new untouched test confirmation. Synthetics here replace supervised target continuations, *not input prompts*, and equal-sized 50/50 mixing is not full historical corpus accumulation. The Extended global q2 class output only covers 50 training symbols, not unseen test-class IDs. Base uses 1M and Extended 300k training episodes per full-budget descendant.

## What is newly committed in this branch

- [Value-ranked reconstructed 48-investigation roadmap](NLP_Assignment_1_Research_Investigation_Roadmap.md) — 31 core items marked completed, 17 open at first audit; three value classes (High/Medium/Low). This is a reconstruction rather than a recovered exact earlier file.
- `scripts/g100_exploratory.py`, `scripts/summarize_g100.py` and `.github/workflows/assignment-g100-exploratory.yml`. A true Extended Omniglot **100-generation** short-budget test with clean-only and recursive target-replacement controls. Only **16,384 episodes per descendant**, unlike 300,000 for primary Extended G10. **LOW assignment value**; no claims of direct full-budget comparability.
- `scripts/summarize_independent_repair.py` and `.github/workflows/assignment-independent-repair.yml`. Exact model-seed paired raw/invalid-q2-plus-label full-budget training for seeds 5054/6054. **HIGH value**.
- `.github/workflows/assignment-three-seed-aggregate.yml`. Waits on original Base replication run success, then aggregates primary + 2 fresh independent model seeds for both Base and Extended with per-seed effects and uncertainty. **HIGH value**.
- `scripts/archive_completed_research_artifacts.py` and `.github/workflows/assignment-archive-existing-evidence.yml`. Copies original actual results: G1–G10 curves, dashboard SVGs, distribution metrics, Base and Extended head interventions, primary six repairs, Extended independent seed metrics, and historical frozen G3 test. Emits each source's run ID, path and SHA256 manifest; commits **nonbinary** reports and figures. Model checkpoint binaries remain in finite-retention Actions artifacts. **HIGH value**.

## Workflows started (source of truth is their GitHub pages)

| Purpose | Run | Latest initial status | Assignment value |
| --- | --- | --- | --- |
| LOW-priority G100 exploratory | [38648425960](https://github.com/huziibee/nlp_ass_one/actions/runs/38648425960) | Running, training step entered | Low |
| Independent repair replication (2 seeds × 2 conditions) | [38648489063](https://github.com/huziibee/nlp_ass_one/actions/runs/38648489063) | Running, training step entered | High |
| G10 Base+Extended three-seed paired aggregation | [38648539187](https://github.com/huziibee/nlp_ass_one/actions/runs/38648539187) | Running, waiting on Base seed results | High |
| Permanent research archive | [38648663762](https://github.com/huziibee/nlp_ass_one/actions/runs/38648663762) | Started | High |
| Original independent Base, seeds 505/605 | [38642335173](https://github.com/huziibee/nlp_ass_one/actions/runs/38642335173) | In progress at last check | High |
| Original independent Extended, seeds 5054/6054 | [38641472801](https://github.com/huziibee/nlp_ass_one/actions/runs/38641472801) | Completed, success | High |

**Never infer results simply because a workflow was started.** For new experiments, numerical results are written in `docs/experiments/{g100,independent_repairs,g10_three_seed,archived_original_runs}/` only if their jobs reach the upload/commit stages successfully. Confirm those paths and see the Actions logs after waking.

## Important work still not automated

1. Interpret the seed-level results, including negative and weak G0 seeds; use model-seed rather than episode-level uncertainty.
2. Incorporate the scientific nuance in the final paper: context-validity loss despite relatively stable numeric ICL; head migration; repair changes both symbol and its linked label.
3. Explicitly justify the lecturer's "adding synthetic data" requirement versus fixed-size replacement/mixing, or perform a separately labeled true growing-corpus accumulation condition if warranted. Do not pretend it was already done.
4. Acquire the *actual Moodle* LaTeX template, NeurIPS 2024 ethics questionnaire and Faculty of Science AI Ethics Statement, and verify the preliminary contribution statement deadline (13 October).
5. After review, merge stacked draft [PR #2](https://github.com/huziibee/nlp_ass_one/pull/2), then [PR #3](https://github.com/huziibee/nlp_ass_one/pull/3), then the new overnight changes. Do not merge unreviewed code or stale docs into `main`.
6. Compile a strictly 2-page main-text extended abstract with methods, figures, results, discussion, and required additional ethics/contribution material. Reproduce clean setup and dependencies before submitting.

**Final submission deadline:** **27 October 2026, 17:00 SAST**.

## Storage safety

Source scripts and documentation committed in Git are durable. Full training outputs/checkpoints in GitHub Actions have **finite retention**, including the new 90-day-artifact request (actual repository limits can be shorter). The archive workflow attempts to commit all important lightweight figures, CSV and JSON. Confirm the archive workflow succeeds and consider an external persistent checkpoint backup if reconstruction cost matters; GitHub binary models are not fully preserved by this documentation alone.
