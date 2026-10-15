# NLP Assignment 1 — Research Investigation Roadmap

**Updated 15 October 2026.** This is a reconstruction of the earlier missing 48-item roadmap, not the identical original file.

**Research branch:** `extended-correctness-and-g10-analysis` | **Deadline:** 27 Oct 2026 17:00 SAST

High = directly critical to assignment and rigorous results; Medium = valuable methodological strengthening; Low = curiosity/supporting exploration. `[x]` means original study step was implemented or its outcome measured, not necessarily replicated.

## Source fidelity and data

- [x] **01 — Pin Singh code & provenance** — **Medium value**
- [x] **02 — Verify original Omniglot HDF5 & features** — **High value**
- [x] **03 — Freeze class/exemplar/label splits** — **High value**
- [x] **04 — Measure feature similarity and cosine baselines** — **Low value**
- [x] **05 — Enforce protected-test separation in model selection** — **High value**
- [x] **06 — Document G3 final-test use and G10 limits** — **High value**

## Base and Extended reproduction

- [x] **07 — Train Singh Base G0 on one million examples** — **High value**
- [x] **08 — Validate Base held-out in-context competence** — **High value**
- [x] **09 — Perform Base head interventions** — **High value**
- [x] **10 — Implement three Extended autoregressive targets** — **High value**
- [x] **11 — Audit Extended gradients, oracle and causal ordering** — **High value**
- [x] **12 — Reconcile Base versus Extended exposure budgets and OOD class-ID limitations** — **Medium value**

## Recursive pipelines

- [x] **13 — Train Base through G10** — **High value**
- [x] **14 — Train Extended through G10** — **High value**
- [x] **15 — Include clean-only control** — **High value**
- [x] **16 — Include 50/50 real-anchor control** — **High value**
- [x] **17 — Include frozen-G0 synthetic control** — **Medium value**
- [x] **18 — Test genuine historical accumulation or justify omission** — **Medium value**

## Behavior and distribution

- [x] **19 — Consolidate eight G1-G10 validation curves** — **High value**
- [x] **20 — Measure Base G10 degradation versus clean** — **High value**
- [x] **21 — Separate teacher-forced and free-running Extended outcomes** — **High value**
- [x] **22 — Measure contextual q2 validity and combined label correctness** — **High value**
- [x] **23 — Measure global q2 entropy and concentration** — **Medium value**
- [x] **24 — Aggregate paired replicated effect sizes and uncertainty** — **High value**

## Mechanistic interpretation

- [x] **25 — Compare Base head circuits G0/G3/G6/G10** — **High value**
- [x] **26 — Compare Extended seven-token circuits G0/G3/G6/G10** — **High value**
- [x] **27 — Verify head ablations as causal interventions** — **High value**
- [x] **28 — Interpret head reorganization versus fixed head decay** — **High value**
- [ ] **29 — Test additional circuit-level mediation** — **Medium value**
- [x] **30 — Explain persistence of numeric ICL despite invalid q2** — **High value**

## Target correction

- [x] **31 — Audit G9 wrong first labels, invalid q2 and wrong second labels** — **High value**
- [x] **32 — Match raw G9 to G10 rerun control** — **High value**
- [x] **33 — Train first-label correction** — **Low value**
- [x] **34 — Train valid-second-label correction** — **Low value**
- [x] **35 — Train invalid-q2 and second-label, joint, and full repairs** — **High value**
- [x] **36 — Independently replicate effective repair on new G10 seeds** — **High value**

## Replication

- [x] **37 — Run two independent Extended G0-G10 seeds** — **High value**
- [x] **38 — Statistically analyze completed Extended seed outputs** — **High value**
- [x] **39 — Finish two independent Base G0-G10 seeds** — **High value**
- [x] **40 — Unit-test aggregation and test-isolation logic** — **High value**
- [x] **41 — Finish seed-paired raw/repair experiments** — **High value**
- [x] **42 — Freeze qualified conclusions without reused test** — **High value**

## Writing and submission

- [x] **43 — Resolve precise scientific research question** — **High value**
- [x] **44 — Finalize reproducible figures and uncertainty** — **High value**
- [ ] **45 — Review stacked PRs, dependencies, provenance, merges** — **High value**
- [x] **46 — Write two-page extended abstract PDF** — **High value**
- [ ] **47 — Provide contributions, NeurIPS ethics and Faculty AI statement** — **High value**
- [ ] **48 — Package code, README.txt, requirements.txt and submit** — **High value**

**Current checked count: 44/48. Remaining #29, #45, #47 and #48 require further action.** A completed GitHub Actions job establishes execution, not automatically a repeated-seed causal conclusion.

## Overnight follow-ups beyond 48

- [x] **HIGH:** Complete independent Base seeds 38642335173 and three-seed G10 aggregation 38648539187. Per-seed results permanently saved.
- [x] **HIGH:** Run independent G9->G10 invalid-symbol-plus-label repairs and save per-seed outcome files. Successful run 38648489063; seed-level results are committed to `docs/experiments/independent_repairs/`.
- [x] **HIGH:** Source, experimental results, provenance and SHA256 metrics stored in Git and persistent checkpoint releases (primary and follow-up).
- [x] **LOW:** Exploratory G100 run 38648425960 completed, kept outside full-budget scientific argument.

## Evidence and warnings

- Base G10: https://github.com/huziibee/nlp_ass_one/actions/runs/38632979396
- Extended G10: https://github.com/huziibee/nlp_ass_one/actions/runs/38633208767
- Six repairs: https://github.com/huziibee/nlp_ass_one/actions/runs/38641963903
- Extended new seeds: https://github.com/huziibee/nlp_ass_one/actions/runs/38641472801
- Base new seeds: https://github.com/huziibee/nlp_ass_one/actions/runs/38642335173
- Authoritative findings: https://github.com/huziibee/nlp_ass_one/blob/extended-correctness-and-g10-analysis/docs/EXTENDED_CORRECTNESS_AND_G10_FINDINGS.md

**Do not conflate:** target-only synthetic feedback and full context generation; 50/50 mixing and historical accumulation; teacher forcing and free generation; fixed-index head decay and entire circuit collapse. The G3 final test was already used and G10/G100 must remain validation-only.

## Newly confirmed 2-seed repair findings (run 38648489063)

Both independent G10 repair comparisons completed at 300,000 training episodes per condition on the original Omniglot features. Fixing invalid q2 **AND its corresponding numeric label** improved free-generated contextual validity from 77.93% to 97.85% (seed 5054) and 80.27% to 97.46% (seed 6054). Paired mean improvement: **18.55 percentage points** over these two seeds. Valid-symbol-plus-correct-second-label consistency also improved, mean **19.53 percentage points**. This is *descriptive n=2 replication*, not a population confidence claim, and does not isolate q2 identity from associated-label repair. Source: `docs/experiments/independent_repairs/summary.json`.

## 15 October independent Base/Extended G10 confirmation update

Full matched runs for Base seeds 505/605 and Extended seeds 5054/6054, combined with the original primary seeds, have succeeded. The G10 multi-seed aggregation workflow [38648539187](https://github.com/huziibee/nlp_ass_one/actions/runs/38648539187) committed actual per-seed tables, paired effects and an indicative n=3 paired-t confidence interval to `docs/experiments/g10_three_seed/`.

- Base G10 held-out label pair clean minus recursive effect: **+8.53 pp**, indicative 95% paired-t interval **[3.67, 13.39] pp**.
- Extended freely generated contextual q2 validity clean minus recursive effect: **+17.06 pp**, interval **[9.63, 24.48] pp**.
- Extended first numeric label is not uniformly preserved: paired changes are highly seed-dependent, and recursive models outperform clean on two of the three seeds; no universal numerical copying claim.
- Primary seed selected head-matching metrics change in clean as well as recursive retraining; *head-index decay* does not establish unique full-circuit degradation.
- The precise working scientific question is whether recursive generated continuations preferentially degrade contextual symbol validity relative to numeric label prediction and whether correcting malformed (q2,number) targets restores the behavior; this is now supported by three-seed observational and two-seed repair comparisons, with explicit limitations.

The additional items checked here (#24, #28, #30, #38, #39, #43) reflect measured completed seed aggregation or **bounded evidence-based interpretations**, not claims of new independent G10 test data or completed circuit mediation. The strictly two-page extended abstract has been **drafted locally using the official template** but requires official ethics/contribution forms and author review; report/submission items remain unchecked.


## 15 October — additional source-verified scientific experiments

- [x] **HIGH — Three-seed G1-G10 onset and all role evaluation**: successful run [38655681721](https://github.com/huziibee/nlp_ass_one/actions/runs/38655681721), source tables `docs/experiments/all_seed_generations/`. Descriptive 5pp gap appears Base G7/G6/G6 and Extended G2/G3/G2; extended exemplar gap +8.14 pp mean across 3 seeds, lower than label-pair gap +17.06 pp. Threshold post hoc; no formal changepoint.
- [x] **HIGH — Actual multi-head clean-recursive causal ablation**: successful [38655884354](https://github.com/huziibee/nlp_ass_one/actions/runs/38655884354), `docs/experiments/extended_joint_heads/`. G0-trained top L2 heads 1/0/7. Triple ablation first-number accuracy loss clean G10 51.56 pp vs recursive G10 10.16 pp; contextual valid next-symbol probability-mass losses clean 10.12 pp and recursive 17.43 pp. **Does not prove circuit mediation**, so core roadmap #29 remains open.
- [x] **HIGH — Graded 0/25/50/75/100% invalid q2+linked-number G10 repair**: successful [38655623103](https://github.com/huziibee/nlp_ass_one/actions/runs/38655623103), `docs/experiments/graded_repair/`; validity 82.81/85.55/87.89/92.19/97.66%, single model seed for intermediate doses.
- [x] **HIGH — Field-separated factorial q2-only vs number-only vs both repair**: successful [38656350306](https://github.com/huziibee/nlp_ass_one/actions/runs/38656350306), `docs/experiments/factorial_invalid_repair/`; raw 82.81%, q2-only 97.27%, number-only 82.03%, joint 97.66% contextual validity; raw 81.45%, q2-only 82.03%, number-only 74.02%, joint 95.70% symbol+label correctness. Number-only is deliberately inconsistent (invalid q2 has no valid contextual label). All 300k exposures, n=1 seed.
- [x] **HIGH — Growing historical corpus**: successful [38655888849](https://github.com/huziibee/nlp_ass_one/actions/runs/38655888849), `docs/experiments/cumulative_corpus/`. Retained cumulative real-context/synthetic-target examples vs growing clean history. G10 clean 87.11% vs accumulated synthetic 83.20% valid next symbol, clean 82.03% vs cumulative 70.70% joint correctness, 100k/g, 1 seed. **Not fully synthetic input prompts**.
- [x] **MEDIUM-HIGH — Matched G1-G10 Base descendant exposures 300k**: successful [38655728846](https://github.com/huziibee/nlp_ass_one/actions/runs/38655728846), `docs/experiments/base_budget_matched/`; clean Base G10 76.95%, recursive 50.78%, compared with original 1M/g 99.41% and 92.38%. Base G0 remains 1M in this pilot, so not fully matched.
- [x] **HIGH — Fully 300k exposure-matched Base G0 through G10**: successful run [38656720477](https://github.com/huziibee/nlp_ass_one/actions/runs/38656720477); clean G10 76.95%, recursive G10 49.41%, one seed. Still not architecture/output matched.
- [x] **HIGH — Persistent original checkpoints**: verified original [GitHub Release](https://github.com/huziibee/nlp_ass_one/releases/tag/nlp-assignment1-original-omniglot-checkpoints-20261015).
- [x] **HIGH — Persistent high-value followup checkpoint backup**: verified [GitHub Release](https://github.com/huziibee/nlp_ass_one/releases/tag/nlp-assignment1-mechanistic-followups-20261015), includes successfully completed source-run artifacts and SHA256 manifests. The stricter 300k G0-through-G10 Base training checkpoints were added to this release by successful run 38657625854.

**Methodological boundary**: New experimental work is substantial and mostly supported by real saved outcomes, but the strict two-main-page LaTeX paper, official NeurIPS 2024 Moodle questionnaire, Faculty AI Ethics Statement, verified contributions and final code submission remain incomplete. No later final-test reuse.


## Final research close-out and submission blockers — 15 October 2026

- [x] **#12 (Medium):** Budget audit finished. Full 300k-example Base G0 plus every G1–G10 descendant (run 38656720477) produces 76.95% clean versus 49.41% recursive accuracy at G10. Not directly architecture-comparable with Extended; training/class-ID limitation disclosed.
- [x] **#42 (High):** Scientifically qualified conclusions frozen: the repeatable primary effect is loss of contextually valid sampled q2 under recursive target-only replacement; target-level symbol-only rescue is shown in one seed and coherent symbol-plus-label rescue in two independent additional seeds; head interventions demonstrate changed contribution without unique circuit mediation. No G10 final-test reuse.
- [x] **#44 (High):** Audited figures and independent-seed uncertainty compiled. Generational trajectory, factorial correction matrix, graded-repair figure, and multi-head intervention table come from archived numerical evidence.
- [x] **#46 (High):** Strict two-page main-text PDF compiled with lecturer-provided CCN LaTeX template; references, group contribution placeholders and mandatory ethics placeholders follow the main text on another page. Compiled PDF and LaTeX source are delivered separately. **Not submission-ready until declarations are completed.**
- [ ] **#29 (Medium):** Full path-level causal mediation, e.g. activation patching, still not executed. Matched causal joint-head knockouts are completed but are not equivalent to an identified training-data-to-circuit mediator.
- [ ] **#45 (High):** Four stacked draft PRs #2, #3, #4 and #5 still require independent code and licensing review, consistency checks and sequential safe merge. Do not imply main is fully merged.
- [ ] **#47 (High):** Complete the original Moodle NeurIPS 2024 questionnaire (.tex), official Faculty of Science AI Ethics Statement and individual group contributions confirmed by all three students. Missing ethics is an explicit not-marked condition.
- [ ] **#48 (High):** Full clean environment installation via requirements.txt, README.txt, licensed source and checkpoint/data availability; final author sign-off and actual Moodle upload by 27 October 2026, 17:00 SAST. Submission has not occurred.

**Status: 44/48 completed; 4 not yet verifiable.** This reconstructed research roadmap is not an official claim of submitted work. Full-budget model snapshots verified at https://github.com/huziibee/nlp_ass_one/releases/tag/nlp-assignment1-original-omniglot-checkpoints-20261015 and https://github.com/huziibee/nlp_ass_one/releases/tag/nlp-assignment1-mechanistic-followups-20261015.

## Submission preflight status (15 October 2026)

- [x] **HIGH — Clean-room dependency and tests check:** GitHub Actions run [38694067217](https://github.com/huziibee/nlp_ass_one/actions/runs/38694067217) succeeded on Ubuntu, Python 3.10, using the actual `requirements.txt`. It checked pinned original data, source split, repository unit tests and saved multi-seed/repair/head/budget/cumulative science summaries without reopening the final test.
- [x] **HIGH — Up-to-date README.md, README.txt and assignment compliance status:** these files on the research branch now explicitly identify the complete G10 findings, persistence releases and ethics/submission blockers. Older historical sections remain below the new verified status headers.
- [x] **HIGH — Original one-time G3 test correctly reported in paper:** frozen 4,096-example original G3 test (Base clean/recursive 93.60/92.70% first label; Extended clean/recursive 95.92/98.32% first label). Source `docs/experiments/archived_original_runs/historical_frozen_g3_test/summary.json`. No subsequent G10 test-reuse claim.
- [x] **HIGH — Two scientific pages verified:** final LaTeX/CCN paper renders main text on pages 1 and 2 and references/form placeholders on page 3; no truncation/overfull box warnings. This PDF is a **draft until exact mandatory forms and author approvals are present**.

Remaining open roadmap numbers are **29** (optional neural-path mediation), **45** (four stacked draft PRs require manual code/provenance review before any merge), **47** (Moodle-provided NeurIPS ethics, Faculty AI ethics, confirmed contributions), and **48** (final code+requirements+README+approved PDF uploaded to Moodle). A passing clean-room workflow is not a substitute for author approval or official ethics paperwork.
