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
- [ ] **12 — Reconcile Base versus Extended exposure budgets and OOD class-ID limitations** — **Medium value**

## Recursive pipelines

- [x] **13 — Train Base through G10** — **High value**
- [x] **14 — Train Extended through G10** — **High value**
- [x] **15 — Include clean-only control** — **High value**
- [x] **16 — Include 50/50 real-anchor control** — **High value**
- [x] **17 — Include frozen-G0 synthetic control** — **Medium value**
- [ ] **18 — Test genuine historical accumulation or justify omission** — **Medium value**

## Behavior and distribution

- [x] **19 — Consolidate eight G1-G10 validation curves** — **High value**
- [x] **20 — Measure Base G10 degradation versus clean** — **High value**
- [x] **21 — Separate teacher-forced and free-running Extended outcomes** — **High value**
- [x] **22 — Measure contextual q2 validity and combined label correctness** — **High value**
- [x] **23 — Measure global q2 entropy and concentration** — **Medium value**
- [ ] **24 — Aggregate paired replicated effect sizes and uncertainty** — **High value**

## Mechanistic interpretation

- [x] **25 — Compare Base head circuits G0/G3/G6/G10** — **High value**
- [x] **26 — Compare Extended seven-token circuits G0/G3/G6/G10** — **High value**
- [x] **27 — Verify head ablations as causal interventions** — **High value**
- [ ] **28 — Interpret head reorganization versus fixed head decay** — **High value**
- [ ] **29 — Test additional circuit-level mediation** — **Medium value**
- [ ] **30 — Explain persistence of numeric ICL despite invalid q2** — **High value**

## Target correction

- [x] **31 — Audit G9 wrong first labels, invalid q2 and wrong second labels** — **High value**
- [x] **32 — Match raw G9 to G10 rerun control** — **High value**
- [x] **33 — Train first-label correction** — **Low value**
- [x] **34 — Train valid-second-label correction** — **Low value**
- [x] **35 — Train invalid-q2 and second-label, joint, and full repairs** — **High value**
- [x] **36 — Independently replicate effective repair on new G10 seeds** — **High value**

## Replication

- [x] **37 — Run two independent Extended G0-G10 seeds** — **High value**
- [ ] **38 — Statistically analyze completed Extended seed outputs** — **High value**
- [ ] **39 — Finish two independent Base G0-G10 seeds** — **High value**
- [x] **40 — Unit-test aggregation and test-isolation logic** — **High value**
- [x] **41 — Finish seed-paired raw/repair experiments** — **High value**
- [ ] **42 — Freeze qualified conclusions without reused test** — **High value**

## Writing and submission

- [ ] **43 — Resolve precise scientific research question** — **High value**
- [ ] **44 — Finalize reproducible figures and uncertainty** — **High value**
- [ ] **45 — Review stacked PRs, dependencies, provenance, merges** — **High value**
- [ ] **46 — Write two-page extended abstract PDF** — **High value**
- [ ] **47 — Provide contributions, NeurIPS ethics and Faculty AI statement** — **High value**
- [ ] **48 — Package code, README.txt, requirements.txt and submit** — **High value**

**Current checked count: 33/48.** A completed GitHub Actions job establishes execution, not automatically a repeated-seed causal conclusion.

## Overnight follow-ups beyond 48

- [ ] **HIGH:** Finish independent Base seeds (run 38642335173) and execute Base/Extended three-seed G10 aggregation.
- [x] **HIGH:** Run independent G9->G10 invalid-symbol-plus-label repairs and save per-seed outcome files. Successful run 38648489063; seed-level results are committed to `docs/experiments/independent_repairs/`.
- [ ] **HIGH:** Store research source, experiment provenance, result CSVs, JSON manifests and conclusions in the Git repository. GitHub Actions checkpoint artifacts expire.
- [ ] **LOW:** Bounded exploratory 100-generation Extended experiment, 512 steps x 32 sequences per generation and matched clean/recursive controls; this is NOT comparable to full-budget 300k/descendant G10.

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
