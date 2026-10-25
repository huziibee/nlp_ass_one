# Matched one-million-episode Base and Extended experiment

**Goal:** Remove unequal training-*episode counts* from the primary between-task comparison while preserving every original observation. The already recorded study remains Base = 1,000,000 episodes and Extended = 300,000. This **new study** compares Base = 1,000,000 and Extended = 1,000,000 at G0 and at each descendant G1–G10.

## What we are doing (and not doing)

- Keep the five independently initialized **Base G0 and G1–G10 1M checkpoints** from fresh run [40194253117](https://github.com/huziibee/nlp_ass_one/actions/runs/40194253117), with its exact archived source commit and original 50/1,473/100 class split.
- Train **five new Extended G0 models from random weights** for a full 1,000,000 episodes each (31,250 updates of 32). A Base G0 artifact supplies the matching dataset/architecture *configuration only*; no trained Base parameters initialize an Extended model.
- For each Extended seed, train one `clean_only` and one `recursive_replacement` arm for ten full generations, **1,000,000 episodes per generation**. All five seeds are retained regardless of G0 quality. That is ten new Extended G1–G10 chains (100 new descendant models), plus five new Extended G0 models.
- Reuse the previous five Base clean and five Base recursive full training chains for comparison (100 historical-*fresh-run* Base descendants); these are newly trained in the referenced run, not the earlier 2026 submission-study checkpoints.
- Independently check the recorded 210 model checkpoint files, 20 paired Base/Extended G10 condition records, training exposure counts and zero final-test access. Store exact SHA-256 hashes and both run IDs, never overwrite earlier results.

This isolates the difference in **training example exposures**. Base still predicts one label and Extended predicts three tokens. Consequently it **does not** equate architecture, output loss, task difficulty, optimizer cost or convergence. The main scientifically controlled contrasts remain *clean vs recursive within each task*.

## Execution

- Training workflow: [`.github/workflows/matched-million-extended.yml`](.github/workflows/matched-million-extended.yml).
- Protocol and metric verifier: [`scripts/fresh_five_seed.py`](scripts/fresh_five_seed.py), with `--extended-exposures 1000000`.
- Reproducibility tests: [`tests/test_fresh_five_seed.py`](tests/test_fresh_five_seed.py).

Inspect the study design without starting expensive training:

```bash
python run.py matched-plan
python run.py tests
```

The workflow starts on the feature-branch workflow addition or via a manual dispatch on `main` once the reviewed code has merged:

```bash
gh workflow run matched-million-extended.yml --ref main
```

The GitHub Actions job stages are: `preflight`, `extended-g0-one-million` (5 seeds), `extended-arms-one-million` (10 runs), `aggregate-matched-one-million` (checks all 210 original-new checkpoints). All inputs in the final aggregator must be from the declared source Base run and the current matched Extended run. This is a compute-intensive workflow and may use paid GitHub Actions minutes.

The new outputs are stored **separately** as a `matched-one-million-five-seed-results` artifact. A successful aggregation yields:

```text
fresh_five_seed_protocol.json
fresh_five_seed_summary.json
fresh_five_seed_rows.csv
fresh_five_seed_checkpoint_manifest.csv
```

Do **not** claim these results are complete merely because the workflow file exists or the first G0 job passes. Completion requires all 5 Extended G0, all 10 Extended arms, access to the 5 original-fresh Base G0 and 10 original-fresh Base arms, and a successful final aggregator validating 210 checkpoints. Preserve its source run, matched run and commit hashes.

After restoring the full set of per-run artifacts under `outputs/matched_five_inputs/` with the naming in `FRESH_FIVE_SEED_REPRODUCTION.md`, independently check them:

```bash
python run.py matched-aggregate
```

## Compact manuscript treatment

Before the results are complete, **do not** replace the original 300K Extended statistics with hypothetical 1M outcomes. When the matched study is complete, prefer a concise sentence in Methods and a concise caveat in Discussion:

**Methods** — “To control for differing training exposure, we additionally train matched Base and Extended cohorts with one million episodes per generation, comparing clean and recursive targets within each task over five model seeds.”

**Discussion** — “Matched-example-budget analyses remove an exposure-count difference but not the tasks’ distinct output spaces or optimization difficulty; we therefore do not interpret effect-size differences as purely architectural susceptibility.”

Insert actual G10 effects and relevant uncertainty only after the completed job verifies them. Do not retrospectively present the new run as part of an earlier frozen five-seed set, and do not discard original anomalous seeds or results.

## Explicit validity boundaries

1. The original 100-class final test was opened at G3. This run is **validation-only** and cannot recreate an untouched final test set.
2. Both models see the same **number of training examples**, not equal gradient FLOPs or converged optimization. Extended learns a three-output autoregressive task.
3. The synthetic target-feedback procedure keeps real Omniglot feature contexts; it does not synthesize full input sequences.
4. Head-level or causal circuit interpretations require separate appropriate interventions and do not follow from equal budgets alone.
