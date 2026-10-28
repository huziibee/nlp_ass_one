# Where Does Recursive Feedback Fail?

This project studies recursive feedback in an Omniglot in-context learning task. The Base model predicts a query character's numerical label. The Extended model also generates another character and its label. Each new model learns from its predecessor's generated answers; the input character features remain real.

The data are 512-dimensional ResNet-18 features supplied by the external Singh et al. repository. There are 50 training classes, 1,473 validation classes and 100 reserved test classes. The character output head covers the 50 training identities. The main experiment trains five seed pairs for each task, with 300,000 episodes per generation, comparing correct targets with recursive targets from G0 through G10. All reported G10 evaluations use validation data. An earlier evaluation accessed the final-test classes, so these runs are not an independent final-test evaluation.

## Install

Use Python 3.10 or 3.11 on Linux or WSL. Run every command from the repository root.

```bash
git submodule update --init --recursive
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py check
python run.py data-check
```

The submodule provides the original transformer code and the Omniglot feature file. The project creates its own 300k Base checkpoint; no one-million-episode checkpoint is needed.

## 1. Run the main experiment

```bash
python -u scripts/run_paper.py train
```

This trains the five Base and five Extended seed pairs from G0 through G10. Both tasks include a real-target control and recursive training. G0 learns correct answers; G1–G10 learn either correct or predecessor-generated answers. All generations belong to the same experiment. Intermediate checkpoints stay inside that run; no separate G3 setup is needed.

This stage saves model checkpoints, validation metrics and generation histories under `outputs/`. Existing completed runs are checked and reused. The main experiment must finish before the five-seed table and figure can be produced.

## 2. Make Table 1 and Figure 1

```bash
python scripts/run_paper.py results
```

This reads the new experiment outputs, checks all five seed pairs, computes the clean-minus-recursive gaps and paired descriptive 95% intervals, and draws the figure. It does not substitute archived results for missing runs.

The report also lists each Extended seed's numerical retrieval and held-out handwriting results, so uneven baselines remain visible alongside the main averages. The headline comparisons use held-out label assignments; handwriting exemplars are an additional robustness check mentioned in Section 8 of the paper. Extended character validity and numerical-label accuracy are different metrics.

| Output | Contents |
| --- | --- |
| `outputs/paper_results/table1.csv` | Table 1 values from the fresh runs |
| `outputs/paper_results/summary.json` | Main effects and experiment metadata |
| `outputs/paper_results/input_manifest.csv` | Input files and checksums |
| `outputs/figure1.png` | Paired Base G10 accuracy and Extended G1–G10 sampled-symbol validity |
| [outputs/evidence/main-results/README.md](outputs/evidence/main-results/README.md) | Readable main-results report and figure link |

## 3. Run each diagnostic or ablation

These commands run the named analysis and then format its evidence as a Markdown report and CSV table under `outputs/evidence/`. Each report includes measured values, input paths, definitions and limitations. The raw JSON outputs are retained for inspection. The one-seed probes compare the primary clean and recursive runs at G0 and G10. They do not require a separately trained G3 model.

Open [outputs/evidence/README.md](outputs/evidence/README.md) for the report index. A partial report lists the observed runs and the inputs still missing; it is not counted as a completed five-seed result.

| Paper component | Command | Required completed work | Evidence report |
| --- | --- | --- | --- |
| Section 6: context-valid probability, off-context mass and entropy | `python scripts/distribution_audit.py` | Main training; primary Extended checkpoints | [outputs/evidence/distribution/README.md](outputs/evidence/distribution/README.md) |
| Section 7: G0 head selection and individual attention-head ablations | `python scripts/head_ablation.py` | Main training; primary Extended checkpoints | [outputs/evidence/head-ablation/README.md](outputs/evidence/head-ablation/README.md) |
| Section 7: joint attention-head removals | `python scripts/joint_ablation.py` | Main training; primary Extended checkpoints | [outputs/evidence/joint-ablation/README.md](outputs/evidence/joint-ablation/README.md) |
| Section 7: response to edited support symbols and labels | `python scripts/support_edits.py` | Main training; primary Extended checkpoints | [outputs/evidence/counterfactual/README.md](outputs/evidence/counterfactual/README.md) |
| Section 6: five matched joint target repairs and G9 target-error counts | `python scripts/target_repair.py` | Main training; recursive G9 and G10 for each seed | [outputs/evidence/repair/README.md](outputs/evidence/repair/README.md) |
| Section 7: within-model activation patching before and after repair | `python scripts/context_patching.py` | Main training and target repair | [outputs/evidence/patching/README.md](outputs/evidence/patching/README.md) |
| Table 2: symbol-only, number-only and joint target repairs | `python scripts/factorial_repair.py` | Main training and primary joint target repair | [outputs/evidence/factorial-repair/README.md](outputs/evidence/factorial-repair/README.md) |
| Sections 3, 4 and 8: fixed-G0, half-real, learning-rate and accumulating-corpus controls | `python scripts/run_controls.py` | Main training; primary Extended G0 | [outputs/evidence/controls/README.md](outputs/evidence/controls/README.md) |

Run target repair before patching or factorial repair. The accumulating-corpus pilot retains its separate 100,000-episode budget, as described in the paper. Held-out label, class and handwriting-exemplar scores are stored in each model's validation metrics.

## Progress and rerunning

```bash
python scripts/run_paper.py status
```

The progress view groups work by the experiments above and describes the current stage. A pending experiment means its outputs are not ready yet. It does not mean the Python implementation is absent. To see the underlying commands before running a stage:

```bash
python scripts/run_paper.py train --dry-run
python scripts/head_ablation.py --dry-run
```

Logs are saved in `outputs/logs/`, with execution records in `outputs/run_status.json`. Rerunning a command skips verified complete work. If a run was interrupted, inspect its log first; `--restart-incomplete` archives that stage's partial directory under `unneeded/interrupted_runs/` and restarts it.

To run the whole sequence in order—training, Table 1 and Figure 1, then the analyses and their reports:

```bash
python -u scripts/run_paper.py all
```

Historical result data live in `data/evidence/`. `python run.py results` formats those recorded results separately under `outputs/reports/`. The main experiment and evidence commands above use fresh local outputs. Archived documentation, workflows, old experiments and tests are under `unneeded/` and are not submission entry points.

## Verification and recorded results

[Final verification](outputs/verification.json) records the completed stages, checkpoint count, input checksums, report-link checks and test result. Fresh rerun measurements can differ from the paper's recorded numbers. The evidence reports preserve the fresh measurements; the paper text is not automatically changed.

## Submission ZIP

Submit `nlp_ass_one_submission.zip`. It contains `README.md`, `requirements.txt`, `run.py`, `scripts/`, `src/`, `outputs/`, `data/evidence/` and the original external checkout. It excludes `unneeded/`, `read/`, Git metadata, virtual environments and Python caches. The repository retains the archived material separately.

The ZIP already includes the external code and feature data. After extracting it, skip the `git submodule update` command and start with virtual-environment creation and dependency installation.
