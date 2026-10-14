# Train–validation–test design for NLP Project 1

The lecturer's PDF (p. 5) explicitly requires a **train–validation–test data split** and careful scientific use of results. The reproduced Singh authors' main training command is at `external/icl-dynamics/ih_paper_runs.sh`. The implementation is authoritative for these split settings, not a generic ML convention.

## Three separate axes, not a simple 80/10/10 sample division

1. **Identity/class split** from `--class_split 50 1473 100`:
   - Train: HDF5 rows **0–49** (50 classes).
   - Val reserve: rows **50–1522** (1,473 classes).
   - Test reserve: rows **1523–1622** (100 classes).
2. **Exemplar split** from `--exemplar_split 1 4 0`:
   - Train: the first feature vector (exemplar 0) of every eligible class.
   - Validation: held-out exemplars 1, 2, 3, 4.
   - Test: no further exemplar-specific slice.
3. **Contextual label-pair split** from `--fs_relabel 5 --fs_relabel_split 8 2 0`:
   - Five labels give ten possible unordered label pairs. Eight pairs appear in training; two are held out as validation mappings. There is no *additional* test-pair partition.
   - The actual JAX key 20 permuted pairs are:
     - Train: (0,4), (1,2), (0,1), (2,4), (0,2), (1,3), (0,3), (1,4)
     - Val: (2,3), (3,4)
     - Test: none
   - Both the symbol ordering and which member gets which label can vary within an episode.

## What the original researchers actually evaluated

The four named evaluators in the published command are:
- `fsl_train`: training classes + training exemplars + training label pairs
- `fsl_val_rl`: training classes + training exemplars + held-out **label pairs**
- `fsl_train_valex`: training classes + held-out **exemplars** + training label pairs
- `fsl_test_class`: **test classes** + training exemplars + training label pairs

**Source-level caution:** The 1,473-class validation reserve is not used by the published command's four named evaluators. Its `fsl_test_class` is run periodically and therefore must **not** be treated as an untouched final test if its scores guide tuning.

## Course-safe experimental split (proposed; not yet trained)

We will explicitly assign:
- Training: 50 training classes, exemplar 0, eight training label pairs.
- Validation A: same 50 classes, held-out exemplar 1–4, training label pairs.
- Validation B: same 50 classes, exemplar 0, the two held-out label pairs.
- Validation C: **1,473 val classes**, exemplar 0, training label pairs.
- **Sealed final test:** only the 100 reserved test classes, exemplar 0, training label pairs, examined **once** after protocol/hyperparameter/head-selection decisions have been frozen.

Other final stress tests (e.g., test classes with novel exemplars or held-out label pairs) must be declared before any access to test metrics. We do **not** claim that this project already trained or evaluated this plan.

### What is allowed during development

It is okay to inspect *training and validation* geometry or behaviour while building the model and selecting hyperparameters. It is NOT okay to inspect final test accuracy, select heads on final test, or modify the hypothesis after learning final results and still claim independent confirmation.

The existing real-data Figure 2/5 exploration used all stored feature classes, including those later designated test classes; that analysis is **pretraining descriptive geometry, not model testing**, but we should avoid additional such analyses on reserved test classes. Our new split-aware diagnostic restricts feature access to training and validation classes.

## Implementation safeguards

- `scripts/reference_split.py` parses the authors' exact main-run command and asserts the three split flags and evaluator assignments have not drifted.
- `scripts/episode_probe.py` generates 2-way, 1-shot contextual-label training/development episodes with explicit oracle checks. Its command line has no final-test sampling option, and the underlying API explicitly guards final-test access.
- `scripts/split_diagnostics.py` (run in GitHub Actions) scores only train and validation feature subsets.
- `tests/test_reference_split.py` asserts class, exemplar and relabel pair disjointness, episode validity, and a final-test access guard.
- `docs/data-audit/reference_split_manifest.json` stores the actual parsed split, and `docs/data-audit/SPLIT_AWARE_ANALYSIS.md` gives measured validation-only feature geometry.

**A dataset split alone does not guarantee model correctness.** A trained checkpoint needs an independently verified split-aware evaluation script, appropriate training/validation loss and accuracy logging, and a one-time final evaluation.
