"""Audit the literal Singh et al. main-run splits; never execute final test evaluation.

Source of truth: pinned external/icl-dynamics/ih_paper_runs.sh and
external/icl-dynamics/main_utils.py; fixed label-pair permutation is the
JAX PRNGKey(20) outcome for 5 labels, verified against the upstream algorithm.
"""
from __future__ import annotations
import argparse
import itertools
import json
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_RUNS = ROOT / "external" / "icl-dynamics" / "ih_paper_runs.sh"
TRAIN_PAIRS = ((0, 4), (1, 2), (0, 1), (2, 4),
               (0, 2), (1, 3), (0, 3), (1, 4))
VALIDATION_PAIRS = ((2, 3), (3, 4))
ALL_PAIRS = tuple(itertools.combinations(range(5), 2))

# Original figure 3a paper configuration (not an invented random split).
EXPECTED = {
    "class_split": (50, 1473, 100),
    "exemplar_split": (1, 4, 0),
    "fs_relabel_split": (8, 2, 0),
    "fs_relabel": (5,),
    "train_context_len": (2,),
}
EVAL_NAMES = ("fsl_train", "fsl_val_rl", "fsl_train_valex", "fsl_test_class")
EVAL_CLASSES = ("train", "train", "train", "test")
EVAL_EXEMPLARS = ("train", "train", "val", "train")
EVAL_RELABEL = ("train", "val", "train", "train")


def original_run_flags(path: Path = ORIGINAL_RUNS) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Upstream reproduction script missing: {path}. "
            "Run: git submodule update --init --recursive"
        )
    matches = [line.strip() for line in path.read_text(encoding="utf8").splitlines()
               if line.strip().startswith("python main.py ") and
               "--run omniglot50_rl5 " in line and
               "--opto_ablate_heads" not in line]
    if len(matches) != 1:
        raise ValueError(f"Expected unique published main run; found {len(matches)}")
    tokens = shlex.split(matches[0])
    result = {}
    options = {
        **{k: "--" + k for k in EXPECTED},
        "pe_names": "--pe_names",
        "pe_classes": "--pe_classes",
        "pe_exemplars": "--pe_exemplars",
        "pe_fs_relabel_scheme": "--pe_fs_relabel_scheme",
    }
    for k, flag in options.items():
        if flag not in tokens:
            raise AssertionError(f"Upstream script lacks {flag}")
        start = tokens.index(flag) + 1
        vals = []
        for value in tokens[start:]:
            if value.startswith("--"):
                break
            vals.append(value)
        result[k] = tuple(int(x) for x in vals) if k in EXPECTED else tuple(vals)
    for key, expected in EXPECTED.items():
        if result[key] != expected:
            raise AssertionError(f"{key}: upstream {result[key]}, expected {expected}")
    for key, expected in [
        ("pe_names", EVAL_NAMES),
        ("pe_classes", EVAL_CLASSES),
        ("pe_exemplars", EVAL_EXEMPLARS),
        ("pe_fs_relabel_scheme", EVAL_RELABEL),
    ]:
        if result[key] != expected:
            raise AssertionError(f"{key}: upstream {result[key]}, expected {expected}")
    return result


def get_class_ids(split: str):
    # These are indices of the original pre-shuffled HDF5 classes.
    if split == "train":
        return tuple(range(0, 50))
    if split == "val":
        return tuple(range(50, 1523))
    if split == "test":
        return tuple(range(1523, 1623))
    raise ValueError(f"Unknown split {split!r}")


def get_exemplar_ids(split: str):
    if split == "train":
        return (0,)
    if split == "val":
        return (1, 2, 3, 4)
    if split == "test":
        return ()
    raise ValueError(f"Unknown exemplar split {split!r}")


def get_label_pairs(split: str):
    if split == "train":
        return TRAIN_PAIRS
    if split == "val":
        return VALIDATION_PAIRS
    if split == "test":
        return ()
    raise ValueError(f"Unknown relabel split {split!r}")


def validate_reference_split():
    cls = [set(get_class_ids(k)) for k in ("train", "val", "test")]
    ex = [set(get_exemplar_ids(k)) for k in ("train", "val", "test")]
    pairs = [set(get_label_pairs(k)) for k in ("train", "val", "test")]
    for sets in (cls, ex, pairs):
        for i in range(3):
            for j in range(i + 1, 3):
                assert not (sets[i] & sets[j]), "Split leakage"
    assert set.union(*cls) == set(range(1623))
    assert set.union(*ex) == set(range(5))
    assert set.union(*pairs) == set(ALL_PAIRS)
    assert [len(s) for s in cls] == [50, 1473, 100]
    assert [len(s) for s in ex] == [1, 4, 0]
    assert [len(s) for s in pairs] == [8, 2, 0]
    assert set(range(5)) <= {i for pair in TRAIN_PAIRS for i in pair}


def build_manifest(source: Path = ORIGINAL_RUNS) -> dict:
    validate_reference_split()
    flags = original_run_flags(source)
    evals = [
        {"name": name, "class_subset": cls, "exemplar_subset": ex,
         "label_pair_subset": pair}
        for name, cls, ex, pair in
        zip(EVAL_NAMES, EVAL_CLASSES, EVAL_EXEMPLARS, EVAL_RELABEL)
    ]
    return {
        "source": "pinned upstream ih_paper_runs.sh, published main run",
        "label_pair_order": "Upstream JAX PRNGKey(20), fs_relabel 5, split [8,2,0]",
        "class_splits": {
            "train": {"count": 50, "range_inclusive": [0, 49]},
            "val": {"count": 1473, "range_inclusive": [50, 1522]},
            "test": {"count": 100, "range_inclusive": [1523, 1622]},
        },
        "exemplar_splits": {"train": [0], "val": [1, 2, 3, 4], "test": []},
        "label_pairs": {
            "train": [list(p) for p in TRAIN_PAIRS],
            "val": [list(p) for p in VALIDATION_PAIRS],
            "test": [],
        },
        "published_main_run_evaluators": evals,
        "important_original_detail": (
            "The 1473-class validation pool is reserved but NOT used by the four "
            "named main-run evaluators. The fsl_test_class evaluator is in the "
            "original periodic evaluation list, so it is not a sealed final test."
        ),
        "course_safe_proposal": {
            "training": "train classes + exemplar 0 + train label pairs",
            "development_seen_classes_new_exemplar":
                "train classes + exemplar 1..4 + train pairs",
            "development_seen_classes_new_pairs":
                "train classes + exemplar 0 + val pairs",
            "development_new_classes":
                "val classes + exemplar 0 + train pairs",
            "final_only": "test classes + exemplar 0 + train pairs",
            "final_test_policy":
                "Never inspect final-test model metrics until model selection and "
                "scientific hypotheses have been frozen."
        },
        "original_flags": {k: list(v) for k, v in flags.items()},
        "data_geometry": {
            "class_count": 1623, "exemplars_per_class": 5,
            "embedding_dim": 512, "num_label_tokens": 5,
            "context_classes": 2
        },
        "test_performance_evaluated": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "docs/data-audit/reference_split_manifest.json")
    args = p.parse_args()
    manifest = build_manifest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print("REFERENCE_SPLIT_MANIFEST=" + json.dumps({
        "class_splits": manifest["class_splits"],
        "exemplar_splits": manifest["exemplar_splits"],
        "label_pairs": manifest["label_pairs"],
        "test_performance_evaluated": False,
        "out": str(args.out),
    }))


if __name__ == "__main__":
    main()
