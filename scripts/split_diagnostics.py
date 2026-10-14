"""Split-aware character-embedding diagnostics, with NO final-test scoring.

For a one-shot nearest-cosine baseline, support is exemplar 0 for each
class, and query is each different exemplar 1..4. We compare 50 train
classes and 1473 validation classes. Test classes [1523, 1622] are never
opened by the analysis below.

Random distractor accuracy for each candidate count k is computed exactly
over uniformly sampled subsets, not via thousands of repeated random draws.
No transformer is trained and no label-copying accuracy is measured.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
from check_data import DATA, KEY, inspect
from reference_split import get_class_ids

ROOT = Path(__file__).resolve().parents[1]
KS = (2, 3, 5, 10, 20, 50, 100, 250, 500, 1473)


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def expected_k_way_accuracy(impostor_counts: np.ndarray, n_classes: int, k: int) -> float:
    """Expected probability true class wins among k sampled distinct classes.

    Context always includes the true class and k-1 distractors chosen without
    replacement from n_classes-1. This is a hypergeometric probability.
    An 'impostor' is any wrong class scoring >= the correct class score.
    """
    if not 2 <= k <= n_classes:
        raise ValueError("Need 2 <= candidate count <= number of classes")
    num_good = n_classes - 1 - impostor_counts.astype(np.float64)
    numerator = np.ones_like(num_good)
    for j in range(k - 1):
        numerator *= np.maximum(num_good - j, 0) / (n_classes - 1 - j)
    return float(np.mean(numerator))


def analyze(features: np.ndarray, name: str, seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    n_classes, n_exemplars, _ = features.shape
    assert n_exemplars == 5
    support = normalize(features[:, 0])
    queries = normalize(features[:, 1:].reshape(-1, features.shape[-1]))
    truth = np.repeat(np.arange(n_classes), n_exemplars - 1)
    sims = queries @ support.T
    correct = sims[np.arange(len(truth)), truth]
    # A uniformly random DIFFERENT class: a pairwise negative similarity.
    negative = rng.integers(0, n_classes - 1, size=len(truth))
    negative += negative >= truth
    distractor = sims[np.arange(len(truth)), negative]
    # Count how many wrong support classes tie with or outperform the true class.
    worse = sims >= correct[:, None]
    worse[np.arange(len(truth)), truth] = False
    impostors = np.count_nonzero(worse, axis=1)
    k_sizes = sorted(set([k for k in KS if k <= n_classes] + [n_classes]))
    k_curve = {str(k): expected_k_way_accuracy(impostors, n_classes, k)
               for k in k_sizes}
    # Note all-class includes all classes within ONE TRAIN OR VALIDATION
    # partition; no held-out final test classes are read or scored.
    results = {
        "partition": name,
        "num_classes": int(n_classes),
        "queries_per_class": int(n_exemplars - 1),
        "num_query_examples": int(len(truth)),
        "exemplar_support": 0,
        "exemplar_query_ids": [1, 2, 3, 4],
        "mean_same_class_cosine": float(np.mean(correct)),
        "mean_random_different_class_cosine": float(np.mean(distractor)),
        "same_class_std": float(np.std(correct)),
        "different_class_std": float(np.std(distractor)),
        "exact_expected_two_choice_accuracy": k_curve["2"],
        "all_partition_class_top1_accuracy": k_curve[str(n_classes)],
        "all_partition_class_random_chance": 1 / n_classes,
        "mean_number_of_better_or_tied_distractors": float(np.mean(impostors)),
        "fraction_with_at_least_one_better_distractor": float(np.mean(impostors > 0)),
        "mean_similarity_margin_vs_random_distractor": float(np.mean(correct-distractor)),
        "candidate_curve": k_curve,
        "assessment_role": ("training-class diagnostic, NOT final-test accuracy"
                            if name == "train" else
                            "validation-class diagnostic, NOT final-test accuracy"),
    }
    vectors = {"same": correct, "different": distractor,
               "impostors": impostors, "margins": correct - distractor}
    return results, vectors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/data-audit")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    verified = inspect(a.file)
    results = {}
    data = {}
    with h5py.File(a.file, "r") as f:
        for name in ("train", "val"):
            selected = get_class_ids(name)
            # SOURCE ACCESS GUARANTEE: only the interval below is loaded.
            # Test indices [1523, 1622] are not used for any measurement.
            x = np.asarray(f[KEY][selected[0]:selected[-1]+1, :, :])
            results[name], data[name] = analyze(x, name, a.seed)
    payload = {
        "source_git_blob_sha1": verified["git_blob_sha1"],
        "seed": a.seed,
        "test_class_indices_ever_scored": False,
        "no_training_or_interventions": True,
        "method": "held-out exemplar cosine matching; exact expectation over uniformly selected distractors",
        "partitions": results,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "split_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf8")

    fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for ax, name, title in zip(
            axs, ("train", "val"),
            ("Training classes (50)", "Validation classes (1,473)")):
        v = data[name]
        ax.hist(v["same"], bins=50, density=True, alpha=.62,
                label="Same class (correct support)")
        ax.hist(v["different"], bins=50, density=True, alpha=.62,
                label="Different class (random distractor)")
        ax.set(title=title, xlabel="Cosine similarity", ylabel="Density",
               xlim=(0.25, 1.01))
        ax.legend(fontsize=8)
    fig.suptitle("Figure 2 explained: same-class features overlap with distractors")
    fig.tight_layout()
    fig.savefig(a.out / "06_split_similarities.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    for name, label in (("train", "Training classes (50)"),
                        ("val", "Validation classes (1,473)")):
        curve = results[name]["candidate_curve"]
        counts = np.asarray([int(x) for x in curve])
        values = np.asarray(list(curve.values())) * 100
        ax.plot(counts, values, "o-", label=label)
    ax.axhline(50, linestyle=":", linewidth=1, label="50% chance for two choices")
    ax.set(xscale="log", ylim=(0, 104),
           xticks=[2, 3, 5, 10, 20, 50, 100, 250, 500, 1473],
           xlabel="Number of candidate characters (includes the true one)",
           ylabel="Expected correct-class matching (%)",
           title="Figure 5 explained: more distractors makes matching harder")
    ax.set_xticklabels(["2", "3", "5", "10", "20", "50", "100", "250", "500", "1473"],
                       rotation=40)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out / "07_matching_vs_candidates.png", dpi=160)
    plt.close(fig)

    print("SPLIT_DIAGNOSTICS=" + json.dumps(payload))


if __name__ == "__main__":
    main()
