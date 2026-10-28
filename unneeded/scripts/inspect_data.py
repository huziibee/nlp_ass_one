"""Visual and numerical audit of official Omniglot ResNet18 features.

No model training or mutation of the upstream data. Generates reproducible
figures and JSON in outputs/data_audit/.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from check_data import DATA, KEY, inspect

ROOT = Path(__file__).resolve().parents[1]


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1) / np.maximum(
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12
    )


def audit(path: Path, out: Path, class_id: int, n_classes: int, seed: int) -> dict:
    provenance = inspect(path)
    with h5py.File(path, "r") as h:
        entries = []
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                entries.append({"key": name, "shape": list(obj.shape), "dtype": str(obj.dtype)})
        h.visititems(visit)
        features = np.array(h[KEY], dtype=np.float64)
    n_total, n_exemplars, dimensions = features.shape
    if not 0 <= class_id < n_total:
        raise ValueError(f"Class ID must be in [0, {n_total-1}]")
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(features, axis=2)
    chosen = rng.choice(n_total, size=min(n_total, 800), replace=False)
    within = cosine(features[chosen, 0], features[chosen, 1])
    between = cosine(features[chosen, 0], features[(chosen + 1) % n_total, 0])

    out.mkdir(parents=True, exist_ok=True)
    summary = {
        **provenance,
        "datasets": entries,
        "feature_mean": float(features.mean()),
        "feature_std": float(features.std()),
        "feature_min": float(features.min()),
        "feature_max": float(features.max()),
        "mean_feature_norm": float(norms.mean()),
        "mean_within_class_cosine": float(within.mean()),
        "mean_between_class_cosine": float(between.mean()),
        "class_id": class_id,
        "seed": seed,
        "interpretation_note": "Class IDs were randomised; PCA is exploratory, not evidence for learned induction heads.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")

    fig, ax = plt.subplots(figsize=(8.3, 4.5))
    ax.hist(norms.ravel(), bins=55)
    ax.set(xlabel="L2 norm of feature vector", ylabel="Exemplar count", title="ResNet18 feature magnitudes")
    fig.tight_layout()
    fig.savefig(out / "01_feature_norms.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.3, 4.5))
    ax.hist(within, bins=38, alpha=0.6, label="Same class, different exemplars")
    ax.hist(between, bins=38, alpha=0.6, label="Different classes")
    ax.set(xlabel="Cosine similarity", ylabel="Pairs", title="Within-class vs between-class similarity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "02_similarity_histograms.png", dpi=160)
    plt.close(fig)

    items = features[class_id]
    matrix = cosine(items[:, None, :], items[None, :, :])
    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(matrix, vmin=-1, vmax=1, cmap="viridis")
    ax.set(
        title=f"Class {class_id}: exemplar cosine similarities",
        xlabel="Exemplar index", ylabel="Exemplar index",
        xticks=np.arange(n_exemplars), yticks=np.arange(n_exemplars),
    )
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "03_exemplar_heatmap.png", dpi=160)
    plt.close(fig)

    num = min(n_total, max(2, n_classes))
    class_ids = rng.choice(n_total, size=num, replace=False)
    points = features[class_ids].reshape(-1, dimensions)
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vectors[:2].T
    fig, ax = plt.subplots(figsize=(8, 5.4))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=np.repeat(np.arange(num), n_exemplars),
        cmap="tab20", s=27, alpha=0.75,
    )
    ax.set(
        title=f"Feature PCA: {num} sampled classes, {n_exemplars} exemplars each",
        xlabel="Principal component 1", ylabel="Principal component 2",
    )
    fig.colorbar(scatter, ax=ax, label="Sampled class index")
    fig.tight_layout()
    fig.savefig(out / "04_feature_pca.png", dpi=160)
    plt.close(fig)

    class_a, class_b = rng.choice(50, size=2, replace=False)
    value_a, value_b = rng.choice(5, size=2, replace=False)
    query_class = int(rng.choice([class_a, class_b]))
    query_value = int(value_a if query_class == class_a else value_b)
    summary["example_episode"] = {
        "context": [[int(class_a), int(value_a)], [int(class_b), int(value_b)]],
        "query_class": query_class,
        "oracle_query_value": query_value,
        "note": "Illustrates one valid random-label episode; does not reconstruct original images.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved four plots to {out.resolve()}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", type=Path, default=DATA)
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "data_audit")
    p.add_argument("--class-id", type=int, default=0)
    p.add_argument("--plot-classes", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    audit(args.file, args.out, args.class_id, args.plot_classes, args.seed)


if __name__ == "__main__":
    main()
