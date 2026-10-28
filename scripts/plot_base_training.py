from __future__ import annotations
import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEVELOPMENT_EVALS = {
    "fsl_train": "Training classes + familiar label pairs",
    "fsl_val_rl": "Training classes + unseen label pairs",
    "fsl_train_valex": "Training classes + held-out drawings",
    "fsl_val_class": "Held-out validation classes",
}


def summarize(log: Path, out: Path) -> dict:
    if not log.is_file():
        raise FileNotFoundError(f"Authors' HDF5 log not found: {log}")
    with h5py.File(log, "r") as h:
        if "train_iter" not in h or "train_loss" not in h or "eval_iter" not in h:
            raise ValueError("Not an original-author training log with expected arrays")
        unwanted = [k for k in h if "test" in k.lower()]
        if unwanted:
            raise ValueError(
                f"Forbidden test-named evaluator in this development log: {unwanted}"
            )
        missing = [k for k in DEVELOPMENT_EVALS if k not in h]
        if missing:
            raise ValueError(f"Missing development evaluators: {missing}")
        train_step = np.array(h["train_iter"], dtype=np.int64)
        train_loss = np.array(h["train_loss"], dtype=np.float64)
        eval_step = np.array(h["eval_iter"], dtype=np.int64)
        tracks = {}
        for key, desc in DEVELOPMENT_EVALS.items():
            g = h[key]
            tracks[key] = {
                "description": desc,
                "accuracy": np.asarray(g["acc"], dtype=np.float64).mean(axis=1),
                "loss": np.asarray(g["loss"], dtype=np.float64).mean(axis=1),
                "n_per_evaluation": int(g["acc"].shape[1]),
            }
    if len(train_loss) < 1 or len(eval_step) < 1:
        raise ValueError("Missing training and evaluation measurements")
    if len(train_step) != len(train_loss):
        raise ValueError("Mismatched training timeline")
    for name, track in tracks.items():
        if len(track["accuracy"]) != len(eval_step):
            raise ValueError(f"Inconsistent evaluation arrays: {name}")

    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(train_step, train_loss, alpha=0.2, label="Per-minibatch cross-entropy")
    k = min(len(train_loss), max(5, len(train_loss) // 100))
    if k > 1:
        xs = train_step[k - 1 :]
        smoothed = np.convolve(train_loss, np.ones(k) / k, mode="valid")
        ax.plot(xs, smoothed, linewidth=1.8, label=f"Mean of {k} updates")
    ax.set(
        xlabel="Training sequences seen",
        ylabel="Cross-entropy",
        title="Original two-layer Base transformer: train loss",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_base_training_loss.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for key, track in tracks.items():
        ax.plot(
            eval_step,
            100 * track["accuracy"],
            "o-",
            markersize=3,
            label=track["description"],
        )
    ax.axhline(20, linestyle=":", linewidth=1, label="5-label uniform guess: 20%")
    ax.set(
        xlabel="Training sequences seen",
        ylabel="Accuracy (%)",
        title="Original Base: development performance by generalization axis",
        ylim=(0, 105),
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "02_base_development_accuracy.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for key, track in tracks.items():
        ax.plot(
            eval_step, track["loss"], "o-", markersize=3, label=track["description"]
        )
    ax.set(
        xlabel="Training sequences seen",
        ylabel="Cross-entropy",
        title="Original Base: development losses",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "03_base_development_losses.png", dpi=170)
    plt.close(fig)

    result = {
        "source": "original pinned Singh et al. main.py log.h5",
        "train_sequences_completed": int(train_step[-1]),
        "updates_logged": int(len(train_step)),
        "training_loss_first": float(train_loss[0]),
        "training_loss_last": float(train_loss[-1]),
        "evaluation_timeline": eval_step.tolist(),
        "evaluation_tracks": {
            key: {
                "description": data["description"],
                "accuracy": data["accuracy"].tolist(),
                "loss": data["loss"].tolist(),
                "n_per_evaluation": data["n_per_evaluation"],
            }
            for key, data in tracks.items()
        },
        "model_final_test_evaluated": False,
        "not_induction_head_evidence": True,
        "interpretation": (
            "These are behavioural metrics only. Induction circuits require "
            "mechanistic analysis of a well-trained checkpoint and controls."
        ),
    }
    (out / "base_training_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    print(
        "BASE_TRAINING_SUMMARY="
        + json.dumps(
            {
                "train_sequences_completed": result["train_sequences_completed"],
                "accuracy_last": {
                    k: round(v["accuracy"][-1], 5)
                    for k, v in result["evaluation_tracks"].items()
                },
                "final_test_evaluated": False,
            }
        )
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("outputs/base_training_summary"))
    args = p.parse_args()
    summarize(args.log, args.out)


if __name__ == "__main__":
    main()
