from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from scripts.base_300k_replication import ARMS, EPISODES, check_g0, check_chain

ROOT = Path(__file__).resolve().parents[1]
EXTENDED_SOURCE = ROOT / "data/evidence/experiments/g10_five_seed/g10_seed_summary.json"
T95_DF4 = 2.7764451051977987


def file_digest(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "file": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def mean_interval(values: list[float]) -> dict:
    if len(values) != 5 or any(not math.isfinite(x) for x in values):
        raise ValueError("Exactly five finite independent seed differences required")
    mu = statistics.mean(values)
    margin = T95_DF4 * statistics.stdev(values) / math.sqrt(5)
    return {
        "n_independent_model_seeds": 5,
        "per_seed_clean_minus_recursive": values,
        "mean_clean_minus_recursive": mu,
        "descriptive_t95_interval": [mu - margin, mu + margin],
    }


def aggregate(inputs: Path, out: Path, extended_source: Path = EXTENDED_SOURCE) -> dict:
    extended = json.loads(extended_source.read_text(encoding="utf-8"))
    if (
        extended["n_model_seeds_per_task"] != 5
        or extended["training_exposures_per_generation"]["extended"] != EPISODES
        or extended["independent_test_set_for_G10"] is not False
    ):
        raise ValueError(
            "The ORIGINAL Extended G10 evidence is not five-seed 300k validation-only"
        )
    ext = extended["paired_effects"]["extended"]["generated_next_symbol_validity"]
    if len(ext["per_seed_clean_minus_recursive"]) != 5:
        raise ValueError("Incomplete original Extended paired-seed effects")
    records, checkpoint_files = [], []
    for index in range(5):
        g0 = inputs / "g0" / f"base300k-g0-{index}"
        check_g0(g0, index)
        checkpoint_files.append(g0 / "safe_train_val_smoke/checkpoints/00000300000.eqx")
        for arm in ARMS:
            source = inputs / "arms" / f"base300k-arm-{index}-{arm}"
            scores = check_chain(source, arm, index)
            records.append(scores)
            checkpoint_files += [
                source / arm / f"g{gen:02d}/model.eqx" for gen in range(1, 11)
            ]
    if len(checkpoint_files) != 105 or len(records) != 10:
        raise ValueError(
            "Must have five independent G0 and 100 G1-G10 Base model checkpoints"
        )
    effects = []
    for i in range(5):
        pair = {
            row["arm"]: row["G10_val_relabel_accuracy"]
            for row in records
            if row["seed_index"] == i
        }
        if set(pair) != set(ARMS):
            raise ValueError(f"Unpaired Base trial for seed index {i}")
        effects.append(pair["clean_only"] - pair["recursive_replacement"])
    base = mean_interval(effects)
    strong = [row for row in records if row["arm"] == "clean_only"]
    base["G10_clean_accuracy_by_seed"] = [
        row["G10_val_relabel_accuracy"] for row in strong
    ]
    base["G10_clean_mean"] = statistics.mean(base["G10_clean_accuracy_by_seed"])
    base["clean_baseline_minimum"] = min(base["G10_clean_accuracy_by_seed"])
    base["weak_clean_baseline_under_60_percent"] = sum(
        x < 0.6 for x in base["G10_clean_accuracy_by_seed"]
    )
    report = {
        "comparison": "matched 300,000 training examples per generation, not matched task complexity",
        "base": base,
        "extended_recorded_original_300k": ext,
        "base_new_G0_models": 5,
        "base_new_G1_G10_models": 100,
        "extended_300k_reused_original_seed_pairs": 5,
        "historical_final_test_accessed_by_this_study": False,
        "validity": "Base five independent seeds; Extended original independent five seeds; validation only",
        "qualification": (
            "Clean Base is weaker at 300k in the earlier one-seed pilot. "
            "All new five-seed clean values are reported rather than excluded. "
            "The tasks, output metrics, optimization cost and historical run origins differ."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    rows = [file_digest(p) for p in checkpoint_files]
    with (out / "base_300k_checkpoint_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=("file", "bytes", "sha256"))
        w.writeheader()
        w.writerows(rows)
    with (out / "base_300k_seed_outcomes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(
            f, fieldnames=("seed_index", "data_seed", "arm", "G10_val_relabel_accuracy")
        )
        w.writeheader()
        w.writerows(records)
    (out / "matched_300k_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Matched-budget (300k) Base and Extended results",
        "",
        "**Five new independent Base seeds** were trained at G0–G10 with 300,000 episodes per generation.",
        "The Extended results come from the separately preserved original 300k five-seed study.",
        "Both results are validation-only. Their outcome definitions differ.",
        "",
        "| Study | G10 metric | Mean clean minus recursive | Descriptive paired 95% interval |",
        "|:--|:--|--:|--:|",
        (
            f"| New Base 300k (n=5) | Held-out contextual-label accuracy | "
            f"{base['mean_clean_minus_recursive']*100:+.2f} pp | "
            f"[{base['descriptive_t95_interval'][0]*100:+.2f}, "
            f"{base['descriptive_t95_interval'][1]*100:+.2f}] pp |"
        ),
        (
            f"| Original Extended 300k (n=5) | Sampled next-character context validity | "
            f"{ext['mean_clean_minus_recursive']*100:+.2f} pp | "
            f"[{ext['indicative_paired_t95_interval'][0]*100:+.2f}, "
            f"{ext['indicative_paired_t95_interval'][1]*100:+.2f}] pp |"
        ),
        "",
        f"Base mean clean G10 accuracy: **{base['G10_clean_mean']:.2%}**. "
        f"Weak Base clean seeds (under 60%): {base['weak_clean_baseline_under_60_percent']}/5.",
        "",
        "## Interpretation",
        "",
        "- Equal episode counts remove only the exposure-count discrepancy.",
        "- We do **not** infer that one task is intrinsically more susceptible from different metrics.",
        "- No G10 final-test partition is untouched; no test evaluation is performed by this study.",
        "- All seeds are included regardless of clean baseline performance.",
        "- This reconstruction reads saved model-associated metrics, not newly recomputed predictions.",
        "",
        "## Evidence",
        "",
        "- Five fresh Base G0 models and 100 G1–G10 model checkpoint hashes (CSV manifest).",
        "- Base original per-seed metrics and their JSON provenance records.",
        "- Extended: data/evidence/experiments/g10_five_seed/g10_seed_summary.json.",
    ]
    (out / "matched_300k_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(
        "MATCHED_300K_FULL_CHAINS_VERIFIED=105 new Base checkpoints, 5 clean/recursive seed pairs",
        flush=True,
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=ROOT / "outputs/base_300k_inputs")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/base_300k_results")
    a = p.parse_args()
    aggregate(a.inputs, a.out)


if __name__ == "__main__":
    main()
