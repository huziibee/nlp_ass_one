#!/usr/bin/env python3
"""Fresh five-model-seed G0--G10 protocol and strict results verifier.

Unlike scripts/aggregate_five_seeds_g10.py, this protocol expects NEW initial
model training artifacts from the same GitHub Actions run. Plan/dry-run needs
no ML libraries; aggregate rejects incomplete or incorrect experiment chains.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import math
import statistics
from pathlib import Path

BASE_INITIALIZATION_SEEDS = (1005, 1105, 1205, 1305, 1405)
BASE_DATA_SEEDS = (39001, 40001, 41001, 42001, 43001)
EXTENDED_SEEDS = (9054, 10054, 11054, 12054, 13054)
ARMS = ("clean_only", "recursive_replacement")
METRICS = {
    "base": (("label_accuracy", "val_relabel", "accuracy"),),
    "extended": (
        ("first_label_accuracy", "val_relabel", "first_label_accuracy"),
        ("next_symbol_context_validity", "val_relabel", "free_next_symbol_context_validity"),
        ("second_label_consistency", "val_relabel", "free_second_label_context_consistency_all"),
    ),
}
T95_DF4 = 2.7764451051977987
SEEDS = {
    "base": BASE_DATA_SEEDS,
    "extended": EXTENDED_SEEDS,
}


def plan(extended_exposures: int = 300_000) -> dict:
    if extended_exposures not in (300_000, 1_000_000):
        raise ValueError("Supported original/matched Extended budgets are 300k or 1M")
    assert len(set(BASE_INITIALIZATION_SEEDS)) == 5
    assert len(set(BASE_DATA_SEEDS)) == 5
    assert len(set(EXTENDED_SEEDS)) == 5
    return {
        "schema": "fresh-v1",
        "budget_protocol": "matched-one-million" if extended_exposures == 1_000_000 else "original-unequal",
        "source": ("independently trained Base 1M run; new Extended 1M G0-G10 run"
                   if extended_exposures == 1_000_000 else
                   "clean checkout; all G0 and G1-G10 trained anew"),
        "independent_initialization_seeds_base": list(BASE_INITIALIZATION_SEEDS),
        "independent_data_seeds_base": list(BASE_DATA_SEEDS),
        "independent_initialization_and_data_seeds_extended": list(EXTENDED_SEEDS),
        "tasks": ["base", "extended"],
        "conditions": list(ARMS),
        "generations": 10,
        "G0_training_examples": {"base": 1_000_000, "extended": extended_exposures},
        "descendant_training_examples_per_generation": {"base": 1_000_000, "extended": extended_exposures},
        "n_training_chains_per_task": 5,
        "expected_G1_G10_arms": 20,
        "historical_checkpoints_used_as_initialization": False,
        "original_feature_data_reused": True,
        "synthetic_input_contexts": False,
        "held_out_final_test_allowed": False,
        "note": "Extended G0 reads fresh Base G0 CONFIG ONLY; no Base trained model weights.",
        "selection": "Include all fixed seeds, regardless of G0 validation accuracy.",
        "comparison_budget_matched_in_examples": extended_exposures == 1_000_000,
        "interpretation": (
            "Same episode count across tasks; tasks, outputs and computational cost still differ. "
            if extended_exposures == 1_000_000 else
            "Exploratory model-seed uncertainty, validation-only; unequal task exposure budgets."
        ),
    }


def checked_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required fresh-run evidence missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def checked_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 1_000:
        raise FileNotFoundError(f"Missing substantive trained checkpoint: {path}")


def validate_g0(root: Path, task: str, i: int, extended_exposures: int = 300_000) -> None:
    base = root / f"{task}_g0" / f"fresh-v1-{task}-g0-{i}"
    if task == "base":
        manifest = checked_json(base / "independent_protocol.json")
        expected = BASE_INITIALIZATION_SEEDS[i]
        if (manifest.get("independent_seed") != expected or
            manifest.get("training_sequences") != 1_000_000 or
            manifest.get("final_test_evaluated") is not False):
            raise ValueError(f"Incorrect fresh Base G0 seed/training metadata: {i}")
        cfg = checked_json(base / "safe_train_val_smoke" / "config.json")
        if cfg.get("train_iters") != 1_000_000 or cfg.get("class_split") != [50, 1473, 100]:
            raise ValueError(f"Invalid Base G0 budget/split: {i}")
        checked_file(base / "safe_train_val_smoke" / "checkpoints" / "00001000000.eqx")
    else:
        meta = checked_json(base / "metrics.json")
        if (meta.get("seed") != EXTENDED_SEEDS[i] or
            meta.get("steps", 0) * meta.get("batch_size", 0) != extended_exposures or
            meta.get("class_split") != [50, 1473, 100] or
            meta.get("final_test_evaluated") is not False):
            raise ValueError(f"Invalid Extended G0 seed/split/budget: {i}")
        checked_file(base / "extended_g0_model.eqx")


def read_chain(root: Path, task: str, i: int, arm: str, extended_exposures: int = 300_000) -> dict:
    folder = root / f"{task}_arms" / f"fresh-v1-{task}-arm-{i}-{arm}"
    summary = checked_json(folder / "summary.json")
    if summary.get("final_test_evaluated") is not False or set(summary.get("arms", {})) != {arm}:
        raise ValueError(f"{folder}: incomplete arm or final-test boundary error")
    records = summary["arms"][arm]
    if len(records) != 10 or [m.get("generation") for m in records] != list(range(1, 11)):
        raise ValueError(f"{folder}: exactly G1...G10 required")
    initial_seed = BASE_INITIALIZATION_SEEDS[i] if task == "base" else EXTENDED_SEEDS[i]
    data_seed = BASE_DATA_SEEDS[i] if task == "base" else EXTENDED_SEEDS[i]
    for g in range(1, 11):
        d = checked_json(folder / arm / f"g{g:02d}" / "metrics.json")
        checked_file(folder / arm / f"g{g:02d}" / "model.eqx")
        if d.get("generation") != g or d.get("final_test_evaluated") is not False:
            raise ValueError(f"{task} seed {i}, {arm}, G{g}: generation/test mismatch")
        if d.get("data_seed") != data_seed:
            raise ValueError(f"{task} seed {i}, {arm}, G{g}: data seed mismatch")
        if task == "base":
            if (d.get("optimizer_exposures") != 1_000_000 or
                d.get("initialisation_seed") != data_seed + 17 * g or
                d.get("class_split") != [50, 1473, 100] or
                d.get("whole_prefix_generated") is not False):
                raise ValueError(f"Base seed {i}, G{g}: full budget/split/context mismatch")
        else:
            if (d.get("training_exposures") != extended_exposures or
                d.get("init_seed") != initial_seed or
                d.get("training_class_split") != [50, 1473, 100] or
                d.get("recursively_generated_input_prefix") is not False or
                d.get("condition") != arm):
                raise ValueError(f"Extended seed {i}, G{g}: full budget/split/source mismatch")
    if task == "extended" and summary.get("all_seed_replication_included_irrespective_of_g0_accuracy") is not True:
        raise ValueError("Extended replicate cannot exclude weak G0 seeds")
    final = checked_json(folder / arm / "g10" / "metrics.json")["history"][-1]["validation"]
    values = {}
    for label, role, key in METRICS[task]:
        value = float(final[role][key])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid {task} seed={i} metric={label}")
        values[label] = value
    return values


def aggregate(root: Path, out: Path, extended_exposures: int = 300_000,
              base_reference_run_id: int | None = None) -> dict:
    p = plan(extended_exposures)
    if extended_exposures == 1_000_000 and not base_reference_run_id:
        raise ValueError("One-million matched comparison requires a verified Base reference run ID")
    if extended_exposures == 300_000 and base_reference_run_id:
        raise ValueError("Original all-fresh protocol should not reuse a Base reference run")
    rows = []
    for task in ("base", "extended"):
        for i in range(5):
            validate_g0(root, task, i, extended_exposures)
            for arm in ARMS:
                values = read_chain(root, task, i, arm, extended_exposures)
                rows.append({"task": task, "seed_index": i,
                             "data_seed": SEEDS[task][i], "arm": arm, **values})
    if len(rows) != 20:
        raise AssertionError("Exactly twenty fresh full training chains required")
    ckpt_rows = []
    for file in sorted(root.rglob("*.eqx")):
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        ckpt_rows.append({
            "path": file.relative_to(root).as_posix(),
            "bytes": file.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    if len(ckpt_rows) != 210:
        raise ValueError(f"Expected exactly 210 freshly trained G0/descendant checkpoint files; found {len(ckpt_rows)}")
    summary = {"protocol": p,
               "base_reference_run_id": base_reference_run_id,
               "base_reused_from_independent_fresh_run": base_reference_run_id is not None,
               "checkpoint_manifest_count": len(ckpt_rows),
               "github_run_id": os.environ.get("GITHUB_RUN_ID"),
               "github_commit_sha": os.environ.get("GITHUB_SHA"),
               "run_kind": "new training from fresh initialization, NOT historical reaggregation",
               "final_test_accessed": False,
               "rows": len(rows), "paired_effects": {}}
    for task in ("base", "extended"):
        summary["paired_effects"][task] = {}
        for metric, _, _ in METRICS[task]:
            deltas = []
            for i in range(5):
                pair = {row["arm"]: row[metric] for row in rows
                        if row["task"] == task and row["seed_index"] == i}
                if set(pair) != set(ARMS):
                    raise ValueError(f"Missing paired arm task={task} seed={i}")
                deltas.append(pair["clean_only"] - pair["recursive_replacement"])
            mean = statistics.mean(deltas)
            margin = T95_DF4 * statistics.stdev(deltas) / math.sqrt(5)
            summary["paired_effects"][task][metric] = {
                "n_independent_seeds": 5,
                "per_seed_clean_minus_recursive": deltas,
                "mean_clean_minus_recursive": mean,
                "descriptive_paired_t95_interval": [mean - margin, mean + margin],
            }
    out.mkdir(parents=True, exist_ok=True)
    (out / "fresh_five_seed_protocol.json").write_text(json.dumps(p, indent=2) + "\n")
    with (out / "fresh_five_seed_checkpoint_manifest.csv").open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(ckpt_rows)
    (out / "fresh_five_seed_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (out / "fresh_five_seed_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("plan", "aggregate"))
    ap.add_argument("--root", type=Path, default=Path("outputs/fresh_five_inputs"))
    ap.add_argument("--out", type=Path, default=Path("outputs/fresh_five_results"))
    ap.add_argument("--extended-exposures", type=int, default=300_000,
                    choices=(300_000, 1_000_000),
                    help="Original Extended 300k or matched Base/Extended 1M exposures")
    ap.add_argument("--base-reference-run-id", type=int, default=None,
                    help="Source run for newly trained Base cohort when reused for 1M comparison")
    args = ap.parse_args()
    if args.command == "plan":
        print(json.dumps(plan(args.extended_exposures), indent=2))
    else:
        result = aggregate(args.root, args.out,
                           extended_exposures=args.extended_exposures,
                           base_reference_run_id=args.base_reference_run_id)
        print("FRESH_FIVE_SEED_FULL_RETRAINING_VERIFIED=" + json.dumps({
            "rows": result["rows"], "checkpoints": result["checkpoint_manifest_count"],
            "final_test_accessed": result["final_test_accessed"],
            "tasks": list(result["paired_effects"]),
        }))


if __name__ == "__main__":
    main()
