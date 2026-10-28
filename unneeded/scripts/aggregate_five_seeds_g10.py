from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ARMS = ("clean_only", "recursive_replacement")
METRICS = {
    "base": (
        ("label_pair_accuracy", "accuracy", "val_relabel"),
        ("validation_class_accuracy", "accuracy", "val_classes"),
    ),
    "extended": (
        ("first_label_accuracy", "first_label_accuracy", "val_relabel"),
        (
            "second_label_teacher_forced_accuracy",
            "teacher_forced_second_label_accuracy",
            "val_relabel",
        ),
        (
            "generated_next_symbol_validity",
            "free_next_symbol_context_validity",
            "val_relabel",
        ),
        (
            "generated_second_number_consistency",
            "free_second_label_context_consistency_all",
            "val_relabel",
        ),
        (
            "generated_valid_symbol_probability_mass",
            "next_symbol_valid_probability_mass",
            "val_relabel",
        ),
        (
            "validation_class_first_label_accuracy",
            "first_label_accuracy",
            "val_classes",
        ),
    ),
}
T95_DF4 = 2.7764451051977987


def read_one(folder: Path, task: str):
    meta_folder = folder
    matches = list(meta_folder.rglob("summary.json"))
    if len(matches) != 1:
        raise AssertionError(f"Expected one summary per artifact: {folder}: {matches}")
    summary = json.loads(matches[0].read_text())
    if summary.get("final_test_evaluated") is not False:
        raise AssertionError("Final test accessed in claimed validation replicate")
    if len(summary.get("arms", {})) != 1:
        raise AssertionError("Expected single-arm checkpoint archive")
    arm = next(iter(summary["arms"]))
    if arm not in ARMS:
        return None
    if len(summary["arms"][arm]) != 10:
        raise AssertionError("Expected 10 trained descendants")
    root = matches[0].parent
    p = root / arm / "g10" / "metrics.json"
    ckpt = root / arm / "g10" / "model.eqx"
    if not p.is_file() or not ckpt.is_file():
        raise FileNotFoundError(
            "Missing final model or full metric provenance " + str(p)
        )
    meta = json.loads(p.read_text())
    if meta.get("final_test_evaluated") is not False or meta.get("generation") != 10:
        raise AssertionError("Unsafe or incomplete G10 metadata")
    if meta.get("arm", meta.get("condition")) != arm:
        raise AssertionError("Wrong arm checkpoint")
    if task == "base":
        seed = int(meta["data_seed"])
        if meta.get("optimizer_exposures") != 1_000_000:
            raise AssertionError("Base not trained with full one-million exposure")
        if meta.get("class_split") != [50, 1473, 100]:
            raise AssertionError("Base class split mismatch")
        if meta.get("whole_prefix_generated") is not False:
            raise AssertionError("Mislabelled synthetic prefix")
    else:
        seed = int(meta["data_seed"])
        if meta.get("training_exposures") != 300_000:
            raise AssertionError("Extended not trained with full 300k exposure")
        if meta.get("training_class_split") != [50, 1473, 100]:
            raise AssertionError("Extended class split mismatch")
        if meta.get("recursively_generated_input_prefix") is not False:
            raise AssertionError("Mislabelled synthetic prefix")
    if summary["arms"][arm][-1]["generation"] != 10:
        raise AssertionError("Mismatched final generation")
    validation = meta["history"][-1]["validation"]
    answer = {
        "task": task,
        "seed": seed,
        "arm": arm,
        "generation": 10,
        "test_evaluated": False,
    }
    for alias, key, role in METRICS[task]:
        x = float(validation[role][key])
        if not math.isfinite(x) or not 0 <= x <= 1:
            raise ValueError("Out-of-range validation accuracy " + alias)
        answer[alias] = x
    return answer


def aggregate(root: Path, out: Path):
    rows = []
    for task in ("base", "extended"):
        for mode in ("primary", "replica", "additional"):
            base = root / f"{task}_{mode}"
            if not base.is_dir():
                raise FileNotFoundError("Expected artifact directory " + str(base))
            subdirs = sorted(x for x in base.iterdir() if x.is_dir())
            if not subdirs:
                raise AssertionError("No downloaded GitHub artifacts in " + str(base))
            for d in subdirs:
                row = read_one(d, task)
                if row is not None:
                    rows.append(row)
    pairs = {}
    for r in rows:
        key = (r["task"], r["seed"])
        a = pairs.setdefault(key, {})
        if r["arm"] in a:
            raise AssertionError("Duplicate arm " + str(key))
        a[r["arm"]] = r
    expected = {
        ("base", 34001),
        ("base", 37001),
        ("base", 38001),
        ("base", 35001),
        ("base", 36001),
        ("extended", 6001),
        ("extended", 7054),
        ("extended", 8054),
        ("extended", 5054),
        ("extended", 6054),
    }
    if set(pairs) != expected:
        raise AssertionError(
            f"Expected five independent full-run seeds per task; found {sorted(pairs)}"
        )
    if len(rows) != 20:
        raise AssertionError("Expected 20 independently trained G10 arms")
    effects = []
    summary = {
        "source": "actual completed original-feature Base/Extended checkpoints",
        "n_model_seeds_per_task": 5,
        "evaluation": "validation-only G10, never reused final-test",
        "independent_test_set_for_G10": False,
        "training_exposures_per_generation": {"base": 1000000, "extended": 300000},
        "model_seed_uncertainty": "paired clean-minus-recursive differences, n=5 model seeds",
        "limitations": [
            "seed effects exploratory with five independent seeds",
            "Base and Extended training exposure budgets differ",
            "target-only synthetic continuation feedback; no recursively generated input contexts",
            "the G3 final 100-class test was previously opened",
        ],
        "paired_effects": {},
    }
    for task in ("base", "extended"):
        summary["paired_effects"][task] = {}
        seeds = sorted(s for t, s in pairs if t == task)
        for metric, _, _ in METRICS[task]:
            diffs = []
            for seed in seeds:
                pair = pairs[(task, seed)]
                if set(pair) != set(ARMS):
                    raise AssertionError("Missing matching clean/recursive for seed")
                clean = pair["clean_only"][metric]
                recursive = pair["recursive_replacement"][metric]
                delta = clean - recursive
                diffs.append(delta)
                effects.append(
                    {
                        "task": task,
                        "seed": seed,
                        "metric": metric,
                        "clean": clean,
                        "recursive": recursive,
                        "clean_minus_recursive": delta,
                    }
                )
            avg = statistics.mean(diffs)
            sd = statistics.stdev(diffs)
            margin = T95_DF4 * sd / math.sqrt(5)
            summary["paired_effects"][task][metric] = {
                "per_seed_clean_minus_recursive": dict(zip(map(str, seeds), diffs)),
                "mean_clean_minus_recursive": avg,
                "sample_sd": sd,
                "indicative_paired_t95_interval": [avg - margin, avg + margin],
                "n_model_seeds": 5,
            }
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("g10_seed_level.csv", rows),
        ("g10_paired_seed_effects.csv", effects),
    ):
        fields = sorted({k for d in data for k in d})
        with (out / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    (out / "g10_seed_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        "FIVE_SEED_G10_COMPLETE="
        + json.dumps(
            {
                "rows": len(rows),
                "base_effects": summary["paired_effects"]["base"],
                "extended_effects": summary["paired_effects"]["extended"],
                "last_final_test_classes_accessed": False,
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    aggregate(a.root, a.out)


if __name__ == "__main__":
    main()
