from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.base_300k_replication import ARMS, check_chain, check_g0

EXTENDED_SEEDS = (6001, 5054, 6054, 7054, 8054)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def valid_probability(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"Invalid probability: {value}")
    return number


def paired_effect(name, pairs):
    differences = [pair["clean_only"] - pair["recursive_replacement"] for pair in pairs]
    average = mean(differences)
    margin = 2.7764451051977987 * stdev(differences) / math.sqrt(5)
    return {
        "outcome": name,
        "n_seeds": 5,
        "clean_mean": mean(pair["clean_only"] for pair in pairs),
        "recursive_mean": mean(pair["recursive_replacement"] for pair in pairs),
        "gap_pp": average * 100,
        "lower_95_pp": (average - margin) * 100,
        "upper_95_pp": (average + margin) * 100,
    }


def extended_paths(root, seed, arm):
    if seed == 6001:
        return root / "extended_primary/g00", root / "extended_primary" / arm
    return root / f"extended_g0_{seed}", root / f"extended_{seed}_{arm}" / arm


def extended_chain(root, seed, arm):
    g0, chain = extended_paths(root, seed, arm)
    initial_seed = 4054 if seed == 6001 else seed
    initial = read_json(g0 / "metrics.json")
    if (
        initial.get("steps", 0) * initial.get("batch_size", 0) != 300000
        or initial.get("seed") != initial_seed
        or initial.get("final_test_evaluated") is not False
        or initial.get("class_split") != [50, 1473, 100]
    ):
        raise ValueError(f"Incorrect Extended G0 metadata: {g0}")
    checkpoint = g0 / "extended_g0_model.eqx"
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1000:
        raise FileNotFoundError(checkpoint)
    values, sources = [], [g0 / "metrics.json", checkpoint]
    for generation in range(1, 11):
        folder = chain / f"g{generation:02d}"
        path = folder / "metrics.json"
        metrics = read_json(path)
        if (
            metrics.get("condition") != arm
            or metrics.get("generation") != generation
            or metrics.get("training_exposures") != 300000
            or metrics.get("data_seed") != seed
            or metrics.get("init_seed") != initial_seed
            or metrics.get("final_test_evaluated") is not False
            or metrics.get("training_class_split") != [50, 1473, 100]
            or metrics.get("recursively_generated_input_prefix") is not False
        ):
            raise ValueError(f"Incorrect Extended generation metadata: {path}")
        checkpoint = folder / "model.eqx"
        if not checkpoint.is_file() or checkpoint.stat().st_size < 1000:
            raise FileNotFoundError(checkpoint)
        history = metrics["history"]
        if history[-1].get("sequence_exposures") != 300000:
            raise ValueError(f"Incomplete Extended generation: {path}")
        scores = history[-1]["validation"]["val_relabel"]
        values.append(valid_probability(scores["free_next_symbol_context_validity"]))
        sources.extend([path, checkpoint])
    return values, sources


def aggregate(root, out):
    base_rows, base_pairs, extended_rows, extended_pairs, sources = [], [], [], [], []
    for index in range(5):
        g0 = root / f"base_300k_g0_{index}"
        check_g0(g0, index)
        sources.extend(
            [
                g0 / "independent_protocol.json",
                g0 / "safe_train_val_smoke/config.json",
                g0 / "safe_train_val_smoke/checkpoints/00000300000.eqx",
            ]
        )
        pair = {}
        for arm in ARMS:
            chain = root / f"base_300k_{index}_{arm}"
            row = check_chain(chain, arm, index)
            base_rows.append(row)
            pair[arm] = valid_probability(row["G10_val_relabel_accuracy"])
            sources.append(chain / "summary.json")
            for generation in range(1, 11):
                folder = chain / arm / f"g{generation:02d}"
                sources.extend([folder / "metrics.json", folder / "model.eqx"])
        base_pairs.append(pair)
    generations = {arm: [[] for _ in range(10)] for arm in ARMS}
    for seed in EXTENDED_SEEDS:
        pair = {}
        for arm in ARMS:
            values, inputs = extended_chain(root, seed, arm)
            sources.extend(inputs)
            for index, value in enumerate(values):
                generations[arm][index].append(value)
            pair[arm] = values[-1]
            _, chain = extended_paths(root, seed, arm)
            validation = read_json(chain / "g10/metrics.json")["history"][-1][
                "validation"
            ]
            extended_rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "G10_sampled_q2_validity": values[-1],
                    "G10_first_label_heldout_labels": valid_probability(
                        validation["val_relabel"]["first_label_accuracy"]
                    ),
                    "G10_first_label_heldout_classes": valid_probability(
                        validation["val_classes"]["first_label_accuracy"]
                    ),
                    "G10_sampled_q2_validity_heldout_exemplars": valid_probability(
                        validation["val_exemplar"]["free_next_symbol_context_validity"]
                    ),
                    "G10_valid_q2_correct_y2": valid_probability(
                        validation["val_relabel"][
                            "free_second_label_context_consistency_all"
                        ]
                    ),
                }
            )
        extended_pairs.append(pair)
    generation_rows = []
    for index in range(10):
        clean = generations["clean_only"][index]
        recursive = generations["recursive_replacement"][index]
        generation_rows.append(
            {
                "task": "extended",
                "generation": index + 1,
                "n_seeds": 5,
                "control_mean": mean(clean),
                "control_sd": stdev(clean),
                "recursive_mean": mean(recursive),
                "recursive_sd": stdev(recursive),
            }
        )
    effects = [
        paired_effect("Base contextual-label accuracy", base_pairs),
        paired_effect("Extended sampled q2 validity", extended_pairs),
    ]
    manifest = []
    for path in sorted(set(sources)):
        manifest.append(
            {
                "file": (
                    path.relative_to(ROOT).as_posix()
                    if path.is_relative_to(ROOT)
                    else path.as_posix()
                ),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "base_300k_seed_outcomes.csv", base_rows)
    write_csv(out / "extended_seed_outcomes.csv", extended_rows)
    write_csv(out / "five_seed_generation_summary.csv", generation_rows)
    write_csv(out / "table1.csv", effects)
    write_csv(out / "input_manifest.csv", manifest)
    report = {
        "source": "fresh local runs only",
        "episodes_per_generation": 300000,
        "final_test_evaluated": False,
        "table1": effects,
    }
    (out / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate five fresh Base and Extended seed pairs"
    )
    parser.add_argument("--root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/paper_results")
    args = parser.parse_args()
    aggregate(args.root, args.out)


if __name__ == "__main__":
    main()
