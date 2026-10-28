from pathlib import Path
import argparse, csv, json, statistics, math

SEEDS = {"base": [34001, 35001, 36001], "extended": [6001, 5054, 6054]}
ARMS = ("clean_only", "recursive_replacement")


def read_one(root, task, seed, arm):
    records = {}
    for file in root.rglob("metrics.json"):
        if not (file.parent.name.startswith("g") and arm in file.parts):
            continue
        gen = int(file.parent.name.removeprefix("g"))
        if gen not in range(1, 11):
            continue
        meta = json.loads(file.read_text())
        if meta.get("generation") != gen:
            continue
        if meta.get("arm", meta.get("condition")) != arm:
            continue
        if int(meta["data_seed"]) != seed:
            continue
        if meta.get("final_test_evaluated") is not False:
            raise AssertionError("Protected test use!")
        exposed = (
            meta["optimizer_exposures"]
            if task == "base"
            else meta["training_exposures"]
        )
        if exposed != (1000000 if task == "base" else 300000):
            raise AssertionError("Unequal exposure")
        if gen in records:
            raise AssertionError(f"Duplicate {task} {seed} {arm} G{gen}")
        records[gen] = meta["history"][-1]["validation"]
    if set(records) != set(range(1, 11)):
        raise AssertionError(f"Incomplete {task} {seed} {arm}: {sorted(records)}")
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = []
    per_seed = []
    role_rows = []
    for task, seeds in SEEDS.items():
        for seed in seeds:
            package = a.root / task / ("primary" if seed == seeds[0] else str(seed))
            pair = {arm: read_one(package, task, seed, arm) for arm in ARMS}
            gaps = []
            for g in range(1, 11):
                c = pair["clean_only"][g]
                r = pair["recursive_replacement"][g]
                role = "val_relabel"
                metric = (
                    "accuracy"
                    if task == "base"
                    else "free_next_symbol_context_validity"
                )
                cl = float(c[role][metric])
                re = float(r[role][metric])
                diff = 100 * (cl - re)
                gaps.append(diff)
                rows.append(
                    {
                        "task": task,
                        "seed": seed,
                        "generation": g,
                        "metric": metric,
                        "clean": cl,
                        "recursive": re,
                        "gap_pp": diff,
                        "validation_only": True,
                    }
                )
                if g == 10:
                    metrics = (
                        (("val_relabel", "accuracy"), ("val_classes", "accuracy"))
                        if task == "base"
                        else (
                            ("val_relabel", "first_label_accuracy"),
                            ("val_relabel", "teacher_forced_second_label_accuracy"),
                            ("val_relabel", "free_next_symbol_context_validity"),
                            (
                                "val_relabel",
                                "free_second_label_context_consistency_all",
                            ),
                            ("val_exemplar", "first_label_accuracy"),
                            ("val_exemplar", "free_next_symbol_context_validity"),
                            ("val_exemplar", "teacher_forced_second_label_accuracy"),
                            ("val_classes", "first_label_accuracy"),
                            ("val_classes", "teacher_forced_second_label_accuracy"),
                        )
                    )
                    for ro, key in metrics:
                        if key not in c[ro] or key not in r[ro]:
                            continue
                        role_rows.append(
                            {
                                "task": task,
                                "seed": seed,
                                "role": ro,
                                "metric": key,
                                "clean": float(c[ro][key]),
                                "recursive": float(r[ro][key]),
                                "clean_minus_recursive_pp": 100
                                * (float(c[ro][key]) - float(r[ro][key])),
                            }
                        )
            onset = next((i + 1 for i, v in enumerate(gaps) if v >= 5), None)
            per_seed.append(
                {
                    "task": task,
                    "seed": seed,
                    "first_at_least_5pp_generation": onset,
                    "g1_gap_pp": gaps[0],
                    "g10_gap_pp": gaps[-1],
                    "gaps_pp": gaps,
                }
            )
    paired = []
    for task in SEEDS:
        for gen in range(1, 11):
            xx = [
                r["gap_pp"]
                for r in rows
                if r["task"] == task and r["generation"] == gen
            ]
            assert len(xx) == 3
            mean = statistics.mean(xx)
            sd = statistics.stdev(xx)
            ci = 4.3026527299 * sd / math.sqrt(3)
            paired.append(
                {
                    "task": task,
                    "generation": gen,
                    "mean_gap_pp": mean,
                    "sd_model_seed_pp": sd,
                    "indicative_t95_lower_pp": mean - ci,
                    "indicative_t95_upper_pp": mean + ci,
                    "positive_gap_in_all_three_seeds": all(x > 0 for x in xx),
                }
            )
    a.out.mkdir(parents=True, exist_ok=True)

    def write(name, rs, columns=None):
        if not rs:
            return
        with (a.out / name).open("w", newline="") as f:
            fields = columns or [x for x in rs[0] if x != "gaps_pp"]
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rs)

    write("all_seed_generations.csv", rows)
    write("per_generation_paired_summary.csv", paired)
    write("g10_validation_roles.csv", role_rows)
    write("individual_seed_onset.csv", per_seed)
    summary = {
        "three_model_seeds_per_task": True,
        "protected_test_reopened": False,
        "formal_changepoint_estimated": False,
        "threshold_definition": "post-hoc 5pp clean minus recursive gap per individual seed",
        "onset_by_seed": per_seed,
        "g10_roles": role_rows,
        "note": "Multiple exploratory generation-wise intervals; not simultaneous inferential intervals",
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("MULTISEED_GENERATION_ONSET=" + json.dumps(per_seed))


if __name__ == "__main__":
    main()
