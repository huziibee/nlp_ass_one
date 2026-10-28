import argparse, csv, json, statistics
from pathlib import Path

KEYS = (
    "first_label_accuracy",
    "teacher_forced_second_label_accuracy",
    "free_next_symbol_context_validity",
    "free_second_label_context_consistency_all",
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = []
    for seed in (5054, 6054):
        row = {"seed": seed, "final_test_evaluated": False}
        for kind in ("raw", "fix_invalid_q2"):
            root = a.root / f"extended-repair-independent-{seed}-{kind}"
            files = [f for f in root.rglob("metrics.json") if "/g10/" in f.as_posix()]
            if len(files) != 1:
                raise AssertionError(
                    f"Missing or ambiguous metrics: {seed}/{kind}: {files}"
                )
            m = json.loads(files[0].read_text())
            if m.get("generation") != 10:
                raise AssertionError("Wrong generation")
            if m.get("final_test_access", m.get("final_test_evaluated")) is not False:
                raise AssertionError("Protected test accessed")
            if kind == "raw":
                if m.get("training_exposures") != 300000 or m.get("init_seed") != seed:
                    raise AssertionError("Raw mismatch")
            else:
                if (
                    m.get("train_exposures") != 300000
                    or m.get("model_initialization_seed") != seed
                    or m.get("repair") != "fix_invalid_q2"
                ):
                    raise AssertionError("Repair mismatch")
            v = m["history"][-1]["validation"]["val_relabel"]
            for k in KEYS:
                row[kind + "_" + k] = float(v[k])
        for k in KEYS:
            row["effect_" + k] = row["fix_invalid_q2_" + k] - row["raw_" + k]
        rows.append(row)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "paired_seed_effects.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    d = {
        "experiment": "Independent validation-only Extended G9 to G10 repair",
        "seeds": [5054, 6054],
        "seed_count": 2,
        "training_exposures_each": 300000,
        "repair_definition": "replace invalid q2 with a support symbol AND replace its associated second label",
        "isolates_symbol_without_label": False,
        "final_test_evaluated": False,
        "rows": rows,
        "mean_repair_minus_raw": {
            k: statistics.mean(row["effect_" + k] for row in rows) for k in KEYS
        },
        "limitations": [
            "Two model seeds: descriptive evidence only",
            "Original G3 final test was already opened",
            "This repairs both symbol and its label, so effect cannot be attributed to symbol alone",
        ],
    }
    (a.out / "summary.json").write_text(json.dumps(d, indent=2))
    (a.out / "README.md").write_text(
        "# Independent Extended repair seed effects\n\n"
        "Full 300k-per-descendant experiments on two independent seeds. "
        "No protected final test. See machine-readable CSV and summary.json. "
        "Full checkpoints are in GitHub Actions artifacts with finite retention.\n"
    )
    print("REPAIR_SEED_SUMMARY=" + json.dumps(d["mean_repair_minus_raw"]))


if __name__ == "__main__":
    main()
