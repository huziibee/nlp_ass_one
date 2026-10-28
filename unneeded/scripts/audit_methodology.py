from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MATCHED = Path("data/evidence/experiments/base_full_budget_300k/summary.json")
CUMULATIVE = Path("data/evidence/experiments/cumulative_corpus/summary.json")
FIVE_SEED = Path("data/evidence/experiments/g10_five_seed/g10_seed_summary.json")
ARMS = ("clean_only", "recursive_replacement")


def read_original(root: Path, relative: Path) -> dict:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing ORIGINAL study file: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def generations(rows: list, label: str) -> None:
    if len(rows) != 10 or [row["generation"] for row in rows] != list(range(1, 11)):
        raise ValueError(f"{label}: expected complete ordered G1--G10 history")


def audit(root: Path = REPO) -> dict:
    matched = read_original(root, MATCHED)
    cumulative = read_original(root, CUMULATIVE)
    five = read_original(root, FIVE_SEED)

    if (matched["g0_exposures"], matched["descendant_exposures"]) != (300000, 300000):
        raise ValueError(
            "Fully matched Base must use 300k examples for G0 and descendants"
        )
    if matched["replicates"] != 1 or matched["protected_final_test"] is not False:
        raise ValueError("Single-seed budget control or final-test isolation invalid")
    if set(matched["arms"]) != set(ARMS):
        raise ValueError("Unexpected fully matched Base conditions")
    for arm in ARMS:
        generations(matched["arms"][arm], f"matched-{arm}")
        for row in matched["arms"][arm]:
            for key in ("val_relabel", "val_classes"):
                if not 0 <= row[key] <= 1:
                    raise ValueError(f"Invalid proportion in matched-{arm}: {key}")

    if (
        cumulative["final_test_evaluated"] is not False
        or cumulative["generated_input_prefixes"] is not False
        or cumulative["historical_examples_replaced"] is not False
        or cumulative["actual_pool_increases_per_generation"] is not True
        or cumulative["clean_history_control_matched_pool_sizes"] is not True
    ):
        raise ValueError("Cumulative study scope or isolation metadata changed")
    if set(cumulative["arms"]) != {"clean_history", "cumulative_recursive"}:
        raise ValueError("Cumulative corpus requires both matched history arms")
    block = cumulative["new_block_per_generation"]
    if block <= 0 or cumulative["fixed_anchor_real_examples"] != block:
        raise ValueError("Unexpected anchor and append blocks")
    for arm, rows in cumulative["arms"].items():
        generations(rows, f"cumulative-{arm}")
        for row in rows:
            g = row["generation"]
            if (
                row["historical_pool_size"] != (g + 1) * block
                or row["training_exposures"]
                != cumulative["per_generation_gradient_exposures"]
                or row["final_test_evaluated"] is not False
            ):
                raise ValueError(
                    f"Unmatched pool size or training budget in {arm}, G{g}"
                )
            if (
                arm == "cumulative_recursive"
                and row["stored_prior_synthetic_blocks"] != g
            ):
                raise ValueError("Cumulative synthetic target history is incomplete")
            if arm == "clean_history" and row["stored_prior_synthetic_blocks"] != 0:
                raise ValueError("Clean-history control contains synthetic blocks")

    if (
        five["n_model_seeds_per_task"] != 5
        or five["independent_test_set_for_G10"] is not False
        or five["training_exposures_per_generation"]
        != {"base": 1000000, "extended": 300000}
    ):
        raise ValueError("Primary five-seed comparison metadata changed")
    a = matched["arms"]["clean_only"][-1]["val_relabel"]
    b = matched["arms"]["recursive_replacement"][-1]["val_relabel"]
    x = cumulative["arms"]["clean_history"][-1]["validation"]["val_relabel"]
    y = cumulative["arms"]["cumulative_recursive"][-1]["validation"]["val_relabel"]
    metric = "free_next_symbol_context_validity"
    if any(metric not in item for item in (x, y)):
        raise ValueError("Original cumulative validity metric is missing")
    return {
        "single_seed_base_300k": {
            "n_model_seeds": 1,
            "g0_exposures": 300000,
            "descendant_exposures": 300000,
            "g10_clean_relabel_accuracy": a,
            "g10_recursive_relabel_accuracy": b,
            "g10_clean_minus_recursive_accuracy": a - b,
            "evidence": MATCHED.as_posix(),
        },
        "single_seed_accumulating_corpus": {
            "n_model_seeds": 1,
            "generation_count": 10,
            "pool_size_g10": cumulative["arms"]["cumulative_recursive"][-1][
                "historical_pool_size"
            ],
            "per_generation_exposures": cumulative["per_generation_gradient_exposures"],
            "g10_clean_history_symbol_validity": x[metric],
            "g10_synthetic_history_symbol_validity": y[metric],
            "g10_clean_minus_synthetic_validity": x[metric] - y[metric],
            "evidence": CUMULATIVE.as_posix(),
        },
        "primary_study": {
            "n_model_seeds": five["n_model_seeds_per_task"],
            "base_descendant_exposures": 1000000,
            "extended_descendant_exposures": 300000,
            "final_test_retained_independent": False,
            "evidence": FIVE_SEED.as_posix(),
        },
        "interpretation": [
            "The brief mandates Base and Extended continuation outputs, not full prompt generation.",
            "Budget-matched Base 300k evidence is a single-seed sensitivity control, not a matched-task effect.",
            "Accumulating corpus uses real input prefixes plus retained generated targets, not generated prompts.",
            "The primary five-seed study used unequal Base/Extended budgets; its cross-task causal comparison remains limited.",
            "The original final test was opened at G3. All these G10 records use validation only.",
        ],
    }


def write_report(report: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "METHOD_CONTROLS.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    b = report["single_seed_base_300k"]
    c = report["single_seed_accumulating_corpus"]
    lines = [
        "# Methodology control audit",
        "",
        "This table is calculated from existing original experimental JSON; it is not new training.",
        "",
        "| Control | Seeds | G10 clean | G10 synthetic | Clean minus synthetic |",
        "|:--|--:|--:|--:|--:|",
        (
            f"| Base 300k G0 + descendants, validation label accuracy | 1 | "
            f"{b['g10_clean_relabel_accuracy']:.2%} | {b['g10_recursive_relabel_accuracy']:.2%} | "
            f"{100*b['g10_clean_minus_recursive_accuracy']:+.2f} pp |"
        ),
        (
            f"| Accumulated training examples, next-symbol context validity | 1 | "
            f"{c['g10_clean_history_symbol_validity']:.2%} | "
            f"{c['g10_synthetic_history_symbol_validity']:.2%} | "
            f"{100*c['g10_clean_minus_synthetic_validity']:+.2f} pp |"
        ),
        "",
        "The 300k Base budget matches Extended's exposure count, **not** its output task or architecture.",
        "The corpus-control experiment retains past examples and appends new generated targets.",
        "Neither single-seed control can replace the original five-seed uncertainty analysis.",
        "",
        "## Sources",
        "",
        f"- \`{MATCHED}\`",
        f"- \`{CUMULATIVE}\`",
        f"- \`{FIVE_SEED}\`",
        "",
        "## Limits",
        "",
    ] + [f"- {s}" for s in report["interpretation"]]
    text = "\n".join(lines).replace("\\`", "`")
    (out / "METHOD_CONTROLS.md").write_text(text + "\n", encoding="utf-8")
    with (out / "METHOD_CONTROLS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "control",
                "model_seeds",
                "g10_clean",
                "g10_synthetic",
                "clean_minus_synthetic",
                "metric",
            )
        )
        writer.writerow(
            (
                "base_300k",
                1,
                b["g10_clean_relabel_accuracy"],
                b["g10_recursive_relabel_accuracy"],
                b["g10_clean_minus_recursive_accuracy"],
                "validation_label_accuracy",
            )
        )
        writer.writerow(
            (
                "cumulative_example_history",
                1,
                c["g10_clean_history_symbol_validity"],
                c["g10_synthetic_history_symbol_validity"],
                c["g10_clean_minus_synthetic_validity"],
                "validation_context_validity",
            )
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=REPO)
    p.add_argument("--out", type=Path, default=REPO / "outputs/reports")
    args = p.parse_args()
    report = audit(args.root)
    write_report(report, args.out)
    print(
        "METHOD_CONTROLS_VERIFIED=single-seed 300k Base and accumulating-corpus, validation-only"
    )
    print(f"Output: {args.out / 'METHOD_CONTROLS.md'}")


if __name__ == "__main__":
    main()
