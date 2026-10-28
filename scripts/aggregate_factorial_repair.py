import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for condition in ("raw", "symbol-only", "number-only", "joint"):
        parser.add_argument(
            f"--{condition}",
            type=Path,
            required=True,
            help="Directory containing one metrics.json",
        )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for condition in ("raw", "symbol_only", "number_only", "joint"):
        folder = getattr(args, condition)
        files = list(folder.rglob("metrics.json"))
        if len(files) != 1:
            raise ValueError(
                f"Expected one metrics.json for {condition}, found {len(files)} in {folder}"
            )
        data = json.loads(files[0].read_text())
        if data.get("final_test_access", data.get("final_test_evaluated")) is not False:
            raise ValueError(f"Final test status missing or invalid for {condition}")
        if data.get("training_exposures", data.get("train_exposures")) != 300000:
            raise ValueError(f"Expected 300000 training exposures for {condition}")
        if (
            condition in ("symbol_only", "number_only")
            and data.get("single_field_repair_mode") != condition
        ):
            raise ValueError(f"Incorrect repair mode for {condition}")
        metrics = data["history"][-1]["validation"]["val_relabel"]
        rows.append(
            {
                "condition": condition,
                "context_validity": metrics["free_next_symbol_context_validity"],
                "valid_symbol_plus_correct_number": metrics[
                    "free_second_label_context_consistency_all"
                ],
                "first_label_accuracy": metrics["first_label_accuracy"],
                "second_teacher_forced_accuracy": metrics[
                    "teacher_forced_second_label_accuracy"
                ],
            }
        )
    result = {
        "results": rows,
        "model_seed": 4054,
        "data_seed": 6001,
        "n_seeds": 1,
        "final_test_evaluated": False,
        "limitations": [
            "On invalid q2 an oracle second label does not exist",
            "Number-only correction applies an otherwise-valid support number to an invalid symbol",
            "Single model seed; do not infer unique mediated cause",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
