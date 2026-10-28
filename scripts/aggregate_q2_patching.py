import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing the ten patch JSON files",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSON file")
    args = parser.parse_args()

    files = list(args.input.rglob("*.json"))
    if len(files) != 10:
        raise ValueError(
            f"Expected ten patch files, found {len(files)} in {args.input}"
        )
    records = [json.loads(path.read_text()) for path in files]
    pairs = {
        (record["arm"], int(record["trained_model_init_seed"])) for record in records
    }
    if len(pairs) != 10:
        raise ValueError("Expected one measurement per arm and seed")

    getters = {
        "natural_context_preference": lambda x: x["layers"]["2"][
            "recipient_margin_mean"
        ]
        - x["layers"]["2"]["donor_margin_mean"],
        "first_layer_patch_shift": lambda x: x["layers"]["1"][
            "whole_layer_patch_margin_shift"
        ],
        "second_layer_patch_shift": lambda x: x["layers"]["2"][
            "whole_layer_patch_margin_shift"
        ],
        "recipient_valid_mass": lambda x: x["layers"]["2"]["recipient_valid_mass"],
        "second_layer_patch_valid_mass": lambda x: x["layers"]["2"][
            "whole_layer_patch_valid_mass"
        ],
    }
    result = {
        "n_independent_model_pairs": 5,
        "n_validation_episodes_per_model": 128,
        "metric": "context-switch logit preference difference; not accuracy",
        "arms": {},
        "notes": [
            "single selected matched pair of contexts used for all models",
            "different model weights across seeds, patches always within one model",
            "patching intervention demonstrates internal information transfer, not causal effect of corrupted training data",
        ],
    }
    for arm in ("clean_only", "recursive_replacement"):
        selected = [record for record in records if record["arm"] == arm]
        if len(selected) != 5:
            raise ValueError(f"Expected five {arm} measurements, found {len(selected)}")
        for record in selected:
            if record["final_test_accessed"]:
                raise ValueError("A measurement accessed the final test set")
            for layer in record["layers"].values():
                if (
                    layer["identity_patch_max_error"] >= 3e-5
                    or layer["future_token_patch_max_error"] >= 3e-5
                ):
                    raise ValueError("Patch identity or future-token check failed")
        result["arms"][arm] = {}
        for name, getter in getters.items():
            values = [getter(record) for record in selected]
            result["arms"][arm][name] = {
                "n": 5,
                "mean": statistics.mean(values),
                "sd": statistics.stdev(values),
                "individual": values,
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
