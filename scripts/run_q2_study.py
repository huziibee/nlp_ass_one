import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {6001: 4054, 5054: 5054, 6054: 6054, 7054: 7054, 8054: 8054}


def run_script(name, *arguments):
    command = [sys.executable, str(ROOT / "scripts" / name), *map(str, arguments)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Train one matched repair and run its activation patches"
    )
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument(
        "--base-root", type=Path, default=ROOT / "outputs/base_300k_g0_0"
    )
    parser.add_argument("--g0-root", type=Path, required=True)
    parser.add_argument("--recursive-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/q2_study")
    args = parser.parse_args()

    for path in (
        args.g0_root / "extended_g0_model.eqx",
        args.recursive_root / "recursive_replacement/g09/model.eqx",
        args.recursive_root / "recursive_replacement/g10/model.eqx",
        args.clean_root / "clean_only/g10/model.eqx",
    ):
        if not path.is_file():
            parser.error(f"Missing checkpoint: {path}")

    repair_root = args.out / "repairs" / str(args.seed)
    if repair_root.exists() and any(repair_root.iterdir()):
        parser.error(f"Repair directory is not empty: {repair_root}")
    run_script(
        "extended_g10_repair.py",
        "--include-all-replication-seeds",
        "--base-root",
        args.base_root,
        "--g0-root",
        args.g0_root,
        "--g10-root",
        args.recursive_root,
        "--repair",
        "fix_invalid_q2",
        "--seed",
        args.seed,
        "--init-seed",
        SEEDS[args.seed],
        "--steps",
        9375,
        "--learning-rate",
        0.0003,
        "--temperature",
        1.0,
        "--out",
        repair_root,
    )

    models = {
        "clean_only": args.clean_root / "clean_only/g10",
        "recursive_replacement": args.recursive_root / "recursive_replacement/g10",
        "fix_invalid_q2": repair_root / "fix_invalid_q2/g10",
    }
    for arm, model in models.items():
        directory = "repaired_patches" if arm == "fix_invalid_q2" else "raw_patches"
        output = args.out / directory / f"{args.seed}-{arm}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        run_script(
            "q2_context_activation_patch_pilot.py",
            "--arm",
            arm,
            "--base-root",
            args.base_root,
            "--model-root",
            model,
            "--out",
            output,
            "--n",
            128,
        )


if __name__ == "__main__":
    main()
