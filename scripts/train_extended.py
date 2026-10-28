import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.extended_g0 import start_g0
from scripts.extended_recursive import ARMS, run


def main():
    parser = argparse.ArgumentParser(description="Train Extended G0 through G10")
    parser.add_argument(
        "--base-artifact", type=Path, default=ROOT / "outputs/base_300k_g0_0"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/extended_primary")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARMS,
        default=["clean_only", "recursive_replacement"],
    )
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument("--initialization-seed", type=int, default=4054)
    parser.add_argument("--steps", type=int, default=9375)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=3125)
    parser.add_argument("--eval-count", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-all-replication-seeds", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.out}")

    g0 = args.out / "g00"
    start_g0(
        args.base_artifact,
        g0,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        eval_count=args.eval_count,
        seed=args.initialization_seed,
        lr=args.learning_rate,
        temperature=args.temperature,
    )
    run(
        base_artifact=args.base_artifact,
        g0_artifact=g0,
        out=args.out,
        arms=args.arms,
        generations=10,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        eval_count=args.eval_count,
        seed=args.seed,
        lr=args.learning_rate,
        temperature=args.temperature,
        require_gate=not args.include_all_replication_seeds,
        init_seed=args.initialization_seed,
    )


if __name__ == "__main__":
    main()
