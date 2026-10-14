"""Smoke-test the unmodified authors' original Base transformer on TRAIN/VAL.

Runs a handful of updates only, NEVER touches the 100 final-test classes,
and never claims to have reproduced learned induction-head mechanisms.
For genuine research reproduction the original paper used 1,000,000
training sequences; the authors' '--train_iters' counts examples, not
optimizer updates. Here '--steps' means minibatch updates.
This wrapper retains its architecture/dataset while replacing its
periodic test-class evaluator with a validation-class evaluator.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "external" / "icl-dynamics"
DATA = PAPER / "omniglot_resnet18_randomized_order_s0.h5"


def source_checked_cmd(steps: int, out: Path, batch_size: int = 8,
                       eval_every_updates: int | None = None,
                       eval_iters: int = 16):
    if not (PAPER / "main.py").is_file() or not DATA.is_file():
        raise FileNotFoundError("Missing authors' pinned submodule; initialize submodules")
    if steps < 1 or steps > 1000:
        raise ValueError("This is only a bounded SMOKE RUN (1..1000 steps)")
    if batch_size < 1 or batch_size > 32:
        raise ValueError("Use batch size 1..32 for this small smoke test")
    if eval_every_updates is None:
        eval_every_updates = steps
    if not 1 <= eval_every_updates <= steps:
        raise ValueError("Evaluation interval must be within run updates")
    if not 1 <= eval_iters <= 2048:
        raise ValueError("Evaluation examples per split must be 1..2048")
    cmd = [
        sys.executable, str(PAPER / "main.py"),
        "--data_file", str(DATA),
        "--mixing_coeffs", "1.0",
        "--pt_burstiness", "1",
        "--train_context_len", "2",
        "--fs_relabel", "5",
        "--fs_relabel_split", "8", "2", "0",
        "--exemplar_split", "1", "4", "0",
        "--class_split", "50", "1473", "100",
        # IMPORTANT: no --pe_classes 'test' while tuning the model.
        "--pe_names", "fsl_train", "fsl_val_rl", "fsl_train_valex", "fsl_val_class",
        "--pe_classes", "train", "train", "train", "val",
        "--pe_exemplars", "train", "train", "val", "train",
        "--pe_fs_relabel_scheme", "train", "val", "train", "train",
        "--pe_burstiness", "1", "1", "1", "1",
        "--train_iters", str(steps * batch_size),  # Authors count SEQUENCES, not updates.
        "--train_bs", str(batch_size),
        "--eval_every", str(eval_every_updates * batch_size),
        "--eval_iters", str(eval_iters),
        "--ckpt_every", str(steps * batch_size),
        "--lr", "0.00001",
        "--d_model", "64",
        "--init_seed", "5",
        "--run", "safe_train_val_smoke",
        "--base_folder", str(out),
        "--raw_name",
    ]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "paper_smoke")
    p.add_argument("--eval-every-updates", type=int, default=None)
    p.add_argument("--eval-iters", type=int, default=16)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cmd = source_checked_cmd(args.steps, args.out, args.batch_size,
                             args.eval_every_updates, args.eval_iters)
    if "--pe_classes" not in cmd:
        raise AssertionError("No evaluator class scopes")
    idx = cmd.index("--pe_classes")
    tail = cmd[idx+1:idx+5]
    if "test" in tail:
        raise AssertionError("Forbidden final-test class evaluator in smoke test")
    meta = {
        "source": "pinned original authors' JAX implementation",
        "optimizer_updates": args.steps,
        "eval_every_optimizer_updates": args.eval_every_updates or args.steps,
        "eval_examples_per_split": args.eval_iters,
        "sequence_exposures": args.steps * args.batch_size,
        "smoke_only_not_trained_icl": True,
        "eval_classes": tail,
        "final_test_scored": False,
        "cmd": cmd,
    }
    print("SMOKE_METADATA=" + json.dumps(meta), flush=True)
    if args.dry_run:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    print("ORIGINAL_BASE_MODEL_SMOKE_SUCCEEDED", flush=True)


if __name__ == "__main__":
    main()
