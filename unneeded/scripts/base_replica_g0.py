from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

from paper_base_smoke import ROOT, source_checked_cmd


def train(seed: int, out: Path):
    if seed in (0, 1, 5):
        raise ValueError("Choose seed different from original/reference/eval defaults")
    out.mkdir(parents=True, exist_ok=True)
    cmd = source_checked_cmd(
        steps=31250,
        out=out,
        batch_size=32,
        eval_every_updates=31250,
        eval_iters=512,
        ckpt_every_updates=31250,
        max_updates=100000,
    )
    cmd[cmd.index("--init_seed") + 1] = str(seed)
    cmd.extend(["--train_seed", str(seed + 37), "--eval_seed", str(seed + 41)])
    if len({seed, seed + 37, seed + 41}) != 3:
        raise AssertionError(
            "Distinct training/initialisation/evaluation seeds required"
        )
    idx = cmd.index("--pe_classes")
    if cmd[idx + 1 : idx + 5] != ["train", "train", "train", "val"]:
        raise AssertionError("Final test evaluator must be withheld")
    proto = {
        "source": "pinned Singh main.py unmodified",
        "independent_seed": seed,
        "training_seed": seed + 37,
        "validation_seed": seed + 41,
        "original_class_split": [50, 1473, 100],
        "training_sequences": 1_000_000,
        "batch_size": 32,
        "learning_rate": 1e-5,
        "model": "original paper two-layer eight-head Base",
        "final_test_evaluated": False,
        "command": cmd,
    }
    (out / "independent_protocol.json").write_text(json.dumps(proto, indent=2))
    subprocess.run(cmd, cwd=ROOT, check=True)
    cp = out / "safe_train_val_smoke" / "checkpoints" / "00001000000.eqx"
    if not cp.is_file():
        raise FileNotFoundError("Original author's final checkpoint missing")
    print(
        "BASE_INDEPENDENT_G0_COMPLETE="
        + json.dumps(
            {
                "init_seed": seed,
                "checkpoint": str(cp),
                "original_paper_task": True,
                "final_test_evaluated": False,
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init-seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    train(a.init_seed, a.out)


if __name__ == "__main__":
    main()
