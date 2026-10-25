#!/usr/bin/env python3
"""Fresh independent Base 300k G0 and ten-generation clean/recursive chains.

Uses the original Singh et al. Omniglot task, the pinned original feature
HDF5 and the same exact 300,000-example training exposure used by Extended.
Never evaluates the historical 100-class test. Every model seed is retained.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INIT_SEEDS = (2105, 2205, 2305, 2405, 2505)
DATA_SEEDS = (44001, 45001, 46001, 47001, 48001)
ARMS = ("clean_only", "recursive_replacement")
EPISODES = 300_000
BATCH_SIZE = 32
STEPS = EPISODES // BATCH_SIZE
CHECKPOINT = "00000300000.eqx"


def protocol(seed_index: int) -> dict:
    if seed_index not in range(5):
        raise ValueError("Choose a preregistered seed_index in 0..4")
    return {
        "study": "independent-five-seed-300k-base",
        "seed_index": seed_index,
        "independent_initialization_seed": INIT_SEEDS[seed_index],
        "independent_data_seed": DATA_SEEDS[seed_index],
        "G0_exposures": EPISODES,
        "descendant_exposures": EPISODES,
        "generations": 10,
        "arms": list(ARMS),
        "class_split": [50, 1473, 100],
        "final_test_evaluated": False,
        "source": "pinned Singh task and original Omniglot features",
        "seed_inclusion": "All five seeded chains reported, no validation-based filtering",
    }


def check_g0(root: Path, seed_index: int) -> dict:
    p = protocol(seed_index)
    existing = json.loads((root / "independent_protocol.json").read_text(encoding="utf-8"))
    if existing != p:
        raise ValueError("G0 provenance does not match predefined 300k protocol")
    cfg = json.loads((root / "safe_train_val_smoke/config.json").read_text(encoding="utf-8"))
    if (cfg.get("train_iters") != EPISODES
            or cfg.get("train_bs") != BATCH_SIZE
            or cfg.get("class_split") != [50, 1473, 100]
            or cfg.get("exemplar_split") != [1, 4, 0]
            or cfg.get("fs_relabel_split") != [8, 2, 0]):
        raise ValueError("Incorrect original Base training budget or data split")
    cp = root / "safe_train_val_smoke/checkpoints" / CHECKPOINT
    if not cp.is_file() or cp.stat().st_size < 1000:
        raise FileNotFoundError(f"Original G0 checkpoint not found: {cp}")
    return p


def train_g0(root: Path, seed_index: int) -> None:
    from scripts.paper_base_smoke import source_checked_cmd
    if root.exists():
        raise FileExistsError(f"Refuse to overwrite a G0 experiment: {root}")
    p = protocol(seed_index)
    command = source_checked_cmd(
        steps=STEPS, out=root, batch_size=BATCH_SIZE,
        eval_every_updates=STEPS, eval_iters=512,
        ckpt_every_updates=STEPS, max_updates=100000,
    )
    command[command.index("--init_seed") + 1] = str(INIT_SEEDS[seed_index])
    command += [
        "--train_seed", str(INIT_SEEDS[seed_index] + 37),
        "--eval_seed", str(INIT_SEEDS[seed_index] + 41)
    ]
    ix = command.index("--pe_classes")
    if command[ix + 1:ix + 5] != ["train", "train", "train", "val"]:
        raise ValueError("Unexpected final-test access configuration")
    root.mkdir(parents=True)
    (root / "independent_protocol.json").write_text(json.dumps(p, indent=2) + "\n")
    (root / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    subprocess.run(command, cwd=ROOT, check=True)
    check_g0(root, seed_index)
    print(f"BASE_300K_FRESH_G0_COMPLETE={seed_index}", flush=True)


def check_chain(root: Path, arm: str, seed_index: int) -> dict:
    if arm not in ARMS:
        raise ValueError("Invalid arm")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("final_test_evaluated") is not False or set(summary.get("arms", {})) != {arm}:
        raise ValueError("Arm, final-test or summary inconsistency")
    rows = summary["arms"][arm]
    if len(rows) != 10 or [v["generation"] for v in rows] != list(range(1, 11)):
        raise ValueError("Expected all ten generations G1--G10")
    for g in range(1, 11):
        d = root / arm / f"g{g:02d}"
        m = json.loads((d / "metrics.json").read_text())
        if (m.get("generation") != g
                or m.get("optimizer_exposures") != EPISODES
                or m.get("data_seed") != DATA_SEEDS[seed_index]
                or m.get("initialisation_seed") != DATA_SEEDS[seed_index] + 17*g
                or m.get("source_G0_training_exposures") != EPISODES
                or m.get("source_G0_initialization_seed") != INIT_SEEDS[seed_index]
                or m.get("final_test_evaluated") is not False
                or m.get("whole_prefix_generated") is not False
                or m.get("class_split") != [50, 1473, 100]):
            raise ValueError(f"Incorrect G{g} training budget/seed/test metadata")
        if not (d / "model.eqx").is_file() or (d / "model.eqx").stat().st_size < 1000:
            raise FileNotFoundError(f"Missing trained G{g} model")
    score = float(rows[-1]["last_validation"]["val_relabel"]["accuracy"])
    if not 0 <= score <= 1:
        raise ValueError("Invalid G10 accuracy")
    return {"seed_index":seed_index, "data_seed":DATA_SEEDS[seed_index],
            "arm":arm, "G10_val_relabel_accuracy":score}


def train_chain(g0_root: Path, out: Path, arm: str, seed_index: int) -> None:
    check_g0(g0_root, seed_index)
    if out.exists():
        raise FileExistsError(f"Refuse to overwrite a descendant chain: {out}")
    import scripts.base_recursive as source_recursive
    from scripts.base_fully_matched_300k import load_g0
    original = source_recursive.load_original_checkpoint
    try:
        source_recursive.load_original_checkpoint = load_g0
        source_recursive.experiment(
            g0_root, out, generations=10,
            steps=STEPS, batch_size=BATCH_SIZE, eval_every=STEPS,
            eval_count=512, seed=DATA_SEEDS[seed_index],
            temperature=1.0, arms=[arm],
        )
    finally:
        source_recursive.load_original_checkpoint = original
    # Correct a description inherited from the untouched historical 1M code.
    # This adds transparent provenance and changes no trained weights or scores.
    for g in range(1,11):
        path = out / arm / f"g{g:02d}" / "metrics.json"
        m = json.loads(path.read_text(encoding="utf-8"))
        m["baseline"] = "Fresh Original-Singh Base G0 at 300000 episodes"
        m["source_G0_training_exposures"] = EPISODES
        m["source_G0_initialization_seed"] = INIT_SEEDS[seed_index]
        path.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    score = check_chain(out, arm, seed_index)
    print("BASE_300K_FULL_CHAIN_COMPLETE=" + json.dumps(score), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "g0", "chain", "verify-g0", "verify-chain"))
    p.add_argument("--seed-index", type=int, default=0, choices=range(5))
    p.add_argument("--arm", choices=ARMS, default="clean_only")
    p.add_argument("--g0-root", type=Path, default=ROOT / "outputs/new_base_300k_g0")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/new_base_300k_arm")
    a = p.parse_args()
    if a.command == "plan":
        print(json.dumps([protocol(i) for i in range(5)], indent=2))
    elif a.command == "g0":
        train_g0(a.g0_root, a.seed_index)
    elif a.command == "chain":
        train_chain(a.g0_root, a.out, a.arm, a.seed_index)
    elif a.command == "verify-g0":
        print(json.dumps(check_g0(a.g0_root, a.seed_index)))
    else:
        print(json.dumps(check_chain(a.out, a.arm, a.seed_index)))


if __name__ == "__main__":
    main()
