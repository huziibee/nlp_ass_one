from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "icl-dynamics"))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from scripts.extended_g0 import load_config, feature_prototypes
from scripts.extended_recursive import (
    load_verified_g0,
    train_generation,
    training_targets,
)
from src.extended_omniglot import ExtendedOmniglot, make_episodes, clean_continuation
from src.extended_learner import build_optimizer, extended_train_step, evaluate_extended

REPAIRS = (
    "raw",
    "fix_invalid_q2",
    "fix_valid_second",
    "fix_joint_q2_second",
    "fix_first",
    "fix_all",
)


def repaired_targets(repair, batch, clean, raw):
    if repair not in REPAIRS:
        raise ValueError("Unknown intervention")
    before = jnp.asarray(raw, dtype=jnp.int32)
    if before.ndim != 2 or before.shape[1] != 3:
        raise ValueError("Extended continuation has 3 integer tokens")
    ctx_ids = batch["class_ids"][:, :2]
    ctx_labels = batch["labels"][:, :2]
    first = before[:, 0]
    q2 = before[:, 1]
    second = before[:, 2]
    valid0 = q2 == ctx_ids[:, 0]
    valid1 = q2 == ctx_ids[:, 1]
    valid = valid0 | valid1
    correct_second = jnp.where(valid0, ctx_labels[:, 0], ctx_labels[:, 1])
    invalid = ~valid
    corrected_q2 = clean["tokens"][:, 1]
    corrected_second = clean["tokens"][:, 2]
    if repair in ("fix_invalid_q2", "fix_joint_q2_second", "fix_all"):
        q2 = jnp.where(invalid, corrected_q2, q2)
        second = jnp.where(invalid, corrected_second, second)
    if repair in ("fix_valid_second", "fix_joint_q2_second", "fix_all"):
        second = jnp.where(valid, correct_second, second)
    if repair in ("fix_first", "fix_all"):
        first = batch["labels"][:, -1]
    out = jnp.stack((first, q2, second), axis=-1)
    if not bool(jnp.all((out[:, 0] >= 0) & (out[:, 0] < 5))):
        raise AssertionError("Invalid repaired first-label token")
    if not bool(jnp.all((out[:, 1] >= 0) & (out[:, 1] < 50))):
        raise AssertionError("Invalid repaired symbol vocabulary")
    if not bool(jnp.all((out[:, 2] >= 0) & (out[:, 2] < 5))):
        raise AssertionError("Invalid repaired second-label token")
    new0 = out[:, 1] == ctx_ids[:, 0]
    new1 = out[:, 1] == ctx_ids[:, 1]
    new_valid = new0 | new1
    new_answer = jnp.where(new0, ctx_labels[:, 0], ctx_labels[:, 1])
    if repair in ("fix_invalid_q2", "fix_joint_q2_second", "fix_all"):
        if not bool(jnp.all(new_valid)):
            raise AssertionError("Structural correction failed")
    if repair in ("fix_joint_q2_second", "fix_all"):
        if not bool(jnp.all(out[:, 2] == new_answer)):
            raise AssertionError("Joint repair failed to produce contextual number")
    if repair == "fix_first" or repair == "fix_all":
        if not bool(jnp.all(out[:, 0] == batch["labels"][:, -1])):
            raise AssertionError("First number was not corrected")
    stats = {
        "source_off_context_symbols": int(np.asarray(invalid).sum()),
        "source_context_valid_wrong_second_number": int(
            np.asarray(valid & (before[:, 2] != correct_second)).sum()
        ),
        "source_wrong_first_numbers": int(
            np.asarray(before[:, 0] != batch["labels"][:, -1]).sum()
        ),
        "intervened_number_of_target_rows": int(
            np.asarray(jnp.any(out != before, axis=-1)).sum()
        ),
        "corrected_off_context_symbols_remaining": int(np.asarray(~new_valid).sum()),
    }
    return out, stats


def load_previous_g9(cfg, root, *, init_seed: int, data_seed: int):
    folder = root / "recursive_replacement" / "g09"
    meta = json.loads((folder / "metrics.json").read_text())
    if (
        meta["generation"] != 9
        or meta["condition"] != "recursive_replacement"
        or meta["final_test_evaluated"] is not False
        or meta["init_seed"] != init_seed
        or meta["data_seed"] != data_seed
        or meta["training_exposures"] != 300000
    ):
        raise AssertionError("G9 generator source is not the frozen primary experiment")
    model = ExtendedOmniglot(config=cfg, seed=int(meta["init_seed"]))
    model = eqx.tree_deserialise_leaves(str(folder / "model.eqx"), model)
    return model


def train_repaired(
    *,
    name,
    source_model,
    g0,
    cfg,
    prototypes,
    sampler,
    val,
    out,
    steps,
    seed,
    temperature,
    lr,
    init_seed,
):
    model = ExtendedOmniglot(config=cfg, seed=init_seed)
    optimizer, opt_state = build_optimizer(model, lr)
    folder = out / name / "g10"
    folder.mkdir(parents=True, exist_ok=True)
    history = []
    total = {
        "source_off_context_symbols": 0,
        "source_context_valid_wrong_second_number": 0,
        "source_wrong_first_numbers": 0,
        "intervened_number_of_target_rows": 0,
        "corrected_off_context_symbols_remaining": 0,
    }
    start = time.time()
    for step in range(steps):
        key = jax.random.PRNGKey(seed + 10 * 103_111 + step)
        k_episode, k_clean, k_synth, k_train = jax.random.split(key, 4)
        batch = sampler(k_episode)
        clean = clean_continuation(batch, key=k_clean)
        raw, _ = training_targets(
            "recursive_replacement",
            batch,
            clean,
            source_model,
            prototypes,
            data_key=k_synth,
            temperature=temperature,
        )
        target, stats = repaired_targets(name, batch, clean, raw)
        for k, v in stats.items():
            total[k] += v
        q2_features = prototypes[target[:, 1]]
        loss, model, opt_state = extended_train_step(
            model, optimizer, opt_state, batch, target, q2_features, k_train
        )
        if (step + 1) % 3125 == 0 or step + 1 == steps:
            metrics = evaluate_extended(
                model,
                val,
                feature_lookup=prototypes,
                seed=seed + 170_000,
                temperature=temperature,
            )
            history.append(
                {
                    "step": step + 1,
                    "train_loss": float(loss),
                    "train_exposures": (step + 1) * 32,
                    "validation": metrics,
                }
            )
            print(
                "EXTENDED_REPAIR_PROGRESS="
                + json.dumps(
                    {
                        "repair": name,
                        "step": step + 1,
                        "validation": metrics["val_relabel"],
                        "final_test_access": False,
                    }
                ),
                flush=True,
            )
    eqx.tree_serialise_leaves(str(folder / "model.eqx"), model)
    evidence = {
        "experiment": "G9-generated targets -> corrected G10 training",
        "repair": name,
        "generation": 10,
        "source_generator": "same verified original Extended recursive G9",
        "class_split": [50, 1473, 100],
        "train_exposures": steps * 32,
        "optimization_steps": steps,
        "batch_size": 32,
        "source_data_seed": seed,
        "model_initialization_seed": init_seed,
        "learning_rate": lr,
        "temperature": temperature,
        "real_prefix_preserved": True,
        "intervention_on_synthetic_training_targets_only": True,
        "all_conditions_match_unrepaired_training_except_targets": True,
        "total_synthetic_target_quality": total,
        "history": history,
        "runtime_seconds": round(time.time() - start, 3),
        "final_test_access": False,
        "model_seed_replicates": 1,
    }
    (folder / "metrics.json").write_text(json.dumps(evidence, indent=2))
    return evidence


def run(
    *,
    base_root,
    g0_root,
    g10_root,
    repair,
    out,
    steps,
    seed,
    lr,
    temperature,
    init_seed,
    include_all_replication_seeds,
):
    if repair not in REPAIRS:
        raise ValueError("Unknown repair")
    if steps != 9375 or abs(lr - 0.0003) > 1e-12 or temperature != 1:
        raise ValueError("Keep frozen 300k G10 training protocol and hyperparameters")
    cfg = load_config(base_root)
    if not include_all_replication_seeds and (seed != 6001 or init_seed != 4054):
        raise ValueError(
            "Only the original frozen G10 or explicitly marked independent replications are supported"
        )
    g0, g0meta = load_verified_g0(
        cfg, g0_root, require_gate=not include_all_replication_seeds
    )
    if int(g0meta["seed"]) != init_seed:
        raise AssertionError("Selected original G0 seed changed")
    source = load_previous_g9(cfg, g10_root, init_seed=init_seed, data_seed=seed)
    train = make_episodes(cfg, role="train", batch_size=32)
    val = {
        role: make_episodes(cfg, role=role, batch_size=512)(
            jax.random.PRNGKey(seed + 911 + i)
        )
        for i, role in enumerate(("val_relabel", "val_exemplar", "val_classes"))
    }
    prototypes = feature_prototypes(cfg)
    existing_meta = json.loads(
        (g10_root / "recursive_replacement" / "g10" / "metrics.json").read_text()
    )
    if (
        existing_meta["final_test_evaluated"]
        or existing_meta["generation"] != 10
        or existing_meta["init_seed"] != init_seed
        or existing_meta["data_seed"] != seed
    ):
        raise AssertionError("Invalid reference G10 model provenance")
    if repair == "raw":
        _, m = train_generation(
            arm="recursive_replacement",
            generation=10,
            predecessor=source,
            g0=g0,
            config=cfg,
            train_sampler=train,
            val=val,
            prototypes=prototypes,
            steps=steps,
            batch_size=32,
            eval_every=3125,
            seed=seed,
            lr=lr,
            temperature=temperature,
            out=out / "raw",
            init_seed=init_seed,
        )
        new = m["history"][-1]["validation"]
    else:
        m = train_repaired(
            name=repair,
            source_model=source,
            g0=g0,
            cfg=cfg,
            prototypes=prototypes,
            sampler=train,
            val=val,
            out=out,
            steps=steps,
            seed=seed,
            temperature=temperature,
            lr=lr,
            init_seed=init_seed,
        )
        new = m["history"][-1]["validation"]
    original = existing_meta["history"][-1]["validation"]
    if repair == "raw":
        for role in original:
            for key in ("first_label_accuracy", "teacher_forced_second_label_accuracy"):
                if abs(original[role][key] - new[role][key]) > 1e-6:
                    raise AssertionError("Raw G10 did not reproduce checkpoint results")
            if role != "val_classes":
                if (
                    abs(
                        original[role]["free_next_symbol_context_validity"]
                        - new[role]["free_next_symbol_context_validity"]
                    )
                    > 1e-6
                ):
                    raise AssertionError("Raw G10 generator result mismatch")
    print(
        "EXTENDED_REPAIR_G10_COMPLETED="
        + json.dumps(
            {
                "repair": repair,
                "reproduced_existing_G10": repair == "raw",
                "last_validation": new["val_relabel"],
                "final_test_access": False,
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-root", type=Path, default=ROOT / "outputs/base_300k_g0_0")
    p.add_argument("--g0-root", type=Path, default=ROOT / "outputs/extended_g0")
    p.add_argument("--g10-root", type=Path, required=True)
    p.add_argument("--repair", choices=REPAIRS, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=9375)
    p.add_argument("--seed", type=int, default=6001)
    p.add_argument("--init-seed", type=int, default=4054)
    p.add_argument(
        "--include-all-replication-seeds",
        action="store_true",
        help="Replicate the same target repair on unselected independent G0 model seeds",
    )
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--temperature", type=float, default=1.0)
    a = p.parse_args()
    run(
        base_root=a.base_root,
        g0_root=a.g0_root,
        g10_root=a.g10_root,
        repair=a.repair,
        out=a.out,
        steps=a.steps,
        seed=a.seed,
        lr=a.learning_rate,
        temperature=a.temperature,
        init_seed=a.init_seed,
        include_all_replication_seeds=a.include_all_replication_seeds,
    )


if __name__ == "__main__":
    main()
