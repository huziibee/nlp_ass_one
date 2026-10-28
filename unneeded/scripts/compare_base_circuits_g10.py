from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "icl-dynamics"))

import equinox as eqx
import numpy as np
import jax
import main_utils

from base_mechanistic import (
    load_original_checkpoint,
    source_episodes,
    make_forward,
    run_fn,
    match_attention,
)

FIXED_G0_INDUCTION_HEAD = 3
ROLES = ("val_relabel", "val_classes")
GENERATIONS = (0, 3, 6, 10)


def load_original_generation(config, root, *, generation, arm):
    folder = root / arm / f"g{generation:02d}"
    path = folder / "model.eqx"
    metric = folder / "metrics.json"
    if not path.is_file() or not metric.is_file():
        raise FileNotFoundError(
            "Missing original Base checkpoint or source metadata: " + str(folder)
        )
    m = json.loads(metric.read_text())
    if m.get("arm") != arm or m.get("generation") != generation:
        raise AssertionError("Wrong model arm or generation")
    if m.get("final_test_evaluated") is not False:
        raise AssertionError("Refuse post-test training source")
    if m.get("class_split") != [50, 1473, 100] or m.get("label_pair_split") != [
        8,
        2,
        0,
    ]:
        raise AssertionError("Wrong paper-sourced data split")
    if m.get("whole_prefix_generated") is not False:
        raise AssertionError("Incorrect provenance claim")
    if m.get("optimizer_exposures") != 1_000_000:
        raise AssertionError("Expected full training, not smoke")
    seed = int(m["initialisation_seed"])
    model = main_utils.get_model_from_opts(
        SimpleNamespace(**{**config, "init_seed": seed})
    )
    model = eqx.tree_deserialise_leaves(str(path), model)
    return model, m


def probe(model, config, batch, *, seed, induction_head, previous_head):
    natural, ce, aux = run_fn(model, make_forward(config), batch, seed=seed)
    scores = match_attention(batch, aux)
    h = induction_head
    drop, drop_loss, _ = run_fn(
        model, make_forward(config, ablate=[f"1:{h}"]), batch, seed=seed
    )
    keep, keep_loss, _ = run_fn(
        model,
        make_forward(config, ablate=[f"1:{other}" for other in range(8) if other != h]),
        batch,
        seed=seed,
    )
    repair, repair_loss, _ = run_fn(
        model, make_forward(config, force_match=[f"1:{h}"]), batch, seed=seed
    )
    prev, prev_loss, _ = run_fn(
        model, make_forward(config, ablate=[f"0:{previous_head}"]), batch, seed=seed
    )
    return {
        "n": len(batch["labels"]),
        "natural_accuracy": natural,
        "natural_cross_entropy": ce,
        "fixed_head3_knockout_accuracy": drop,
        "fixed_head3_knockout_ce": drop_loss,
        "fixed_head3_retain_only_accuracy": keep,
        "fixed_head3_retain_only_ce": keep_loss,
        "fixed_head3_oracle_match_accuracy": repair,
        "fixed_head3_oracle_match_ce": repair_loss,
        "fixed_previous_head_knockout_accuracy": prev,
        "fixed_previous_head_knockout_ce": prev_loss,
        "fixed_head3_query_label_matching_margin": scores[
            "per_head_L2_matching_margin"
        ][h],
        "fixed_head3_correct_context_label_attention": scores[
            "per_head_L2_correct_label_attention"
        ][h],
        "fixed_head3_wrong_context_label_attention": scores[
            "per_head_L2_wrong_label_attention"
        ][h],
        "fixed_head3_negative_match_fraction": scores[
            "per_head_L2_negative_margin_fraction"
        ][h],
        "fixed_previous_head_attention": scores["per_head_L1_previous_token_attention"][
            previous_head
        ],
        "all_heads_L1_previous_token_attention": scores[
            "per_head_L1_previous_token_attention"
        ],
        "all_heads_L2_match_margin": scores["per_head_L2_matching_margin"],
    }


def evaluate(*, base_artifact, g3_root, g10_root, arm, out, n, seed):
    g0, config = load_original_checkpoint(base_artifact)
    train = source_episodes(config, role="train", n=256, seed=3201)
    _, _, ret = run_fn(g0, make_forward(config), train, seed=3211)
    mech = match_attention(train, ret)

    if int(np.argmax(mech["per_head_L2_matching_margin"])) != FIXED_G0_INDUCTION_HEAD:
        raise AssertionError("Original G0 training-only head identity inconsistent")
    previous_head = int(np.argmax(mech["per_head_L1_previous_token_attention"]))
    samples = {
        role: source_episodes(config, role=role, n=n, seed=seed + 200 + i * 43)
        for i, role in enumerate(ROLES)
    }
    result = {
        "task": "Original Singh 2-layer Base circuit comparison",
        "arm": arm,
        "generations": list(GENERATIONS),
        "chosen_induction_head_index": FIXED_G0_INDUCTION_HEAD,
        "chosen_previous_head_index_from_train_G0": previous_head,
        "heads_selected_from": "original G0 train episodes",
        "validation_roles": list(ROLES),
        "test_evaluated": False,
        "model_seed_replicates": 1,
        "optogenetic_test_time_interventions_not_training_causal": True,
        "results": {},
    }
    for generation in GENERATIONS:
        root = g3_root if generation == 3 else g10_root
        model, m = (
            (g0, None)
            if generation == 0
            else load_original_generation(config, root, generation=generation, arm=arm)
        )
        result["results"][str(generation)] = {
            role: probe(
                model,
                config,
                samples[role],
                seed=seed + 500 + i * 43,
                induction_head=FIXED_G0_INDUCTION_HEAD,
                previous_head=previous_head,
            )
            for i, role in enumerate(ROLES)
        }
        print(
            "BASE_MECHANISTIC_GENERATION="
            + json.dumps(
                {
                    "arm": arm,
                    "generation": generation,
                    "validation": {
                        role: {
                            "natural": v["natural_accuracy"],
                            "head3_margin": v[
                                "fixed_head3_query_label_matching_margin"
                            ],
                            "head3_knockout": v["fixed_head3_knockout_accuracy"],
                            "oracle_repair": v["fixed_head3_oracle_match_accuracy"],
                        }
                        for role, v in result["results"][str(generation)].items()
                    },
                    "test_evaluated": False,
                }
            ),
            flush=True,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(
        "BASE_GENERATIONAL_MECHANISM_COMPLETE="
        + json.dumps(
            {
                "arm": arm,
                "generations": list(GENERATIONS),
                "source": "original opto.py",
                "test_evaluated": False,
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-artifact", type=Path, default=ROOT / "outputs/base_full_1m")
    p.add_argument("--g3-root", type=Path, required=True)
    p.add_argument("--g10-root", type=Path, required=True)
    p.add_argument(
        "--arm",
        required=True,
        choices=[
            "clean_only",
            "frozen_g0",
            "real_anchor_half",
            "recursive_replacement",
        ],
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--seed", type=int, default=3201)
    a = p.parse_args()
    if not 32 <= a.n <= 1024:
        raise ValueError("Validation sample count out of range")
    evaluate(
        base_artifact=a.base_artifact,
        g3_root=a.g3_root,
        g10_root=a.g10_root,
        arm=a.arm,
        out=a.out,
        n=a.n,
        seed=a.seed,
    )


if __name__ == "__main__":
    main()
