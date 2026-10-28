from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "icl-dynamics"))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from scripts.extended_g0 import load_config, feature_prototypes
from src.extended_omniglot import ExtendedOmniglot, make_episodes, clean_continuation
from src.extended_learner import autoregressive_sample


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    x = p[p > 0]
    return float(-np.sum(x * np.log(x)))


def jensen_shannon(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2
    mask1 = p > 0
    mask2 = q > 0
    return float(
        0.5 * (p[mask1] * np.log(p[mask1] / m[mask1])).sum()
        + 0.5 * (q[mask2] * np.log(q[mask2] / m[mask2])).sum()
    )


def load_model(config, root: Path, *, generation: int, init_seed: int):
    meta_path = root / "metrics.json"
    path = root / ("extended_g0_model.eqx" if generation == 0 else "model.eqx")
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            "Missing original Extended checkpoint and metadata: " + str(path)
        )
    m = json.loads(meta_path.read_text())
    if m.get("final_test_evaluated") is not False:
        raise AssertionError(
            "Diagnostic cannot consume final-test-derived model metrics"
        )
    if generation == 0:
        if m["class_split"] != [50, 1473, 100]:
            raise AssertionError("Wrong G0 class split")
        model_seed = int(m["seed"])
    else:
        if m["generation"] != generation or m["training_class_split"] != [
            50,
            1473,
            100,
        ]:
            raise AssertionError("Incorrect generation checkpoint")
        model_seed = int(m["init_seed"])
        if model_seed != init_seed:
            raise AssertionError("Inconsistent fixed initialization seeds")
    template = ExtendedOmniglot(config=config, seed=model_seed)
    return eqx.tree_deserialise_leaves(str(path), template), m, path


def classify_generated(batch, tokens):
    ids = np.asarray(batch["class_ids"], dtype=np.int32)
    labels = np.asarray(batch["labels"], dtype=np.int32)
    generated = np.asarray(tokens, dtype=np.int32)
    n = len(generated)
    if (
        generated.shape != (n, 3)
        or np.min(generated[:, 1]) < 0
        or np.max(generated[:, 1]) >= 50
    ):
        raise ValueError("Invalid original Extended output")
    q2 = generated[:, 1]
    valid0 = q2 == ids[:, 0]
    valid1 = q2 == ids[:, 1]
    legal = valid0 | valid1
    correct_number = np.where(valid0, labels[:, 0], labels[:, 1])
    right_second = (generated[:, 2] == correct_number) & legal
    right_first = generated[:, 0] == labels[:, -1]
    histogram = np.bincount(q2, minlength=50)
    if len(histogram) != 50 or histogram.sum() != n:
        raise AssertionError("Bad marginal class frequency accounting")
    query_ids = ids[:, -1]
    return {
        "count": n,
        "first_correct": int(right_first.sum()),
        "q2_context_valid": int(legal.sum()),
        "q2_outside_context": int((~legal).sum()),
        "valid_q2_second_number_correct": int(right_second.sum()),
        "first_and_valid_q2_and_correct_second": int(
            (right_first & right_second).sum()
        ),
        "second_correct_given_valid_numer": int(right_second.sum()),
        "second_correct_given_valid_denom": int(legal.sum()),
        "rare_query_count": int((query_ids >= 40).sum()),
        "rare_query_valid_count": int((legal & (query_ids >= 40)).sum()),
        "top10_query_count": int((query_ids < 10).sum()),
        "top10_query_valid_count": int((legal & (query_ids < 10)).sum()),
        "generated_symbols_histogram": histogram,
    }


def diagnostics(
    *,
    config_root: Path,
    model_root: Path,
    generation: int,
    role: str,
    n: int,
    repeats: int,
    seed: int,
    temperature: float,
    init_seed: int,
    out: Path,
):
    if role not in ("val_relabel", "val_exemplar"):
        raise PermissionError(
            "Diagnostic may use only held-out pairs/exemplars with train classes"
        )
    if not (64 <= n <= 1024 and 1 <= repeats <= 32 and temperature > 0):
        raise ValueError("Invalid prespecified sample count/repeats/temperature")
    cfg = load_config(config_root)
    model, meta, checkpoint = load_model(
        cfg, model_root, generation=generation, init_seed=init_seed
    )
    prototypes = feature_prototypes(cfg)

    batch = make_episodes(cfg, role=role, batch_size=n)(jax.random.PRNGKey(seed + 19))
    clean = clean_continuation(batch, key=jax.random.PRNGKey(seed + 20))
    true_hist = np.bincount(np.asarray(clean["tokens"][:, 1]), minlength=50)
    ids = np.asarray(batch["class_ids"])
    assert ids.max() < 50 and ids.min() >= 0
    counts = {}
    totals = {}
    for repeat in range(repeats):
        sampled = autoregressive_sample(
            model,
            batch,
            jax.random.PRNGKey(seed + 1000 + repeat),
            jnp.asarray(temperature, dtype=jnp.float32),
            prototypes,
        )
        cc = classify_generated(batch, sampled)
        for key, value in cc.items():
            if key == "generated_symbols_histogram":
                counts[key] = counts.get(key, np.zeros(50, dtype=np.int64)) + value
            else:
                totals[key] = totals.get(key, 0) + value
    total = totals["count"]
    p = counts["generated_symbols_histogram"] / total
    q = true_hist / true_hist.sum()
    h = entropy(p)
    mean_rates = {
        "free_first_label_oracle_accuracy": totals["first_correct"] / total,
        "free_next_symbol_context_validity": totals["q2_context_valid"] / total,
        "free_next_symbol_off_context_rate": totals["q2_outside_context"] / total,
        "free_second_label_context_consistency_all": totals[
            "valid_q2_second_number_correct"
        ]
        / total,
        "free_all_three_correct_and_context_valid": totals[
            "first_and_valid_q2_and_correct_second"
        ]
        / total,
        "free_second_label_correct_given_valid": (
            totals["second_correct_given_valid_numer"]
            / max(1, totals["second_correct_given_valid_denom"])
        ),
        "rare_training_class_query_validity": (
            totals["rare_query_valid_count"] / max(1, totals["rare_query_count"])
        ),
        "top10_training_class_query_validity": (
            totals["top10_query_valid_count"] / max(1, totals["top10_query_count"])
        ),
    }
    record = {
        "task": "original Omniglot-feature Extended G0/G3/G6/G10 diagnostic",
        "generation": generation,
        "role": role,
        "n_unique_validation_episodes": n,
        "n_independent_autoregressive_draws_per_episode": repeats,
        "n_generated_sequences": total,
        "seed": seed,
        "temperature": temperature,
        "model_training_seed": int(
            meta["seed"] if generation == 0 else meta["data_seed"]
        ),
        "saved_checkpoint_file": checkpoint.name,
        "saved_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "class_partition": "training identities, development label pairs/exemplars; NO 100 final classes",
        "final_test_access": False,
        "marginal_next_symbol_counts": counts["generated_symbols_histogram"].tolist(),
        "clean_oracle_next_symbol_counts": true_hist.tolist(),
        "next_symbol_marginal_entropy_nats": h,
        "next_symbol_marginal_entropy_normalized": h / np.log(50),
        "next_symbol_unique_count": int(
            (counts["generated_symbols_histogram"] > 0).sum()
        ),
        "next_symbol_top5_mass": float(np.sort(p)[-5:].sum()),
        "jensen_shannon_to_clean_next_symbol_marginal_nats": jensen_shannon(p, q),
        "next_symbol_conditional_contextual_metrics": mean_rates,
        "caveat": "G0 and selected G3/G6/G10 checkpoints have ONE trained seed each; multiple draws only reduce sampling error, not model-seed uncertainty.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(
        "EXTENDED_DIST_DIAGNOSTIC="
        + json.dumps(
            {
                "generation": generation,
                "role": role,
                "n": total,
                "entropy": record["next_symbol_marginal_entropy_nats"],
                "js": record["jensen_shannon_to_clean_next_symbol_marginal_nats"],
                "context_valid": mean_rates["free_next_symbol_context_validity"],
                "second_consistency": mean_rates[
                    "free_second_label_context_consistency_all"
                ],
                "rare_query_validity": mean_rates["rare_training_class_query_validity"],
                "top10_query_validity": mean_rates[
                    "top10_training_class_query_validity"
                ],
                "final_test_access": False,
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-root", type=Path, default=ROOT / "outputs/base_300k_g0_0")
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--generation", type=int, required=True)
    p.add_argument(
        "--role", choices=["val_relabel", "val_exemplar"], default="val_relabel"
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261401)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--init-seed", type=int, default=4054)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    diagnostics(
        config_root=a.config_root,
        model_root=a.checkpoint_root,
        generation=a.generation,
        role=a.role,
        n=a.batch_size,
        repeats=a.repeats,
        seed=a.seed,
        temperature=a.temperature,
        init_seed=a.init_seed,
        out=a.out,
    )


if __name__ == "__main__":
    main()
