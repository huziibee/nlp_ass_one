from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "icl-dynamics"))
import jax
import jax.numpy as jnp
import numpy as np
from scripts.extended_g0 import load_config
from scripts.compare_extended_circuits_g10 import (
    load_model,
    batch_aux,
    add_generation_arguments,
)
from src.extended_omniglot import make_episodes, clean_continuation


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def entropy(p):
    a = np.asarray(p, dtype=np.float64)
    return -np.sum(a * np.log(np.maximum(a, 1e-15)), axis=-1)


def ece(p, truth, bin_count=10):
    confidence = np.max(p, axis=-1)
    predicted = np.argmax(p, axis=-1)
    good = predicted == truth
    out = 0.0
    for k in range(bin_count):
        lower = k / bin_count
        upper = (k + 1) / bin_count
        selected = (confidence >= lower) & (
            confidence < upper if k < bin_count - 1 else confidence <= upper
        )
        if selected.any():
            out += selected.mean() * abs(
                float(good[selected].mean()) - float(confidence[selected].mean())
            )
    return float(out)


def describe(p, ids, labels, p_y1, episode_seed):
    n = len(p)
    assert p.shape == (n, 50) and len(ids) == n and np.all(ids[:, 0] != ids[:, 1])
    p0 = p[np.arange(n), ids[:, 0]]
    p1 = p[np.arange(n), ids[:, 1]]
    mass = p0 + p1
    assert np.all(mass <= 1 + 1e-5)
    best = np.argmax(p, axis=1)
    legal = (best == ids[:, 0]) | (best == ids[:, 1])
    inside = np.stack((p0, p1), axis=1) / np.maximum(1e-12, mass[:, None])
    h_inside = entropy(inside)
    marginal = np.mean(p, axis=0)
    log_odds = np.log(np.maximum(mass, 1e-15) / np.maximum(1 - mass, 1e-15))
    expected = labels[:, -1]
    confidence_correct = p_y1[np.arange(n), expected]
    probs_y1 = np.maximum(p_y1, 1e-15)
    return {
        "n_validation_episodes": n,
        "probability_mass_valid_supports": float(mass.mean()),
        "probability_mass_off_context": float((1 - mass).mean()),
        "valid_mass_std_over_episodes": float(mass.std(ddof=1)),
        "valid_mass_episode_q05_q50_q95": np.quantile(mass, [0.05, 0.5, 0.95]).tolist(),
        "fraction_invalid_symbol_is_top1": float((~legal).mean()),
        "fraction_valid_mass_below_half": float((mass < 0.5).mean()),
        "mean_contextual_vs_outside_log_odds": float(log_odds.mean()),
        "mean_per_episode_symbol_entropy_nats": float(entropy(p).mean()),
        "global_marginal_symbol_entropy_nats": float(entropy(marginal)),
        "global_marginal_top5_mass": float(np.sort(marginal)[-5:].sum()),
        "effective_global_symbol_count_exp_entropy": float(np.exp(entropy(marginal))),
        "mutual_information_episode_to_symbol_nats": float(
            entropy(marginal) - entropy(p).mean()
        ),
        "conditional_binary_entropy_between_two_legal_symbols_nats": float(
            h_inside.mean()
        ),
        "absolute_support_preference_among_legal_symbols": float(
            np.mean(np.abs(p0 - p1) / np.maximum(1e-12, mass))
        ),
        "mean_first_label_probability_assigned_to_oracle": float(
            confidence_correct.mean()
        ),
        "first_label_greedy_accuracy": float(
            (np.argmax(p_y1, axis=-1) == expected).mean()
        ),
        "first_label_expected_categorical_accuracy": float(confidence_correct.mean()),
        "first_label_nll": float(-np.log(np.maximum(confidence_correct, 1e-15)).mean()),
        "first_label_confidence_ece_10_bins": ece(p_y1, expected),
        "seed": episode_seed,
    }, {
        "support_ids": ids[:, :2].tolist(),
        "correct_numeric_first_label": expected.tolist(),
        "valid_mass": mass.tolist(),
        "p_support0": p0.tolist(),
        "p_support1": p1.tolist(),
        "p_off_context": (1 - mass).tolist(),
        "argmax_is_context_valid": legal.tolist(),
        "p_first_label_oracle": confidence_correct.tolist(),
        "q2_probs_50": p.tolist(),
    }


def run(args):
    config = load_config(args.base_root)
    g0, meta0 = load_model(config, args.g0_root, gen=0)
    assert meta0["final_test_evaluated"] is False
    data = {
        "experiment": "original Omniglot Extended token-level probability audit",
        "arm": args.arm,
        "generations": list(args.generations),
        "roles": ["val_relabel", "val_exemplar"],
        "original_final_test_access": False,
        "source_class_split": [50, 1473, 100],
        "same_episodes_across_models": True,
        "n_model_seeds": 1,
        "first_label_distribution": "Extended first-head logits; fixed evaluation rng",
        "q2_distribution": "50-way model softmax, exact mixture over all five possible first labels, conditional on fixed evaluation rng",
        "q2_teacher_forced_y1_distribution": "50-way q2 softmax given oracle correct first label",
        "do_not_conflate_with_autoregressive_sample_rate": True,
        "results": {},
    }
    for g in args.generations:
        checkpoint = (
            args.g0_root if g == 0 else args.generation_root / args.arm / f"g{g:02d}"
        )
        model, metadata = (
            (g0, meta0) if g == 0 else load_model(config, checkpoint, gen=g)
        )
        assert metadata["final_test_evaluated"] is False
        data["results"][str(g)] = {}
        for i, role in enumerate(("val_relabel", "val_exemplar")):
            key = args.seed + i * 101
            episode = make_episodes(config, role=role, batch_size=args.n)(
                jax.random.PRNGKey(key)
            )
            oracle = clean_continuation(episode, key=jax.random.PRNGKey(key + 1))
            trace = batch_aux(model, episode, oracle, key=key + 2)
            p_y1 = softmax(trace[0])
            p_true = softmax(trace[1])
            probs = []
            for y in range(5):
                new_oracle = {**oracle, "tokens": oracle["tokens"].at[:, 0].set(y)}
                changed = batch_aux(model, episode, new_oracle, key=key + 2)

                np.testing.assert_allclose(
                    np.asarray(changed[0]), np.asarray(trace[0]), atol=3e-5, rtol=3e-5
                )
                probs.append(softmax(changed[1]))
            p_exact = np.sum(np.stack(probs, axis=1) * p_y1[:, :, None], axis=1)
            np.testing.assert_allclose(p_exact.sum(axis=-1), 1.0, atol=1e-6)
            ids = np.asarray(episode["class_ids"], dtype=np.int32)
            labs = np.asarray(episode["labels"], dtype=np.int32)
            result, raw = describe(p_exact, ids, labs, p_y1, key)
            oracle_details, _ = describe(p_true, ids, labs, p_y1, key)
            result["q2_teacher_forced_first_label_valid_mass"] = oracle_details[
                "probability_mass_valid_supports"
            ]
            result["valid_mass_gain_when_first_label_teacher_forced"] = (
                oracle_details["probability_mass_valid_supports"]
                - result["probability_mass_valid_supports"]
            )
            result["q2_teacher_forced_first_label_entropy"] = oracle_details[
                "mean_per_episode_symbol_entropy_nats"
            ]
            data["results"][str(g)][role] = result
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"g{g:02d}_{role}_episodes.json").write_text(
                json.dumps(
                    {
                        "arm": args.arm,
                        "generation": g,
                        "role": role,
                        "fixed_sample_count": args.n,
                        "protected_final_test_accessed": False,
                        **raw,
                    }
                )
            )
            (args.out / "summary.json").write_text(json.dumps(data, indent=2))
            print(
                "CONDITIONAL_SYMBOL_DISTRIBUTION="
                + json.dumps(
                    {
                        "arm": args.arm,
                        "generation": g,
                        "role": role,
                        "valid_mass": result["probability_mass_valid_supports"],
                        "mean_conditional_entropy": result[
                            "mean_per_episode_symbol_entropy_nats"
                        ],
                        "global_entropy": result["global_marginal_symbol_entropy_nats"],
                        "invalid_top1_fraction": result[
                            "fraction_invalid_symbol_is_top1"
                        ],
                    }
                ),
                flush=True,
            )
    print(
        "DISTRIBUTION_MECHANISM_AUDIT_COMPLETE="
        + json.dumps(
            {
                "arm": args.arm,
                "no_final_test": True,
                "generations": len(args.generations),
                "roles": 2,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-root", type=Path, required=True)
    p.add_argument("--g0-root", type=Path, required=True)
    add_generation_arguments(p)
    p.add_argument(
        "--arm", choices=("clean_only", "recursive_replacement"), required=True
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--seed", type=int, default=93211)
    args = p.parse_args()
    if args.n not in (64, 128, 256):
        raise ValueError("Bounded validation batch")
    run(args)
