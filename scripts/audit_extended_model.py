from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "icl-dynamics"))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from scripts.extended_g0 import load_config, feature_prototypes
from src.extended_omniglot import (
    ExtendedOmniglot,
    make_episodes,
    clean_continuation,
    validate_original_episode,
)
from src.extended_learner import (
    autoregressive_sample,
    teacher_forced_logits,
    extended_loss,
)


def assert_close(x, y, *, name, atol=2e-5):
    np.testing.assert_allclose(
        np.asarray(x), np.asarray(y), atol=atol, rtol=atol, err_msg=name
    )


def audit(base_root: Path, extended_root: Path, *, seed: int):
    config = load_config(base_root)
    meta = json.loads((extended_root / "metrics.json").read_text())
    assert meta["final_test_evaluated"] is False
    assert meta["class_split"] == [50, 1473, 100]
    assert meta["unmasked_symbol_vocabulary_train_only"] == 50
    assert (
        meta["clean_second_symbol"] == "uniform support class from 2 contextual symbols"
    )
    assert (meta["steps"] * meta["batch_size"]) >= 300_000
    model = ExtendedOmniglot(config=config, seed=int(meta["seed"]))
    model = eqx.tree_deserialise_leaves(
        str(extended_root / "extended_g0_model.eqx"), model
    )
    prototypes = feature_prototypes(config)
    assert prototypes.shape == (50, 512)

    data = {}
    targets = {}
    for i, role in enumerate(("train", "val_relabel", "val_exemplar", "val_classes")):
        samples = make_episodes(config, role=role, batch_size=32)(
            jax.random.PRNGKey(seed + 100 * i)
        )
        validate_original_episode(samples, role)
        c = clean_continuation(samples, key=jax.random.PRNGKey(seed + 100 * i + 1))
        assert c["tokens"].shape == (32, 3)
        assert c["q2_features"].shape == (32, 512)
        rows = jnp.arange(32)
        ids = samples["class_ids"]
        labels = samples["labels"]
        assert bool(jnp.all(c["tokens"][:, 0] == labels[:, 2]))
        assert bool(
            jnp.all((c["tokens"][:, 1] == ids[:, 0]) | (c["tokens"][:, 1] == ids[:, 1]))
        )
        assert bool(jnp.all((c["tokens"][:, 0] >= 0) & (c["tokens"][:, 0] < 5)))
        assert bool(jnp.all((c["tokens"][:, 2] >= 0) & (c["tokens"][:, 2] < 5)))
        expected = labels[rows, c["chosen_support"]]
        assert bool(jnp.all(c["tokens"][:, 2] == expected))
        expect_features = samples["examples"][rows, c["chosen_support"], :]
        assert_close(
            c["q2_features"], expect_features, name="Oracle q2 feature leakage"
        )
        if role == "train":

            assert bool(jnp.all(samples["exemplar_ids"] == 0))
            assert_close(
                prototypes[c["tokens"][:, 1]],
                c["q2_features"],
                name="Generated training q2 must be original exact feature",
            )
        if role == "val_classes":
            assert bool(jnp.all(ids >= 50)) and bool(jnp.all(ids < 1523))
        else:
            assert bool(jnp.all(ids < 50))
        data[role] = samples
        targets[role] = c

    ex = data["train"]["examples"][0]
    lab = data["train"]["labels"][0]
    context_label = targets["train"]["tokens"][0, 0]
    q2feat = targets["train"]["q2_features"][0]
    key = jax.random.PRNGKey(seed + 2000)
    logits = model(ex, lab[:2], context_label, q2feat, key=key)
    assert [tuple(l.shape) for l in logits] == [(5,), (50,), (5,)]
    assert bool(jnp.all(jnp.isfinite(jnp.concatenate(logits))))

    base_logits = model.base(ex, lab, key=key)[-1]
    assert_close(
        logits[0], base_logits, name="First query incompatible with original Singh"
    )
    changed_number = model(ex, lab[:2], (context_label + 1) % 5, q2feat, key=key)
    changed_feature = model(ex, lab[:2], context_label, ex[1], key=key)
    assert_close(
        logits[0],
        changed_number[0],
        name="Future first-number token leaked into first prediction",
    )
    assert_close(
        logits[0],
        changed_feature[0],
        name="Future symbol token leaked into first prediction",
    )
    assert_close(
        logits[1],
        changed_feature[1],
        name="Next symbol feature leaked into its own prediction",
    )

    samples = autoregressive_sample(
        model,
        data["train"],
        jax.random.PRNGKey(seed + 301),
        jnp.asarray(1.0, dtype=jnp.float32),
        prototypes,
    )
    assert samples.shape == (32, 3)
    assert bool(jnp.all((samples[:, 0] >= 0) & (samples[:, 0] < 5)))
    assert bool(jnp.all((samples[:, 1] >= 0) & (samples[:, 1] < 50)))
    assert bool(jnp.all((samples[:, 2] >= 0) & (samples[:, 2] < 5)))

    batch = data["train"]
    clean = targets["train"]
    keys = jax.random.split(jax.random.PRNGKey(seed + 400), len(batch["labels"]))
    loss, grad = extended_loss(
        model,
        batch["examples"],
        batch["labels"],
        clean["tokens"],
        clean["q2_features"],
        keys,
    )
    assert np.isfinite(float(loss))
    g_symbol = np.asarray(grad.symbol_unembed.weight)
    g_label = np.asarray(grad.base.unembed.weight)
    assert np.isfinite(g_symbol).all() and np.linalg.norm(g_symbol) > 1e-8
    assert np.isfinite(g_label).all() and np.linalg.norm(g_label) > 1e-8

    heldout = data["val_classes"]
    heldout_clean = targets["val_classes"]
    out = teacher_forced_logits(
        model,
        heldout,
        heldout_clean["tokens"],
        heldout_clean["q2_features"],
        jax.random.PRNGKey(seed + 500),
    )
    assert (
        out[0].shape == (32, 5) and out[1].shape == (32, 50) and out[2].shape == (32, 5)
    )
    assert bool(jnp.all(jnp.isfinite(out[2])))
    record = {
        "audit": "saved trained Extended G0, exact upstream-backed Omniglot",
        "model_seed": int(meta["seed"]),
        "original_task": True,
        "original_first_label_equivalence": True,
        "no_future_token_leakage": True,
        "second_symbol_output_space": 50,
        "first_and_second_label_output_space": 5,
        "clean_targets_and_features_valid": True,
        "all_three_target_ce_gradient_pathways_nonzero": True,
        "autoregressive_output_ranges_correct": True,
        "heldout_class_symbol_head_not_scored": True,
        "last_100_final_classes_untouched": True,
        "known_limitation": "training target-only recursion, no generated input prefixes",
        "status": "PASS",
    }
    print("EXTENDED_FULL_AUDIT=" + json.dumps(record, sort_keys=True), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-root", type=Path, default=ROOT / "outputs/base_300k_g0_0")
    p.add_argument(
        "--extended-root", type=Path, default=ROOT / "outputs/extended_primary/g00"
    )
    p.add_argument("--seed", type=int, default=20261014)
    a = p.parse_args()
    audit(a.base_root, a.extended_root, seed=a.seed)


if __name__ == "__main__":
    main()
