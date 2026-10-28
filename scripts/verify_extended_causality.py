from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "icl-dynamics"))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from scripts.extended_g0 import load_config
from src.extended_omniglot import ExtendedOmniglot, make_episodes, clean_continuation


def check(artifact_root, output_path, *, seed):
    config = load_config(artifact_root)
    report = json.loads((output_path / "metrics.json").read_text())
    model = ExtendedOmniglot(config=config, seed=int(report["seed"]))
    model = eqx.tree_deserialise_leaves(
        str(output_path / "extended_g0_model.eqx"), model
    )
    samples = make_episodes(config, role="train", batch_size=8)(
        jax.random.PRNGKey(seed)
    )
    targets = clean_continuation(samples, key=jax.random.PRNGKey(seed + 1))
    x = samples["examples"][0]
    y = samples["labels"][0]
    real_y1 = targets["tokens"][0, 0]
    q2_feature = targets["q2_features"][0]
    k = jax.random.PRNGKey(seed + 2)
    normal = model(x, y[:2], real_y1, q2_feature, key=k)
    future_symbol_changed = model(x, y[:2], real_y1, x[1], key=k)
    future_label_changed = model(x, y[:2], (real_y1 + 1) % 5, q2_feature, key=k)
    orig_base = model.base(x, y, key=k)[-1]

    np.testing.assert_allclose(
        np.asarray(normal[0]),
        np.asarray(future_symbol_changed[0]),
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(normal[0]), np.asarray(future_label_changed[0]), atol=2e-5, rtol=2e-5
    )

    np.testing.assert_allclose(
        np.asarray(normal[1]),
        np.asarray(future_symbol_changed[1]),
        atol=2e-5,
        rtol=2e-5,
    )

    np.testing.assert_allclose(
        np.asarray(normal[0]), np.asarray(orig_base), atol=2e-5, rtol=2e-5
    )
    assert (
        normal[0].shape == (5,) and normal[1].shape == (50,) and normal[2].shape == (5,)
    )
    assert int(jnp.max(samples["class_ids"])) < 50
    assert report["final_test_evaluated"] is False
    print(
        "EXTENDED_CAUSALITY_PASSED: y1 unaffected by future targets; "
        "q2 unaffected by future q2 feature; original Base logits recovered"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--artifact-root", type=Path, default=ROOT / "outputs" / "base_300k_g0_0"
    )
    p.add_argument("--trained", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1212)
    a = p.parse_args()
    check(a.artifact_root, a.trained, seed=a.seed)


if __name__ == "__main__":
    main()
