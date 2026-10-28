from __future__ import annotations
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import sys

import equinox as eqx
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "external" / "icl-dynamics"
DATA = PAPER / "omniglot_resnet18_randomized_order_s0.h5"
sys.path.insert(0, str(PAPER))
import main_utils
import samplers

ROLES = {
    "train": ("train", "train", "train"),
    "val_relabel": ("train", "train", "val"),
    "val_exemplar": ("train", "val", "train"),
    "val_classes": ("val", "train", "train"),
}
LABEL_COUNT = 5
SYMBOL_VOCAB = 50


class ExtendedOmniglot(eqx.Module):

    base: eqx.Module
    symbol_unembed: eqx.nn.Linear
    symbol_vocab: int = eqx.field(static=True)

    def __init__(self, *, config: dict, seed: int, base_model=None):
        self.base = (
            base_model
            if base_model is not None
            else main_utils.get_model_from_opts(
                SimpleNamespace(**{**config, "init_seed": int(seed)})
            )
        )
        if self.base.num_classes != LABEL_COUNT:
            raise ValueError("Expected Singh five-contextual-label baseline")
        self.symbol_vocab = SYMBOL_VOCAB
        self.symbol_unembed = eqx.nn.Linear(
            in_features=int(config["d_model"]),
            out_features=SYMBOL_VOCAB,
            use_bias=True,
            key=jax.random.PRNGKey(int(seed) + 997),
        )

    def __call__(self, examples, context_labels, y1_in, q2_feature_in, *, key):
        if examples.shape != (3, 512):
            raise ValueError("Expected 3 original ResNet18 features of width 512")
        if context_labels.shape != (2,):
            raise ValueError("Expected two contextual number tokens")
        if q2_feature_in.shape != (512,):
            raise ValueError("Expected one next-symbol feature embedding")
        e = jnp.concatenate([examples, q2_feature_in[None, :]], axis=0)
        base = self.base
        xemb = base.example_embed_drop(
            jax.vmap(base.example_embed)(e),
            key=jax.random.PRNGKey(jnp.sum(e).astype(int)),
            inference=False,
        )
        values = jnp.concatenate([context_labels, y1_in[None]], axis=0)
        vemb = jax.vmap(base.label_embed)(jax.nn.one_hot(values, LABEL_COUNT))
        tokens = jnp.stack(
            (xemb[0], vemb[0], xemb[1], vemb[1], xemb[2], vemb[2], xemb[3])
        )

        kpos, kforward = jax.random.split(key)
        pos = jax.nn.one_hot(jnp.arange(7, dtype=jnp.float32), 7)
        pos_embed = base.pos_embed_drop(jax.vmap(base.pos_embed)(pos), key=kpos)
        hidden = base.transformer(tokens + pos_embed, key=kforward)
        return (
            base.unembed(hidden[4]),
            self.symbol_unembed(hidden[5]),
            base.unembed(hidden[6]),
        )


def make_episodes(
    config: dict, *, role: str, batch_size: int, source_zipf: bool = True
):
    if role not in ROLES:
        raise ValueError(f"Forbidden split or unknown role: {role}")
    if not 2 <= batch_size <= 2048:
        raise ValueError("batch size must be 2..2048")
    opts = SimpleNamespace(**config)
    data = main_utils.get_data_from_opts(opts)
    splits = main_utils.get_splits_from_opts(opts, data.shape)
    cls, exe, pairs = ROLES[role]
    classes = splits["class"][cls]
    weights = jnp.ones(len(classes), dtype=jnp.float32)
    if role == "train" and source_zipf:
        weights = 1 / jnp.arange(1, len(classes) + 1, dtype=jnp.float32) ** float(
            opts.zipf_alpha
        )
    weights = weights / weights.sum()

    def one_batch(key):
        kclass, kex, kv = jax.random.split(key, 3)
        index = samplers.get_constant_burst_seq_idxs(
            kclass,
            classes=classes,
            class_distr=weights,
            num_seqs=batch_size,
            context_len=2,
            burstiness=1,
            distractor=True,
            no_support=False,
            unique_rest=False,
        )
        examples_ids = index["class_idxs"]
        types = index["idx_types"]
        exemplar_ids = samplers.get_exemplar_inds(
            kex,
            allowed_inds=splits["exemplar"][exe],
            match_query_and_distractors=bool(opts.match_query_and_distractors),
            idx_types=types,
        )
        labels = samplers.fewshot_relabel(
            kv,
            class_idxs=examples_ids,
            idx_types=types,
            labels=splits["relabeling"][pairs],
        )
        return {
            "examples": data[examples_ids, exemplar_ids],
            "labels": labels,
            "class_ids": examples_ids,
            "exemplar_ids": exemplar_ids,
        }

    return jax.jit(one_batch)


def clean_continuation(batch: dict, *, key):
    ids = batch["class_ids"]
    labels = batch["labels"]
    n = ids.shape[0]
    choice = jax.random.randint(key, shape=(n,), minval=0, maxval=2)
    rows = jnp.arange(n)
    return {
        "tokens": jnp.stack(
            (labels[:, -1], ids[rows, choice], labels[rows, choice]), axis=-1
        ).astype(jnp.int32),
        "q2_features": batch["examples"][rows, choice, :],
        "chosen_support": choice,
    }


def training_symbol_features(feature_data, q2_ids):
    return feature_data[q2_ids, 0, :]


def validate_original_episode(batch: dict, role: str):
    if role not in ROLES:
        raise ValueError("Invalid role")
    ids = jnp.asarray(batch["class_ids"])
    vals = jnp.asarray(batch["labels"])
    assert ids.shape[1] == 3 and vals.shape == ids.shape
    assert bool(jnp.all(ids[:, 0] != ids[:, 1]))
    assert bool(jnp.all((ids[:, 2] == ids[:, 0]) | (ids[:, 2] == ids[:, 1])))
    assert bool(jnp.all(vals[:, 0] != vals[:, 1]))
    correct = jnp.where(ids[:, 2] == ids[:, 0], vals[:, 0], vals[:, 1])
    assert bool(jnp.all(vals[:, 2] == correct))
    if role == "train":
        assert bool(jnp.all((ids >= 0) & (ids < 50)))
    if role == "val_classes":
        assert bool(jnp.all((ids >= 50) & (ids < 1523)))
    assert not bool(jnp.any(ids >= 1523))
    return True
