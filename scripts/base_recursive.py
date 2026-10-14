"""Original-Singh Base generational target-feedback experiment (no test access).

Every run uses the pinned authors' HDF5, SequenceClassifier, optimizer and
episodic JAX sampler, with original 50/1473/100 classes and 8/2 label pairs.
G0 is the verified 1M-sequence model from the existing workflow artifact.

G>=1 starts from scratch on matched clean context prefixes; only the query
LABEL used as training target may be substituted by predecessor samples.
This is target-only recursive feedback, not full autoregressive resampling
of prefixes or a growing-corpus accumulation experiment.

Conditions:
 clean_only: oracle-labelled samples, independent of synthetic models.
 recursive_replacement: G-1 samples all query labels.
 real_anchor_half: exactly half G-1 samples and half real oracle labels.
 frozen_g0: same all-synthetic policy but generator remains frozen at G0.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from base_mechanistic import (
    load_original_checkpoint, source_episodes, make_forward, PAPER, DATA,
)
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import main as original_main
import main_utils
import samplers

ARMS = ("clean_only", "recursive_replacement", "real_anchor_half", "frozen_g0")


def make_original_train_sampler(config, batch_size):
    opts = SimpleNamespace(**config)
    data = main_utils.get_data_from_opts(opts)
    splits = main_utils.get_splits_from_opts(opts, data.shape)
    # Match original paper (including its training Zipf parameter).
    ranks = jnp.arange(1, 51, dtype=jnp.float32)
    probabilities = 1.0 / ranks ** float(opts.zipf_alpha)
    probabilities = probabilities / probabilities.sum()
    class_sampler = partial(
        samplers.get_constant_burst_seq_idxs,
        classes=splits["class"]["train"],
        class_distr=probabilities,
        num_seqs=batch_size,
        context_len=2,
        burstiness=1,
        distractor=True,
        no_support=False,
        unique_rest=False,
    )
    exemplar_sampler = partial(
        samplers.get_exemplar_inds,
        allowed_inds=splits["exemplar"]["train"],
        match_query_and_distractors=bool(opts.match_query_and_distractors),
    )
    sampler = samplers.make_data_sampler(
        class_sampler, exemplar_sampler,
        fs_relabel=splits["relabeling"]["train"],
        noise_scale=float(opts.noise_scale_train),
        assign_query_label_random=bool(opts.assign_query_label_random),
    )
    return jax.jit(partial(sampler, data=data))


@eqx.filter_jit
def sample_query_labels(model, x, y, key, temperature):
    # The query y[-1] never appears among the transformer's input tokens.
    keys = jax.random.split(key, x.shape[0] + 1)
    logits = jax.vmap(lambda xi, yi, ki:
        model(xi, yi, key=ki)[-1])(x, y, keys[:-1])
    samples = jax.random.categorical(keys[-1], logits / temperature, axis=-1)
    return samples.astype(jnp.int32)


def get_validation_data(config, n, seed):
    return {
        k: source_episodes(config, role=k, n=n, seed=seed + i * 101)
        for i, k in enumerate(("val_relabel", "val_classes"))
    }


def validation_metrics(model, config, val_data, seed):
    fn = make_forward(config)
    scores = {}
    for j, (name, sample) in enumerate(val_data.items()):
        ret = original_main.eval_step(
            model, fn, sample["examples"], sample["labels"],
            jax.random.PRNGKey(seed + j * 31))
        scores[name] = {
            "accuracy": float(np.asarray(ret["acc"]).mean()),
            "loss": float(np.asarray(ret["loss"]).mean()),
            "accuracy_in_context_masked": float(
                np.asarray(ret["in_context_acc"]).mean()),
            "n": int(len(sample["labels"])),
        }
    return scores


def create_fresh_model(config, init_seed):
    # Do not inadvertently modify the cached original Base config.
    opts_dict = dict(config)
    opts_dict["init_seed"] = int(init_seed)
    model = main_utils.get_model_from_opts(SimpleNamespace(**opts_dict))
    optimizer = main_utils.get_optimizer_from_opts(SimpleNamespace(**opts_dict))
    return model, optimizer, optimizer.init(eqx.filter(model, eqx.is_array))


def batch_targets(arm, *, truth, synthetic, key):
    if arm == "clean_only":
        return truth, jnp.ones(len(truth), dtype=bool)
    if synthetic is None:
        raise ValueError("Synthetic condition needs previous model outputs")
    if arm in ("recursive_replacement", "frozen_g0"):
        return synthetic, jnp.zeros(len(truth), dtype=bool)
    if arm == "real_anchor_half":
        # Exactly half of a minibatch is real for a stable effective fraction.
        random_positions = jax.random.permutation(key, len(truth))
        real = jnp.zeros(len(truth), dtype=bool).at[
            random_positions[:len(truth)//2]].set(True)
        return jnp.where(real, truth, synthetic), real
    raise ValueError(f"Unknown condition: {arm}")


def train_arm_generation(
    *, arm, generation, predecessor, g0, config, sampler, val_data,
    steps, batch_size, eval_every, seed, out, temperature, verbose=True,
):
    if arm not in ARMS or generation < 1:
        raise ValueError("Only nonzero generational arms are allowed")
    # Identical initial weights across arms for each generation/seed.
    init_seed = seed + generation * 17
    model, optimizer, opt_state = create_fresh_model(config, init_seed)
    fn = make_forward(config)
    arm_dir = out / arm / f"g{generation:02d}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    history = []
    total_real = 0
    total_synthetic = 0
    for step in range(steps):
        # Same data order across arms (matched episodes; no final-test classes).
        batch_key = jax.random.PRNGKey(seed + 103_111 * generation + step)
        input_key, generator_key, mix_key, training_key = jax.random.split(batch_key, 4)
        batch = sampler(input_key)
        x = batch["examples"]
        y = batch["labels"]
        truth = y[:, -1]
        if arm == "clean_only":
            generated = None
        else:
            source_model = g0 if arm == "frozen_g0" else predecessor
            generated = sample_query_labels(
                source_model, x, y, generator_key, temperature)
        train_y, real_mask = batch_targets(
            arm, truth=truth, synthetic=generated, key=mix_key)
        # Only targets are replaced: all support examples and labels stay
        # unchanged. The original model code drops the final y token.
        train_labels = y.at[:, -1].set(train_y)
        stats, model, opt_state = original_main.train_step(
            model, fn, optimizer, opt_state, batch_size,
            float(config.get("weight_decay", 0.0)),
            x, train_labels, training_key,
        )
        n_real = int(np.asarray(real_mask).sum())
        total_real += n_real
        total_synthetic += batch_size - n_real
        if (step + 1) % eval_every == 0 or step == 0 or step + 1 == steps:
            vals = validation_metrics(
                model, config, val_data, seed=seed+100_000+step)
            item = {
                "generation": generation, "arm": arm,
                "seen_training_sequences": (step + 1) * batch_size,
                "last_train_loss": float(np.asarray(stats["loss"])),
                "validation": vals,
                "real_training_target_fraction": total_real / ((step+1)*batch_size),
            }
            history.append(item)
            if verbose:
                print("BASE_PROGRESS="+json.dumps(item), flush=True)
    eqx.tree_serialise_leaves(str(arm_dir/"model.eqx"), model)
    metadata = {
        "task": "Original Singh 2-way one-shot Omniglot Base",
        "arm": arm, "generation": generation,
        "generator_generation": (
            0 if arm == "frozen_g0" else generation-1
        ) if arm != "clean_only" else None,
        "target_only_synthetic": True,
        "whole_prefix_generated": False,
        "class_split": [50, 1473, 100],
        "exemplar_split": [1, 4, 0],
        "label_pair_split": [8, 2, 0],
        "final_test_evaluated": False,
        "temperature": float(temperature),
        "optimizer_steps": steps, "batch_size": batch_size,
        "optimizer_exposures": steps*batch_size,
        "real_target_count": total_real,
        "synthetic_target_count": total_synthetic,
        "initialisation_seed": init_seed, "data_seed": seed,
        "baseline": "original 1M-sequence pinned-Singh G0 checkpoint",
        "duration_seconds": round(time.time()-started, 3),
        "history": history,
    }
    (arm_dir/"metrics.json").write_text(json.dumps(metadata, indent=2))
    return model, metadata


def experiment(artifact_root, out, *, generations, steps, batch_size,
               eval_every, eval_count, seed, temperature, arms):
    if generations < 1 or steps < 1 or batch_size < 2 or batch_size % 2:
        raise ValueError("Generations/steps positive; batch size even >=2")
    if eval_every < 1 or eval_count < 32 or temperature <= 0:
        raise ValueError("Invalid evaluation interval/count or temperature")
    if not set(arms) <= set(ARMS):
        raise ValueError("Unknown requested conditions")
    if len(arms) != len(set(arms)):
        raise ValueError("Duplicate arm names")
    out.mkdir(parents=True, exist_ok=True)
    g0, config = load_original_checkpoint(artifact_root)
    assert config["class_split"] == [50, 1473, 100]
    assert config["fs_relabel_split"] == [8, 2, 0]
    sampler = make_original_train_sampler(config, batch_size)
    validation = get_validation_data(config, eval_count, seed+501)
    evidence = {"g0_validation": validation_metrics(g0,config,validation,seed+800),
                "arms":{}, "final_test_evaluated":False,
                "training_prefix_generated":False}
    for arm in arms:
        predecessor = g0
        evidence["arms"][arm] = []
        for g in range(1,generations+1):
            predecessor, meta = train_arm_generation(
                arm=arm, generation=g, predecessor=predecessor,
                g0=g0, config=config, sampler=sampler,
                val_data=validation, steps=steps, batch_size=batch_size,
                eval_every=eval_every, seed=seed, out=out,
                temperature=temperature)
            evidence["arms"][arm].append({
                "generation": g,
                "last_validation": meta["history"][-1]["validation"],
                "real_target_count": meta["real_target_count"],
                "synthetic_target_count": meta["synthetic_target_count"],
            })
            (out/"summary.json").write_text(json.dumps(evidence,indent=2))
    print("BASE_RECURSION_FINISHED="+json.dumps(evidence), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root",type=Path,
                   default=ROOT/"outputs"/"base_full_1m")
    p.add_argument("--out",type=Path,default=ROOT/"outputs"/"base_recursive")
    p.add_argument("--generations",type=int,default=2)
    p.add_argument("--steps",type=int,default=32)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--eval-every",type=int,default=16)
    p.add_argument("--eval-count",type=int,default=128)
    p.add_argument("--seed",type=int,default=34001)
    p.add_argument("--temperature",type=float,default=1.0)
    p.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    a=p.parse_args()
    experiment(a.artifact_root,a.out,generations=a.generations,
               steps=a.steps,batch_size=a.batch_size,eval_every=a.eval_every,
               eval_count=a.eval_count,seed=a.seed,
               temperature=a.temperature,arms=a.arms)


if __name__=="__main__":
    main()
