"""Validation-only mechanistic analysis of the original Singh Base checkpoint.

Loads the authors' exact JAX checkpoint, creates source-native evaluation
episodes, measures causal head ablations/retention via the ORIGINAL opto.py,
and measures how second-layer attention selects the correct contextual
label. No final-test classes are sampled or scored. Test-time interventions
do NOT establish causes of training-data corruption or model collapse.

Run from the repository root using authors-compatible requirements:
    python scripts/base_mechanistic.py \
      --artifact-root outputs/base_full_1m --out outputs/base_mechanistic
"""
from __future__ import annotations
import argparse
import json
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "external" / "icl-dynamics"
DATA = PAPER / "omniglot_resnet18_randomized_order_s0.h5"
sys.path.insert(0, str(PAPER))

import jax
import jax.numpy as jnp
import equinox as eqx
import h5py
import main_utils
import opto
import samplers

DEVELOPMENT = {
    "train": ("train", "train", "train"),
    "val_relabel": ("train", "train", "val"),
    "val_classes": ("val", "train", "train"),
}


def load_original_checkpoint(artifact_root: Path):
    folder = artifact_root / "safe_train_val_smoke"
    cfg = folder / "config.json"
    ckpt = folder / "checkpoints" / "00001000000.eqx"
    if not cfg.is_file() or not ckpt.is_file():
        raise FileNotFoundError(f"Expected original-source model config/checkpoint under {folder}")
    config = json.loads(cfg.read_text())
    if config["train_iters"] != 1_000_000:
        raise AssertionError("Mechanism study requires completed million-sequence Base")
    assert config["class_split"] == [50, 1473, 100]
    assert config["exemplar_split"] == [1, 4, 0]
    assert config["fs_relabel_split"] == [8, 2, 0]
    assert config["model_output_classes"] == 5
    config["data_file"] = str(DATA)
    opts = SimpleNamespace(**config)
    model = main_utils.get_model_from_opts(opts)
    optimizer = main_utils.get_optimizer_from_opts(opts)
    template = {
        "iter": -1,
        "seeds": {
            "eval_model_seed": jax.random.PRNGKey(0),
            "train_data_seed": jax.random.PRNGKey(0),
            "train_model_seed": jax.random.PRNGKey(0),
        },
        "opt_state": optimizer.init(eqx.filter(model, eqx.is_array)),
        "model": model,
    }
    restored = eqx.tree_deserialise_leaves(str(ckpt), template)
    assert int(restored["iter"]) == 1_000_000
    return restored["model"], config


def source_episodes(config: dict, *, role: str, n: int, seed: int):
    """Source-aligned sampler for exactly two context symbol/label pairs."""
    if role not in DEVELOPMENT:
        raise ValueError("Only training/validation roles can be sampled")
    opts = SimpleNamespace(**config)
    data = main_utils.get_data_from_opts(opts)
    splits = main_utils.get_splits_from_opts(opts, data.shape)
    cs, es, ps = DEVELOPMENT[role]
    classes = splits["class"][cs]
    class_sampler = partial(
        samplers.get_constant_burst_seq_idxs,
        classes=classes,
        class_distr=jnp.ones(len(classes)) / len(classes),
        num_seqs=n,
        context_len=2,
        burstiness=1,
        distractor=True,
        no_support=False,
        unique_rest=False,
    )
    exemplar_sampler = partial(
        samplers.get_exemplar_inds,
        allowed_inds=splits["exemplar"][es],
        match_query_and_distractors=False,
    )
    sampler = samplers.make_data_sampler(
        class_sampler, exemplar_sampler,
        fs_relabel=splits["relabeling"][ps],
        noise_scale=0.0,
        assign_query_label_random=False,
    )
    sample = sampler(jax.random.PRNGKey(seed), data)
    return sample


def make_forward(config: dict, *, ablate: list[str] | None = None,
                 force_match: list[str] | None = None):
    opts = SimpleNamespace(**config)
    opts.opto_ablate_heads = ablate
    opts.opto_induction_heads = force_match
    return opto.make_fn_from_opts(opts)


def run_fn(model, fn, samples, seed=13):
    n = len(samples["labels"])
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    ret = jax.vmap(lambda x, y, key: fn(model=model, x=x, y=y, key=key))(
        samples["examples"], samples["labels"], keys
    )
    logits = np.asarray(ret["out"])[:, -1, :]
    truth = np.asarray(samples["labels"])[:, -1]
    accuracy = float(np.mean(np.argmax(logits, axis=-1) == truth))
    logit_shift = logits - np.max(logits, axis=-1, keepdims=True)
    logps = logit_shift - np.log(np.sum(np.exp(logit_shift), axis=-1, keepdims=True))
    ce = float(-np.mean(logps[np.arange(len(truth)), truth]))
    return accuracy, ce, ret


def match_attention(samples, result):
    x = np.asarray(samples["examples"])  # [n, 3, 512]
    eq0 = np.all(np.isclose(x[:, 2, :], x[:, 0, :], rtol=0, atol=1e-6), axis=1)
    eq1 = np.all(np.isclose(x[:, 2, :], x[:, 1, :], rtol=0, atol=1e-6), axis=1)
    if not np.all(eq0 ^ eq1):
        raise AssertionError(
            "Reference-aligned query exemplar must match exactly one of two support vectors"
        )
    attn = result["transformer_output"]["block_outputs"]
    a1 = np.asarray(attn[0]["attn_output"]["attn_scores"])
    a2 = np.asarray(attn[1]["attn_output"]["attn_scores"])
    assert a1.shape[1:] == (1, 8, 5, 5), f"Unexpected L1 attention {a1.shape}"
    assert a2.shape[1:] == (1, 8, 5, 5), f"Unexpected L2 attention {a2.shape}"
    prev = (a1[:, 0, :, 1, 0] + a1[:, 0, :, 3, 2]) / 2
    label_attn = a2[:, 0, :, 4, :]  # query token position 4
    correct = np.where(eq0, 1, 3)
    wrong = np.where(eq0, 3, 1)
    a_corr = np.take_along_axis(label_attn, correct[:, None, None], axis=-1)[:, :, 0]
    a_wrong = np.take_along_axis(label_attn, wrong[:, None, None], axis=-1)[:, :, 0]
    margin = a_corr - a_wrong
    return {
        "per_head_L1_previous_token_attention": np.mean(prev, axis=0).tolist(),
        "per_head_L2_matching_margin": np.mean(margin, axis=0).tolist(),
        "per_head_L2_correct_label_attention": np.mean(a_corr, axis=0).tolist(),
        "per_head_L2_wrong_label_attention": np.mean(a_wrong, axis=0).tolist(),
        "per_head_L2_negative_margin_fraction": np.mean(margin < 0, axis=0).tolist(),
        "per_head_L2_matching_margin_5th_percentile": np.quantile(margin, .05, axis=0).tolist(),
        "query_support_equality_confirmed": True,
    }


def audit(artifact_root: Path, out: Path, n: int = 256, seed: int = 3201):
    model, config = load_original_checkpoint(artifact_root)
    clean_fn = make_forward(config)
    out.mkdir(parents=True, exist_ok=True)
    # Head choice is based on TRAIN-only examples; evaluation rows are disjoint
    # sources and use different seeds. Do not use a final-test sample here.
    train = source_episodes(config, role="train", n=n, seed=seed)
    tr_acc, tr_ce, tr_result = run_fn(model, clean_fn, train, seed=seed+10)
    train_mechanism = match_attention(train, tr_result)
    chosen = int(np.argmax(train_mechanism["per_head_L2_matching_margin"]))
    result = {
        "checkpoint": "original Singh JAX Base model, 1_000_000 sequences",
        "source": "pinned upstream JAX/Equinox model + opto framework",
        "train_head_selection": {
            "samples": n, "seed": seed, "accuracy": tr_acc,
            "positive_L2_matching_head": chosen,
            "L2_margin_by_head": train_mechanism["per_head_L2_matching_margin"],
            "L1_previous_token_attention_by_head": train_mechanism["per_head_L1_previous_token_attention"],
        },
        "train_split_final_test_evaluated": False,
        "validation": {},
    }
    print("CHOSEN_TRAIN_ONLY_IH_HEAD="+str(chosen),flush=True)
    for key, role in (("heldout_label_pairs", "val_relabel"),
                      ("heldout_validation_classes", "val_classes")):
        samples = source_episodes(config, role=role, n=n, seed=seed+100)
        natural, natural_loss, aux = run_fn(model, clean_fn, samples, seed=seed+200)
        mech = match_attention(samples, aux)
        d = {"n": n, "accuracy": natural, "loss": natural_loss,
             "mechanistic_scores": mech, "L2_knockout": [], "L2_retain_only": [],
             "L1_knockout": []}
        for head in range(8):
            knock, _, _ = run_fn(model, make_forward(config, ablate=[f"1:{head}"]),
                                 samples, seed=seed+200)
            retained, _, _ = run_fn(model, make_forward(
                config, ablate=[f"1:{h}" for h in range(8) if h != head]),
                samples, seed=seed+200)
            prev_knock, _, _ = run_fn(model, make_forward(config, ablate=[f"0:{head}"]),
                                      samples, seed=seed+200)
            d["L2_knockout"].append(float(knock))
            d["L2_retain_only"].append(float(retained))
            d["L1_knockout"].append(float(prev_knock))
            print(f"{key} h{head} accuracy={natural:.3f} L2_ko={knock:.3f} L2_only={retained:.3f} L1_ko={prev_knock:.3f}",flush=True)
        forced, _, _ = run_fn(model, make_forward(config, force_match=[f"1:{chosen}"]),
                               samples, seed=seed+200)
        d["forced_match_selected_head_accuracy"] = float(forced)
        d["selected_head"] = chosen
        result["validation"][key] = d
        (out / "mechanistic_dev.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key, d in result["validation"].items():
        ids = np.arange(8)
        fig, ax = plt.subplots(figsize=(9,4.8))
        ax.bar(ids-.20, np.asarray(d["L2_knockout"])*100, width=.4, label="Knockout this L2 head")
        ax.bar(ids+.20, np.asarray(d["L2_retain_only"])*100,width=.4,label="Retain only this L2 head")
        ax.axhline(100*d["accuracy"], linestyle="--",color="black",label="Natural network")
        ax.set(xlabel="0-based second-layer head index",ylabel="Validation accuracy (%)",
               title=f"Original Singh Base: head interventions ({key})",ylim=(0,105),
               xticks=ids)
        ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/f"head_ablations_{key}.png",dpi=170);plt.close(fig)
    print("BASE_MECHANISTIC_ANALYSIS="+json.dumps({
        "selected_training_head":chosen,
        "validation":{
            k:{"natural":v["accuracy"],
               "forced_match":v["forced_match_selected_head_accuracy"],
               "largest_l2_knockout_accuracy_drop":
                   float(v["accuracy"]-min(v["L2_knockout"])),
               "max_l2_matching_margin":float(max(v["mechanistic_scores"]["per_head_L2_matching_margin"]))}
            for k,v in result["validation"].items()
        },
        "final_test_evaluated":False,
    }),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root",type=Path,default=ROOT/"outputs"/"base_full_1m")
    p.add_argument("--out",type=Path,default=ROOT/"outputs"/"base_mechanistic")
    p.add_argument("--num-eval",type=int,default=256)
    p.add_argument("--seed",type=int,default=3201)
    a=p.parse_args()
    if a.num_eval < 32 or a.num_eval > 2000:
        raise ValueError("Need 32..2000 development examples")
    audit(a.artifact_root,a.out,n=a.num_eval,seed=a.seed)

if __name__=="__main__":
    main()
