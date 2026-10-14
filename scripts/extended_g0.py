"""Train the *literal original-feature* Extended G0 before recursive fitting.

Use the published JAX/Equinox 2-layer, 8-head Singh attention-only backbone,
the original 50/1473/100 class split, and the original 8/2/0 label pairs.
The new 50-way UNMASKED symbol head extends the original five-way label
classifier with two additional autoregressive output tokens.
Only development conditions are sampled. Final test stays sealed.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import main as original_main
import main_utils

from src.extended_omniglot import (
    ExtendedOmniglot, make_episodes, clean_continuation,
    validate_original_episode, ROOT, DATA,
)
from src.extended_learner import (
    extended_train_step, build_optimizer, evaluate_extended,
)

CONFIG_PATH=ROOT/"outputs"/"base_full_1m"/"safe_train_val_smoke"/"config.json"


def load_config(artifact_root:Path):
    f=artifact_root/"safe_train_val_smoke"/"config.json"
    if not f.is_file():
        raise FileNotFoundError(f"Missing verified Base configuration: {f}")
    c=json.loads(f.read_text())
    if c["train_iters"]!=1_000_000:
        raise AssertionError("Use original 1M sequence reference Base config")
    for field,val in (("class_split",[50,1473,100]),
                      ("exemplar_split",[1,4,0]),
                      ("fs_relabel_split",[8,2,0]),
                      ("model_output_classes",5)):
        if c[field]!=val: raise AssertionError(f"Bad source field {field}")
    c["data_file"]=str(DATA)
    return c


def feature_prototypes(config):
    from types import SimpleNamespace
    full=main_utils.get_data_from_opts(SimpleNamespace(**config))
    assert full.shape==(1623,5,512)
    return full[:50,0,:]


def start_g0(artifact_root:Path,out:Path,*,steps:int,batch_size:int,
             eval_every:int,eval_count:int,seed:int,lr:float,temperature:float):
    if not (steps>=1 and 2<=batch_size<=32 and eval_every>=1):
        raise ValueError("Invalid training configuration")
    if not (32<=eval_count<=1024) or temperature<=0:
        raise ValueError("Invalid evaluation configuration")
    cfg=load_config(artifact_root)
    out.mkdir(parents=True,exist_ok=True)
    model=ExtendedOmniglot(config=cfg,seed=seed)
    opt,opt_state=build_optimizer(model,lr)
    train_sampler=make_episodes(cfg,role="train",batch_size=batch_size)
    val={
        k:make_episodes(cfg,role=k,batch_size=eval_count)(
            jax.random.PRNGKey(seed+911+i))
        for i,k in enumerate(("val_relabel","val_exemplar","val_classes"))
    }
    for k,batch in val.items(): validate_original_episode(batch,k)
    prototypes=feature_prototypes(cfg)
    history=[]
    started=time.time()
    for update in range(steps):
        key=jax.random.PRNGKey(seed+31_000+update)
        input_key,next_key,training_key=jax.random.split(key,3)
        batch=train_sampler(input_key)
        clean=clean_continuation(batch,key=next_key)
        loss,model,opt_state=extended_train_step(
            model,opt,opt_state,batch,clean["tokens"],
            clean["q2_features"],training_key)
        if (update+1)%eval_every==0 or update==0 or update+1==steps:
            scores=evaluate_extended(
                model,val,feature_lookup=prototypes,
                seed=seed+130_000,temperature=temperature)
            entry={
                "sequence_exposures":(update+1)*batch_size,
                "update":update+1,
                "train_three_token_mean_loss":float(loss),
                "validation":scores,
            }
            history.append(entry)
            print("EXTENDED_G0_PROGRESS="+json.dumps(entry),flush=True)
    eqx.tree_serialise_leaves(str(out/"extended_g0_model.eqx"),model)
    meta={
        "task":"original Omniglot Singh context task + three-token continuation",
        "initialization":"fresh, NOT initialized from trained Base G0",
        "source_config":str(CONFIG_PATH),
        "original_singh_2layer_backbone":True,
        "unmasked_symbol_vocabulary_train_only":50,
        "label_vocabulary":5,
        "clean_second_symbol":"uniform support class from 2 contextual symbols",
        "learning_rate":lr,
        "optimizer":"Adam",
        "loss":"equal mean of first label, next symbol, next label CE",
        "train_target_semantics":"teacher-forced future tokens",
        "eval_generation_semantics":"sample y1, then q2, then y2",
        "class_split":[50,1473,100],
        "exemplar_split":[1,4,0],
        "relabel_pair_split":[8,2,0],
        "steps":steps,"batch_size":batch_size,
        "seed":seed,"temperature":temperature,
        "final_test_evaluated":False,
        "runtime_seconds":round(time.time()-started,3),
        "history":history,
    }
    (out/"metrics.json").write_text(json.dumps(meta,indent=2))
    return meta


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifact-root",type=Path,default=ROOT/"outputs"/"base_full_1m")
    p.add_argument("--out",type=Path,default=ROOT/"outputs"/"extended_g0")
    p.add_argument("--steps",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--eval-every",type=int,default=8)
    p.add_argument("--eval-count",type=int,default=64)
    p.add_argument("--seed",type=int,default=4054)
    p.add_argument("--learning-rate",type=float,default=1e-5)
    p.add_argument("--temperature",type=float,default=1.0)
    a=p.parse_args()
    start_g0(a.artifact_root,a.out,steps=a.steps,batch_size=a.batch_size,
             eval_every=a.eval_every,eval_count=a.eval_count,seed=a.seed,
             lr=a.learning_rate,temperature=a.temperature)


if __name__=="__main__":
    main()
