"""Extended recursive Omniglot target-feedback training (valid G0 required).

Unlike Base, Extended generates [query number, next symbol, next number].
All three are sampled autoregressively; no context-symbol mask is applied to
the 50-way symbol head. The generated q2 ID is mapped to its genuine
original ResNet18 exemplar-0 feature for the third next-token prediction.

Every generation starts with fresh original Singh transformer parameters.
Real contextual input prefixes are held clean and unchanged; only target
continuations change. We do not call this full-context recursion or true
historical corpus accumulation. Held-out test classes are never sampled.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"external"/"icl-dynamics"))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from scripts.extended_g0 import load_config, feature_prototypes
from src.extended_omniglot import ExtendedOmniglot, make_episodes, clean_continuation
from src.extended_learner import (
    build_optimizer, extended_train_step,
    autoregressive_sample, evaluate_extended,
)

ARMS=("clean_only","recursive_replacement","real_anchor_half","frozen_g0")


def load_verified_g0(cfg, artifact_root, *, require_gate=True):
    metrics_path=artifact_root/"metrics.json"
    model_path=artifact_root/"extended_g0_model.eqx"
    if not metrics_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("Extended G0 model/metrics artifact missing")
    meta=json.loads(metrics_path.read_text())
    if meta["final_test_evaluated"] or meta["class_split"]!=[50,1473,100]:
        raise AssertionError("Expected protected Omniglot original class split")
    if require_gate:
        final=meta["history"][-1]["validation"]["val_relabel"]
        # Frozen after a validation-only 300k-example LR comparison:
        # 3e-4 yielded 100%/100% label accuracies and 96.96% valid Q2 mass.
        # No protected test class was inspected in this choice.
        if meta["steps"]*meta["batch_size"]<300_000 or (
            final["first_label_accuracy"]<.985 or
            final["teacher_forced_second_label_accuracy"]<.985 or
            final["next_symbol_valid_probability_mass"]<.95):
            raise ValueError("Extended G0 has not passed the validation-selected gate")
    m=ExtendedOmniglot(config=cfg,seed=int(meta["seed"]))
    model=eqx.tree_deserialise_leaves(str(model_path),m)
    return model,meta


def training_targets(arm,batch,clean,model_source,prototypes,
                     *,data_key,temperature):
    y_clean=clean["tokens"]
    if arm=="clean_only":
        return y_clean,{"num_real":len(y_clean),"num_synthetic":0}
    if model_source is None:
        raise ValueError("Synthetic conditions need source generation model")
    synthetic=autoregressive_sample(
        model_source,batch,data_key,temperature,prototypes)
    if arm=="real_anchor_half":
        k=jax.random.fold_in(data_key,911)
        n=len(y_clean)
        subset=jax.random.permutation(k,n)[:n//2]
        real=jnp.zeros(n,dtype=bool).at[subset].set(True)
        target=jnp.where(real[:,None],y_clean,synthetic)
        return target,{"num_real":n//2,"num_synthetic":n-n//2}
    if arm in ("recursive_replacement","frozen_g0"):
        return synthetic,{"num_real":0,"num_synthetic":len(synthetic)}
    raise ValueError(f"Unsupported arm {arm}")


def train_generation(*,arm,generation,predecessor,g0,config,train_sampler,
                     val,prototypes,steps,batch_size,eval_every,seed,
                     lr,temperature,out):
    if arm not in ARMS or generation<1:
        raise ValueError("Unknown experimental condition/generation")
    model=ExtendedOmniglot(config=config,seed=seed+generation*17)
    opt,opt_state=build_optimizer(model,lr)
    folder=out/arm/f"g{generation:02d}"
    folder.mkdir(parents=True,exist_ok=True)
    started=time.time()
    track=[]
    true_real=true_synthetic=0
    for step in range(steps):
        batch_key=jax.random.PRNGKey(seed+generation*103_111+step)
        ks,clean_key,synth_key,train_key=jax.random.split(batch_key,4)
        batch=train_sampler(ks)
        clean=clean_continuation(batch,key=clean_key)
        generated_by=(g0 if arm=="frozen_g0" else predecessor)
        targets,origin=training_targets(
            arm,batch,clean,generated_by,prototypes,
            data_key=synth_key,temperature=temperature)
        q2features=prototypes[targets[:,1]]
        loss,model,opt_state=extended_train_step(
            model,opt,opt_state,batch,targets,q2features,train_key)
        true_real+=origin["num_real"]
        true_synthetic+=origin["num_synthetic"]
        if (step+1)%eval_every==0 or step==0 or step+1==steps:
            measurements=evaluate_extended(
                model,val,feature_lookup=prototypes,
                seed=seed+170_000,temperature=temperature)
            row={
                "arm":arm,"generation":generation,
                "sequence_exposures":(step+1)*batch_size,
                "mean_three_token_train_loss":float(loss),
                "real_target_fraction":true_real/(true_real+true_synthetic),
                "validation":measurements,
            }
            print("EXTENDED_RECURSIVE_PROGRESS="+json.dumps(row),flush=True)
            track.append(row)
    eqx.tree_serialise_leaves(str(folder/"model.eqx"),model)
    data={
        "task":"original Omniglot ResNet18 Extended",
        "condition":arm,"generation":generation,
        "synthetic_source_generation": (
            None if arm=="clean_only" else 0 if arm=="frozen_g0"
            else generation-1),
        "clean_original_context_prefix":True,
        "recursively_generated_input_prefix":False,
        "growing_history_corpus_accumulation":False,
        "no_symbol_validity_mask":True,
        "training_class_split":[50,1473,100],
        "exemplar_split":[1,4,0],
        "label_pair_split":[8,2,0],
        "batch_size":batch_size,"optimizer_steps":steps,
        "training_exposures":steps*batch_size,
        "num_real_targets":true_real,
        "num_synthetic_targets":true_synthetic,
        "init_seed":seed+generation*17,
        "data_seed":seed,
        "lr":float(lr),"temperature":float(temperature),
        "runtime_seconds":round(time.time()-started,3),
        "final_test_evaluated":False,
        "history":track,
    }
    (folder/"metrics.json").write_text(json.dumps(data,indent=2))
    return model,data


def run(*,base_artifact,g0_artifact,out,arms,generations,steps,
        batch_size,eval_every,eval_count,seed,lr,temperature,require_gate):
    if not (1<=generations<=20 and steps>=1 and 2<=batch_size<=32
            and batch_size%2==0 and 1<=eval_every<=steps):
        raise ValueError("Bad update/size/generation configuration")
    if not (32<=eval_count<=1024 and temperature>0):
        raise ValueError("Invalid validation size/temperature")
    if not set(arms)<=set(ARMS) or len(set(arms))!=len(arms):
        raise ValueError("Invalid conditions")
    c=load_config(base_artifact)
    g0,g0meta=load_verified_g0(c,g0_artifact,require_gate=require_gate)
    prototypes=feature_prototypes(c)
    train_sampler=make_episodes(c,role="train",batch_size=batch_size)
    val={name:make_episodes(c,role=name,batch_size=eval_count)(
         jax.random.PRNGKey(seed+911+i))
         for i,name in enumerate(("val_relabel","val_exemplar","val_classes"))}
    out.mkdir(parents=True,exist_ok=True)
    evidence={
        "original_base_config":str(base_artifact),
        "extended_g0_seed":int(g0meta["seed"]),
        "g0_validation":g0meta["history"][-1]["validation"],
        "final_test_evaluated":False,
        "recursively_generated_input_prefix":False,
        "arms":{},
    }
    for arm in arms:
        parent=g0
        evidence["arms"][arm]=[]
        for gen in range(1,generations+1):
            parent,meta=train_generation(
                arm=arm,generation=gen,predecessor=parent,
                g0=g0,config=c,train_sampler=train_sampler,
                val=val,prototypes=prototypes,steps=steps,
                batch_size=batch_size,eval_every=eval_every,
                seed=seed,lr=lr,temperature=temperature,out=out)
            evidence["arms"][arm].append({
                "generation":gen,
                "last_validation":meta["history"][-1]["validation"],
                "num_real_targets":meta["num_real_targets"],
                "num_synthetic_targets":meta["num_synthetic_targets"],
            })
            (out/"summary.json").write_text(json.dumps(evidence,indent=2))
    print("EXTENDED_RECURSION_FINISHED="+json.dumps({
        "generations":generations,"arms":list(arms),
        "final_test_evaluated":False}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base-artifact",type=Path,default=ROOT/"outputs"/"base_full_1m")
    p.add_argument("--extended-g0-artifact",type=Path,default=ROOT/"outputs"/"extended_g0_full")
    p.add_argument("--out",type=Path,default=ROOT/"outputs"/"extended_recursive")
    p.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    p.add_argument("--generations",type=int,default=2)
    p.add_argument("--steps",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--eval-every",type=int,default=8)
    p.add_argument("--eval-count",type=int,default=64)
    p.add_argument("--seed",type=int,default=6001)
    p.add_argument("--learning-rate",type=float,default=1e-5)
    p.add_argument("--temperature",type=float,default=1.0)
    p.add_argument("--allow-incomplete-g0-for-smoke",action="store_true")
    a=p.parse_args()
    run(base_artifact=a.base_artifact,g0_artifact=a.extended_g0_artifact,
        out=a.out,arms=a.arms,generations=a.generations,steps=a.steps,
        batch_size=a.batch_size,eval_every=a.eval_every,
        eval_count=a.eval_count,seed=a.seed,lr=a.learning_rate,
        temperature=a.temperature,
        require_gate=not a.allow_incomplete_g0_for_smoke)


if __name__=="__main__":
    main()
