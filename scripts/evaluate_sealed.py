"""One-time frozen 100-class final evaluator for original-Omniglot Base/Extended.

IMPORTANT: this is deliberately NOT part of train-time or validation workflows.
Invoke ONLY after the entire task, architecture, training and hyperparameters
are frozen. Unlike the authors' periodically monitored test evaluator, this
uses the sealed last 100 HDF5 classes for a one-time independent result.

Extended global 50-class q2 symbol logits cannot label these unseen test
symbols. Therefore final Extended metrics are the first label and
teacher-forced second label on *clean* held-out test-class Q2 features.
No invented zero-shot Q2 accuracy and no tuning after the test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"external"/"icl-dynamics"))

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np

import main_utils
from base_mechanistic import load_original_checkpoint
from extended_g0 import load_config
from episode_probe import generate_episodes, validate_episodes
from src.extended_omniglot import ExtendedOmniglot, DATA


def sealed_episodes(n:int,seed:int):
    if n<128: raise ValueError("Final sample count must be >=128")
    sampled=generate_episodes("final_test",n,seed,allow_final=True)
    validate_episodes(sampled,"final_test")
    ids=sampled["symbol_ids"]
    with h5py.File(DATA,"r") as f:
        all_feats=np.asarray(f["resnet18/224/feat"])
    e=sampled["exemplar_ids"]
    x=all_feats[ids,e]
    y=np.column_stack((sampled["context_labels"],sampled["targets"]))
    if not (np.min(ids)>=1523 and np.max(ids)<1623):
        raise AssertionError("Final episodes must use only the 100 held-out classes")
    return x.astype(np.float32),y.astype(np.int32),ids.astype(np.int32)


def logit_metrics(logits,labels):
    logits=np.asarray(logits,dtype=np.float64)
    labels=np.asarray(labels,dtype=np.int32)
    z=logits-logits.max(axis=-1,keepdims=True)
    logps=z-np.log(np.exp(z).sum(axis=-1,keepdims=True))
    return {"accuracy":float(np.mean(np.argmax(logits,axis=-1)==labels)),
            "cross_entropy":float(-np.mean(logps[np.arange(len(labels)),labels])),
            "n":int(len(labels))}


def predictions_base(model, x, y, *,batch_size,seed):
    outputs=[]
    for start in range(0,len(y),batch_size):
        end=min(len(y),start+batch_size)
        keys=jax.random.split(jax.random.PRNGKey(seed+start),end-start)
        out=jax.vmap(
            lambda ex,lab,k: model(ex,lab,key=k)[-1]
        )(jnp.asarray(x[start:end]),jnp.asarray(y[start:end]),keys)
        outputs.append(np.asarray(out))
    return np.concatenate(outputs,axis=0)


def predictions_extended(model,x,y,*,batch_size,seed):
    rng=np.random.default_rng(seed+741)
    choose=rng.integers(0,2,size=len(y))
    q2feat=x[np.arange(len(y)),choose,:]
    second_target=y[np.arange(len(y)),choose]
    first_logits,second_logits=[],[]
    for start in range(0,len(y),batch_size):
        end=min(len(y),start+batch_size)
        keys=jax.random.split(jax.random.PRNGKey(seed+start),end-start)
        logits=jax.vmap(
            lambda ex,lab,f,k:model(ex,lab[:2],lab[2],f,key=k)
        )(jnp.asarray(x[start:end]),jnp.asarray(y[start:end]),
          jnp.asarray(q2feat[start:end]),keys)
        first_logits.append(np.asarray(logits[0]))
        second_logits.append(np.asarray(logits[2]))
    return (np.concatenate(first_logits),np.concatenate(second_logits),
            second_target)


def load_frozen_model(kind, cfg_root, checkpoint, *,init_seed):
    c=load_config(cfg_root)
    if kind=="original_base_g0":
        model,source=load_original_checkpoint(cfg_root)
        return model,c,{"source":"verified original Singh 1M Base G0"}
    if checkpoint is None:
        raise ValueError("Checkpoint is mandatory for trained recursive/Extended models")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    meta_file=checkpoint.parent/"metrics.json"
    meta=json.loads(meta_file.read_text())
    if meta.get("final_test_evaluated") is True:
        raise AssertionError("Cannot use post-test model provenance for sealed confirmation")
    if kind=="base_recursive":
        cfg_dict={**c,"init_seed":int(init_seed)}
        template=main_utils.get_model_from_opts(SimpleNamespace(**cfg_dict))
    elif kind in ("extended_g0","extended_recursive"):
        if kind=="extended_g0" and "seed" in meta:
            init_seed=int(meta["seed"])
        template=ExtendedOmniglot(config=c,seed=int(init_seed))
    else: raise ValueError(f"Invalid model kind {kind}")
    model=eqx.tree_deserialise_leaves(str(checkpoint),template)
    return model,c,{"checkpoint":str(checkpoint),
                    "training_metadata":meta_file.name,
                    "source_model_kind":kind,"initialization_seed":init_seed}


def final_evaluation(*,kind,config_root,checkpoint,out,n,chunk,seed,
                     init_seed,confirm):
    if not confirm:
        raise PermissionError("Explicit --confirm-final-frozen required")
    if out.exists():
        raise FileExistsError("Once-only final result already exists: "+str(out))
    if chunk<1 or n%chunk!=0:
        raise ValueError("Use fixed chunk sizes dividing final evaluation size")
    model,c,provenance=load_frozen_model(
        kind,config_root,checkpoint,init_seed=init_seed)
    x,y,ids=sealed_episodes(n,seed)
    scores={}
    if kind.startswith("extended"):
        l1,l2,y2=predictions_extended(model,x,y,batch_size=chunk,seed=seed+1000)
        scores["first_label"]=logit_metrics(l1,y[:,-1])
        scores["second_label_given_clean_q2"]=logit_metrics(l2,y2)
        scores["next_symbol_global_class_id_accuracy"]="UNDEFINED_ON_UNSEEN_CLASSES"
    else:
        logits=predictions_base(model,x,y,batch_size=chunk,seed=seed+1000)
        scores["first_label"]=logit_metrics(logits,y[:,-1])
    split_data=(ROOT/"docs/data-audit/reference_split_manifest.json").read_bytes()
    meta={
        "model_kind":kind,"scores":scores,
        "sealed_class_ids_min":int(ids.min()),
        "sealed_class_ids_max":int(ids.max()),
        "final_n":n,"fixed_evaluation_seed":seed,
        "reference_split_manifest_sha256":hashlib.sha256(split_data).hexdigest(),
        "data_provenance":"original pinned Singh ResNet18 HDF5, class indices 1523..1622",
        "next_symbol_test_scoring":"disabled: output head only has 50 training classes",
        "once_only_final_protocol":True,
        "source":provenance,
    }
    out.mkdir(parents=True)
    (out/"sealed_final_scores.json").write_text(json.dumps(meta,indent=2))
    print("FROZEN_FINAL_EVALUATION="+json.dumps(meta),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-kind",required=True,choices=[
        "original_base_g0","base_recursive","extended_g0","extended_recursive"])
    p.add_argument("--config-root",type=Path,default=ROOT/"outputs/base_full_1m")
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--n",type=int,default=4096)
    p.add_argument("--chunk",type=int,default=128)
    p.add_argument("--seed",type=int,default=20261027)
    p.add_argument("--init-seed",type=int,default=0)
    p.add_argument("--confirm-final-frozen",action="store_true")
    a=p.parse_args()
    final_evaluation(kind=a.model_kind,config_root=a.config_root,
        checkpoint=a.checkpoint,out=a.out,n=a.n,chunk=a.chunk,
        seed=a.seed,init_seed=a.init_seed,confirm=a.confirm_final_frozen)


if __name__=="__main__":
    main()
