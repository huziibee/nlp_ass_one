"""Source-aligned seven-token Extended mechanistic validation across G0/G3/G6/G10.

Constructs the exact Extended causal embeddings, reads original Singh
transformer attention arrays for tokens 4 and 6, and performs genuine
test-time head-output ablation by zeroing the corresponding L2 projection
columns (leaving the original saved model untouched).

Freeze candidate heads by G0 TRAIN-only attention; do not select by G10
validation performance. These interventions do not establish training-
data causality, and are not equivalent to model-fixing ablations.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"external"/"icl-dynamics"))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from scripts.extended_g0 import load_config
from src.extended_omniglot import ExtendedOmniglot,make_episodes,clean_continuation

ROLES=("val_relabel","val_exemplar","val_classes")
GENERATIONS=(0,3,6,10)

def load_model(cfg,path,*,gen:int):
    config_path=path/"metrics.json"
    ckpt=path/("extended_g0_model.eqx" if gen==0 else "model.eqx")
    if not config_path.is_file() or not ckpt.is_file():
        raise FileNotFoundError("Missing Extended checkpoint/metrics "+str(path))
    meta=json.loads(config_path.read_text())
    if meta["final_test_evaluated"] is not False:
        raise AssertionError("Post-test checkpoint not permitted")
    if gen==0:
        assert meta["class_split"]==[50,1473,100]
        seed=int(meta["seed"])
    else:
        assert meta["generation"]==gen and meta["training_class_split"]==[50,1473,100]
        seed=int(meta["init_seed"])
    m=ExtendedOmniglot(config=cfg,seed=seed)
    return eqx.tree_deserialise_leaves(str(ckpt),m),meta

def original_aux_one(model,x,context_labels,y1,q2feat,*,key):
    base=model.base
    features=jnp.concatenate([x,q2feat[None,:]],axis=0)
    v=jnp.concatenate([context_labels,y1[None]],axis=0)
    e=base.example_embed_drop(
        jax.vmap(base.example_embed)(features),
        key=jax.random.PRNGKey(jnp.sum(features).astype(int)),
        inference=False)
    labels=jax.vmap(base.label_embed)(
        jax.nn.one_hot(v,base.num_classes))
    tokens=jnp.stack((e[0],labels[0],e[1],labels[1],e[2],labels[2],e[3]))
    k_pos,k_forward=jax.random.split(key)
    pos=jax.nn.one_hot(jnp.arange(7,dtype=jnp.float32),7)
    pos_emb=base.pos_embed_drop(jax.vmap(base.pos_embed)(pos),key=k_pos)
    aux=base.transformer.call_with_all_aux(tokens+pos_emb,key=k_forward)
    hidden=aux["out"]
    attn=aux["block_outputs"]
    return (
        base.unembed(hidden[4]),
        model.symbol_unembed(hidden[5]),
        base.unembed(hidden[6]),
        attn[0]["attn_output"]["attn_scores"],
        attn[1]["attn_output"]["attn_scores"],
    )

def batch_aux(model,batch,clean,key):
    keys=jax.random.split(jax.random.PRNGKey(key),len(batch["labels"]))
    return jax.vmap(
        lambda x,y,t,q2,k: original_aux_one(
            model,x,y[:2],t[0],q2,key=k)
    )(batch["examples"],batch["labels"],clean["tokens"],
      clean["q2_features"],keys)

def attention_metrics(batch,clean,a1,a2):
    ids=np.asarray(batch["class_ids"])
    q2=np.asarray(clean["tokens"][:,1])
    attn1=np.asarray(a1)[:,0]
    attn2=np.asarray(a2)[:,0]
    n=len(ids)
    assert attn1.shape==(n,8,7,7) and attn2.shape==(n,8,7,7)
    correct_first=np.where(ids[:,2]==ids[:,0],1,3)
    correct_second=np.where(q2==ids[:,0],1,3)
    def margins(pos,correct):
        wrong=np.where(correct==1,3,1)
        per=attn2[:,:,pos,:]
        right=np.take_along_axis(per,correct[:,None,None],axis=-1)[:,:,0]
        wrongs=np.take_along_axis(per,wrong[:,None,None],axis=-1)[:,:,0]
        return (right-wrongs).mean(axis=0).tolist(),right.mean(axis=0).tolist()
    margin_first,attention_first=margins(4,correct_first)
    margin_second,attention_second=margins(6,correct_second)
    return {
        "l1_prev_token_mean_by_head":(
            (attn1[:,:,1,0]+attn1[:,:,3,2])/2).mean(axis=0).tolist(),
        "l2_query_first_number_matching_margin_by_head":margin_first,
        "l2_query_second_number_matching_margin_by_head":margin_second,
        "l2_correct_label_first_attention_by_head":attention_first,
        "l2_correct_label_second_attention_by_head":attention_second,
        "l2_q2_prediction_attention_to_support_symbols_by_head":(
            attn2[:,:,5,0]+attn2[:,:,5,2]).mean(axis=0).tolist(),
    }

def accuracy_scores(batch,clean,logits):
    labels=np.asarray(batch["labels"])
    outputs=np.asarray(clean["tokens"])
    first=np.asarray(logits[0])
    next_=np.asarray(logits[1])
    second=np.asarray(logits[2])
    result={
        "first_label_accuracy":float((first.argmax(axis=-1)==labels[:,-1]).mean()),
        "second_label_teacher_forced_accuracy":float(
            (second.argmax(axis=-1)==outputs[:,2]).mean()),
    }
    if np.asarray(batch["class_ids"]).max()<50:
        ids=np.asarray(batch["class_ids"])[:,:2]
        e=np.exp(next_-next_.max(axis=-1,keepdims=True))
        p=e/e.sum(axis=-1,keepdims=True)
        result["next_symbol_valid_probability_mass"]=float(
            (p[np.arange(len(ids)),ids[:,0]]+
             p[np.arange(len(ids)),ids[:,1]]).mean())
    else:
        result["next_symbol_global_id_metric"]="NOT_DEFINED_UNSEEN_CLASSES"
    return result

def knock_out_head(model,layer,head):
    if layer!=1:raise ValueError("Probe currently supports L2 projection-head removal")
    weight=model.base.transformer.blocks[1].attn.proj.weight
    d=int(weight.shape[1])
    if d%8:raise ValueError("Head dimension mismatch")
    width=d//8
    modified=weight.at[:,head*width:(head+1)*width].set(0)
    return eqx.tree_at(
        lambda m:m.base.transformer.blocks[1].attn.proj.weight,
        model,modified)

def evaluate(*,base_root,g0_root,g3_root,g10_root,arm,out,n,seed):
    cfg=load_config(base_root)
    g0,source=load_model(cfg,g0_root,gen=0)
    train=make_episodes(cfg,role="train",batch_size=256)(
        jax.random.PRNGKey(seed+37))
    clean=clean_continuation(train,key=jax.random.PRNGKey(seed+38))
    outputs=batch_aux(g0,train,clean,key=seed+39)
    # Verify that the manual seven-token auxiliary trace and actual trained
    # Extended model return exactly the same logits.
    key=jax.random.PRNGKey(seed+39)
    original=g0(
        train["examples"][0],train["labels"][0,:2],
        clean["tokens"][0,0],clean["q2_features"][0],
        key=jax.random.split(key,256)[0])
    for i in range(3):
        np.testing.assert_allclose(
            np.asarray(outputs[i][0]),np.asarray(original[i]),
            atol=3e-5,rtol=3e-5)
    base_mech=attention_metrics(train,clean,outputs[3],outputs[4])
    selected_first=int(np.argmax(
        base_mech["l2_query_first_number_matching_margin_by_head"]))
    selected_second=int(np.argmax(
        base_mech["l2_query_second_number_matching_margin_by_head"]))
    result={
        "task":"Extended original seven-token transformer mechanism",
        "arm":arm,
        "generations":list(GENERATIONS),
        "head_selection":"G0 training-only marginal matching margin",
        "selected_l2_first_head":selected_first,
        "selected_l2_second_head":selected_second,
        "actual_extended_logits_equal_manual_seven_token_aux":True,
        "test_evaluated":False,
        "model_replicates":1,
        "test_time_ablation_does_not_prove_data_causality":True,
        "results":{},
    }
    for gen in GENERATIONS:
        folder=(g0_root if gen==0 else
                g3_root/arm/"g03" if gen==3 else
                g10_root/arm/f"g{gen:02d}")
        model,_=(g0,source) if gen==0 else load_model(cfg,folder,gen=gen)
        outputs_gen={}
        for i,role in enumerate(ROLES):
            batch=make_episodes(cfg,role=role,batch_size=n)(
                jax.random.PRNGKey(seed+1000+i*101))
            clean=clean_continuation(batch,key=jax.random.PRNGKey(seed+1001+i*101))
            kwargs=dict(key=seed+1100+i*101)
            z=batch_aux(model,batch,clean,**kwargs)
            z_first=batch_aux(knock_out_head(model,1,selected_first),batch,clean,**kwargs)
            z_second=batch_aux(knock_out_head(model,1,selected_second),batch,clean,**kwargs)
            metrics={
                "natural":accuracy_scores(batch,clean,z[:3]),
                "ablated_first_candidate":accuracy_scores(batch,clean,z_first[:3]),
                "ablated_second_candidate":accuracy_scores(batch,clean,z_second[:3]),
                "attention":attention_metrics(batch,clean,z[3],z[4]),
                "num_validation_episodes":n,
            }
            outputs_gen[role]=metrics
        result["results"][str(gen)]=outputs_gen
        print("EXTENDED_CIRCUIT_GENERATION="+json.dumps({
            "arm":arm,"generation":gen,
            "selected_first_head":selected_first,
            "selected_second_head":selected_second,
            "validation":{
                role:{
                    "first":v["natural"]["first_label_accuracy"],
                    "second":v["natural"]["second_label_teacher_forced_accuracy"],
                    "first_candidate_ablation":v["ablated_first_candidate"]["first_label_accuracy"],
                    "second_candidate_ablation":v["ablated_second_candidate"]["second_label_teacher_forced_accuracy"],
                } for role,v in outputs_gen.items()
            },
            "test_evaluated":False,
        }),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print("EXTENDED_SEVENTOKEN_MECHANISTIC_AUDIT_PASSED="+json.dumps({
        "arm":arm,"generations":list(GENERATIONS),
        "selected_first_head":selected_first,
        "selected_second_head":selected_second,
        "no_final_test":True}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-root",type=Path,default=ROOT/"outputs/base_full_1m")
    p.add_argument("--g0-root",type=Path,default=ROOT/"outputs/extended_g0")
    p.add_argument("--g3-root",type=Path,required=True)
    p.add_argument("--g10-root",type=Path,required=True)
    p.add_argument("--arm",choices=[
        "clean_only","frozen_g0","real_anchor_half","recursive_replacement"
    ],required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--n",type=int,default=128)
    p.add_argument("--seed",type=int,default=4817)
    a=p.parse_args()
    if not 32<=a.n<=512:raise ValueError("Invalid validation size")
    evaluate(base_root=a.base_root,g0_root=a.g0_root,
             g3_root=a.g3_root,g10_root=a.g10_root,arm=a.arm,
             out=a.out,n=a.n,seed=a.seed)
if __name__=="__main__":
    main()
