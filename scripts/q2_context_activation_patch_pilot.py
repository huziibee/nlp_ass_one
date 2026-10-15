"""Exploratory within-model activation patching of contextual next-symbol prediction.

Intervention: on paired Omniglot val_relabel prompts, replace the nonquery
support character B with a different legal TRAIN character C. A and query are
unchanged; B's numeric label is assigned to C. Thus the correct next-symbol
support set changes from {A,B} to {A,C}. Within the SAME trained model,
patch donor {A,B} attention-head activations at token 5 into target {A,C}
and measure the shifted q2 logit margin logit(C)-logit(B).

This tests conditional symbol information flow; NOT proof that recursive
corrupt training targets caused any discovered circuit. No final test.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"external"/"icl-dynamics")]
import jax
import jax.numpy as jnp
import numpy as np
from scripts.extended_g0 import load_config, feature_prototypes
from scripts.compare_extended_circuits_g10 import load_model
from src.extended_omniglot import make_episodes

def forward_one(model,features,context_labels,y1,feature_last,key,
                donor=None,layer=1,mask=None):
    base=model.base
    ex=jnp.concatenate([features,feature_last[None,:]],axis=0)
    emb=base.example_embed_drop(
        jax.vmap(base.example_embed)(ex),
        key=jax.random.PRNGKey(jnp.sum(ex).astype(int)),inference=False)
    lab=jax.vmap(base.label_embed)(jax.nn.one_hot(
        jnp.concatenate([context_labels,y1[None]],axis=0),5))
    tokens=jnp.stack((emb[0],lab[0],emb[1],lab[1],
                      emb[2],lab[2],emb[3]))
    kpos,kforward=jax.random.split(key)
    pos=jax.nn.one_hot(jnp.arange(7,dtype=jnp.float32),7)
    inp=tokens+base.pos_embed_drop(jax.vmap(base.pos_embed)(pos),key=kpos)
    if donor is None:
        aux=base.transformer.call_with_all_aux(inp,key=kforward)
    else:
        nodes=[{},{}]; masks=[{},{}]
        nodes[layer]={"attn_output":{"values_pre_proj":donor}}
        masks[layer]={"attn_output":{"values_pre_proj":mask}}
        aux=base.transformer.call_with_all_aux(
            inp,key=kforward,
            cache={"block_outputs":nodes},
            cache_mask={"block_outputs":masks})
    return (model.symbol_unembed(aux["out"][5]),
            aux["block_outputs"][layer]["attn_output"]["values_pre_proj"])

def create_prompt_pair(config,n,seed):
    raw=make_episodes(config,role="val_relabel",batch_size=n*3)(
        jax.random.PRNGKey(seed))
    ids=np.asarray(raw["class_ids"])
    take=np.flatnonzero(ids[:,2]==ids[:,0])[:n]
    if len(take)!=n:raise AssertionError("Insufficient original A-query episodes")
    batch={k:jnp.asarray(v)[take] for k,v in raw.items()}
    ids=np.asarray(batch["class_ids"]); a=ids[:,0];b=ids[:,1]
    rng=np.random.default_rng(seed+66)
    c=np.asarray([rng.choice([k for k in range(50) if k not in (int(x),int(y))])
                  for x,y in zip(a,b)],dtype=np.int32)
    proto=feature_prototypes(config)
    revised={k:jnp.array(v) for k,v in batch.items()}
    revised["examples"]=batch["examples"].at[:,1,:].set(proto[c])
    revised["class_ids"]=batch["class_ids"].at[:,1].set(c)
    revised["exemplar_ids"]=batch["exemplar_ids"].at[:,1].set(0)
    assert bool(jnp.all(revised["labels"][:,-1]==revised["labels"][:,0]))
    return batch,revised,a,b,c

def run(args):
    config=load_config(args.base_root)
    model,meta=load_model(config,args.model_root,gen=10)
    assert meta["final_test_evaluated"] is False
    source,target,a,b,c=create_prompt_pair(config,args.n,args.seed)
    assert np.all(a!=b) and np.all(a!=c) and np.all(b!=c)
    feat=source["examples"][:,0]
    label=source["labels"][:,-1]
    keys=jax.random.split(jax.random.PRNGKey(args.seed+83),args.n)

    # Compile per layer using exact source forward: each patch uses identical
    # model weights and target input. Only donor head vectors are substituted.
    def natural(batch,layer):
        return jax.jit(jax.vmap(
            lambda x,labels,y,q,k:forward_one(
                model,x,labels[:2],y,q,k,layer=layer)
        ))(batch["examples"],batch["labels"],label,feat,keys)

    def patched(layer,donors,recipient,mask):
        return jax.jit(jax.vmap(
            lambda x,labels,y,q,k,d:forward_one(
                model,x,labels[:2],y,q,k,donor=d,layer=layer,mask=mask)
        ))(recipient["examples"],recipient["labels"],label,feat,keys,donors)[0]

    results={"arm":args.arm,"n":args.n,"seed":args.seed,
             "evaluation":"validation heldout label pairs, training character vocabulary",
             "intervention":"within-model original B support -> alternative C support",
             "patched_position":5,"metric":"logit(C)-logit(B)",
             "reported_generation":10,"trained_model_init_seed":meta["init_seed"],
             "final_test_accessed":False,"layers":{}}
    for layer in [0,1]:
        src,srcvec=natural(source,layer)
        tar,tarvec=natural(target,layer)
        d=int(srcvec.shape[-1]);hsize=d//8
        if d%8:raise AssertionError("Head dims")
        orig=model(
            source["examples"][0],source["labels"][0,:2],
            label[0],feat[0],key=keys[0])[1]
        if layer==0:
            np.testing.assert_allclose(np.asarray(src[0]),np.asarray(orig),
                                       rtol=3e-5,atol=3e-5)
        scores=np.asarray(tar)
        donor=np.asarray(src)
        rows=np.arange(args.n)
        def margin(z):return np.asarray(z)[rows,c]-np.asarray(z)[rows,b]
        def valid(z):
            p=jax.nn.softmax(z,axis=-1)
            p=np.asarray(p)
            return float((p[rows,a]+p[rows,c]).mean())
        target_margin=margin(tar); donor_margin=margin(src)
        source_preference=donor_margin.mean()
        target_preference=target_margin.mean()
        shifts={}
        for head in range(8):
            mask=jnp.zeros((7,d),dtype=bool).at[5,head*hsize:(head+1)*hsize].set(True)
            patched_logits=patched(layer,srcvec,target,mask)
            deltas=margin(patched_logits)-target_margin
            shifts[str(head)]={
                "q2_old_B_vs_new_C_margin_shift":float(deltas.mean()),
                "mean_absolute_shift":float(np.mean(np.abs(deltas))),
                "valid_mass_after_patch":valid(patched_logits)}
        mask=jnp.zeros((7,d),dtype=bool).at[5,:].set(True)
        full=patched(layer,srcvec,target,mask)
        identity=patched(layer,tarvec,target,mask)
        mask_future=jnp.zeros((7,d),dtype=bool).at[6,:].set(True)
        future=patched(layer,srcvec,target,mask_future)
        np.testing.assert_allclose(np.asarray(identity),np.asarray(tar),
                                   atol=3e-5,rtol=3e-5)
        np.testing.assert_allclose(np.asarray(future),np.asarray(tar),
                                   atol=3e-5,rtol=3e-5)
        results["layers"][str(layer+1)]={
            "donor_margin_mean":float(source_preference),
            "recipient_margin_mean":float(target_preference),
            "donor_minus_recipient_mean":float(source_preference-target_preference),
            "recipient_valid_mass":valid(tar),
            "source_valid_mass_on_target_set":valid(src),
            "whole_layer_patch_margin_shift":float((margin(full)-target_margin).mean()),
            "whole_layer_patch_valid_mass":valid(full),
            "identity_patch_max_error":float(np.abs(np.asarray(identity)-np.asarray(tar)).max()),
            "future_token_patch_max_error":float(np.abs(np.asarray(future)-np.asarray(tar)).max()),
            "head_level":shifts}
        print("Q2_PATCH_LAYER="+json.dumps({
            "arm":args.arm,"layer":layer+1,
            "original_to_changed_margin":float(target_preference-source_preference),
            "whole_layer_shift":results["layers"][str(layer+1)]["whole_layer_patch_margin_shift"],
            "max_single_head_shift":max(abs(x["q2_old_B_vs_new_C_margin_shift"]) for x in shifts.values())
        }),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(results,indent=2))
    print("Q2_PATCH_COMPLETE="+json.dumps({"arm":args.arm,"file":str(args.out)}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--arm",choices=["clean_only","recursive_replacement"],required=True)
    p.add_argument("--base-root",type=Path,required=True)
    p.add_argument("--model-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--n",type=int,default=128)
    p.add_argument("--seed",type=int,default=67822)
    o=p.parse_args()
    if o.n not in (64,128,256):raise ValueError("Use bounded validation sample")
    run(o)
