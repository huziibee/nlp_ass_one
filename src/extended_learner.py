"""JAX training/generation/validation for the Omniglot Extended sequence.

One teacher-forced forward pass predicts: y_query, q2_symbol, y_q2.
Autoregressive generator samples y_query then q2 then y_q2, with
separate causal passes. All logits are unmasked, allowing invalid Q2
relative to the current context. No final-test class evaluation occurs.
"""
from __future__ import annotations
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from src.extended_omniglot import (
    ExtendedOmniglot, clean_continuation, training_symbol_features,
    SYMBOL_VOCAB
)


@eqx.filter_value_and_grad
def extended_loss(model, examples, labels, tokens, q2features, keys):
    out=jax.vmap(
        lambda x,y,t,f,k: model(
            x,y[:2],t[0],f,key=k)
    )(examples,labels,tokens,q2features,keys)
    y1,q2,y2=out
    n=tokens.shape[0]
    idx=jnp.arange(n)
    loss1=-jnp.mean(jax.nn.log_softmax(y1)[idx,tokens[:,0]])
    lossq=-jnp.mean(jax.nn.log_softmax(q2)[idx,tokens[:,1]])
    loss2=-jnp.mean(jax.nn.log_softmax(y2)[idx,tokens[:,2]])
    return (loss1+lossq+loss2)/3.0


@eqx.filter_jit
def extended_train_step(model, optimizer, opt_state, batch, tokens,
                        q2features, key):
    keys=jax.random.split(key,batch["examples"].shape[0])
    loss, grad=extended_loss(
        model,batch["examples"],batch["labels"],tokens,q2features,keys)
    updates,opt_state=optimizer.update(grad,opt_state)
    model=eqx.apply_updates(model,updates)
    return loss,model,opt_state


def build_optimizer(model,lr:float):
    if not 0<lr<0.01: raise ValueError("Invalid LR")
    optimizer=optax.adam(learning_rate=lr)
    return optimizer,optimizer.init(eqx.filter(model,eqx.is_array))


def _sample_one(model, x, lab, key, temperature, feature_lookup):
    a,b,c,d,e,f=jax.random.split(key,6)
    # A: no query label is supplied in original Base network's input.
    label_logits=model.base(x,lab,key=a)[-1]
    y1=jax.random.categorical(b,label_logits/temperature).astype(jnp.int32)
    # B: 7-token causal pass with arbitrary last symbol. Token 6 cannot
    # influence position-5 logits under causal attention.
    _,symbol_logits,_=model(x,lab[:2],y1,x[0],key=c)
    q2=jax.random.categorical(d,symbol_logits/temperature).astype(jnp.int32)
    if feature_lookup.shape[0]!=SYMBOL_VOCAB:
        raise ValueError("Expected 50 training class feature prototypes")
    q2feat=feature_lookup[q2]
    # C: the third generated token is conditioned on the *sampled* y1,q2.
    _,_,y2_logits=model(x,lab[:2],y1,q2feat,key=e)
    y2=jax.random.categorical(f,y2_logits/temperature).astype(jnp.int32)
    return jnp.stack((y1,q2,y2))


@eqx.filter_jit
def autoregressive_sample(model, batch, key, temperature, feature_lookup):
    keys=jax.random.split(key,batch["examples"].shape[0])
    return jax.vmap(
        lambda x,lab,k: _sample_one(model,x,lab,k,temperature,feature_lookup)
    )(batch["examples"],batch["labels"],keys)


@eqx.filter_jit
def teacher_forced_logits(model,batch,tokens,q2feat,key):
    keys=jax.random.split(key,batch["examples"].shape[0])
    return jax.vmap(
        lambda x,lab,t,feat,k: model(
            x,lab[:2],t[0],feat,key=k)
    )(batch["examples"],batch["labels"],tokens,q2feat,keys)


def metrics_from_logits(batch,tokens,logits, *,role):
    y1,q2,y2=[jnp.asarray(x) for x in logits]
    n=tokens.shape[0]
    rows=jnp.arange(n)
    supports=batch["class_ids"][:,:2]
    expected=(batch["labels"][:,-1]).astype(jnp.int32)
    p1=jax.nn.log_softmax(y1)
    p2=jax.nn.log_softmax(y2)
    result={
        "n":int(n),
        "first_label_accuracy":float(jnp.mean(jnp.argmax(y1,axis=-1)==expected)),
        "first_label_loss":float(-jnp.mean(p1[rows,expected])),
        "teacher_forced_second_label_accuracy":float(jnp.mean(
            jnp.argmax(y2,axis=-1)==tokens[:,2])),
        "teacher_forced_second_label_loss":float(-jnp.mean(
            p2[rows,tokens[:,2]])),
        "final_test_evaluated":False,
    }
    if role!="val_classes":
        ps=jax.nn.softmax(q2,axis=-1)
        mass=jnp.sum(
            jnp.take_along_axis(ps,supports,axis=-1),axis=-1)
        pred=jnp.argmax(q2,axis=-1)
        result.update({
            "next_symbol_valid_probability_mass":float(mass.mean()),
            "next_symbol_greedy_context_valid_rate":float(
                jnp.mean(jnp.any(pred[:,None]==supports,axis=-1))),
            "next_symbol_teacher_forced_cross_entropy":float(
                -jnp.mean(jax.nn.log_softmax(q2)[rows,tokens[:,1]])),
            "next_symbol_teacher_forced_accuracy":float(
                jnp.mean(pred==tokens[:,1])),
            "next_symbol_entropy":float(
                jnp.mean(-jnp.sum(ps*jnp.log(jnp.maximum(ps,1e-20)),axis=-1))),
        })
    else:
        result["next_symbol_global_class_id_evaluation"]="NOT_DEFINED_for_unseen_character_classes"
    return result


def evaluate_extended(model, validation, *,feature_lookup, seed, temperature):
    """Return clean teacher-forced and valid-scope free-running metrics."""
    out={}
    for j,(role,batch) in enumerate(validation.items()):
        key=jax.random.PRNGKey(seed+j*117)
        key_targets,key_fwd,key_gen=jax.random.split(key,3)
        c=clean_continuation(batch,key=key_targets)
        tokens=c["tokens"]
        logits=teacher_forced_logits(
            model,batch,tokens,c["q2_features"],key_fwd)
        metrics=metrics_from_logits(batch,tokens,logits,role=role)
        if role!="val_classes":
            generated=autoregressive_sample(
                model,batch,key_gen,temperature,feature_lookup)
            ids=batch["class_ids"][:,:2]
            labels=batch["labels"][:,:2]
            valid=(generated[:,1,None]==ids)
            legal=jnp.any(valid,axis=-1)
            mapped=jnp.where(valid[:,0],labels[:,0],labels[:,1])
            metrics["free_first_label_oracle_accuracy"]=float(jnp.mean(
                generated[:,0]==batch["labels"][:,-1]))
            metrics["free_next_symbol_context_validity"]=float(jnp.mean(legal))
            metrics["free_second_label_context_consistency_all"]=float(
                jnp.mean(legal & (generated[:,2]==mapped)))
            metrics["free_second_label_correct_given_valid"]=float(
                jnp.sum(legal & (generated[:,2]==mapped))/
                jnp.maximum(1,jnp.sum(legal)))
            metrics["free_next_symbol_unique_count"]=int(jnp.unique(
                generated[:,1],size=SYMBOL_VOCAB,fill_value=-1)[0:SYMBOL_VOCAB].__ne__(-1).sum())
        out[role]=metrics
    return out
