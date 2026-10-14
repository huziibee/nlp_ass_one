"""Verify the frozen-evaluation code WITHOUT opening final-test examples.

Loads original Base and Extended G0, scores *validation* classes with the
same evaluators used later by the once-only script, and verifies that
missing confirmation blocks all final-test data access.
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts"))

import numpy as np

from base_mechanistic import load_original_checkpoint, source_episodes
from evaluate_sealed import (
    final_evaluation, load_frozen_model,
    predictions_base, predictions_extended, logit_metrics,
)


def main():
    root=ROOT/"outputs"/"base_full_1m"
    ext=ROOT/"outputs"/"extended_g0_selected"/"extended_g0_model.eqx"
    fake=ROOT/"outputs"/"final_evaluator_guard_sentinel"
    try:
        final_evaluation(
            kind="original_base_g0",config_root=root,checkpoint=None,
            out=fake,n=128,chunk=32,seed=20261027,
            init_seed=0,confirm=False)
    except PermissionError:
        pass
    else:
        raise AssertionError("Final evaluator failed to require confirmation")
    assert not fake.exists(), "No test artifact may exist"
    base,conf=load_original_checkpoint(root)
    batch=source_episodes(conf,role="val_classes",n=128,seed=2121)
    x,y=np.asarray(batch["examples"]),np.asarray(batch["labels"])
    assert x.shape==(128,3,512)
    b=predictions_base(base,x,y,batch_size=32,seed=818)
    sb=logit_metrics(b,y[:,-1])
    assert 0<=sb["accuracy"]<=1 and np.isfinite(sb["cross_entropy"])
    extended,_,_ = load_frozen_model(
        "extended_g0",root,ext,init_seed=4054)
    y1,y2,t2=predictions_extended(
        extended,x,y,batch_size=32,seed=818)
    se1=logit_metrics(y1,y[:,-1])
    se2=logit_metrics(y2,t2)
    assert all(np.isfinite(v["cross_entropy"]) for v in (se1,se2))
    assert sb["n"]==se1["n"]==se2["n"]==128
    assert x.max()>-float("inf")
    print("EVALUATOR_VALIDATION_SMOKE_PASSED",{
        "base_val_class":sb,
        "extended_val_class_y1":se1,
        "extended_val_class_y2_clean_teacher":se2,
        "final_test_evaluated":False,
    })


if __name__=="__main__":
    main()
