"""One-million-sequence Base induction-head training using ORIGINAL Singh et al. JAX code.

IMPORTANT: this trains a genuine research baseline; inspect validation only,
not final 100 held-out classes. Do not infer ICL until validation metrics and
head causal checks support it. Training architecture/data hyperparameters are
taken from the upstream main figure configuration. Periodic evaluation and
checkpoint frequency are changed for compute efficiency, disclosed here.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from paper_base_smoke import ROOT, source_checked_cmd

UPDATES = 31_250
BATCH_SIZE = 32
TOTAL_SEQUENCES = UPDATES * BATCH_SIZE
assert TOTAL_SEQUENCES == 1_000_000

def main():
    out = ROOT / "outputs" / "base_full_1m"
    out.mkdir(parents=True, exist_ok=True)
    cmd = source_checked_cmd(
        steps=UPDATES,
        out=out,
        batch_size=BATCH_SIZE,
        eval_every_updates=625,     # every 20,000 training sequences
        eval_iters=512,
        ckpt_every_updates=3125,    # every 100,000 sequences
        max_updates=100_000,
    )
    eval_idx = cmd.index("--pe_classes")
    assert cmd[eval_idx+1:eval_idx+5] == ["train","train","train","val"]
    assert "test" not in cmd[eval_idx+1:eval_idx+5]
    protocol={
        "source": "original authors' pinned JAX/Equinox code",
        "primary_task": "Base: query feature -> contextual numeric label",
        "train_sequences": TOTAL_SEQUENCES,
        "optimizer_updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "d_model": 64,
        "num_heads": 8,
        "source_lr": 1e-5,
        "eval_every_sequences": 20000,
        "checkpoint_every_sequences": 100000,
        "eval_size_per_condition": 512,
        "final_test_class_indices_evaluated": False,
        "intentional_change_from_authors":
            "fourth periodic evaluator is val classes, not test classes; "
            "eval and checkpoint frequency reduced for compute. Model and "
            "training data/optimizer parameters follow original main run.",
        "command": cmd,
    }
    (out / "course_safe_base_protocol.json").write_text(
        json.dumps(protocol,indent=2),encoding="utf8")
    print("BASE_FULL_PROTOCOL="+json.dumps({
        k:v for k,v in protocol.items() if k != "command"
    }),flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)
    print("BASE_FULL_RUN_COMPLETED_WITHOUT_FINAL_TEST",flush=True)

if __name__=="__main__":
    main()
