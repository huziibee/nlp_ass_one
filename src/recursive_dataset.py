"""Reproducible train-only target datasets for Base/Extended recursion experiments.

This module assembles TARGET continuations. The contextual input prefix is
never replaced by synthetic tokens. It is NOT an autoregressive generator,
a complete model trainer, or evidence that recursive model collapse occurred.

Training-only protection: call sites must provide contexts from the 50
training character classes. All alternative evaluation classes stay sealed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

import numpy as np

from scripts.reference_split import get_class_ids
from src.target_repairs import Batch, oracle_query_labels

Task = Literal["base", "extended"]
Condition = Literal[
    "clean_only", "recursive_replacement", "real_anchor_half", "fixed_g0_half"
]
CONDITIONS = (
    "clean_only",
    "recursive_replacement",
    "real_anchor_half",
    "fixed_g0_half",
)


@dataclass(frozen=True)
class TargetDataset:
    task: str
    condition: str
    targets: np.ndarray
    is_real: np.ndarray
    generation: int
    generated_by: int | None
    seed: int
    metadata: dict


def _verify_training_context(batch: Batch) -> None:
    allowed = np.asarray(get_class_ids("train"), dtype=np.int32)
    if not np.isin(batch.context_symbols, allowed).all():
        raise PermissionError("Training data must not contain validation/test classes")
    if not np.isin(batch.query_symbols, allowed).all():
        raise PermissionError("Training query must not contain validation/test classes")


def _verify_targets(task: Task, x: np.ndarray, n: int) -> np.ndarray:
    if task not in ("base", "extended"):
        raise ValueError("Task must be base or extended")
    y = np.asarray(x)
    width = 1 if task == "base" else 3
    if y.shape != (n, width) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError(f"{task} targets must be an integer [{n},{width}] array")
    if not np.all((0 <= y[:, 0]) & (y[:, 0] < 5)):
        raise ValueError("first numeric label must be 0..4")
    if task == "extended":
        if not np.all((0 <= y[:, 2]) & (y[:, 2] < 5)):
            raise ValueError("second numeric label must be 0..4")
        # This is the 50-class symbol-vocabulary TRAINING protocol, not
        # the protected final character classification space.
        train_classes = np.asarray(get_class_ids("train"))
        if not np.isin(y[:, 1], train_classes).all():
            raise ValueError("generated next symbol must belong to training vocabulary")
    return np.array(y, dtype=np.int32, copy=True)


def clean_targets(batch: Batch, task: Task, *, seed: int) -> np.ndarray:
    """Generate the clean one-label or three-token teaching targets.

    Extended oracle: choose one of the TWO contextual symbols uniformly;
    second label is the value assigned to that symbol in this episode.
    This next-symbol distribution is a RESEARCH CHOICE (not set by PDF).
    """
    _verify_training_context(batch)
    n = len(batch.query_symbols)
    y1 = oracle_query_labels(batch).astype(np.int32)
    if task == "base":
        return _verify_targets(task, y1[:, None], n)
    if task != "extended":
        raise ValueError(f"Unknown task {task!r}")
    rng = np.random.default_rng(seed)
    choice = rng.integers(0, 2, size=n)
    rows = np.arange(n)
    q2 = np.asarray(batch.context_symbols)[rows, choice].astype(np.int32)
    y2 = np.asarray(batch.context_labels)[rows, choice].astype(np.int32)
    return _verify_targets(task, np.column_stack((y1, q2, y2)), n)


def _sha(x: np.ndarray) -> str:
    arr = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(tuple(arr.shape)).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def compose_targets(
    batch: Batch,
    clean: np.ndarray,
    generated: np.ndarray | None,
    *,
    task: Task,
    condition: Condition,
    generation: int,
    seed: int,
    generated_by: int | None = None,
) -> TargetDataset:
    """Build fixed-size, equal-compute controlled TARGET-only training rows.

    clean_only: use the exact clean targets (G0 or clean-only control).
    recursive_replacement: 100% previous-generation continuations.
    real_anchor_half: randomly choose floor(N/2) clean rows; rest are from
      generation G-1 of the preceding model (matched real-data anchor).
    fixed_g0_half: half clean rows and half from a FROZEN G0 generator.

    This last control isolates repeated sampling/model changes from
    a stationary, imperfect generator. These are NOT whole-sequence
    accumulation: original context prefixes are retained in every arm.

    The caller must sample generated targets *conditioned on the exact
    same batch rows*. We explicitly track provenance but cannot verify
    a generative model's counterfactual conditioning here.
    """
    _verify_training_context(batch)
    n = len(batch.query_symbols)
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}")
    if not isinstance(generation, int) or generation < 0:
        raise ValueError("generation must be a nonnegative integer")
    if generation == 0 and condition != "clean_only":
        raise ValueError("Generation zero is trained only from clean targets")
    clean_y = _verify_targets(task, clean, n)
    if condition == "clean_only":
        if generated is not None or generated_by is not None:
            raise ValueError("Clean-only condition cannot consume synthetic targets")
        out = clean_y
        is_real = np.ones(n, dtype=bool)
    else:
        if generation < 1 or generated is None:
            raise ValueError("Synthetic condition requires generation >= 1 and outputs")
        if condition in ("recursive_replacement", "real_anchor_half"):
            if generated_by != generation - 1:
                raise ValueError("Recursive source must be generation G-1")
        else:
            if generated_by != 0:
                raise ValueError("Fixed-generator control must use G0 outputs")
        synth_y = _verify_targets(task, generated, n)
        is_real = np.zeros(n, dtype=bool)
        if condition in ("real_anchor_half", "fixed_g0_half"):
            rng = np.random.default_rng(seed)
            is_real[rng.permutation(n)[:n // 2]] = True
        out = np.where(is_real[:, None], clean_y, synth_y)
    result = np.array(out, dtype=np.int32, copy=True)
    origin = np.array(is_real, dtype=bool, copy=True)
    meta = {
        "task": task, "condition": condition, "generation": generation,
        "generated_by": generated_by, "seed": seed,
        "rows": n, "clean_rows": int(origin.sum()),
        "synthetic_rows": int((~origin).sum()),
        "realized_real_fraction": float(origin.mean()),
        "clean_target_sha256": _sha(clean_y),
        "training_target_sha256": _sha(result),
        "uses_sealed_final_test": False,
        "target_only_synthetic_feedback": True,
        "canonical_split_source": "scripts/reference_split.py",
    }
    if condition != "clean_only":
        meta["generated_target_sha256"] = _sha(synth_y)
    return TargetDataset(
        task=task, condition=condition, targets=result, is_real=origin,
        generation=generation, generated_by=generated_by, seed=seed,
        metadata=meta,
    )
