"""Deterministic, unit-testable synthetic continuation repair operators.

Scientific use: experiments on the extended, contextual two-way induction
task; do NOT interpret these functions as a trained model, a causal finding,
or a substitute for matched retraining. The original clean prefix remains
unchanged. The continuation layout is [first_label, next_symbol, next_label].

Unlike a global character classifier, the oracle value of a next symbol is
defined *only if the symbol is one of the two support characters*.
An invalid next symbol therefore cannot have a context-defined oracle
label. Every repair below preserves that fact.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np

Mode = Literal["raw", "first_label_only", "valid_second_label_only",
               "label_where_defined", "joint_repair"]
MODES = ("raw", "first_label_only", "valid_second_label_only",
         "label_where_defined", "joint_repair")


@dataclass(frozen=True)
class Batch:
    """Two-way context and a query; all arrays have n rows."""
    context_symbols: np.ndarray  # (n, 2): two distinct HDF5 class IDs
    context_labels: np.ndarray   # (n, 2): two distinct labels from {0,...,4}
    query_symbols: np.ndarray    # (n,): one of context_symbols columns

    def __post_init__(self):
        sym = np.asarray(self.context_symbols)
        lab = np.asarray(self.context_labels)
        q = np.asarray(self.query_symbols)
        n = len(q)
        if sym.shape != (n, 2) or lab.shape != (n, 2):
            raise ValueError("Contexts must be [n,2] and queries [n]")
        if n == 0:
            raise ValueError("Batch must not be empty")
        if not np.issubdtype(sym.dtype, np.integer) or not np.issubdtype(lab.dtype, np.integer):
            raise TypeError("Symbol and label arrays must be integer")
        if not np.issubdtype(q.dtype, np.integer):
            raise TypeError("Query array must be integer")
        if np.any(sym[:, 0] == sym[:, 1]):
            raise ValueError("Support symbols must be distinct")
        if np.any(lab[:, 0] == lab[:, 1]):
            raise ValueError("Contextual labels must be distinct")
        if np.any(lab < 0) or np.any(lab >= 5):
            raise ValueError("Labels must be in [0,4]")
        if not np.all((q == sym[:, 0]) | (q == sym[:, 1])):
            raise ValueError("Query must match one of two contextual symbols")


@dataclass(frozen=True)
class RepairResult:
    mode: str
    targets: np.ndarray
    changed_mask: np.ndarray
    original_metrics: dict
    repaired_metrics: dict
    seed: int


def _coerce_targets(batch: Batch, targets: np.ndarray) -> np.ndarray:
    y = np.asarray(targets)
    n = len(batch.query_symbols)
    if y.shape != (n, 3) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("targets must be integer array [n,3] = [y1,q2,y2]")
    if np.any(y[:, (0, 2)] < 0) or np.any(y[:, (0, 2)] >= 5):
        raise ValueError("Generated value tokens must be in [0,4]")
    if np.any(y[:, 1] < 0):
        raise ValueError("Generated symbol IDs must be nonnegative")
    return y.astype(np.int32, copy=True)


def _matching_context_index(batch: Batch, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sym = np.asarray(batch.context_symbols)
    match_a = symbols == sym[:, 0]
    match_b = symbols == sym[:, 1]
    valid = match_a | match_b
    index = np.where(match_a, 0, 1)
    return valid, index


def oracle_query_labels(batch: Batch) -> np.ndarray:
    _, pos = _matching_context_index(batch, np.asarray(batch.query_symbols))
    return np.asarray(batch.context_labels)[np.arange(len(pos)), pos]


def continuation_metrics(batch: Batch, targets: np.ndarray) -> dict:
    y = _coerce_targets(batch, targets)
    n = len(y)
    valid, pos = _matching_context_index(batch, y[:, 1])
    ctx_val = np.asarray(batch.context_labels)[np.arange(n), pos]
    consistent = valid & (y[:, 2] == ctx_val)
    first_ok = y[:, 0] == oracle_query_labels(batch)
    return {
        "count": n,
        "first_value_oracle_accuracy": float(first_ok.mean()),
        "next_symbol_context_validity": float(valid.mean()),
        "second_value_correct_given_valid": float(consistent.sum()/max(1, valid.sum())),
        "second_value_context_consistency_total": float(consistent.mean()),
        "valid_next_symbol_count": int(valid.sum()),
        "invalid_next_symbol_count": int((~valid).sum()),
        "exact_valid_continuation_fraction": float((first_ok & consistent).mean()),
    }


def repair_continuations(
    batch: Batch, targets: np.ndarray, mode: Mode,
    *, seed: int = 42
) -> RepairResult:
    """Return a copied, repaired target array and per-field provenance.

    * raw: no changes.
    * first_label_only: correct the first predicted label.
    * valid_second_label_only: correct y2 only where q2 is in the context.
    * label_where_defined: correct first label, and y2 only where valid.
    * joint_repair: also replace invalid q2 by a randomly selected context
      symbol and then correct y2 using the context. This changes the
      *joint distribution* of q2 and y2; it is not a pure causal manipulation
      of q2 alone. No synthetic symbols are fed into later input prefixes by
      this operator.

    The RNG is seeded and independent of any model training RNG.
    """
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    source = _coerce_targets(batch, targets)
    out = source.copy()
    n = len(out)
    labels = np.asarray(batch.context_labels)
    symbols = np.asarray(batch.context_symbols)
    if mode in ("first_label_only", "label_where_defined", "joint_repair"):
        out[:, 0] = oracle_query_labels(batch)
    valid, pos = _matching_context_index(batch, source[:, 1])
    if mode in ("valid_second_label_only", "label_where_defined"):
        repaired = out[:, 2].copy()
        correct = labels[np.arange(n), pos]
        repaired[valid] = correct[valid]
        out[:, 2] = repaired
    if mode == "joint_repair":
        rng = np.random.default_rng(seed)
        choose = rng.integers(0, 2, size=n)
        replacement = symbols[np.arange(n), choose]
        out[:, 1] = np.where(valid, source[:, 1], replacement)
        _, new_pos = _matching_context_index(batch, out[:, 1])
        out[:, 2] = labels[np.arange(n), new_pos]
    return RepairResult(
        mode=mode,
        targets=out,
        changed_mask=out != source,
        original_metrics=continuation_metrics(batch, source),
        repaired_metrics=continuation_metrics(batch, out),
        seed=int(seed),
    )
