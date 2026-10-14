"""Generate and validate Singh-style TWO-way, one-shot symbol/label episodes.

No neural model is trained. The default CLI never samples final test classes.
For scientific fidelity, numeric symbols select stored 512-dimensional image
features; labels are random per episode, not permanent character identities.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from reference_split import get_class_ids, get_exemplar_ids, get_label_pairs


MODES = {
    "train": ("train", "train", "train"),
    "val_new_drawing": ("train", "val", "train"),
    "val_new_label_pair": ("train", "train", "val"),
    "val_new_class": ("val", "train", "train"),
    "final_test": ("test", "train", "train"),
}


def generate_episodes(mode: str, n: int, seed: int, *, allow_final: bool = False) -> dict:
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if mode == "final_test" and not allow_final:
        raise PermissionError(
            "Final test is protected. Do not use it for design/tuning. "
            "Only pass allow_final=True when you have frozen your protocol."
        )
    cls_split, exemplar_split, pairs_split = MODES[mode]
    classes = np.asarray(get_class_ids(cls_split), dtype=np.int32)
    exemplars = np.asarray(get_exemplar_ids(exemplar_split), dtype=np.int32)
    label_pairs = np.asarray(get_label_pairs(pairs_split), dtype=np.int32)
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    a_idx = rng.integers(0, len(classes), size=n)
    b_idx = rng.integers(0, len(classes) - 1, size=n)
    b_idx += b_idx >= a_idx
    a = classes[a_idx]
    b = classes[b_idx]
    pair = label_pairs[rng.integers(0, len(label_pairs), size=n)]
    swapped = rng.integers(0, 2, size=n).astype(bool)
    labels_a = np.where(swapped, pair[:, 1], pair[:, 0])
    labels_b = np.where(swapped, pair[:, 0], pair[:, 1])
    # Sample independently from the allowed exemplar subset, as in a
    # generalized episodic sampler. With train only exemplar 0 is possible.
    exemplars_a = rng.choice(exemplars, size=n)
    exemplars_b = rng.choice(exemplars, size=n)
    query_exemplars = rng.choice(exemplars, size=n)
    query_a = rng.integers(0, 2, size=n).astype(bool)
    query_cls = np.where(query_a, a, b)
    target_label = np.where(query_a, labels_a, labels_b)
    out = {
        "symbol_ids": np.stack([a, b, query_cls], axis=1),
        "context_labels": np.stack([labels_a, labels_b], axis=1),
        "exemplar_ids": np.stack([exemplars_a, exemplars_b, query_exemplars], axis=1),
        "targets": target_label.astype(np.int32),
    }
    validate_episodes(out, mode)
    return out


def validate_episodes(x: dict, mode: str) -> None:
    a, b, q = x["symbol_ids"].T
    v_a, v_b = x["context_labels"].T
    t = x["targets"]
    assert len(a) > 0
    assert np.all(a != b), "support classes must differ"
    assert np.all(v_a != v_b), "two labels must differ within episode"
    assert np.all((q == a) | (q == b)), "query must occur in context"
    assert np.all(t == np.where(q == a, v_a, v_b)), "incorrect oracle label"
    class_subset, ex_subset, pair_subset = MODES[mode]
    classes = set(get_class_ids(class_subset))
    ex = set(get_exemplar_ids(ex_subset))
    labels = set(get_label_pairs(pair_subset))
    assert set(np.unique(x["symbol_ids"])) <= classes
    assert set(np.unique(x["exemplar_ids"])) <= ex
    assert all(tuple(sorted(map(int, p))) in labels for p in x["context_labels"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=[k for k in MODES if k != "final_test"], default="train")
    p.add_argument("--num", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    samples = generate_episodes(args.mode, args.num, args.seed)
    example = []
    for s, l, e, y in zip(samples["symbol_ids"], samples["context_labels"],
                          samples["exemplar_ids"], samples["targets"]):
        example.append({
            "context": [[int(s[0]), int(l[0]), int(e[0])],
                        [int(s[1]), int(l[1]), int(e[1])]],
            "query": [int(s[2]), int(e[2])],
            "oracle_value": int(y),
        })
    print(json.dumps({
        "mode": args.mode,
        "seed": args.seed,
        "n": args.num,
        "label_mapping_per_episode": "randomly permuted contextual pair",
        "examples": example,
        "final_test_evaluated": False,
    }, indent=2))


if __name__ == "__main__":
    main()
