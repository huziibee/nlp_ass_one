"""Preview experimental target repairs on explicitly ARTIFICIAL toy continuations.

No actual model samples, model fitting or final test metrics are used.
This is a unit-tested contract preview, not a model-collapse result.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.target_repairs import Batch, MODES, repair_continuations

def main():
    batch = Batch(
        context_symbols=np.array([[3, 7], [3, 7], [5, 9], [5, 9], [11, 12]], dtype=np.int32),
        context_labels=np.array([[0, 4], [0, 4], [1, 2], [1, 2], [2, 3]], dtype=np.int32),
        query_symbols=np.array([3, 7, 5, 9, 11], dtype=np.int32),
    )
    synthetic = np.array([
        [0, 7, 4],     # valid and consistent
        [1, 8, 4],     # invalid second symbol
        [1, 5, 4],     # valid second symbol, WRONG second label
        [2, 9, 2],     # valid
        [0, 0, 1],     # both first label and next symbol wrong
    ], dtype=np.int32)
    output = {}
    for mode in MODES:
        r = repair_continuations(batch, synthetic, mode, seed=2026)
        output[mode] = {
            "targets": r.targets.tolist(),
            "changed_fields": r.changed_mask.astype(int).tolist(),
            "metrics": r.repaired_metrics,
        }
    print(json.dumps({
        "warning": "ARTIFICIAL TOY EXAMPLES: not sampled from a neural network",
        "original_targets": synthetic.tolist(),
        "modes": output,
        "model_trained": False,
        "test_evaluated": False,
    }, indent=2))

if __name__ == "__main__":
    main()
