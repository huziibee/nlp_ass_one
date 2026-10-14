"""Unit checks of candidate-difficulty math, no test classes or training."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split_diagnostics import expected_k_way_accuracy, analyze


class SplitDiagnosticsTests(unittest.TestCase):
    def test_exact_uniform_subsets(self):
        m = np.asarray([0, 1, 4])
        self.assertAlmostEqual(expected_k_way_accuracy(m, 5, 2),
                               (1 + 3/4 + 0) / 3)
        self.assertAlmostEqual(expected_k_way_accuracy(m, 5, 3),
                               (1 + 0.5 + 0) / 3)
        self.assertAlmostEqual(expected_k_way_accuracy(m, 5, 5), 1/3)
        with self.assertRaises(ValueError):
            expected_k_way_accuracy(m, 5, 6)

    def test_easy_separated_classes(self):
        # Synthetic feature geometry ONLY; no Omniglot final classes used.
        rng = np.random.default_rng(42)
        array = np.zeros((12, 5, 12), dtype=np.float32)
        for i in range(12):
            array[i, :, i] = 1.0
        result, _ = analyze(array, "toy", seed=3)
        self.assertAlmostEqual(result["exact_expected_two_choice_accuracy"], 1)
        self.assertAlmostEqual(result["all_partition_class_top1_accuracy"], 1)
        self.assertEqual(result["num_query_examples"], 48)
        self.assertEqual(result["candidate_curve"]["2"], 1)

    def test_matching_difficulty_monotone(self):
        rng = np.random.default_rng(4)
        array = rng.normal(size=(50, 5, 16)).astype(np.float32)
        result, _ = analyze(array, "toy", seed=3)
        counts = [int(k) for k in result["candidate_curve"]]
        vals = [result["candidate_curve"][str(k)] for k in counts]
        self.assertEqual(counts[0], 2)
        self.assertGreaterEqual(vals[0], vals[-1])
        self.assertTrue(all(a >= b-1e-8 for a,b in zip(vals, vals[1:])))


if __name__ == "__main__":
    unittest.main()
