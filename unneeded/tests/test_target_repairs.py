import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.target_repairs import Batch, MODES, continuation_metrics, repair_continuations


class TargetRepairTests(unittest.TestCase):
    def setUp(self):
        self.batch = Batch(
            context_symbols=np.array([[1, 2], [1, 2], [3, 4], [8, 9]], dtype=np.int32),
            context_labels=np.array([[0, 3], [0, 3], [1, 2], [2, 4]], dtype=np.int32),
            query_symbols=np.array([1, 2, 3, 8], dtype=np.int32),
        )
                                                             
                                             
        self.generated = np.array([
            [0, 2, 3],
            [0, 17, 4],
            [1, 3, 4],
            [0, 20, 0],
        ], dtype=np.int32)

    def test_modes_exactly_declared(self):
        self.assertEqual(len(MODES), 6)

    def test_never_mutates_inputs(self):
        before = self.generated.copy()
        for mode in MODES:
            r = repair_continuations(self.batch, self.generated, mode, seed=6)
            self.assertTrue(np.array_equal(self.generated, before))
            self.assertFalse(np.shares_memory(r.targets, self.generated))
            self.assertEqual(r.changed_mask.shape, self.generated.shape)

    def test_raw_is_identity(self):
        r = repair_continuations(self.batch, self.generated, "raw")
        np.testing.assert_array_equal(r.targets, self.generated)
        self.assertEqual(int(r.changed_mask.sum()), 0)

    def test_first_label_only_leaves_second_fields_untouched(self):
        r = repair_continuations(self.batch, self.generated, "first_label_only")
        np.testing.assert_array_equal(r.targets[:, 1:], self.generated[:, 1:])
        np.testing.assert_array_equal(r.targets[:, 0], [0, 3, 1, 2])

    def test_second_label_only_does_not_invent_label_for_invalid_symbol(self):
        r = repair_continuations(self.batch, self.generated, "valid_second_label_only")
        np.testing.assert_array_equal(r.targets[:, :2], self.generated[:, :2])
        np.testing.assert_array_equal(r.targets[:, 2], [3, 4, 1, 0])
                                                                 
        np.testing.assert_array_equal(r.targets[[1, 3]], self.generated[[1, 3]])

    def test_label_where_defined_retains_invalid_next_symbols(self):
        r = repair_continuations(self.batch, self.generated, "label_where_defined")
        np.testing.assert_array_equal(r.targets[:, 1], self.generated[:, 1])
        self.assertLess(r.repaired_metrics["next_symbol_context_validity"], 1.0)
        self.assertEqual(r.repaired_metrics["first_value_oracle_accuracy"], 1.0)

    def test_joint_repair_recovers_fully_consistent_contextual_continuations(self):
        r = repair_continuations(self.batch, self.generated, "joint_repair", seed=10)
        self.assertEqual(r.repaired_metrics["next_symbol_context_validity"], 1.0)
        self.assertEqual(r.repaired_metrics["second_value_context_consistency_total"], 1.0)
        self.assertEqual(r.repaired_metrics["first_value_oracle_accuracy"], 1.0)
                                                               
        np.testing.assert_array_equal(r.targets[[0, 2], 1], self.generated[[0, 2], 1])
                                                                          
        for row in range(4):
            self.assertIn(r.targets[row, 1], self.batch.context_symbols[row])

    def test_structure_repair_preserves_first_value(self):
        r = repair_continuations(self.batch, self.generated,
                                 "structure_second_only", seed=10)
        np.testing.assert_array_equal(r.targets[:, 0], self.generated[:, 0])
        self.assertEqual(r.repaired_metrics["next_symbol_context_validity"], 1.0)
        self.assertEqual(r.repaired_metrics["second_value_context_consistency_total"], 1.0)
        self.assertLess(r.repaired_metrics["first_value_oracle_accuracy"], 1.0)

    def test_joint_repair_is_seed_deterministic(self):
        a = repair_continuations(self.batch, self.generated, "joint_repair", seed=42)
        b = repair_continuations(self.batch, self.generated, "joint_repair", seed=42)
        np.testing.assert_array_equal(a.targets, b.targets)

    def test_invalid_query_or_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            Batch(self.batch.context_symbols, self.batch.context_labels,
                  np.array([1, 2, 3, 100]))
        with self.assertRaises(ValueError):
            Batch(self.batch.context_symbols, np.ones((4, 2), dtype=int),
                  self.batch.query_symbols)
        with self.assertRaises(ValueError):
            repair_continuations(self.batch, np.zeros((4, 4), dtype=int), "raw")
        with self.assertRaises(ValueError):
            repair_continuations(self.batch, self.generated, "unsupported")

    def test_no_hidden_ground_truth_for_invalid_q2(self):
        metrics = continuation_metrics(self.batch, self.generated)
        self.assertEqual(metrics["invalid_next_symbol_count"], 2)
        self.assertEqual(metrics["valid_next_symbol_count"], 2)
        self.assertEqual(metrics["second_value_context_consistency_total"], .25)


if __name__ == "__main__":
    unittest.main()
