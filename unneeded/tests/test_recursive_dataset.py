import unittest

import numpy as np

from scripts.reference_split import get_class_ids
from src.target_repairs import Batch, oracle_query_labels
from src.recursive_dataset import clean_targets, compose_targets, CONDITIONS


class RecursiveTargetTests(unittest.TestCase):
    def setUp(self):
        self.batch = Batch(
            context_symbols=np.array([[i % 25, (i % 25) + 25] for i in range(20)]),
            context_labels=np.array([[0, 4] if i % 2 else [1, 3] for i in range(20)]),
            query_symbols=np.array([i % 25 if i % 2 else (i % 25) + 25 for i in range(20)]),
        )

    def test_source_of_truth_is_original_paper_class_split(self):
        self.assertEqual(len(get_class_ids("train")), 50)
        self.assertEqual(len(get_class_ids("val")), 1473)
        self.assertEqual(len(get_class_ids("test")), 100)
        self.assertEqual(set(get_class_ids("train")) & set(get_class_ids("test")), set())

    def test_clean_base_follows_contextual_query_oracle(self):
        clean = clean_targets(self.batch, "base", seed=11)
        self.assertEqual(clean.shape, (20, 1))
        np.testing.assert_array_equal(clean[:, 0], oracle_query_labels(self.batch))

    def test_clean_extended_follows_three_token_oracle(self):
        x = clean_targets(self.batch, "extended", seed=12)
        self.assertEqual(x.shape, (20, 3))
        np.testing.assert_array_equal(x[:, 0], oracle_query_labels(self.batch))
        for i in range(20):
            self.assertIn(x[i, 1], self.batch.context_symbols[i])
            j = int(x[i, 1] == self.batch.context_symbols[i, 1])
            self.assertEqual(x[i, 2], self.batch.context_labels[i, j])
        np.testing.assert_array_equal(x, clean_targets(self.batch, "extended", seed=12))

    def test_clean_and_recursive_replacement(self):
        clean = clean_targets(self.batch, "base", seed=10)
        synth = np.full_like(clean, 4)
        a = compose_targets(self.batch, clean, None, task="base",
                            condition="clean_only", generation=0, seed=6)
        b = compose_targets(self.batch, clean, synth, task="base",
                            condition="recursive_replacement", generation=3,
                            generated_by=2, seed=6)
        np.testing.assert_array_equal(a.targets, clean)
        np.testing.assert_array_equal(b.targets, synth)
        self.assertEqual(a.metadata["synthetic_rows"], 0)
        self.assertEqual(b.metadata["clean_rows"], 0)
        self.assertFalse(b.metadata["uses_sealed_final_test"])

    def test_anchored_half_real_reproducible_and_no_mutations(self):
        clean = clean_targets(self.batch, "extended", seed=10)
        synth = clean.copy()
        synth[:, 0] = (synth[:, 0] + 1) % 5
        original = synth.copy()
        kwargs = dict(task="extended", condition="real_anchor_half",
                      generation=2, generated_by=1, seed=2026)
        a = compose_targets(self.batch, clean, synth, **kwargs)
        b = compose_targets(self.batch, clean, synth, **kwargs)
        self.assertEqual(a.metadata["clean_rows"], 10)
        self.assertEqual(a.metadata["synthetic_rows"], 10)
        np.testing.assert_array_equal(a.targets, b.targets)
        np.testing.assert_array_equal(a.is_real, b.is_real)
        np.testing.assert_array_equal(a.targets[a.is_real], clean[a.is_real])
        np.testing.assert_array_equal(a.targets[~a.is_real], synth[~a.is_real])
        np.testing.assert_array_equal(synth, original)
        self.assertFalse(np.shares_memory(a.targets, synth))

    def test_fixed_g0_control_provenance(self):
        clean = clean_targets(self.batch, "base", seed=1)
        fixed = np.full_like(clean, 2)
        out = compose_targets(self.batch, clean, fixed, task="base",
                              condition="fixed_g0_half", generation=4,
                              generated_by=0, seed=2)
        self.assertEqual(out.generated_by, 0)
        self.assertEqual(out.metadata["clean_rows"], 10)

    def test_no_test_or_val_classes_can_be_used_to_train(self):
        bad = Batch(
            np.array([[1523, 1]]),
            np.array([[0, 4]]),
            np.array([1]),
        )
        with self.assertRaises(PermissionError):
            clean_targets(bad, "base", seed=1)
        with self.assertRaises(PermissionError):
            compose_targets(bad, np.array([[4]]), None, task="base",
                            condition="clean_only", generation=0, seed=1)

    def test_validation_label_pairs_are_rejected_for_training(self):
        invalid = Batch(
            np.array([[0, 1]]),
            np.array([[2, 3]]),                                    
            np.array([0]),
        )
        with self.assertRaises(PermissionError):
            clean_targets(invalid, "extended", seed=1)

    def test_reject_bad_shapes_values_and_generator_provenance(self):
        clean = clean_targets(self.batch, "extended", seed=1)
        wrong_vocab = clean.copy()
        wrong_vocab[0, 1] = 1523
        with self.assertRaises(ValueError):
            compose_targets(self.batch, clean, wrong_vocab, task="extended",
                            condition="recursive_replacement", generation=1,
                            generated_by=0, seed=1)
        with self.assertRaises(ValueError):
            compose_targets(self.batch, clean, np.ones((20, 1), dtype=int),
                            task="extended", condition="recursive_replacement",
                            generation=1, generated_by=0, seed=1)
        with self.assertRaises(ValueError):
            compose_targets(self.batch, clean, clean, task="extended",
                            condition="real_anchor_half", generation=2,
                            generated_by=0, seed=1)
        with self.assertRaises(ValueError):
            compose_targets(self.batch, clean, clean, task="extended",
                            condition="fixed_g0_half", generation=2,
                            generated_by=1, seed=1)
        with self.assertRaises(ValueError):
            compose_targets(self.batch, clean, clean, task="extended",
                            condition="fixed_g0_half", generation=0,
                            generated_by=0, seed=1)
        self.assertEqual(len(CONDITIONS), 4)


if __name__ == "__main__":
    unittest.main()
