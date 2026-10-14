"""Fast, CPU-only guardrail tests. No test-set scoring."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
from reference_split import (
    TRAIN_PAIRS, VALIDATION_PAIRS, ALL_PAIRS,
    build_manifest, validate_reference_split, get_class_ids,
    get_exemplar_ids, original_run_flags,
)
from episode_probe import MODES, generate_episodes, validate_episodes


class ReferenceDataTests(unittest.TestCase):
    def test_original_script_matches_frozen_expected_flags(self):
        flags = original_run_flags()
        self.assertEqual(flags["class_split"], (50, 1473, 100))
        self.assertEqual(flags["exemplar_split"], (1, 4, 0))
        self.assertEqual(flags["fs_relabel_split"], (8, 2, 0))
        self.assertEqual(len(flags["pe_names"]), 4)

    def test_disjoint_class_exemplar_and_label_pair_splits(self):
        validate_reference_split()
        self.assertEqual(len(get_class_ids("train")), 50)
        self.assertEqual(len(get_class_ids("val")), 1473)
        self.assertEqual(len(get_class_ids("test")), 100)
        self.assertFalse(set(get_class_ids("train")) & set(get_class_ids("test")))
        self.assertEqual(get_exemplar_ids("val"), (1, 2, 3, 4))
        self.assertEqual(set(TRAIN_PAIRS) | set(VALIDATION_PAIRS), set(ALL_PAIRS))

    def test_manifest_preserves_original_test_evaluator_warning(self):
        m = build_manifest()
        self.assertFalse(m["test_performance_evaluated"])
        self.assertTrue(any(x["class_subset"] == "test" for x in
                            m["published_main_run_evaluators"]))
        self.assertTrue("NOT used" in m["important_original_detail"])

    def test_episode_invariants_for_every_development_mode(self):
        for i, mode in enumerate(MODES):
            if mode == "final_test":
                continue
            examples = generate_episodes(mode, 160, seed=20 + i)
            validate_episodes(examples, mode)
            self.assertEqual(examples["symbol_ids"].shape, (160, 3))
            self.assertEqual(examples["context_labels"].shape, (160, 2))
            self.assertEqual(examples["targets"].shape, (160,))
            self.assertTrue(np.isin(examples["context_labels"], np.arange(5)).all())
            self.assertFalse(np.isin(examples["symbol_ids"], np.arange(1523, 1623)).any())

    def test_seed_determinism_and_contextual_label_swap(self):
        a = generate_episodes("train", 80, seed=123)
        b = generate_episodes("train", 80, seed=123)
        for key in a:
            self.assertTrue(np.array_equal(a[key], b[key]))

        # The identity-to-label assignment is contextual, not global.
        ids = a["symbol_ids"][:, 0]
        assigned = a["context_labels"][:, 0]
        self.assertGreater(len(set(assigned[ids == ids[0]])), 1)

    def test_final_test_guard(self):
        with self.assertRaises(PermissionError):
            generate_episodes("final_test", 1, seed=1)
        with self.assertRaises(ValueError):
            generate_episodes("train", 0, seed=1)


if __name__ == "__main__":
    unittest.main()
