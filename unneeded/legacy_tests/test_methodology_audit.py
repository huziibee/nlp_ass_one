from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_methodology import (
    MATCHED, CUMULATIVE, FIVE_SEED, REPO, audit, write_report,
)


class MethodologyEvidenceTests(unittest.TestCase):
    def load_sample(self, root: Path):
        for p in (MATCHED, CUMULATIVE, FIVE_SEED):
            dest = root / p
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((REPO / p).read_bytes())

    def mutate(self, root: Path, relative: Path, update):
        path = root / relative
        original = json.loads(path.read_text())
        update(original)
        path.write_text(json.dumps(original), encoding="utf-8")

    def test_original_control_evidence_is_complete(self):
        report = audit()
        self.assertEqual(report["single_seed_base_300k"]["n_model_seeds"], 1)
        self.assertEqual(report["single_seed_accumulating_corpus"]["generation_count"], 10)
        self.assertEqual(report["primary_study"]["n_model_seeds"], 5)
        self.assertFalse(report["primary_study"]["final_test_retained_independent"])
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            write_report(report, out)
            for suffix in ("json", "md", "csv"):
                self.assertGreater((out / f"METHOD_CONTROLS.{suffix}").stat().st_size, 60)
            self.assertIn("single-seed", (out / "METHOD_CONTROLS.md").read_text())

    def test_rejects_wrong_matched_base_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.load_sample(root)
            self.mutate(root, MATCHED, lambda o: o.__setitem__("g0_exposures", 1000000))
            with self.assertRaises(ValueError):
                audit(root)

    def test_rejects_missing_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.load_sample(root)
            self.mutate(root, MATCHED, lambda o: o["arms"]["clean_only"].pop())
            with self.assertRaises(ValueError):
                audit(root)

    def test_rejects_invalid_corpus_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.load_sample(root)
            self.mutate(root, CUMULATIVE, lambda o: o["arms"]["cumulative_recursive"][3].__setitem__("stored_prior_synthetic_blocks", 0))
            with self.assertRaises(ValueError):
                audit(root)

    def test_rejects_false_claim_of_independent_final_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.load_sample(root)
            self.mutate(root, FIVE_SEED, lambda o: o.__setitem__("independent_test_set_for_G10", True))
            with self.assertRaises(ValueError):
                audit(root)


if __name__ == "__main__":
    unittest.main()
