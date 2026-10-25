"""The student runner must read committed evidence and never re-evaluate test data."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run


class StudentRunnerTests(unittest.TestCase):
    def test_original_results_are_formatted_without_training(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run.summarize(Path(folder))
            self.assertFalse(result)
            text = (Path(folder) / "RESULTS_SUMMARY.md").read_text()
            self.assertIn("+9.22 pp", text)
            self.assertIn("+19.73 pp", text)
            self.assertIn("+21.17 pp", text)
            self.assertIn("+5.94 logits", text)
            with (Path(folder) / "RESULTS_SUMMARY.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertTrue((Path(folder) / "RESULTS_INDEX.md").is_file())
            manifest = Path(folder) / "RESULTS_FILE_MANIFEST.csv"
            self.assertTrue(manifest.is_file())
            with manifest.open(newline="") as handle:
                evidence_rows = list(csv.DictReader(handle))
            self.assertGreater(len(evidence_rows), 100)
            self.assertEqual(len(evidence_rows[0]["sha256"]), 64)

    def test_refuses_five_seed_data_marked_as_final_test(self):
        original = run.load
        def poisoned(path):
            output = original(path)
            if path == run.G10:
                output["independent_test_set_for_G10"] = True
            return output
        with mock.patch.object(run, "load", side_effect=poisoned):
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    run.summarize(Path(folder))

    def test_input_commands_never_call_sealed_evaluator(self):
        source = Path(run.__file__).read_text(encoding="utf-8")
        self.assertNotIn('run_command([sys.executable, "scripts/evaluate_sealed.py"', source)


if __name__ == "__main__":
    unittest.main()
