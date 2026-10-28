from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.fresh_five_seed import (
    ARMS, BASE_INITIALIZATION_SEEDS, BASE_DATA_SEEDS,
    EXTENDED_SEEDS, METRICS, plan, aggregate,
)


class FreshFiveSeedTests(unittest.TestCase):
    def test_plan_has_five_distinct_initializations_per_task(self):
        p = plan()
        self.assertEqual(len(set(BASE_INITIALIZATION_SEEDS)), 5)
        self.assertEqual(len(set(EXTENDED_SEEDS)), 5)
        self.assertEqual(len(set(BASE_DATA_SEEDS)), 5)
        self.assertEqual(p["expected_G1_G10_arms"], 20)
        self.assertFalse(p["historical_checkpoints_used_as_initialization"])
        self.assertFalse(p["held_out_final_test_allowed"])

    def fixture(self, root: Path, extended_exposures: int = 300_000):
        for task in ("base", "extended"):
            for i in range(5):
                g0 = root / f"{task}_g0" / f"fresh-v1-{task}-g0-{i}"
                g0.mkdir(parents=True)
                if task == "base":
                    (g0 / "independent_protocol.json").write_text(json.dumps({
                        "independent_seed": BASE_INITIALIZATION_SEEDS[i],
                        "training_sequences": 1000000,
                        "final_test_evaluated": False,
                    }))
                    loc = g0 / "safe_train_val_smoke"
                    (loc / "checkpoints").mkdir(parents=True)
                    (loc / "config.json").write_text(json.dumps({
                        "train_iters": 1000000, "class_split": [50, 1473, 100]
                    }))
                    (loc / "checkpoints" / "00001000000.eqx").write_bytes(b"x" * 1100)
                else:
                    (g0 / "metrics.json").write_text(json.dumps({
                        "seed": EXTENDED_SEEDS[i], "steps": extended_exposures // 32,
                        "batch_size": 32, "class_split": [50, 1473, 100],
                        "final_test_evaluated": False,
                    }))
                    (g0 / "extended_g0_model.eqx").write_bytes(b"x" * 1100)
                for arm in ARMS:
                    folder = root / f"{task}_arms" / f"fresh-v1-{task}-arm-{i}-{arm}"
                    generations = []
                    for g in range(1, 11):
                        output = folder / arm / f"g{g:02d}"
                        output.mkdir(parents=True)
                        (output / "model.eqx").write_bytes(b"x" * 1100)
                        validation = {
                            role: {metric[2]: 0.8 - 0.02 * (arm == "recursive_replacement")
                                    for metric in METRICS[task] if metric[1] == role}
                            for role in ("val_relabel",)
                        }
                        evidence = {
                            "generation": g,
                            "data_seed": BASE_DATA_SEEDS[i] if task == "base" else EXTENDED_SEEDS[i],
                            "final_test_evaluated": False,
                            "history": [{"validation": validation}],
                        }
                        if task == "base":
                            evidence.update({
                                "optimizer_exposures": 1000000,
                                "initialisation_seed": BASE_DATA_SEEDS[i] + 17 * g,
                                "class_split": [50, 1473, 100],
                                "whole_prefix_generated": False,
                            })
                        else:
                            evidence.update({
                                "training_exposures": extended_exposures,
                                "init_seed": EXTENDED_SEEDS[i],
                                "training_class_split": [50, 1473, 100],
                                "recursively_generated_input_prefix": False,
                                "condition": arm,
                            })
                        (output / "metrics.json").write_text(json.dumps(evidence))
                        generations.append({"generation": g})
                    (folder / "summary.json").write_text(json.dumps({
                        "final_test_evaluated": False, "arms": {arm: generations},
                        "all_seed_replication_included_irrespective_of_g0_accuracy": True,
                    }))

    def test_complete_fresh_chains_aggregate_and_save(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            x = aggregate(root, root / "report")
            self.assertEqual(x["rows"], 20)
            self.assertEqual(x["checkpoint_manifest_count"], 210)
            self.assertTrue((root / "report" / "fresh_five_seed_checkpoint_manifest.csv").is_file())
            self.assertFalse(x["final_test_accessed"])
            self.assertEqual(len(x["paired_effects"]["extended"]), 3)
            self.assertEqual(len(x["paired_effects"]["base"]), 1)
            self.assertTrue((root / "report" / "fresh_five_seed_rows.csv").is_file())

    def test_matched_one_million_baseline_and_extended(self):
        p = plan(1_000_000)
        self.assertEqual(p["G0_training_examples"]["base"], 1_000_000)
        self.assertEqual(p["G0_training_examples"]["extended"], 1_000_000)
        self.assertTrue(p["comparison_budget_matched_in_examples"])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root, extended_exposures=1_000_000)
            report = aggregate(root, root / "report",
                               extended_exposures=1_000_000,
                               base_reference_run_id=40194253117)
            self.assertEqual(report["checkpoint_manifest_count"], 210)
            self.assertTrue(report["base_reused_from_independent_fresh_run"])
            self.assertEqual(report["base_reference_run_id"], 40194253117)
            self.assertEqual(report["rows"], 20)

    def test_one_million_requires_independent_base_run_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root, extended_exposures=1_000_000)
            with self.assertRaisesRegex(ValueError, "Base reference run ID"):
                aggregate(root, root / "report", extended_exposures=1_000_000)

    def test_one_million_rejects_300k_extended_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root, extended_exposures=1_000_000)
            f = root / "extended_g0/fresh-v1-extended-g0-2/metrics.json"
            m = json.loads(f.read_text())
            m["steps"] = 9375
            f.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                aggregate(root, root / "report", extended_exposures=1_000_000,
                          base_reference_run_id=40194253117)

    def test_missing_checkpoint_is_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            bad = root / "extended_arms/fresh-v1-extended-arm-2-clean_only/clean_only/g06/model.eqx"
            bad.unlink()
            with self.assertRaises(FileNotFoundError):
                aggregate(root, root / "report")

    def test_wrong_seed_is_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            bad = root / "base_arms/fresh-v1-base-arm-0-clean_only/clean_only/g10/metrics.json"
            meta = json.loads(bad.read_text())
            meta["data_seed"] = 999
            bad.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                aggregate(root, root / "report")

    def test_final_test_access_is_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            bad = root / "extended_g0/fresh-v1-extended-g0-3/metrics.json"
            meta = json.loads(bad.read_text())
            meta["final_test_evaluated"] = True
            bad.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                aggregate(root, root / "report")


if __name__ == "__main__":
    unittest.main()
