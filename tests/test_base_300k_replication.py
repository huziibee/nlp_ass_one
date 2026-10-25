"""Tests for Base 300k five-seed G0/G1--G10 verifier (no JAX needed)."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.base_300k_replication import (
    ARMS, BATCH_SIZE, CHECKPOINT, DATA_SEEDS, EPISODES, INIT_SEEDS,
    check_g0, check_chain, protocol,
)
from scripts.aggregate_base_300k import aggregate


class Base300kReplicationTests(unittest.TestCase):
    def populate(self, root: Path) -> None:
        for i in range(5):
            g0 = root / "g0" / f"base300k-g0-{i}"
            config = g0 / "safe_train_val_smoke"
            (config / "checkpoints").mkdir(parents=True)
            (g0 / "independent_protocol.json").write_text(json.dumps(protocol(i)))
            (config / "config.json").write_text(json.dumps({
                "train_iters": EPISODES,
                "train_bs": BATCH_SIZE,
                "class_split": [50, 1473, 100],
                "exemplar_split": [1, 4, 0],
                "fs_relabel_split": [8, 2, 0],
            }))
            (config / "checkpoints" / CHECKPOINT).write_bytes(b"testfixture" * 110)
            for arm in ARMS:
                folder = root / "arms" / f"base300k-arm-{i}-{arm}"
                generations = []
                for gen in range(1, 11):
                    p = folder / arm / f"g{gen:02d}"
                    p.mkdir(parents=True)
                    (p / "model.eqx").write_bytes(b"testfixture" * 110)
                    score = 0.80 if arm == "clean_only" else 0.55
                    val = {"val_relabel": {"accuracy": score}}
                    (p / "metrics.json").write_text(json.dumps({
                        "generation": gen,
                        "optimizer_exposures": EPISODES,
                        "data_seed": DATA_SEEDS[i],
                        "initialisation_seed": DATA_SEEDS[i] + 17*gen,
                        "class_split": [50, 1473, 100],
                        "final_test_evaluated": False,
                        "whole_prefix_generated": False,
                        "history": [{"validation": val}],
                    }))
                    generations.append({
                        "generation": gen,
                        "last_validation": val,
                    })
                (folder / "summary.json").write_text(json.dumps({
                    "arms": {arm: generations},
                    "final_test_evaluated": False,
                }))

    def test_protocol_has_five_independent_seeds_and_300k_budget(self):
        self.assertEqual(len(set(INIT_SEEDS)),5)
        self.assertEqual(len(set(DATA_SEEDS)),5)
        self.assertEqual(EPISODES,300_000)
        self.assertEqual(len(ARMS),2)
        self.assertEqual(protocol(0)["G0_exposures"],300_000)
        self.assertFalse(protocol(0)["final_test_evaluated"])

    def test_all_105_checkpoint_artifacts_and_results_export(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.populate(root)
            outcome = aggregate(root,root/"report")
            self.assertEqual(outcome["base_new_G0_models"],5)
            self.assertEqual(outcome["base_new_G1_G10_models"],100)
            self.assertEqual(outcome["base"]["n_independent_model_seeds"],5)
            self.assertAlmostEqual(outcome["base"]["mean_clean_minus_recursive"], .25)
            self.assertEqual(outcome["extended_300k_reused_original_seed_pairs"],5)
            with (root/"report/base_300k_checkpoint_manifest.csv").open(newline="") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),105)
            self.assertTrue(all(len(v["sha256"])==64 for v in rows))
            self.assertTrue((root/"report/matched_300k_results.md").is_file())

    def test_g0_rejects_wrong_exposure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.populate(root)
            p = root/"g0/base300k-g0-0/safe_train_val_smoke/config.json"
            cfg=json.loads(p.read_text())
            cfg["train_iters"]=200000
            p.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                check_g0(root/"g0/base300k-g0-0",0)

    def test_chain_rejects_wrong_seed_and_final_test(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            self.populate(root)
            folder=root/"arms/base300k-arm-1-clean_only"
            p=folder/"clean_only/g04/metrics.json"
            obj=json.loads(p.read_text())
            obj["data_seed"]=100
            p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):
                check_chain(folder,"clean_only",1)
            obj["data_seed"]=DATA_SEEDS[1]
            obj["final_test_evaluated"]=True
            p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):
                check_chain(folder,"clean_only",1)

    def test_chain_rejects_missing_generation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            self.populate(root)
            folder=root/"arms/base300k-arm-0-recursive_replacement"
            (folder/"recursive_replacement/g10/model.eqx").unlink()
            with self.assertRaises(FileNotFoundError):
                check_chain(folder,"recursive_replacement",0)

    def test_invalid_seed_index(self):
        with self.assertRaises(ValueError):
            protocol(5)


if __name__=="__main__":
    unittest.main()
