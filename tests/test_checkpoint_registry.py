"""Offline tests of the authenticated, non-overwriting checkpoint utility."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import checkpoint_registry as ck


class CheckpointRegistryTests(unittest.TestCase):
    def fixture(self, root: Path):
        base = root / ck.ASSETS["base"]["root"]
        (base / "safe_train_val_smoke/checkpoints").mkdir(parents=True)
        (base / "safe_train_val_smoke/config.json").write_text(json.dumps({
            "train_iters": 1000000,
            "class_split": [50, 1473, 100],
            "exemplar_split": [1, 4, 0],
            "fs_relabel_split": [8, 2, 0],
            "model_output_classes": 5,
        }), encoding="utf-8")
        (base / "safe_train_val_smoke/checkpoints/00001000000.eqx").write_bytes(b"model-base" * 100)
        ext = root / ck.ASSETS["extended"]["root"]
        ext.mkdir(parents=True)
        (ext / "metrics.json").write_text(json.dumps({
            "steps": 9375, "batch_size": 32,
            "final_test_evaluated": False,
            "class_split": [50, 1473, 100]
        }), encoding="utf-8")
        (ext / "extended_g0_model.eqx").write_bytes(b"model-extended" * 100)

    def test_verified_checkpoint_sha_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            out = ck.verify(root)
            self.assertEqual(set(out["assets"]), {"base", "extended"})
            self.assertFalse(out["independent_published_checksum_validation"])
            self.assertFalse(out["original_final_test_accessed_by_this_tool"])
            self.assertEqual(len(out["assets"]["base"]["files"]["safe_train_val_smoke/config.json"]["sha256"]), 64)

    def test_rejects_final_test_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / ck.ASSETS["extended"]["root"] / "metrics.json"
            val = json.loads(path.read_text())
            val["final_test_evaluated"] = True
            path.write_text(json.dumps(val))
            with self.assertRaises(ValueError):
                ck.verify(root)

    def test_rejects_invalid_base_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / ck.ASSETS["base"]["root"] / "safe_train_val_smoke/config.json"
            val = json.loads(path.read_text())
            val["class_split"] = [50, 1472, 101]
            path.write_text(json.dumps(val))
            with self.assertRaises(ValueError):
                ck.verify(root)

    def test_restore_never_overwrites_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            with mock.patch.object(ck.shutil, "which", return_value="/fake/gh"):
                with mock.patch.object(ck.subprocess, "run") as run:
                    with self.assertRaises(FileExistsError):
                        ck.restore(root, ("base",))
                    run.assert_not_called()

    def test_restore_requires_authenticated_cli_program(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch.object(ck.shutil, "which", return_value=None):
                with self.assertRaises(RuntimeError):
                    ck.restore(root, ("base",))


if __name__ == "__main__":
    unittest.main()
