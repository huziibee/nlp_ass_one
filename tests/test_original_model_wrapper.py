"""Pre-run guarantee: only train/validation evaluators are passed to authors' main.py."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paper_base_smoke import source_checked_cmd


class OriginalModelWrapperTests(unittest.TestCase):
    def test_dry_run_config_has_no_test_class_monitor(self):
        cmd = source_checked_cmd(4, Path("/tmp/paper_smoke"), 8)
        self.assertIn("--pe_classes", cmd)
        i = cmd.index("--pe_classes")
        self.assertEqual(cmd[i+1:i+5], ["train","train","train","val"])
        self.assertNotIn("test", cmd[i+1:i+5])
        self.assertEqual(cmd[cmd.index("--class_split")+1:cmd.index("--class_split")+4],
                         ["50","1473","100"])
        self.assertEqual(cmd[cmd.index("--fs_relabel_split")+1:cmd.index("--fs_relabel_split")+4],
                         ["8","2","0"])

    def test_smoke_run_is_intentionally_short(self):
        with self.assertRaises(ValueError):
            source_checked_cmd(0, Path("/tmp/x"))
        with self.assertRaises(ValueError):
            source_checked_cmd(1001, Path("/tmp/x"))


if __name__=="__main__":
    unittest.main()
