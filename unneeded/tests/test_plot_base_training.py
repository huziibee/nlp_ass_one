import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import h5py
import numpy as np
from plot_base_training import summarize, DEVELOPMENT_EVALS

class PlotTrainingTests(unittest.TestCase):
    def test_toy_author_log_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"log.h5"
            out=Path(folder)/"plots"
            with h5py.File(p,"w") as h:
                h.create_dataset("train_iter",data=np.arange(8,81,8))
                h.create_dataset("train_loss",data=np.linspace(1.9,1.0,10))
                h.create_dataset("eval_iter",data=np.array([0,40,80]))
                for key in DEVELOPMENT_EVALS:
                    group=h.create_group(key)
                    group.create_dataset("acc",data=np.asarray([
                        [False]*8,[False]*4+[True]*4,[True]*8],dtype=np.bool_))
                    group.create_dataset("loss",data=np.stack([
                        np.repeat(1.8,8),np.repeat(1.4,8),np.repeat(1.0,8)]))
            result=summarize(p,out)
            self.assertEqual(result["train_sequences_completed"],80)
            self.assertEqual(result["updates_logged"],10)
            self.assertFalse(result["model_final_test_evaluated"])
            self.assertEqual(len(list(out.glob("*.png"))),3)
            self.assertTrue((out/"base_training_summary.json").is_file())

if __name__=="__main__":
    unittest.main()
