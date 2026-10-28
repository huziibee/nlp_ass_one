import json
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_independent_seeds_g10 import aggregate


def write_fixture(base, *, task, seed, arm, primary):
    prefix=(f"original-base-g10-{arm}" if task=="base" else
            f"extended-original-g10-{arm}")
    if not primary:
        prefix=(f"original-base-independent-g10-{505 if seed==35001 else 605}-{arm}"
                if task=="base" else
                f"extended-independent-g10-{seed}-{arm}")
    folder=base/f"{task}_{'primary' if primary else 'replica'}"/prefix
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"summary.json").write_text(json.dumps({
        "final_test_evaluated":False,
        "arms":{arm:[{"generation":i,
                      "last_validation":{"val_relabel":{}}
                      } for i in range(1,11)]},
    }))
    p=folder/arm/"g10"
    p.mkdir(parents=True)
    (p/"model.eqx").write_bytes(b"checkpoint-synthetic-fixture")
    acc=(.95 if arm=="clean_only" else .80)
    if task=="base":
        meta={
            "arm":arm,"generation":10,
            "data_seed":seed,
            "optimizer_exposures":1_000_000,
            "class_split":[50,1473,100],
            "whole_prefix_generated":False,
            "final_test_evaluated":False,
            "history":[{"validation":{
                "val_relabel":{"accuracy":acc},
                "val_classes":{"accuracy":acc-.02},
            }}],
        }
    else:
        meta={
            "condition":arm,"generation":10,
            "data_seed":seed,
            "training_exposures":300_000,
            "training_class_split":[50,1473,100],
            "recursively_generated_input_prefix":False,
            "final_test_evaluated":False,
            "history":[{"validation":{
                "val_relabel":{
                    "first_label_accuracy":acc,
                    "teacher_forced_second_label_accuracy":acc,
                    "free_next_symbol_context_validity":acc,
                    "free_second_label_context_consistency_all":acc,
                    "next_symbol_valid_probability_mass":acc,
                },
                "val_classes":{"first_label_accuracy":acc},
            }}],
        }
    (p/"metrics.json").write_text(json.dumps(meta))
    return p


class ThreeSeedTests(unittest.TestCase):
    def build(self,root):
        for task,seeds in (
            ("base",(34001,35001,36001)),
            ("extended",(6001,5054,6054)),
        ):
            for k,seed in enumerate(seeds):
                for arm in ("clean_only","recursive_replacement"):
                    write_fixture(root,task=task,seed=seed,arm=arm,primary=k==0)

    def test_full_seed_comparison_pairs_all_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.build(root)
            aggregate(root,root/"out")
            result=json.loads((root/"out"/"g10_seed_summary.json").read_text())
            self.assertEqual(result["n_model_seeds_per_task"],3)
            self.assertAlmostEqual(
                result["paired_effects"]["base"]["label_pair_accuracy"]["mean_clean_minus_recursive"],.15)
            self.assertTrue((root/"out"/"g10_seed_level.csv").is_file())

    def test_reject_missing_or_test_contaminated_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.build(root)
            p=root/"base_primary"/"original-base-g10-clean_only"/"clean_only"/"g10"/"metrics.json"
            x=json.loads(p.read_text())
            x["final_test_evaluated"]=True
            p.write_text(json.dumps(x))
            with self.assertRaises(AssertionError):
                aggregate(root,root/"out")


if __name__=="__main__":
    unittest.main()
