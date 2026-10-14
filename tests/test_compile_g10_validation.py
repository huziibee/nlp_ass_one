"""Synthetic fixture tests for G1-G10 validation aggregation. No test data."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.compile_g10_validation import compile_arm, extract

BASE_VAL={
    "val_relabel":{"accuracy":0.8,"loss":0.4},
    "val_classes":{"accuracy":0.75,"loss":0.5},
}
EXT_VAL={
    "val_relabel":{
        "first_label_accuracy":0.9,
        "teacher_forced_second_label_accuracy":0.85,
        "free_next_symbol_context_validity":0.8,
        "free_second_label_context_consistency_all":0.7,
        "next_symbol_valid_probability_mass":0.83},
    "val_exemplar":{
        "first_label_accuracy":0.7,
        "teacher_forced_second_label_accuracy":0.75},
    "val_classes":{
        "first_label_accuracy":0.67,
        "teacher_forced_second_label_accuracy":0.68},
}

def fixture(root1,root2,task,arm):
    val=BASE_VAL if task=="base" else EXT_VAL
    recorded=[]
    for g in range(1,11):
        root=root1 if g<=3 else root2
        loc=root/arm/f"g{g:02d}"
        loc.mkdir(parents=True,exist_ok=True)
        content={
            "generation":g,"final_test_evaluated":False,
            "history":[{"validation":val}],
        }
        if task=="base":
            content.update(arm=arm,class_split=[50,1473,100],
                           whole_prefix_generated=False,
                           optimizer_exposures=1_000_000)
        else:
            content.update(condition=arm,training_class_split=[50,1473,100],
                           recursively_generated_input_prefix=False,
                           training_exposures=300_000)
        (loc/"metrics.json").write_text(json.dumps(content))
        (loc/"model.eqx").write_bytes(b"mock-not-valid-checkpoint")
        recorded.append({"generation":g,"last_validation":val})
    (root2/"summary.json").write_text(json.dumps({
        "final_test_evaluated":False,
        "arms":{arm:recorded},
    }))


class TrajectoryTests(unittest.TestCase):
    def test_base_all_ten_and_no_early_test(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)
            a=p/"prior"
            b=p/"later"
            fixture(a,b,"base","recursive_replacement")
            rows,digest=compile_arm("base","recursive_replacement",a,b)
            self.assertEqual([r["generation"] for r in rows],list(range(1,11)))
            self.assertEqual(rows[-1]["val_classes_accuracy"],0.75)
            self.assertEqual(len(digest),10)
            self.assertTrue(all(r["validation_only"] for r in rows))
            self.assertEqual(rows[-1]["independent_model_seeds"],1)

    def test_extended_keeps_teacher_forcing_distinct(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)
            a=p/"early"
            b=p/"late"
            fixture(a,b,"extended","real_anchor_half")
            rows,_=compile_arm("extended","real_anchor_half",a,b)
            self.assertEqual(rows[-1]["val_relabel_second_teacher_forced_accuracy"],.85)
            self.assertEqual(rows[-1]["val_relabel_generated_symbol_context_validity"],.8)
            self.assertNotIn("final_test_accuracy",rows[-1])

    def test_corrupted_generation_or_test_access_fails(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)
            a=p/"early"
            b=p/"late"
            fixture(a,b,"base","clean_only")
            f=b/"clean_only"/"g10"/"metrics.json"
            m=json.loads(f.read_text())
            m["final_test_evaluated"]=True
            f.write_text(json.dumps(m))
            with self.assertRaises(AssertionError):
                compile_arm("base","clean_only",a,b)

    def test_missing_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)
            a=p/"early"
            b=p/"late"
            fixture(a,b,"extended","frozen_g0")
            (b/"frozen_g0"/"g10"/"model.eqx").unlink()
            with self.assertRaises(FileNotFoundError):
                compile_arm("extended","frozen_g0",a,b)


if __name__=="__main__":
    unittest.main()
