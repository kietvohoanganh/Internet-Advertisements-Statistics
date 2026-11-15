from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import publish_results as release


def fixture_rows():
    rows=[]
    for name in release.MODELS:
        rows.append(dict(protocol="nested_group_oof_train_only",model=name,objective="f2_precision",
            selected=str(name=="similarity_svm").lower(),n="4",tp="1",fp="1",tn="2",fn="0",
            accuracy=".75",precision=".5",recall="1",f1=str(2/3),f2=str(5/6),
            f2_ci_low=".2",f2_ci_high="1",precision_ci_low=".1",precision_ci_high="1",
            roc_auc="" if name=="pattern_correction" else ".8",
            average_precision="" if name=="pattern_correction" else ".75"))
    rows.append(dict(rows[-2],protocol="development_exposed_historical_test"))
    return rows


class PublishTest(unittest.TestCase):
    def test_metric_counts_and_correction_ranking(self):
        rows=fixture_rows();release.validate_model_table(rows)
        bad=deepcopy(rows);bad[0]["accuracy"]="1"
        with self.assertRaises(release.ReleaseError):release.validate_model_table(bad)
        bad=deepcopy(rows);bad[5]["roc_auc"]=".99"
        with self.assertRaises(release.ReleaseError):release.validate_model_table(bad)
        bad=deepcopy(rows);bad[1]["n"]="5"
        with self.assertRaises(release.ReleaseError):release.validate_model_table(bad)

    def test_missing_inputs_do_not_modify_public_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"results").mkdir();(root/"results/notes.txt").write_text("keep")
            (root/"README.md").write_text("unchanged")
            with self.assertRaises(release.ReleaseError):release.publish(root)
            self.assertEqual((root/"results/notes.txt").read_text(),"keep")
            self.assertEqual((root/"README.md").read_text(),"unchanged")

    def test_atomic_failure_and_unexpected_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/"outputs";out.mkdir()
            for name in ("validation.txt","source_checksums.csv","benchmark/selection.json","benchmark/metadata.json","models/selected.csv"):
                path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("fixture")
            release.write_csv(out/"run_status.csv",("step","status"),[dict(step="scripts/05_benchmark.R",status="passed")])
            (root/"results").mkdir();(root/"results/user-note.txt").write_text("keep")
            (root/"README.md").write_text("unchanged")
            with patch.object(release.subprocess,"run"),patch.object(release,"generated_tables",return_value=(fixture_rows(),[])):
                with self.assertRaises(release.ReleaseError):release.publish(root)
            self.assertEqual((root/"results/user-note.txt").read_text(),"keep")
            (root/"results/user-note.txt").unlink()
            # A staging failure must preserve the previous public files.
            with patch.object(release.subprocess,"run",side_effect=[None,OSError("render failed")]),patch.object(release,"generated_tables",return_value=(fixture_rows(),[])):
                with self.assertRaises(OSError):release.publish(root)
            self.assertTrue((root/"results").is_dir())
            self.assertEqual((root/"README.md").read_text(),"unchanged")
            self.assertFalse(list(root.glob(".publish-*")))

    def test_readme_updates_are_deterministic(self):
        template="head\n<!-- dataset:start -->\nstale\n<!-- dataset:end -->\ntail"
        once=release.replace_block(template,"dataset","verified")
        self.assertEqual(once,release.replace_block(once,"dataset","verified"))
        self.assertTrue(once.startswith("head"));self.assertTrue(once.endswith("tail"))
        with self.assertRaises(release.ReleaseError):release.replace_block("no markers","dataset","verified")


if __name__=="__main__":unittest.main()
