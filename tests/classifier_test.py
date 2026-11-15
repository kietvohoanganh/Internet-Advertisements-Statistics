import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "classifier"))
import correction
from model import SimilarityClassifier
from patterns import counts, patterns, revise, rulebook


class RetainedClassifierTest(unittest.TestCase):
    def test_conflicting_and_agreeing_evidence(self):
        book = [{("size",): 4}, {("ad_word",): 6}]
        self.assertEqual(revise([("size",)], book, 1, 0)[0], 0)
        self.assertEqual(revise([("size",), ("ad_word",)], book, 1, 0)[0], 1)
        self.assertEqual(revise([("size",)], book, 1, 1)[0], 1)
        mixed = counts([[("same",)], [("same",)]], [0, 1])
        self.assertEqual(rulebook(mixed, 1), [{}, {}])

    def test_invalid_features_and_labels(self):
        raw=np.array([[10,20,2,0,0],[20,40,2,1,1]],float)
        names=["height","width","aratio","local","url*advert"]
        model=SimilarityClassifier(names,names,1,1,0).fit(raw,[0,1])
        for bad in [raw[:0],np.array([[np.inf,20,2,0,1]]),np.array([[10,20,2,2,1]])]:
            with self.assertRaises(ValueError):model.predict(bad)
        with self.assertRaises(ValueError):SimilarityClassifier(names,names,1,1,0).fit(raw,[0,2])

    def test_feature_only_scoring_and_model_fingerprints(self):
        raw = np.array([[10, 20, 2, 0, 0], [11, 20, 1.8, 0, 0],
                        [60, 468, 7.8, 0, 1], [61, 468, 7.7, 0, 1]], dtype=float)
        names = ["height", "width", "aratio", "local", "url*advert"]
        model = SimilarityClassifier(names, names, 1, 4, 0).fit(raw, np.array([0, 0, 1, 1]))
        svm_class = int(model.predict(raw[:1])[0])
        original = 1-svm_class
        book = [{}, {}]
        book[svm_class] = {patterns(raw[:1], names)[0][0]: 4}
        with TemporaryDirectory() as tmp:
            stage = Path(tmp)
            elastic, svm, rules = [stage/name for name in ("elastic.rds", "svm.joblib", "rules.joblib")]
            elastic.write_bytes(b"R component fingerprint fixture")
            joblib.dump(model, svm)
            joblib.dump(dict(columns=names, names=names, book=book,
                elastic_sha256=correction.fingerprint(elastic), svm_sha256=correction.fingerprint(svm)), rules)
            with (stage/"features.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(names[::-1])
                writer.writerow(raw[0, ::-1])
            with (stage/"baseline.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["ad_prob", "predicted_class"])
                writer.writeheader()
                writer.writerow(dict(ad_prob=.9 if original else .1,
                                     predicted_class="ad" if original else "nonad"))
            with patch.object(correction, "MODEL_R", elastic), patch.object(correction, "MODEL_SVM", svm), patch.object(correction, "RULES", rules):
                correction.score(stage, stage/"result.csv")
                with (stage/"result.csv").open() as stream:
                    result, = csv.DictReader(stream)
                self.assertEqual(result["predicted_class"], "ad" if svm_class else "nonad")
                elastic.write_bytes(b"Changed R component")
                with self.assertRaisesRegex(ValueError, "models changed"):
                    correction.score(stage, stage/"result.csv")


if __name__ == "__main__":
    unittest.main()
