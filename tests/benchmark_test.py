from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import joblib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"classifier"))
from benchmark import forest, select_default
from evaluation import choose_threshold, group_intervals, metrics, paired_interval
from similarity import read, write


class BenchmarkTest(unittest.TestCase):
    def test_metrics_and_ties(self):
        m=metrics(np.array([1,0,0,0]),np.array([1,1,0,0]),np.array([.9,.6,.3,.1]))
        self.assertAlmostEqual(m["f2"],5/6)
        self.assertAlmostEqual(m["balanced_accuracy"],5/6)
        self.assertAlmostEqual(m["mcc"],2/np.sqrt(12))
        tied=metrics(np.array([1,0]),np.array([0,0]),np.array([.2,.2]))
        self.assertEqual(tied["roc_auc"],.5);self.assertEqual(tied["average_precision"],.5)
        self.assertTrue(np.isnan(metrics([1,0],[1,0])["roc_auc"]))
        for y,p in [([],[]),([0,2],[0,1]),([1,0],[1])]:
            with self.assertRaises(ValueError):metrics(y,p)

    def test_precision_constraint_and_infeasible_fallback(self):
        y=np.array([0,1,1,0]);s=np.array([.9,.8,.7,.1])
        result=choose_threshold(y,s,"f2_precision",.9)
        self.assertTrue(result["fallback"]);self.assertLess(result["precision"],.9)
        result=choose_threshold([1,1,0,0],[.9,.8,.7,.1],"f2_precision",.9)
        self.assertFalse(result["fallback"]);self.assertEqual(result["recall"],1)
        # Tied classes cannot produce positives under a zero-FP budget.
        result=choose_threshold([1,0],[.5,.5],"accuracy",fp_budget=0)
        self.assertEqual(result["fp"],0);self.assertEqual(result["tp"],0)

    def test_fold_local_preprocessing_and_roundtrip(self):
        x=np.array([[10,20,2,0,0],[20,40,2,1,1],[30,60,2,0,0],[40,80,2,1,1]],float)
        fit=forest(dict(n_estimators=16,max_depth=3),7,1,5).fit(x,np.array([0,1,0,1]))
        before=fit[0].named_transformers_["geometry"].statistics_.copy()
        changed=x.copy();changed[:,:3]=1e12
        fit.predict_proba(changed)
        np.testing.assert_array_equal(before,fit[0].named_transformers_["geometry"].statistics_)
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"fit.joblib";joblib.dump(fit,path)
            np.testing.assert_array_equal(fit.predict_proba(x),joblib.load(path).predict_proba(x))
            empty=Path(tmp)/"empty.csv";write(empty,[],fieldnames=["row_id","score"])
            self.assertEqual(read(empty),[])
            missing=Path(tmp)/"missing.csv";write(missing,[dict(value=np.nan)])
            self.assertEqual(read(missing),[dict(value="")])

    def test_accuracy_selection_does_not_apply_precision_constraint(self):
        candidates=[dict(model="elastic_net",objective="accuracy",accuracy=.99,precision=.8,f2=.9),
                    dict(model="lasso",objective="accuracy",accuracy=.95,precision=.95,f2=.85)]
        ablation=[dict(objective="accuracy",candidate="pattern_correction",metric="f2",low=-.1,delta=0)]
        folds=[dict(fold=f,model=m,objective="accuracy",f2=.8) for f in (1,2,3) for m in ("elastic_net","pattern_correction")]
        result=select_default(candidates,ablation,folds,dict(objective="accuracy",min_precision=.9,
            correction_min_gain=.005,outer_folds=3,seed=2026))
        self.assertEqual(result["model"],"elastic_net")
        self.assertFalse(result["precision_constraint_applied"])
        self.assertEqual(result["fallback"],"none")

    def test_group_bootstrap_determinism(self):
        args=([1,1,0,0],[1,0,0,0],[1,1,2,3])
        self.assertEqual(group_intervals(*args,reps=20),group_intervals(*args,reps=20))
        result=paired_interval([1,1,0,0],[1,0,0,0],[1,1,0,0],[1,1,2,3],reps=30)
        self.assertGreater(result["delta"],0);self.assertLessEqual(result["low"],result["high"])


if __name__=="__main__":unittest.main()
