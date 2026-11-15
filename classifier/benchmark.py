# Compare models using nested grouped CV.
import argparse
import hashlib
from importlib import metadata as package_metadata
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import scipy
import sklearn
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from correction import select_rule
from evaluation import choose_threshold, group_intervals, metrics, paired_interval, rank_choice
from model import SimilarityClassifier, encode, similarity, token_mapping
from patterns import counts, patterns, revise, rulebook
from similarity import dataset, numbers, read, write


def forest(params, seed, jobs, n_features):
    transform = ColumnTransformer([
        ("geometry", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), list(range(3))),
        ("binary", SimpleImputer(strategy="most_frequent", keep_empty_features=True), list(range(3, n_features)))])
    return make_pipeline(transform, RandomForestClassifier(**params, random_state=seed, n_jobs=jobs))


def fit_forest(params, seed, jobs, x, y):
    fit = forest(params, seed, jobs, x.shape[1]).fit(x,y)
    # Seed tree construction and reduce scores sequentially for repeatability.
    fit[-1].set_params(n_jobs=1)
    return fit


def forest_search(x, y, folds, cfg, seed):
    trees = cfg["rf_trees"]
    if isinstance(trees, int): trees = [trees]
    small, large = min(trees), max(trees)
    grid = [dict(n_estimators=n, max_depth=d, min_samples_leaf=l, max_features=f, class_weight=w)
            for n, d, l, f, w in [(small,None,1,"sqrt",None),(large,None,1,"sqrt",None),
              (small,12,2,"sqrt","balanced_subsample"),(large,None,3,.1,"balanced_subsample"),
              (small,20,1,.1,None),(large,12,3,"sqrt",None)]]
    choices = {}
    trials = []
    for i, params in enumerate(grid):
        probability = np.full(len(y), np.nan)
        for fold in np.unique(folds):
            valid = folds == fold
            fit = fit_forest(params, seed, cfg["jobs"], x[~valid], y[~valid])
            probability[valid] = fit.predict_proba(x[valid])[:, 1]
        for objective in ("accuracy", "f2_precision"):
            result = choose_threshold(y, probability, objective, cfg["min_precision"])
            trials.append(dict(trial=i, objective=objective, parameters=json.dumps(params, sort_keys=True), **result))
            if objective not in choices or rank_choice(result, objective) > rank_choice(choices[objective][0], objective):
                choices[objective] = (result, params, probability.copy())
    return choices, trials


def svm_search(x, y, folds, names, cfg, seed, budgets):
    mapping = token_mapping(names)
    encoded = encode(x, mapping); matrices = similarity(encoded, encoded)
    choices = {}
    for kind, matrix in enumerate(matrices):
        for penalty in (1, 4, 16):
            scores = np.full(len(y), np.nan)
            for fold in np.unique(folds):
                valid = folds == fold
                fit = SVC(kernel="precomputed", C=penalty).fit(matrix[np.ix_(~valid, ~valid)], y[~valid])
                scores[valid] = fit.decision_function(matrix[np.ix_(valid, ~valid)])
            for objective in ("accuracy", "f2_precision"):
                result = choose_threshold(y, scores, objective, cfg["min_precision"], fp_budget=budgets[objective])
                if objective not in choices or rank_choice(result, objective) > rank_choice(choices[objective][0], objective):
                    choices[objective] = (result, kind, penalty, scores.copy())
    return choices


def permutation_families(fit, x, y, groups, names, seed):
    # Permute predefined feature families within validation groups.
    fields = {name: np.array([j]) for j, name in enumerate(names[:4])}
    for field in ("url", "origurl", "ancurl", "alt", "caption"):
        fields[field] = np.array([j for j, name in enumerate(names) if name.startswith(field+"*")], dtype=int)
    score = metrics(y, fit.predict(x), fit.predict_proba(x)[:, 1])["average_precision"]
    unique, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    rng = np.random.default_rng(seed); result = []
    for name, columns in fields.items():
        if len(columns) == 0: continue
        drops = []
        for repeat in range(3):
            donors = first[rng.permutation(len(unique))][inverse]
            changed = x.copy(); changed[:, columns] = x[donors][:, columns]
            drops.append(score - metrics(y, fit.predict(changed), fit.predict_proba(changed)[:, 1])["average_precision"])
        result.append(dict(feature_family=name, ap_drop_mean=np.mean(drops), ap_drop_sd=np.std(drops)))
    return result


def execute_case(directory, raw, names, columns, cfg, seed, final=False):
    parts = read(directory/"partitions.csv")
    train = [r for r in parts if r["part"] == "train"]
    valid = [r for r in parts if r["part"] == "validation"]
    tids, vids = ([int(r["row_id"])-1 for r in rows] for rows in (train, valid))
    x, xv = raw[tids], raw[vids]
    y = np.array([int(r["target"]) for r in train]); folds = np.array([int(r["fold"]) for r in train])
    groups = np.array([int(r["group_id"]) for r in train])
    if set(groups) & {int(r["group_id"]) for r in valid}: raise ValueError("Outer group leakage.")
    for fold in np.unique(folds):
        if set(groups[folds==fold]) & set(groups[folds!=fold]): raise ValueError("Inner group leakage.")
    linear = read(directory/"linear_predictions.csv")
    settings = read(directory/"linear_settings.csv")
    outputs = []; artifacts = {}; importance = []; durations = {}
    for r in linear:
        if r["part"] == "validation":
            outputs.append(dict(model=r["model"], objective=r["objective"], row_id=int(r["row_id"]),
                group_id=int(r["group_id"]), target=int(r["target"]), score=float(r["score"]),
                prediction=int(r["prediction"]), threshold=float(r["threshold"]), score_type="probability"))
    for r in settings:
        key=(r["model"],r["objective"]); durations[key]=float(r["elapsed_seconds"])
        if final: artifacts[key]=dict(engine="r",artifact=r["artifact"], threshold=float(r["threshold"]))
    def linear_rows(objective, part):
        return [r for r in linear if r["model"] == "elastic_net" and r["objective"] == objective and r["part"] == part]
    began = time.perf_counter()
    rf, trials = forest_search(x,y,folds,cfg,seed)
    write(directory/"forest_search.csv",trials)
    for objective,(result,params,oof) in rf.items():
        fit=fit_forest(params,seed,cfg["jobs"],x,y)
        probability=fit.predict_proba(xv)[:,1]
        for r,p in zip(valid,probability):
            outputs.append(dict(model="random_forest",objective=objective,row_id=int(r["row_id"]),group_id=int(r["group_id"]),
                target=int(r["target"]),score=p,prediction=int(p>=result["threshold"]),threshold=result["threshold"],score_type="probability"))
        durations["random_forest",objective]=time.perf_counter()-began
        if final:
            path=directory/f"random_forest_{objective}.joblib"
            joblib.dump(dict(kind="random_forest",model=fit,columns=columns,threshold=result["threshold"]),path,compress=3)
            artifacts["random_forest",objective]=dict(engine="python",artifact=str(path),threshold=result["threshold"])
        elif objective == "f2_precision":
            importance=permutation_families(fit,xv,np.array([int(r["target"]) for r in valid]),
                np.array([int(r["group_id"]) for r in valid]),names,seed)
    budgets={o:sum(int(r["prediction"])==1 and int(r["target"])==0 for r in linear_rows(o,"inner")) for o in rf}
    began=time.perf_counter(); svms=svm_search(x,y,folds,names,cfg,seed,budgets)
    keys=patterns(x,names); valid_keys=patterns(xv,names)
    for objective,(result,kind,penalty,margins) in svms.items():
        fit=SimilarityClassifier(names,columns,kind,penalty,result["threshold"]).fit(x,y)
        scores=fit.decision_function(xv); labels=scores>=result["threshold"]
        for r,p,margin in zip(valid,labels,scores):
            outputs.append(dict(model="similarity_svm",objective=objective,row_id=int(r["row_id"]),group_id=int(r["group_id"]),
                target=int(r["target"]),score=float(expit(margin)),prediction=int(p),threshold=float(expit(result["threshold"])),score_type="sigmoid_margin"))
        durations["similarity_svm",objective]=time.perf_counter()-began
        lin=linear_rows(objective,"inner")
        if [int(r["row_id"])-1 for r in lin] != tids: raise ValueError("Misaligned inner linear predictions.")
        cv=[dict(row_id=r["row_id"],group_id=r["group_id"],target=r["target"],fold=str(k),baseline=r["prediction"])
            for r,k in zip(lin,folds)]
        svm_cv=[dict(r,candidate=int(m>=result["threshold"])) for r,m in zip(cv,margins)]
        rule_started=time.perf_counter()
        support,_,grid=select_rule(keys,cv,svm_cv)
        book=rulebook(counts(keys,y),support)
        base=linear_rows(objective,"validation")
        decisions=[revise(k,book,int(b["prediction"]),int(s))[0] for k,b,s in zip(valid_keys,base,labels)]
        for r,p in zip(valid,decisions):
            outputs.append(dict(model="pattern_correction",objective=objective,row_id=int(r["row_id"]),group_id=int(r["group_id"]),
                target=int(r["target"]),score="",prediction=int(p),threshold="",score_type="decision_only"))
        durations["pattern_correction",objective]=durations["elastic_net",objective]+durations["similarity_svm",objective]+time.perf_counter()-rule_started
        write(directory/f"correction_search_{objective}.csv",grid)
        if final:
            for name,bundle in [("similarity_svm",dict(kind="svm",model=fit,columns=columns,threshold=result["threshold"])),
                ("pattern_correction",dict(kind="correction",model=fit,columns=columns,names=names,book=book,
                                          linear=artifacts["elastic_net",objective]["artifact"]))]:
                path=directory/f"{name}_{objective}.joblib";joblib.dump(bundle,path,compress=3)
                artifacts[name,objective]=dict(engine="python",artifact=str(path),threshold=result["threshold"],
                    parent_artifact=bundle.get("linear",""))
    if final:
        threshold_curves=[]
        series={}
        for objective in ("accuracy","f2_precision"):
            for name in ("logistic","lasso","elastic_net"):
                series[name,objective]=np.array([float(r["score"]) for r in linear
                    if r["model"]==name and r["objective"]==objective and r["part"]=="inner"])
            series["random_forest",objective]=rf[objective][2]
            series["similarity_svm",objective]=expit(svms[objective][3])
        for (name,objective),score in series.items():
            for threshold in np.linspace(.01,.99,99):
                result=metrics(y,score>=threshold)
                threshold_curves.append(dict(model=name,objective=objective,threshold=threshold,
                    precision=result["precision"],recall=result["recall"],f2=result["f2"],
                    context="inner_oof_selection_only"))
        write(directory/"threshold_curve.csv",threshold_curves)
    return outputs,artifacts,durations,importance


def select_default(summary, ablation, fold_metrics, cfg):
    candidates=[r for r in summary if r["objective"]==cfg["objective"] and r["model"] not in ("elastic_net_clip","elastic_net_filter")]
    correction_evidence=next(r for r in ablation if r["objective"]==cfg["objective"] and r["candidate"]=="pattern_correction" and r["metric"]=="f2")
    fold_wins=sum(next(r for r in fold_metrics if r["model"]=="pattern_correction" and r["objective"]==cfg["objective"] and r["fold"]==f)["f2"] >
                  next(r for r in fold_metrics if r["model"]=="elastic_net" and r["objective"]==cfg["objective"] and r["fold"]==f)["f2"]
                  for f in range(1,cfg["outer_folds"]+1))
    repeatable=(correction_evidence["low"]>0 and correction_evidence["delta"]>=cfg["correction_min_gain"]
                and fold_wins>=int(np.ceil(.6*cfg["outer_folds"])))
    if not repeatable:
        candidates=[r for r in candidates if r["model"]!="pattern_correction"]
    feasible=[r for r in candidates if r["precision"]>=cfg["min_precision"]]
    apply_constraint=cfg["objective"]=="f2_precision"
    fallback=apply_constraint and not bool(feasible)
    pool=(feasible or candidates) if apply_constraint else candidates
    complexity={"logistic":0,"lasso":1,"elastic_net":2,"random_forest":3,"similarity_svm":4,"pattern_correction":5}
    chosen=max(pool,key=lambda r:((r["f2"] if cfg["objective"]=="f2_precision" else r["accuracy"]),-complexity[r["model"]]))
    selection=dict(model=chosen["model"],objective=cfg["objective"],precision_requirement=cfg["min_precision"],
        constraint_met=bool(chosen["precision"]>=cfg["min_precision"]),precision_constraint_applied=apply_constraint,fallback="no_model_meets_precision" if fallback else "none",
        protocol="nested_group_oof_train_only",historical_test="development_exposed",seed=cfg["seed"],
        correction_repeatable=repeatable,correction_fold_wins=fold_wins)
    return selection


def run(cfg):
    root=Path(".");out=Path("outputs/benchmark")
    raw,source_target,names,columns=dataset(root)
    nested=[];timing=[];importance=[]
    for fold in range(1,cfg["outer_folds"]+1):
        partition=read(out/f"fold{fold}"/"partitions.csv")
        if any(int(r["target"]) != source_target[int(r["row_id"])-1] for r in partition):
            raise ValueError("Partition labels disagree with the immutable source.")
        print(f"Python nested benchmark {fold}/{cfg['outer_folds']}",flush=True)
        rows,_,times,permutation=execute_case(out/f"fold{fold}",raw,names,columns,cfg,cfg["seed"]+fold)
        nested.extend(dict(r,fold=fold) for r in rows)
        timing.extend(dict(fold=fold,model=m,objective=o,seconds=v,
            scope="measured parent components plus rule" if m=="pattern_correction" else
                  "SVM search/refit; EN budget cost recorded separately" if m=="similarity_svm" else
                  "family search/refit shared across objectives") for (m,o),v in times.items())
        importance.extend(dict(r,fold=fold) for r in permutation)
    write(out/"oof_predictions.csv",nested);write(out/"timing.csv",timing)
    if importance:write(out/"permutation_importance.csv",importance)
    combinations=sorted({(r["model"],r["objective"]) for r in nested})
    summary=[];intervals=[];fold_metrics=[];ablation=[];threshold_results=[]
    lookup={}
    for model,objective in combinations:
        rows=sorted([r for r in nested if r["model"]==model and r["objective"]==objective],key=lambda r:r["row_id"])
        y,p,g=[np.array([r[k] for r in rows]) for k in ("target","prediction","group_id")]
        score=None if model=="pattern_correction" else np.array([r["score"] for r in rows])
        result=metrics(y,p,score);lookup[model,objective]=(rows,y,p,g)
        if score is not None:
            threshold_results.extend([dict(model=model,objective=objective,operating_point="fixed_0.5",**metrics(y,score>=.5,score)),
                                      dict(model=model,objective=objective,operating_point="inner_optimized",**result)])
        summary.append(dict(protocol="nested_group_oof_train_only",model=model,objective=objective,**result))
        intervals.extend(dict(model=model,objective=objective,**r) for r in group_intervals(y,p,g,reps=cfg["reps"],seed=cfg["seed"]))
        for fold in range(1,cfg["outer_folds"]+1):
            ix=np.array([r["fold"]==fold for r in rows])
            fold_metrics.append(dict(fold=fold,model=model,objective=objective,**metrics(y[ix],p[ix],None if score is None else score[ix])))
    for objective in ("accuracy","f2_precision"):
        base,y,b,g=lookup["elastic_net",objective]
        for candidate in ("similarity_svm","pattern_correction","elastic_net_clip","elastic_net_filter"):
            if (candidate,objective) not in lookup:continue
            rows,y2,p,g2=lookup[candidate,objective]
            if [r["row_id"] for r in base]!=[r["row_id"] for r in rows] or not np.array_equal(g,g2):raise ValueError("Unpaired comparison.")
            for metric in ("accuracy","f2","precision","recall"):
                ablation.append(dict(objective=objective,baseline="elastic_net",candidate=candidate,
                    **paired_interval(y,b,p,g,metric,reps=cfg["reps"],seed=cfg["seed"])))
    write(out/"threshold_comparison.csv",threshold_results)
    write(out/"comparison.csv",summary);write(out/"intervals.csv",intervals)
    write(out/"fold_metrics.csv",fold_metrics);write(out/"ablation.csv",ablation)
    # Select from outer CV predictions before historical-test evaluation.
    selection=select_default(summary,ablation,fold_metrics,cfg)
    (out/"selection.json").write_text(json.dumps(selection,indent=2))
    print("Locked selection:",json.dumps(selection),flush=True)
    rows,artifacts,_,_=execute_case(out/"full",raw,names,columns,cfg,cfg["seed"]+100,final=True)
    write(out/"development_test_predictions.csv",rows)
    held=[]
    for model,objective in combinations:
        items=[r for r in rows if r["model"]==model and r["objective"]==objective]
        y,p=[np.array([r[k] for r in items]) for k in ("target","prediction")]
        score=None if model=="pattern_correction" else np.array([r["score"] for r in items])
        held.append(dict(protocol="development_exposed_historical_test",model=model,objective=objective,**metrics(y,p,score)))
    write(out/"development_test_metrics.csv",held)
    # Keep historical comparisons separate from CV rankings.
    before_after=[]
    history=Path("outputs/history/classifier_metrics.csv")
    if history.exists():
        old=next((r for r in read(history) if r["partition"]=="test" and r["model"]=="correction"),None)
        if old is not None:
            tp,fp,tn,fn=(int(old[k]) for k in ("tp","fp","tn","fn"))
            div=lambda a,b:a/b if b else np.nan
            before_after.append(dict(protocol="archived_development_exposed_test",model="previous_pattern_correction",
                n=int(old["n"]),accuracy=div(tp+tn,int(old["n"])),precision=div(tp,tp+fp),recall=div(tp,tp+fn),
                f1=div(2*tp,2*tp+fp+fn),f2=div(5*tp,5*tp+4*fn+fp),tp=tp,fp=fp,tn=tn,fn=fn))
    for data,protocol in [(summary,"nested_training_cv_used_for_selection"),(held,"development_exposed_test_after")]:
        measured=next(r for r in data if r["model"]==selection["model"] and r["objective"]==selection["objective"])
        before_after.append(dict(measured,protocol=protocol))
    fields=("protocol","model","n","accuracy","precision","recall","f1","f2","tp","fp","tn","fn")
    write(out/"before_after.csv",[{k:r.get(k,"") for k in fields} for r in before_after])
    artifact=artifacts[selection["model"],selection["objective"]]
    quality,=read(Path("outputs/tables/01_data_quality.csv"))
    md5=lambda p:hashlib.md5(Path(p).read_bytes()).hexdigest()
    manifest=dict(**selection,**artifact,artifact_md5=md5(artifact["artifact"]),config_md5=md5("config.R"),
                  source_file=quality["source_file"],source_md5=quality["source_md5"],
                  dictionary_md5=md5("data/processed/feature_dictionary.csv"),schema_version=1)
    if artifact.get("parent_artifact"):manifest["parent_md5"]=md5(artifact["parent_artifact"])
    write(Path("outputs/models/selected.csv"),[manifest])
    metadata=dict(config=cfg,python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,
                  source_md5=quality["source_md5"],uncertainty="group bootstrap conditional on fitted fold decisions; excludes retraining variation",protocol="outer CV conditional on the development training partition; candidate selection uses these scores; not external validation")
    metadata["dependencies"]={}
    for name in ("scikit-learn","numpy","scipy","joblib","threadpoolctl","cloudpickle"):
        try:metadata["dependencies"][name]=package_metadata.version(name)
        except package_metadata.PackageNotFoundError:metadata["dependencies"][name]="not installed"
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2))


def predict(stage, artifact, output):
    bundle=joblib.load(artifact);rows=read(stage/"features.csv")
    if not rows:raise ValueError("Empty prediction input.")
    if set(bundle["columns"])-set(rows[0]):raise ValueError("Missing predictor columns.")
    x=np.array([numbers([r[c] for c in bundle["columns"]]) for r in rows])
    if not np.isfinite(x[~np.isnan(x)]).all():raise ValueError("Non-finite features.")
    if bundle["kind"]=="random_forest":
        scores=bundle["model"].predict_proba(x)[:,1];labels=scores>=bundle["threshold"];kind="probability"
    elif bundle["kind"]=="svm":
        margins=bundle["model"].decision_function(x);scores=expit(margins);labels=margins>=bundle["threshold"];kind="sigmoid_margin"
    else:
        baseline=read(stage/"baseline.csv")
        if len(baseline)!=len(rows):raise ValueError("Misaligned correction input.")
        labels=[revise(k,bundle["book"],int(r["predicted_class"]=="ad"),int(s))[0]
            for k,r,s in zip(patterns(x,bundle["names"]),baseline,bundle["model"].predict(x))]
        scores=[float(r["ad_prob"]) for r in baseline];kind="elastic_net_component_probability"
    write(output,[dict(ad_score=float(s),predicted_class="ad" if p else "nonad",score_type=kind) for s,p in zip(scores,labels)])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description='Matched nested grouped comparisons. Historical test labels never select a model.')
    parser.add_argument("command",choices=("run","predict"));parser.add_argument("arguments",nargs="+")
    args=parser.parse_args()
    if args.command=="run":run(json.loads(Path(args.arguments[0]).read_text()))
    else:predict(Path(args.arguments[0]),Path(args.arguments[1]),Path(args.arguments[2]))
