import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "logistic": "Geometry logistic", "lasso": "LASSO", "elastic_net": "Elastic Net",
    "random_forest": "Random Forest", "similarity_svm": "Similarity SVM",
    "pattern_correction": "Pattern correction",
}
FIGURES = ("class_distribution.png", "model_comparison.png", "confusion_matrix.png", "precision_recall.png")
FILES = {"model_comparison.csv", "statistical_summary.csv", *("figures/" + x for x in FIGURES)}
MODEL_FIELDS = ("protocol", "model", "objective", "selected", "n", "accuracy", "precision", "recall", "f1", "f2",
                "roc_auc", "average_precision", "tp", "fp", "tn", "fn", "f2_ci_low", "f2_ci_high",
                "precision_ci_low", "precision_ci_high")
STAT_FIELDS = ("kind", "variable", "method", "estimate", "estimate_type", "ci_low", "ci_high", "p_value",
               "p_adjusted", "effect_size", "effect_size_type", "n_ad", "n_nonad", "n", "note")


class ReleaseError(ValueError):
    pass


def read_csv(path):
    if not path.is_file():
        raise ReleaseError(f"Missing required input: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    if value is None or str(value).strip() in ("", "NA", "NaN", "nan"):
        return math.nan
    return float(value)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if isinstance(row.get(key), float) and not math.isfinite(row[key])
                             else row.get(key, "") for key in fields})


def digest(path, algorithm="sha256"):
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()


def analysis_signature(root):
    paths = [root / "config.R", root / "requirements.txt"]
    paths.extend(root / name for name in ("renv.lock", ".Rprofile", "renv/activate.R")
                 if (root / name).is_file())
    for folder in ("R", "classifier"):
        paths.extend(p for p in (root / folder).glob("*") if p.suffix in (".R", ".py"))
    paths.extend((root / "scripts").glob("0[1-5]_*.R"))
    value = hashlib.sha256()
    for path in sorted(set(paths)):
        value.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return value.hexdigest()


def generated_tables(root):
    out = root / "outputs"
    selection = json.loads((out / "benchmark/selection.json").read_text())
    metadata = json.loads((out / "benchmark/metadata.json").read_text())
    manifest, = read_csv(out / "models/selected.csv")
    if selection["model"] != manifest["model"] or selection["objective"] != manifest["objective"]:
        raise ReleaseError("Selection JSON and deployed manifest disagree.")
    if digest(root / "config.R", "md5") != manifest["config_md5"]:
        raise ReleaseError("Configuration changed; run make run before publishing.")
    objective = selection["objective"]
    comparison = read_csv(out / "benchmark/comparison.csv")
    intervals = read_csv(out / "benchmark/intervals.csv")
    rows = []
    for name in MODELS:
        matches = [r for r in comparison if r["model"] == name and r["objective"] == objective]
        if len(matches) != 1:
            raise ReleaseError(f"Expected one matched CV result for {name}.")
        row = dict(matches[0], selected=str(name == selection["model"]).lower())
        for metric in ("f2", "precision"):
            ci = [r for r in intervals if r["model"] == name and r["objective"] == objective and r["metric"] == metric]
            if len(ci) != 1:
                raise ReleaseError(f"Missing {metric} interval for {name}.")
            row[metric + "_ci_low"], row[metric + "_ci_high"] = ci[0]["low"], ci[0]["high"]
        rows.append(row)
    held = read_csv(out / "benchmark/development_test_metrics.csv")
    chosen, = [r for r in held if r["model"] == selection["model"] and r["objective"] == objective]
    rows.append(dict(chosen, selected="true"))
    legacy = out / "classifier/metrics.csv"
    legacy_current = any(r["step"] == "scripts/05_finalize.R" and r["status"] == "passed"
                         for r in read_csv(out / "run_status.csv"))
    if legacy.exists() and legacy_current:
        old, = [r for r in read_csv(legacy) if r["partition"] == "test" and r["model"] == "correction"]
        tp, fp, tn, fn = (int(old[k]) for k in ("tp", "fp", "tn", "fn"))
        div = lambda a, b: a / b if b else math.nan
        rows.append(dict(protocol="development_exposed_legacy_accuracy", model="legacy_pattern_correction",
            objective="accuracy", selected="false", n=int(old["n"]), tp=tp, fp=fp, tn=tn, fn=fn,
            accuracy=div(tp+tn, int(old["n"])), precision=div(tp,tp+fp), recall=div(tp,tp+fn),
            f1=div(2*tp,2*tp+fp+fn), f2=div(5*tp,5*tp+4*fn+fp)))
    stats = []
    quality, = read_csv(out / "tables/01_data_quality.csv")
    classes = {r["class"]: int(r["n"]) for r in read_csv(out / "tables/01_class_distribution.csv")}
    dataset = {k: int(quality[k]) for k in ("rows", "predictors", "duplicate_rows", "unique_rows",
                                             "conflicting_label_groups", "rows_missing_geometry")}
    dataset.update(ad=classes["ad"], nonad=classes["nonad"])
    stats.extend(dict(kind="dataset", variable=k, estimate=v, estimate_type="count") for k,v in dataset.items())
    for r in read_csv(out / "tables/03_hypothesis_results.csv"):
        if r["test"] not in ("Welch t-test", "Pearson chi-square", "Fisher exact", "Not estimable"):
            continue
        welch = r["test"] == "Welch t-test"
        stats.append(dict(kind="primary_test", variable=r["variable"], method=r["test"],
            estimate=r["diff_ad_minus_nonad"] if welch else r["effect_size"],
            estimate_type="ad minus nonad mean" if welch else "phi", ci_low=r["ci_low"], ci_high=r["ci_high"],
            p_value=r["p_value"], p_adjusted=r["p_adj_BH"], effect_size=r["effect_size"] if welch else "",
            effect_size_type="Cohen d" if welch else "", n_ad=r["n_ad"], n_nonad=r["n_nonad"],
            note="BH family: four primary tests; individual mean CIs are unadjusted"))
    split = read_csv(out / "tables/01_split_overview.csv")
    train, = [r for r in split if r["stage"] == "train_fit"]
    for r in read_csv(out / "tables/04_logistic_odds_ratios.csv"):
        if r["term"] not in ("height", "width", "aratio", "local"):
            continue
        stable = r["stable_wald_interval"].upper() == "TRUE"
        stats.append(dict(kind="logistic_effect", variable=r["term"], method="Geometry logistic; Wald interval",
            estimate=r["odds_ratio"] if stable else "", estimate_type="conditional odds ratio",
            ci_low=r["ci_low"] if stable else "", ci_high=r["ci_high"] if stable else "", p_value=r["p_value"],
            n=int(train["rows"]), n_ad=int(train["ad_count"]), n_nonad=int(train["rows"])-int(train["ad_count"]),
            note="One training SD for geometry; 0-to-1 for local. Unstable estimates omitted."))
    for r in read_csv(out / "tables/03_missing_geometry_association.csv"):
        stats.append(dict(kind="missingness_test", variable=r["variable"], method=r["test"],
            estimate=r["phi"], estimate_type="phi", p_value=r["p_value"], p_adjusted=r["p_adj_BH"], n=r["n"],
            note="Missing indicator vs class; separate exploratory BH family of three tests"))
    provenance = dict(source_md5=quality["source_md5"], config_md5=manifest["config_md5"],
                      analysis_signature=analysis_signature(root), settings=json.dumps(metadata["config"],sort_keys=True),
                      python_versions=json.dumps(metadata.get("dependencies", {}),sort_keys=True))
    stats.extend(dict(kind="provenance", variable=k, note=v) for k,v in provenance.items())
    return rows, stats


def validate_model_table(rows):
    cv = [r for r in rows if r["protocol"] == "nested_group_oof_train_only"]
    if {r["model"] for r in cv} != set(MODELS) or len(cv) != len(MODELS):
        raise ReleaseError("Published CV candidates are incomplete or duplicated.")
    if len({r["n"] for r in cv}) != 1 or sum(r["selected"] == "true" for r in cv) != 1:
        raise ReleaseError("CV rows must be matched and identify one selected candidate.")
    for r in rows:
        tp, fp, tn, fn = (int(r[k]) for k in ("tp", "fp", "tn", "fn"))
        n = int(r["n"])
        if min(tp,fp,tn,fn) < 0 or tp+fp+tn+fn != n:
            raise ReleaseError("Invalid published confusion counts.")
        for name,a,b in (("accuracy",tp+tn,n),("precision",tp,tp+fp),("recall",tp,tp+fn),
                         ("f1",2*tp,2*tp+fp+fn),("f2",5*tp,5*tp+4*fn+fp)):
            expected = a/b if b else math.nan
            actual = number(r.get(name))
            if not ((math.isnan(expected) and math.isnan(actual)) or math.isclose(expected,actual,abs_tol=1e-12)):
                raise ReleaseError(f"Published {name} disagrees with confusion counts.")
        for name in ("f2", "precision"):
            lo,hi = number(r.get(name+"_ci_low")),number(r.get(name+"_ci_high"))
            if not math.isnan(lo) and not (0 <= lo <= hi <= 1):
                raise ReleaseError("Invalid published confidence interval.")
        if r["model"] in ("pattern_correction","legacy_pattern_correction"):
            if any(math.isfinite(number(r.get(name))) for name in ("roc_auc","average_precision")):
                raise ReleaseError("Correction has no standalone probability ranking.")


def replace_block(text, name, content):
    start,end=f"<!-- {name}:start -->",f"<!-- {name}:end -->"
    if text.count(start)!=1 or text.count(end)!=1:
        raise ReleaseError(f"README needs exactly one {name} marker pair.")
    return re.sub(re.escape(start)+r"[\s\S]*?"+re.escape(end),lambda _:start+"\n"+content+"\n"+end,text)


def readme_blocks(models, stats):
    pct=lambda value:f"{100*number(value):.2f}%"
    cv=[r for r in models if r["protocol"]=="nested_group_oof_train_only"]
    lines=[f"**Nested grouped CV on {int(cv[0]['n']):,} training records.** Intervals are 95% group-bootstrap intervals conditional on fitted fold predictions.","",
           "| Model | Accuracy | Ad precision | Ad recall | F2 [95% CI] |", "|---|---:|---:|---:|---:|"]
    for r in cv:
        lines.append(f"| {MODELS[r['model']]}{' (selected)' if r['selected']=='true' else ''} | {pct(r['accuracy'])} | {pct(r['precision'])} | {pct(r['recall'])} | {number(r['f2']):.3f} [{number(r['f2_ci_low']):.3f}, {number(r['f2_ci_high']):.3f}] |")
    test,=[r for r in models if r['protocol']=="development_exposed_historical_test"]
    lines.extend(["",f"On the **development-exposed historical test** ({int(test['n']):,} records), the selected model achieves **{pct(test['accuracy'])} accuracy**, **{pct(test['precision'])} precision** and **{pct(test['recall'])} recall**."])
    old=[r for r in models if r['protocol']=="development_exposed_legacy_accuracy"]
    if old:lines.extend(["",f"The accuracy-oriented legacy classifier achieves **{pct(old[0]['accuracy'])} accuracy** on that same historical partition. The CV-selected model does **not** establish a test-performance improvement."])
    lines.extend(["","CV results inform candidate selection; neither this CV comparison nor the development-exposed test is independent external validation."])
    data={r['variable']:int(number(r['estimate'])) for r in stats if r['kind']=="dataset"}
    dataset=(f"**{data['rows']:,} records · {data['predictors']:,} predictors · {data['ad']:,} ad · {data['nonad']:,} nonad.** "
             f"There are {data['rows_missing_geometry']:,} rows with missing geometry, {data['duplicate_rows']:,} exact duplicate rows and "
             f"{data['conflicting_label_groups']} predictor-profile groups with conflicting labels. Labels are preserved; counts come from the source records.")
    return {"key-results":"\n".join(lines),"dataset":dataset}


def check_snapshot(root, directory=None, text=None):
    directory=directory or root/"results"
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    if actual!=FILES or any(p.is_symlink() for p in directory.rglob('*')):
        raise ReleaseError("results/ must contain exactly the two CSVs and four regular PNG files.")
    models=read_csv(directory/"model_comparison.csv");stats=read_csv(directory/"statistical_summary.csv")
    validate_model_table(models)
    provenance={r['variable']:r['note'] for r in stats if r['kind']=="provenance"}
    if provenance.get('config_md5')!=digest(root/'config.R','md5') or provenance.get('analysis_signature')!=analysis_signature(root):
        raise ReleaseError("Published results do not match the frozen analysis/configuration.")
    data_counts={r['variable']:int(number(r['estimate'])) for r in stats if r['kind']=='dataset'}
    if data_counts.get('ad',0)+data_counts.get('nonad',0)!=data_counts.get('rows') or \
       data_counts.get('duplicate_rows',0)+data_counts.get('unique_rows',0)!=data_counts.get('rows'):
        raise ReleaseError("Published dataset counts are inconsistent.")
    for row in stats:
        if row['kind'] in ('primary_test','logistic_effect','missingness_test'):
            for key in ('p_value','p_adjusted'):
                value=number(row[key])
                if not math.isnan(value) and not 0 <= value <= 1:
                    raise ReleaseError("Invalid published p-value.")
            lo,hi=number(row['ci_low']),number(row['ci_high'])
            if not math.isnan(lo) and lo>hi:raise ReleaseError("Invalid statistical confidence interval.")
    for name in FIGURES:
        p=directory/'figures'/name;data=p.read_bytes()
        if len(data)<5000 or data[:8]!=b'\x89PNG\r\n\x1a\n' or min(struct.unpack('>II',data[16:24]))<500:
            raise ReleaseError(f"Invalid or undersized figure: {name}")
        if provenance.get('figure:'+name)!=digest(p):
            raise ReleaseError(f"Figure integrity mismatch: {name}")
    text=(root/'README.md').read_text() if text is None else text
    for reference in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
        if '://' in reference:continue
        target=reference.split('#')[0]
        path=directory/target.removeprefix('results/') if target.startswith('results/') else root/target
        if not path.is_file():raise ReleaseError(f"Broken README link: {reference}")
    for name,content in readme_blocks(models,stats).items():
        if replace_block(text,name,content)!=text:
            raise ReleaseError(f"README {name} block disagrees with the published data.")
    return models,stats


def publish(root, rscript='Rscript'):
    required=('run_status.csv','validation.txt','source_checksums.csv','benchmark/selection.json',
              'benchmark/metadata.json','models/selected.csv')
    for name in required:
        if not (root/'outputs'/name).is_file():raise ReleaseError(f"Missing validated output: outputs/{name}; run make run.")
    status=read_csv(root/'outputs/run_status.csv')
    if not status or any(r['status']!='passed' for r in status) or not any(r['step']=='scripts/05_benchmark.R' for r in status):
        raise ReleaseError("The complete benchmark pipeline must pass before publishing.")
    # Replay the saved predictor before publication.
    subprocess.run([rscript,'tests/validate_outputs.R'],cwd=root,check=True)
    models,stats=generated_tables(root);validate_model_table(models)
    destination=root/'results'
    if destination.exists():
        extras={str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()}-FILES
        if extras or destination.is_symlink() or any(p.is_symlink() for p in destination.rglob('*')):
            raise ReleaseError("Refusing to replace results/ containing unexpected files or symlinks.")
    original=(root/'README.md').read_text()
    with tempfile.TemporaryDirectory(prefix='.publish-',dir=root) as tmp:
        stage=Path(tmp);new=stage/'results';(new/'figures').mkdir(parents=True)
        write_csv(new/'model_comparison.csv',MODEL_FIELDS,models)
        write_csv(new/'statistical_summary.csv',STAT_FIELDS,stats)
        subprocess.run([rscript,'scripts/publish_figures.R',str(new)],cwd=root,check=True)
        versions=json.loads((stage/'r_versions.json').read_text())
        stats.append(dict(kind='provenance',variable='r_versions',note=json.dumps(versions,sort_keys=True)))
        stats.extend(dict(kind='provenance',variable='figure:'+name,note=digest(new/'figures'/name)) for name in FIGURES)
        write_csv(new/'statistical_summary.csv',STAT_FIELDS,stats)
        updated=original
        for name,content in readme_blocks(models,stats).items():updated=replace_block(updated,name,content)
        check_snapshot(root,new,updated)
        (stage/'README.md').write_text(updated)
        backup=stage/'previous'
        if destination.exists():destination.rename(backup)
        try:
            new.rename(destination)
            os.replace(stage/'README.md',root/'README.md')
        except BaseException:
            if destination.exists():shutil.rmtree(destination)
            if backup.exists():backup.rename(destination)
            raise
    print('Published six validated artifacts in results/ and synchronized README tables.')


def main():
    parser=argparse.ArgumentParser(description='Validate and atomically publish the six small, reproducible release artifacts.')
    parser.add_argument('--check',action='store_true',help='Verify the committed snapshot without training data.')
    args=parser.parse_args()
    try:
        if args.check:
            check_snapshot(ROOT);print('PASS: published metrics, source/configuration, figure hashes and README consistency.')
        else:publish(ROOT,os.environ.get('RSCRIPT','Rscript'))
    except (ReleaseError,OSError,subprocess.CalledProcessError,KeyError,ValueError) as error:
        parser.exit(1,f'Publication failed: {error}\n')


if __name__=='__main__':main()
