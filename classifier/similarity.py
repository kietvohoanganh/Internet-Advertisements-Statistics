# Align SVM predictions with the grouped training folds.
import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile

import joblib
import numpy as np
from sklearn.svm import SVC

from model import SimilarityClassifier, encode, similarity, token_mapping


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write(path, rows, fieldnames=None):
    # Keep the schema for empty tables and write missing values as blanks.
    path = Path(path); rows = list(rows)
    if fieldnames is None:
        if not rows: raise ValueError("Empty CSV needs fieldnames.")
        fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=path.parent,
                                prefix=".csv_", delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: "" if isinstance(value, (float, np.floating)) and not np.isfinite(value)
                                 else value for key, value in row.items()})
        temporary.replace(path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def column(rows, key):
    return np.array([int(r[key]) for r in rows])


def numbers(values):
    return [np.nan if v.strip() in {"?", "", "NA", "NaN"} else float(v) for v in values]


def dataset(root):
    # Check row alignment with the import used to build folds.
    quality, = read(root / "outputs/tables/01_data_quality.csv")
    source = root / "data/raw" / quality["source_file"]
    if hashlib.md5(source.read_bytes()).hexdigest() != quality["source_md5"]:
        raise ValueError("Source changed. Run make run to rebuild the training tables.")
    dictionary = read(root / "data/processed/feature_dictionary.csv")
    names = [r["original_name"] for r in dictionary]
    columns = [r["feature"] for r in dictionary]
    features, target = [], []
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for line_no, row in enumerate(csv.reader(handle)):
            label = row[-1].strip().lower().rstrip(".")
            if line_no == 0 and label not in ("ad", "nonad") and source.suffix.lower() == ".csv":
                continue
            if label not in ("ad", "nonad") or len(row) not in (len(columns) + 1, len(columns) + 2):
                raise ValueError(f"Invalid labelled UCI record at line {line_no + 1}.")
            features.append(numbers(row[-len(columns)-1:-1]))
            target.append(int(label == "ad"))
    if len(features) != int(quality["rows"]):
        raise ValueError("Row count disagrees with the R import.")
    return np.array(features), np.array(target), names, columns


def select(matrices, y, folds, groups, fp_budget):
    # Pairwise similarities use only the two inputs and the UCI dictionary.
    # Fit the SVM and thresholds within the inner folds.
    if not 0 <= fp_budget < sum(y == 0):
        raise ValueError("Invalid baseline false-positive budget.")
    options = []
    for kind, matrix in enumerate(matrices):
        for penalty in (1, 4, 16):
            scores = np.full(len(y), np.nan)
            for fold in np.unique(folds):
                valid = folds == fold
                if np.intersect1d(groups[valid], groups[~valid]).size:
                    raise ValueError("A predictor group crosses folds.")
                svm = SVC(kernel="precomputed", C=penalty).fit(matrix[np.ix_(~valid, ~valid)], y[~valid])
                scores[valid] = svm.decision_function(matrix[np.ix_(valid, ~valid)])
            if not np.isfinite(scores).all():
                raise ValueError("Incomplete inner-fold predictions.")
            # Exclude negatives tied at the false-positive budget boundary.
            boundary = np.sort(scores[y == 0])[::-1][fp_budget]
            cutoffs = np.maximum([-.45, -.25, 0., .20], np.nextafter(boundary, np.inf))
            for cutoff in np.unique(cutoffs):
                pred = scores >= cutoff
                if sum(pred & (y == 0)) > fp_budget:
                    raise ValueError("Threshold exceeds the inner-fold false-positive budget.")
                rank = (sum(pred == y), sum(pred & (y == 1)), -penalty, -abs(cutoff), -kind)
                options.append((rank, (kind, penalty, float(cutoff))))
    return max(options)[1]


def reference_budget(row, y):
    return int(sum(y == 0) - (int(row["correct"]) - int(row["tp"])))


def comparison(part, y, groups, baseline, candidate):
    win = (candidate == y) & (baseline != y)
    loss = (candidate != y) & (baseline == y)
    _, index = np.unique(groups, return_inverse=True)
    sizes = np.bincount(index)
    changes = np.bincount(index, weights=win.astype(int)-loss.astype(int))
    draws = np.random.default_rng(2026).integers(len(sizes), size=(2000, len(sizes)))
    delta = changes[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    lo, hi = np.quantile(delta, [.025, .975])
    return dict(partition=part, fixed=sum(win), lost=sum(loss), delta=changes.sum()/len(y),
                low=lo, high=hi)


def run(root, out):
    table = root / "outputs/tables"
    geometry = root / "outputs/training/elastic"
    raw, target, names, columns = dataset(root)
    rows = read(geometry / "predictions.csv")
    ids, folds, groups = (column(rows, k) for k in ("row_id", "fold", "group_id"))
    x, y = raw[ids-1], target[ids-1]
    ref = column(rows, "baseline")
    partitions = read(table / "01_row_partitions.csv")
    training = [r for r in partitions if r["used_for_training"] == "TRUE"]
    if (not np.array_equal(ids, column(training, "row_id"))
            or not np.array_equal(groups, column(training, "group_id"))
            or not np.array_equal(y, column(rows, "target"))):
        raise ValueError("Training features, group IDs and baseline predictions must align.")
    lookup = {(int(r["outer_fold"]), int(r["row_id"])): int(r["inner_fold"])
              for r in read(geometry / "inner_folds.csv")}
    reference_settings = {int(r["outer_fold"]): r for r in read(geometry / "settings.csv")
                          if r["model"] == "baseline"}
    encoded = encode(x, token_mapping(names))
    matrices = similarity(encoded, encoded)
    pred = np.empty(len(y), dtype=int)
    margin = np.full(len(y), np.nan)
    settings = []
    for k in np.unique(folds):
        valid = folds == k
        inner = np.array([lookup[k, i] for i in ids[~valid]])
        budget = reference_budget(reference_settings[k], y[~valid])
        fit_matrices = [m[np.ix_(~valid, ~valid)] for m in matrices]
        kind, penalty, cutoff = select(fit_matrices, y[~valid], inner, groups[~valid], budget)
        svm = SVC(kernel="precomputed", C=penalty).fit(fit_matrices[kind], y[~valid])
        margin[valid] = svm.decision_function(matrices[kind][np.ix_(valid, ~valid)])
        pred[valid] = margin[valid] >= cutoff
        settings.append(dict(fold=k, kernel=kind, C=penalty, threshold=cutoff, fp_budget=budget))
    ref_cv = read(table / "04_classifier_cv.csv")[0]
    budget = reference_budget(ref_cv, y)
    kind, penalty, cutoff = select(matrices, y, folds, groups, budget)
    settings.append(dict(fold="full", kernel=kind, C=penalty, threshold=cutoff, fp_budget=budget))
    fitted = SimilarityClassifier(names, columns, kind, penalty, cutoff).fit(x, y)
    joblib.dump(fitted, out / "model.joblib", compress=3)
    write(out / "settings.csv", settings)
    write(out / "oof.csv", [dict(row_id=i, group_id=g, fold=k, target=t, baseline=b,
                                 margin=s, candidate=p) for i,g,k,t,b,s,p in zip(ids, groups, folds, y, ref, margin, pred)])

    # Evaluate the test set only after saving the final model and threshold.
    held = [r for r in partitions if r["partition"] == "test"]
    test = read(table / "04_test_predictions.csv")
    it, gt = column(held, "row_id"), column(held, "group_id")
    yt = target[it-1]
    if (not np.array_equal(it, column(test, "row_id")) or not np.array_equal(yt, column(test, "target"))
            or np.intersect1d(groups, gt).size):
        raise ValueError("Invalid historical holdout alignment or group overlap.")
    base = np.array([{"ad":1, "nonad":0}[r["predicted_class"]] for r in test])
    score = joblib.load(out / "model.joblib").decision_function(raw[it-1])
    proposed = (score >= cutoff).astype(int)
    write(out / "test.csv", [dict(row_id=i, group_id=g, target=t, baseline=b, margin=s, candidate=p)
                             for i,g,t,b,s,p in zip(it, gt, yt, base, score, proposed)])
    print(json.dumps(settings[-1], ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit the SVM component on grouped training folds and export aligned predictions.')
    commands = parser.add_subparsers(dest="command", required=True)
    trial = commands.add_parser("run")
    trial.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "run":
        output = args.root / "outputs/training/similarity"
        output.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output) as tmp:
            run(args.root, Path(tmp))
            for artifact in Path(tmp).iterdir():
                destination = args.root / "outputs/models/similarity.joblib" if artifact.name == "model.joblib" else output / artifact.name
                artifact.replace(destination)
