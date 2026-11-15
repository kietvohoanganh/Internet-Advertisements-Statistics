# Learn correction rules from training groups.
import argparse
from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import joblib
import numpy as np

from similarity import column, comparison, dataset, numbers, read, write
from patterns import counts, patterns, revise, rulebook

MODEL_R = Path("outputs/models/04_ad_classifier.rds")
MODEL_SVM = Path("outputs/models/similarity.joblib")
RULES = Path("outputs/models/correction.joblib")
AD_CUE = re.compile(r"(?:^|[^a-z])(?:ad|ads|advert\w*|sponsor\w*|banners?)(?:[^a-z]|$)|adclick|adserver|doubleclick|memberbanners")


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def align(rows, other, keys):
    if len(rows) != len(other) or any(any(a[k] != b[k] for k in keys) for a, b in zip(rows, other)):
        raise ValueError(f"Misaligned records ({', '.join(keys)})")


def score_counts(y, base, predicted):
    return dict(n=len(y), correct=int(sum(predicted == y)), tp=int(sum((predicted == 1) & (y == 1))),
                fn=int(sum((predicted == 0) & (y == 1))), fp=int(sum((predicted == 1) & (y == 0))),
                tn=int(sum((predicted == 0) & (y == 0))),
                fixed=int(sum((predicted == y) & (base != y))), introduced=int(sum((predicted != y) & (base == y))))


def select_rule(keys, cv, svm_cv):
    align(cv, svm_cv, ("row_id", "group_id", "target", "fold", "baseline"))
    y, base, groups, folds = (column(cv, name) for name in ("target", "baseline", "group_id", "fold"))
    svm = column(svm_cv, "candidate")
    candidates = {n: {} for n in (4, 8, 16, 32, 64)}
    for fold in np.unique(folds):
        valid = folds == fold
        if np.intersect1d(groups[valid], groups[~valid]).size:
            raise ValueError("The same profile appears in fitting and validation.")
        freq = counts((keys[i] for i in np.flatnonzero(~valid)), y[~valid])
        for support, decisions in candidates.items():
            book = rulebook(freq, support)
            for i in np.flatnonzero(valid):
                decisions[i] = revise(keys[i], book, base[i], svm[i])
    grid, eligible = [], []
    for support, decisions in candidates.items():
        p = np.array([decisions[i][0] for i in range(len(cv))])
        stats = score_counts(y, base, p)
        grid.append(dict(min_support=support, **stats))
        if stats["introduced"] == 0 and stats["fixed"] > 0:
            eligible.append((stats["fixed"], support))
    # Keep the baseline if every trial loses a correct OOF decision.
    support = max(eligible)[1] if eligible else len(y) + 1
    selected = candidates.get(support, {i: (int(v), "", 0) for i,v in enumerate(base)})
    return support, selected, grid


def audit(rows, features, names, evidence, cuts, svm_classes, revisions, group_sizes, label_counts):
    result = []
    for r, x, terms, t, s, (pred, rule, support) in zip(
            rows, features, evidence, cuts, svm_classes, revisions):
        y, b, g = int(r["target"]), int(r["baseline"]), int(r["group_id"])
        flags = [names[j] for j in np.flatnonzero(x[4:] == 1) + 4]
        cues = [term for term in flags if AD_CUE.search(term.partition("*")[2].lower())]
        prob = float(r["baseline_prob"])
        item = dict(row_id=r["row_id"], group_id=g, rows_in_partition_with_profile=group_sizes[g],
                    same_profile_ad=label_counts[g][1], same_profile_nonad=label_counts[g][0],
                    target=y, baseline=b, baseline_prob=prob, threshold=t,
                    geometry_missing=bool(np.isnan(x[:3]).any()), height=x[0], width=x[1],
                    active_terms=len(flags), ad_cues=" | ".join(cues), all_terms=" | ".join(flags),
                    svm_class=int(s), corrected_class=int(pred), rule=rule, supporting_train_rows=support,
                    outcome="fixed" if b != y and pred == y else "introduced" if b == y and pred != y else
                            "still_wrong" if b != y else "still_correct")
        if all(label_counts[g][label] for label in (0, 1)):
            item["review_action"] = "Inspect source images/pages for identical feature profiles with conflicting labels."
        elif y == 1 and item["geometry_missing"]:
            item["review_action"] = "Recover original dimensions and link context for ads with missing geometry."
        elif y == 1 and len(flags) <= 2:
            item["review_action"] = "Recover the destination URL, surrounding text or source image for sparse text evidence."
        elif y == 1:
            item["review_action"] = "Compare positive and negative model contributions with the source image and landing page."
        elif cues:
            item["review_action"] = "Inspect the context of advertising terms and the actual destination of the source link."
        else:
            item["review_action"] = "Compare dimensions and high-weight terms with the actual function of the image."
        if terms is not None:
            item.update(intercept=terms["intercept"], log_odds=terms["log_odds"],
                        strongest_toward_ad=terms["toward_ad"], strongest_toward_nonad=terms["toward_nonad"])
        result.append(item)
    return result


def run(out, evidence_file):
    raw, target, names, columns = dataset(Path("."))
    cv = read(Path("outputs/training/elastic/predictions.csv"))
    svm_cv = read(Path("outputs/training/similarity/oof.csv"))
    fit_ids = column(cv, "row_id")
    partitions = read(Path("outputs/tables/01_row_partitions.csv"))
    align(cv, [r for r in partitions if r["used_for_training"] == "TRUE"], ("row_id", "group_id", "target"))
    if not np.array_equal(target[fit_ids-1], column(cv, "target")) or len(set(fit_ids)) != len(cv):
        raise ValueError("Invalid training IDs or labels.")
    keys = patterns(raw[fit_ids-1], names)
    support, selected, grid = select_rule(keys, cv, svm_cv)
    # Full-dataset counts are audit-only; exclude them from prediction.
    group_labels = defaultdict(Counter)
    for r in partitions:
        group_labels[int(r["group_id"])][int(r["target"])] += 1
    book = rulebook(counts(keys, column(cv, "target")), support)
    # Bind the correction to its parent models before test evaluation.
    bundle = dict(book=book, names=names, columns=columns, support=support,
                  elastic_sha256=fingerprint(MODEL_R), svm_sha256=fingerprint(MODEL_SVM))
    joblib.dump(bundle, out / "rules.joblib", compress=3)
    write(out / "training_grid.csv", grid)
    fold_cuts = {r["outer_fold"]:float(r["threshold"]) for r in read(Path("outputs/training/elastic/settings.csv"))
                 if r["model"] == "baseline"}
    cv_audit = audit(cv, raw[fit_ids-1], names, [None]*len(cv), [fold_cuts[r["fold"]] for r in cv],
                     column(svm_cv, "candidate"), [selected[i] for i in range(len(cv))],
                     Counter(column(cv, "group_id")), group_labels)

    held = [r for r in partitions if r["partition"] == "test"]
    test = read(Path("outputs/tables/04_test_predictions.csv"))
    svm_test = read(Path("outputs/training/similarity/test.csv"))
    evidence = read(evidence_file)
    align(held, test, ("row_id", "target"))
    align(held, svm_test, ("row_id", "target", "group_id"))
    align(held, evidence, ("row_id",))
    test_ids = column(held, "row_id")
    test_x = raw[test_ids-1]
    if np.intersect1d(column(held, "group_id"), column(cv, "group_id")).size:
        raise ValueError("A predictor profile crosses train and test.")
    svm = joblib.load(MODEL_SVM)
    predicted_svm = svm.predict(test_x)
    reconstructed = np.array([1/(1+math.exp(-float(row["log_odds"]))) for row in evidence])
    if not np.allclose(reconstructed, [float(r["tuned_prob"]) for r in test], rtol=0, atol=1e-10):
        raise ValueError("Elastic Net attributions disagree with stored test probabilities.")
    if not np.array_equal(predicted_svm, column(svm_test, "candidate")):
        raise ValueError("Saved SVM predictions disagree with the test reference.")
    base = np.array([{"ad":1,"nonad":0}[r["predicted_class"]] for r in test])
    if not np.array_equal(base, column(svm_test, "baseline")):
        raise ValueError("Mismatched Elastic Net labels.")
    decisions = [revise(k, book, b, s) for k, b, s in zip(patterns(test_x, names), base, predicted_svm)]
    test_rows = [dict(r, baseline=b, baseline_prob=p["tuned_prob"]) for r,b,p in zip(held,base,test)]
    held_audit = audit(test_rows, test_x, names, evidence, [float(r["cutoff"]) for r in evidence],
                       predicted_svm, decisions, Counter(column(held, "group_id")), group_labels)
    write(out / "test_errors.csv", [r for r in held_audit if r["outcome"] != "still_correct"], fieldnames=list(held_audit[0]))
    write(out / "training_errors.csv", [r for r in cv_audit if r["outcome"] != "still_correct"], fieldnames=list(cv_audit[0]))
    write(out / "test_predictions.csv", [{k:r[k] for k in ("row_id","group_id","target","baseline",
          "baseline_prob","svm_class","corrected_class","rule","supporting_train_rows","outcome")} for r in held_audit])

    unique = list({(r["group_id"], r["target"]):r for r in held_audit}.values())
    metrics, intervals = [], []
    for part, rows in (("training_oof",cv_audit), ("test",held_audit), ("test_unique",unique)):
        y, b, p, g = (column(rows,k) for k in ("target","baseline","corrected_class","group_id"))
        for model, pred in (("elastic_net",b), ("correction",p)):
            metrics.append(dict(partition=part, model=model, **score_counts(y,b,pred)))
        intervals.append(comparison(part,y,g,b,p))
    write(out / "metrics.csv", metrics)
    write(out / "paired.csv", intervals)


def score(staging, output):
    bundle = joblib.load(RULES)
    if (bundle["elastic_sha256"] != fingerprint(MODEL_R)
            or bundle["svm_sha256"] != fingerprint(MODEL_SVM)):
        raise ValueError("The base models changed. Run make run to rebuild the classifier.")
    features = read(staging / "features.csv")
    base_rows = read(staging / "baseline.csv")
    if not features or len(features) != len(base_rows):
        raise ValueError("Feature rows and baseline predictions must match.")
    x = np.array([numbers([row[c] for c in bundle["columns"]]) for row in features])
    svm = joblib.load(MODEL_SVM).predict(x)
    rows = []
    for keys, r, s in zip(patterns(x,bundle["names"]), base_rows, svm):
        b = {"ad":1,"nonad":0}[r["predicted_class"]]
        pred, reason, n = revise(keys, bundle["book"], b, s)
        rows.append(dict(elastic_ad_prob=r["ad_prob"], elastic_class=r["predicted_class"],
                         predicted_class="ad" if pred else "nonad", changed=pred!=b,
                         rule=reason, supporting_train_rows=n))
    write(output, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and score the retained Elastic Net, SVM and pattern correction classifier.')
    actions = parser.add_subparsers(dest="command", required=True)
    experiment = actions.add_parser("run")
    experiment.add_argument("contributions", type=Path)
    inference = actions.add_parser("predict")
    inference.add_argument("stage", type=Path)
    inference.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        out = Path("outputs/classifier")
        out.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=out) as stage:
            run(Path(stage), args.contributions)
            for p in sorted(Path(stage).iterdir()):
                destination = RULES if p.name == "rules.joblib" else out/p.name
                p.replace(destination)
    else:
        score(args.stage, args.output)
