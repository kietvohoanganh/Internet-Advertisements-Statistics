# Select thresholds on training data and bootstrap whole profile groups.
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def metrics(y, predicted, score=None):
    y, predicted = np.asarray(y), np.asarray(predicted)
    if (y.ndim != 1 or len(y) == 0 or y.shape != predicted.shape
            or not np.isin(y, [0, 1]).all() or not np.isin(predicted, [0, 1]).all()):
        raise ValueError("Supply aligned non-empty binary targets and decisions.")
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else np.nan
    precision, recall, specificity = div(tp, tp+fp), div(tp, tp+fn), div(tn, tn+fp)
    result = dict(n=len(y), tp=tp, fp=fp, tn=tn, fn=fn, correct=tp+tn,
                  accuracy=(tp+tn)/len(y), precision=precision, recall=recall,
                  specificity=specificity, f1=div(2*tp, 2*tp+fp+fn),
                  f2=div(5*tp, 5*tp+4*fn+fp), balanced_accuracy=(recall+specificity)/2,
                  mcc=div(tp*tn-fp*fn, np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))),
                  roc_auc=np.nan, average_precision=np.nan)
    if score is not None:
        score = np.asarray(score)
        if score.shape != y.shape or not np.isfinite(score).all():
            raise ValueError("Scores must be finite and aligned.")
        if len(np.unique(y)) == 2:
            result.update(roc_auc=roc_auc_score(y, score), average_precision=average_precision_score(y, score))
    return result


def choose_threshold(y, scores, objective="accuracy", min_precision=.9, fp_budget=None):
    y, scores = np.asarray(y), np.asarray(scores)
    if (objective not in ("accuracy", "f2_precision") or not 0 < min_precision <= 1
            or y.ndim != 1 or y.size == 0 or y.shape != scores.shape
            or not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all()):
        raise ValueError("Invalid threshold-selection inputs.")
    order = np.argsort(-scores, kind="stable")
    p, truth = scores[order], y[order]
    last = np.flatnonzero(np.r_[p[:-1] != p[1:], True])
    thresholds = np.r_[np.nextafter(p[0], np.inf), p[last]]
    tp = np.r_[0., np.cumsum(truth)[last].astype(float)]
    fp = np.r_[0., last+1] - tp; fn = truth.sum()-tp; tn = (len(y)-truth.sum())-fp
    precision = np.divide(tp, tp+fp, out=np.full_like(tp, np.nan), where=tp+fp>0)
    f2 = np.divide(5*tp, 5*tp+4*fn+fp, out=np.zeros_like(tp), where=5*tp+4*fn+fp > 0)
    eligible = precision >= min_precision
    budget_ok = np.ones_like(eligible) if fp_budget is None else fp <= fp_budget
    feasible = eligible & budget_ok
    allowed = feasible if objective == "f2_precision" and feasible.any() else budget_ok
    if not allowed.any(): allowed = np.ones_like(eligible)
    # Accuracy also considers the all-negative decision.
    candidates = [(bool(allowed[i]), f2[i] if objective == "f2_precision" else tp[i]+tn[i],
                   tp[i] if objective == "accuracy" else np.nan_to_num(precision[i], nan=-1), -abs(float(thresholds[i])-.5),
                   float(thresholds[i]), i) for i in range(len(thresholds))]
    chosen = max(candidates)[-1]
    threshold = float(thresholds[chosen])
    if objective == "accuracy" and (len(y)-truth.sum()) > tp[chosen]+tn[chosen]:
        threshold = float(np.nextafter(p[0], np.inf))
    measured = metrics(y, scores >= threshold, scores)
    measured.update(threshold=threshold, constraint_met=bool(measured["precision"] >= min_precision),
                    fallback=bool(objective == "f2_precision" and not feasible.any()))
    return measured


def rank_choice(result, objective):
    if objective == "accuracy": return (result["correct"], result["tp"])
    return (not result["fallback"], result["f2"], np.nan_to_num(result["precision"], nan=-1), result["correct"])


def group_intervals(y, predicted, groups, score=None, reps=500, seed=2026):
    y, predicted, groups = map(np.asarray, (y, predicted, groups))
    if len(groups) != len(y) or reps < 20: raise ValueError("Invalid group bootstrap inputs.")
    ids = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    fields = ("accuracy", "precision", "recall", "f1", "f2", "balanced_accuracy", "mcc")
    values = []
    for _ in range(reps):
        ix = np.concatenate([ids[i] for i in rng.integers(len(ids), size=len(ids))])
        # Keep ranking estimates separate from corrected class decisions.
        values.append([metrics(y[ix], predicted[ix])[f] for f in fields])
    data = np.array(values)
    return [dict(metric=f, low=float(np.nanquantile(data[:, j], .025)) if np.isfinite(data[:,j]).any() else np.nan,
                 high=float(np.nanquantile(data[:, j], .975)) if np.isfinite(data[:,j]).any() else np.nan, valid=int(np.isfinite(data[:, j]).sum()))
            for j, f in enumerate(fields)]


def paired_interval(y, base, candidate, groups, metric="f2", reps=500, seed=2026):
    y, base, candidate, groups = map(np.asarray, (y, base, candidate, groups))
    ids = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(reps):
        ix = np.concatenate([ids[i] for i in rng.integers(len(ids), size=len(ids))])
        deltas.append(metrics(y[ix], candidate[ix])[metric]-metrics(y[ix], base[ix])[metric])
    return dict(metric=metric, delta=metrics(y, candidate)[metric]-metrics(y, base)[metric],
                low=float(np.nanquantile(deltas, .025)) if np.isfinite(deltas).any() else np.nan,
                high=float(np.nanquantile(deltas, .975)) if np.isfinite(deltas).any() else np.nan)
