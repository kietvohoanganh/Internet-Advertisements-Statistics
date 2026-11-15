# Learn pure correction patterns from training data.
from collections import Counter
from itertools import combinations

import numpy as np


def patterns(raw, names):
    # Activate measured sizes, the local flag and observed URL/text terms.
    result = []
    for row in raw:
        terms = [names[j] for j in np.flatnonzero(row[4:] == 1) + 4]
        fields = set(terms)
        for label, value in zip(("height", "width", "local"), (row[0], row[1], row[3])):
            if np.isfinite(value):
                fields.add(f"{label}={value:g}")
        fields.add("geometry=missing" if np.isnan(row[:3]).any() else "geometry=complete")
        ordered = sorted(fields)
        result.append([(f,) for f in ordered] + list(combinations(ordered, 2)))
    return result


def counts(pattern_rows, labels):
    classes = [Counter(), Counter()]
    for keys, label in zip(pattern_rows, labels):
        classes[int(label)].update(keys)
    return classes


def revise(keys, book, original, svm):
    # Reject a pattern if any training label contradicts it.
    if original == svm:
        return original, "", 0
    evidence = []
    for label in (0, 1):
        hits = [k for k in keys if k in book[label]]
        best = max(hits, key=lambda k: (book[label][k], -len(k), k), default=None)
        evidence.append((book[label].get(best, 0), best))
    if evidence[original][0] or not evidence[1-original][0]:
        return original, "", 0
    n, rule = evidence[1-original]
    return 1-original, " & ".join(rule), n


def rulebook(classes, support):
    # Retain pure patterns with enough training support.
    return [{k: count for k, count in source.items() if count >= support and k not in classes[1-label]}
            for label, source in enumerate(classes)]
