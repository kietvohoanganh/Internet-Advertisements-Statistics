# Compare image geometry and active UCI terms with an SVM.
import re

import numpy as np
from scipy import sparse
from sklearn.svm import SVC


def token_mapping(names):
    vocabulary = {}
    rows, columns = [], []
    patterns = {
        "advert": r"(?:^|[^a-z])(?:ad|ads|adverts?|advertis\w*)(?:[^a-z]|$)|adclick|adserver|adbanner|adlink",
        "banner": "banner", "sponsor": "sponsor", "click": "click", "promotion": "promo|affiliate",
    }
    for i, name in enumerate(names[4:]):
        field, sep, phrase = name.lower().partition("*")
        if not sep:
            continue
        keys = {"field:" + field, "phrase:" + phrase}
        keys.update("word:" + w for w in re.findall(r"[a-z]{2,}|[0-9]+", phrase))
        for stem, pattern in patterns.items():
            if re.search(pattern, phrase):
                keys.update(("stem:" + stem, "stem:" + field + ":" + stem))
        for key in sorted(keys):
            rows.append(i)
            columns.append(vocabulary.setdefault(key, len(vocabulary)))
    if not vocabulary:
        raise ValueError("The trial needs the UCI token names from ad.names.")
    return sparse.csr_matrix((np.ones(len(rows)), (rows, columns)),
                             shape=(len(names) - 4, len(vocabulary)))


def encode(raw, mapping):
    raw = np.asarray(raw, dtype=float)
    if np.any(~np.isfinite(raw) & ~np.isnan(raw)):
        raise ValueError("Non-finite feature measurements.")
    if raw.ndim != 2 or raw.shape[1] != mapping.shape[0] + 4:
        raise ValueError("Supply all UCI predictors in training order.")
    if not np.all(np.isnan(raw[:,3:]) | np.isin(raw[:,3:], [0,1])):
        raise ValueError("Unknown binary feature category.")
    tokens = sparse.csr_matrix(raw[:, 4:] == 1, dtype=float)
    words = tokens @ mapping
    words.data[:] = 1
    return tokens, words, raw[:, :4].copy()


def jaccard(a, b):
    intersection = (a @ b.T).toarray()
    union = a.getnnz(axis=1)[:, None] + b.getnnz(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def similarity(left, right):
    tokens, words, x = left
    reference_tokens, reference_words, y = right
    text = 0.5 * jaccard(tokens, reference_tokens) + 0.5 * jaccard(words, reference_words)
    closeness, equality = [], []
    for column in range(3):
        a, b = x[:, column, None], y[None, :, column]
        known = np.isfinite(a) & np.isfinite(b)
        both_missing = np.isnan(a) & np.isnan(b)
        difference = np.log1p(np.maximum(a, 0)) - np.log1p(np.maximum(b, 0))
        closeness.append(np.where(known, np.exp(-2 * difference**2), 0.25 * both_missing))
        equality.append(np.where(known, a == b, 0.25 * both_missing))
    smooth = (sum(closeness) + closeness[0] * closeness[1]) / 4
    exact = (sum(equality) + equality[0] * equality[1]) / 4
    geometry = 0.5 * (smooth + exact)
    local = x[:, 3, None] == y[None, :, 3]
    return (text + 0.30 * geometry + 0.02 * local,
            text * (1 + geometry) + 0.35 * geometry + 0.02 * local)


class SimilarityClassifier:
    def __init__(self, names, columns, kernel, penalty, threshold):
        self.mapping = token_mapping(names)
        self.columns = columns
        self.kernel = kernel
        self.penalty = penalty
        self.threshold = threshold

    def fit(self, x, y, sample_weight=None):
        y = np.asarray(y)
        if len(x) == 0 or len(x) != len(y) or not np.isin(y,[0,1]).all() or len(np.unique(y)) != 2:
            raise ValueError("Supply non-empty aligned features and both binary classes.")
        self.reference = encode(x, self.mapping)
        matrix = similarity(self.reference, self.reference)[self.kernel]
        self.model = SVC(kernel="precomputed", C=self.penalty).fit(matrix, y, sample_weight=sample_weight)
        return self

    def decision_function(self, x):
        if len(x) == 0: raise ValueError("Empty prediction features.")
        # Batch large inputs to limit memory use.
        scores = []
        for start in range(0, len(x), 128):
            batch = encode(x[start:start + 128], self.mapping)
            matrix = similarity(batch, self.reference)[self.kernel]
            scores.append(self.model.decision_function(matrix))
        return np.concatenate(scores) if scores else np.array([])

    def predict(self, x):
        return (self.decision_function(x) >= self.threshold).astype(int)
