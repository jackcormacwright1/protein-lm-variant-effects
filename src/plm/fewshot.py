"""How many measured variants does it take to beat the zero-shot score?

Supervised models are trained on n measured single mutants from an assay and tested on held-out ones, using
ProteinGym's five-fold splits (Notin et al., 2023). Three ways of splitting matter:

  random      variants assigned to folds at random, so a test variant's position has usually been seen in
              training (with a different substitution);
  modulo      positions assigned to folds in rotation (1, 6, 11, ... in one fold): test positions are new, but
              their neighbours were seen;
  contiguous  the sequence cut into five blocks: test positions are new and so are their neighbours.

Three predictors (following Hsu et al., 2022):

  zero-shot   the language model's score, no training;
  one-hot     ridge regression on indicators of the position, of (position, amino acid) and of the mutant
              amino acid: the information in a one-hot encoding of the whole mutated sequence;
  augmented   the same, plus the zero-shot score as a feature that is barely penalised.

The comparison asks two things: how many labels the supervised model needs, and whether the answer depends
on how the test variants relate to the training ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

from .data import AA_INDEX, Assay, parse

SCHEMES = ("random", "modulo", "contiguous")
ALPHAS = np.logspace(-1, 3, 9)
ZERO_SHOT_SCALE = 30.0    # the zero-shot column is scaled up so the ridge penalty barely touches it


def features(assay: Assay, mutants: pd.Series, zero_shot: np.ndarray | None = None) -> sparse.csr_matrix:
    """Indicators of the position, (position, amino acid) and mutant amino acid, plus optionally the zero-shot
    score. For single mutants this is the information in a one-hot encoding of the whole mutated sequence
    (Hsu et al., 2022): losing the wild-type residue at a position is shared by every substitution there."""
    L = len(assay.sequence)
    rows, cols = [], []
    for i, mut in enumerate(mutants):
        m = parse(mut)[0]
        rows += [i, i, i]
        cols += [m.pos - 1, L + (m.pos - 1) * 20 + AA_INDEX[m.mt], L * 21 + AA_INDEX[m.mt]]
    X = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(mutants), L * 21 + 20))
    if zero_shot is not None:
        z = (zero_shot - zero_shot.mean()) / (zero_shot.std() + 1e-12)
        X = sparse.hstack([X, sparse.csr_matrix(ZERO_SHOT_SCALE * z[:, None])]).tocsr()
    return X


def _spearman(a, b) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(spearmanr(a, b).statistic)


def learning_curve(assay: Assay, zero_shot: np.ndarray, sizes: tuple[int, ...], scheme: str,
                   rng: np.random.Generator) -> pd.DataFrame:
    """For each fold and training size: test-set Spearman of the three predictors."""
    v = assay.variants
    fold = v[f"fold_{scheme}_5"].to_numpy()
    y = v["DMS_score"].to_numpy()
    X_one = features(assay, v["mutant"])
    X_aug = features(assay, v["mutant"], zero_shot)
    rows = []
    for k in range(5):
        test = np.flatnonzero(fold == k)
        pool = np.flatnonzero(fold != k)
        if len(test) < 5:
            continue
        rows.append({"scheme": scheme, "fold": k, "n_train": 0, "model": "zero-shot",
                     "spearman": _spearman(zero_shot[test], y[test])})
        order = rng.permutation(pool)
        for n in sizes:
            if n > len(pool):
                break
            train = order[:n]
            for name, X in (("one-hot", X_one), ("augmented", X_aug)):
                model = RidgeCV(alphas=ALPHAS).fit(X[train], y[train])
                rows.append({"scheme": scheme, "fold": k, "n_train": n, "model": name,
                             "spearman": _spearman(model.predict(X[test]), y[test])})
    return pd.DataFrame(rows)
