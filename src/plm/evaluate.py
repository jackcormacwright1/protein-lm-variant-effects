"""Accuracy of zero-shot scores against the measured effects.

Each assay's accuracy is the Spearman rank correlation between the model's scores and the measured DMS
scores over all its variants. ProteinGym's headline number does not simply average the 217 assays, because
some proteins have many assays and organismal-fitness assays outnumber binding assays six to one: it averages
assays within each protein, then proteins within each of the five functions, then the five functions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import DATA, load, reference
from .scoring import LogProbs

CACHE = DATA / "cache"
FUNCTIONS = ["Activity", "Binding", "Expression", "OrganismalFitness", "Stability"]


def aggregate(per_assay: pd.Series, ref: pd.DataFrame) -> float:
    """ProteinGym's average: assays within a protein (per function), proteins within a function, then
    functions."""
    df = ref[["UniProt_ID", "coarse_selection_type"]].join(per_assay.rename("value"), how="inner")
    by_protein = df.groupby(["coarse_selection_type", "UniProt_ID"])["value"].mean()
    by_function = by_protein.groupby(level=0).mean()
    return float(by_function.mean())


def by_function(per_assay: pd.Series, ref: pd.DataFrame) -> pd.Series:
    df = ref[["UniProt_ID", "coarse_selection_type"]].join(per_assay.rename("value"), how="inner")
    return df.groupby(["coarse_selection_type", "UniProt_ID"])["value"].mean().groupby(level=0).mean()


def bootstrap_se(per_assay: pd.Series, ref: pd.DataFrame, n: int = 1000, seed: int = 0) -> float:
    """Standard error of the aggregate from resampling assays with replacement."""
    rng = np.random.default_rng(seed)
    ids = per_assay.index.to_numpy()
    vals = []
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        s = per_assay.loc[pick]
        s.index = [f"{i}#{k}" for k, i in enumerate(pick)]
        r = ref.loc[pick].copy()
        r.index = s.index
        vals.append(aggregate(s, r))
    return float(np.std(vals, ddof=1))


def score_assay(dms_id: str, cache_dir: Path, ref: pd.DataFrame) -> pd.DataFrame:
    assay = load(dms_id, ref)
    lp = LogProbs.load(cache_dir / f"{dms_id}.npz")
    v = assay.variants.copy()
    v["score"] = lp.score(v["mutant"])
    return v


def spearman_table(cache_name: str, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-assay Spearman (all variants, and single mutants only) for one cache (e.g. '8M_masked')."""
    ref = ref if ref is not None else reference()
    cache_dir = CACHE / cache_name
    rows = []
    for path in sorted(cache_dir.glob("*.npz")):
        v = score_assay(path.stem, cache_dir, ref)
        singles = v[v["n_mutations"] == 1]
        lp = LogProbs.load(path)
        rows.append({"DMS_id": path.stem, "spearman": spearmanr(v["score"], v["DMS_score"]).statistic,
                     "spearman_singles": spearmanr(singles["score"], singles["DMS_score"]).statistic,
                     "n": len(v), "n_singles": len(singles), "wt_loglik": lp.wildtype_log_likelihood()})
    return pd.DataFrame(rows).set_index("DMS_id")
