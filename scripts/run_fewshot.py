"""Learning curves: supervised models trained on n measured single mutants against the zero-shot score.

For every assay, each of ProteinGym's three five-fold splits (random, modulo, contiguous) and each training
size, trains ridge regressions on n variants drawn from four folds and tests on the fifth. The zero-shot score
is ESM-2 650M's wild-type marginal (data/cache/650M_wt). Writes results/fewshot.csv (one row per assay,
split, fold, size and predictor) and results/fewshot_summary.csv (the learning curves, averaged in
ProteinGym's scheme over the assays large enough for each training size).

Usage:  python scripts/run_fewshot.py [--workers 2] [--zero-shot 650M_wt]
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plm.data import DATA, load, reference  # noqa: E402
from plm.evaluate import aggregate  # noqa: E402
from plm.fewshot import SCHEMES, learning_curve, summarise  # noqa: E402
from plm.scoring import LogProbs  # noqa: E402

SIZES = (24, 48, 96, 192, 384, 768, 1536)


def job(args):
    dms_id, cache = args
    ref = reference()
    assay = load(dms_id, ref, folds=True)
    zs = LogProbs.load(DATA / "cache" / cache / f"{dms_id}.npz").score(assay.variants["mutant"])
    seed = sum(ord(c) for c in dms_id)
    parts = [learning_curve(assay, zs, SIZES, s, np.random.default_rng(seed)) for s in SCHEMES]
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "DMS_id", dms_id)
    out["n_singles"] = len(assay.variants)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--zero-shot", default="650M_wt")
    args = ap.parse_args()
    ref = reference()
    ids = list(ref.sort_values("DMS_number_single_mutants", ascending=False).index)
    t0 = time.time()
    parts = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(job, [(d, args.zero_shot) for d in ids]), 1):
            parts.append(res)
            if i % 20 == 0 or i == len(ids):
                print(f"  {i}/{len(ids)} assays ({time.time() - t0:.0f}s)", flush=True)
    out = pd.concat(parts, ignore_index=True)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "fewshot.csv", index=False)
    summary = summarise(out, ref, aggregate)
    summary.to_csv(ROOT / "results" / "fewshot_summary.csv", index=False)
    print(summary.pivot(index=["scheme", "n_train"], columns="model",
                        values="spearman").round(3).to_string())
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
