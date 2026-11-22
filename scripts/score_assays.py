"""Score ProteinGym's substitution assays with an ESM-2 model.

Saves each assay's log-probability table to data/cache/<model>_<strategy>/<DMS_id>.npz (skipping assays
already done, so the run can be stopped and resumed), and logs the time taken to data/cache/timings.csv.

Usage:
  python scripts/score_assays.py --model 8M --strategy masked
  python scripts/score_assays.py --model 650M --strategy wt
  python scripts/score_assays.py --model 150M --strategy masked --max-length 250   (short proteins only)

Weights are read from --weights-dir (default ../esm_weights), as downloaded from
https://dl.fbaipublicfiles.com/fair-esm/models/<name>.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plm.data import DATA, load, reference  # noqa: E402
from plm.scoring import MODELS, Scorer  # noqa: E402

CACHE = DATA / "cache"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--strategy", choices=["wt", "masked"], required=True)
    ap.add_argument("--weights-dir", type=Path, default=ROOT.parent / "esm_weights")
    ap.add_argument("--max-length", type=int, default=None, help="only proteins up to this length")
    ap.add_argument("--assays", nargs="*", help="DMS ids (default: all)")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    ref = reference()
    ids = args.assays or list(ref.sort_values("seq_len").index)
    if args.max_length:
        ids = [i for i in ids if ref.loc[i, "seq_len"] <= args.max_length]
    out_dir = CACHE / f"{args.model}_{args.strategy}"
    todo = [i for i in ids if not (out_dir / f"{i}.npz").exists()]
    print(f"{args.model} {args.strategy}: {len(ids)} assays, {len(todo)} to do", flush=True)
    if not todo:
        return
    scorer = Scorer(args.weights_dir / f"{MODELS[args.model]}.pt", threads=args.threads)
    t0 = time.time()
    timings = CACHE / "timings.csv"
    for n, dms_id in enumerate(todo, 1):
        assay = load(dms_id, ref)
        lp, secs = scorer.score_assay(assay, args.strategy)
        lp.save(out_dir / f"{dms_id}.npz")
        new = not timings.exists()
        with open(timings, "a") as f:
            if new:
                f.write("model,strategy,DMS_id,seq_len,positions,seconds\n")
            f.write(f"{args.model},{args.strategy},{dms_id},{len(assay.sequence)},{len(lp.positions)},{secs:.2f}\n")
        if n % 10 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
