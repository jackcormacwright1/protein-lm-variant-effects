"""Zero-shot accuracy: reproduction, scaling, and which assays a protein language model gets right.

Reads the score caches written by score_assays.py and ProteinGym's published results, and writes:

  results/assays.csv          one row per assay: features, published Spearman for every ESM-2 size, this
                              project's Spearman (wild-type and masked marginals) and wild-type pseudo-likelihoods
  results/scaling.csv         ProteinGym-style aggregate Spearman by model size, overall and by function,
                              taxon and alignment depth, published and reproduced
  results/hierarchy_650M.json posterior summaries of the partially pooled model of ESM-2 650M's accuracy
  results/hierarchy_scale.json the same for the change from 650M to 15B

Usage:  python scripts/run_analysis.py [--skip-model] [--reuse]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plm.data import PUBLISHED, reference  # noqa: E402
from plm.evaluate import CACHE, aggregate, bootstrap_se, by_function, spearman_tables  # noqa: E402
from plm.scoring import LogProbs  # noqa: E402

RESULTS = ROOT / "results"
SIZES = ["8M", "35M", "150M", "650M", "3B", "15B"]
OWN = ["8M", "35M", "150M", "650M"]       # sizes scored here
PARAMS = {"8M": 8e6, "35M": 35e6, "150M": 150e6, "650M": 650e6, "3B": 3e9, "15B": 15e9}


def pll_table(ref: pd.DataFrame) -> pd.DataFrame:
    """The models' preference for each wild-type sequence. Over every position: the single-pass PLL and the
    plain unmasked log-likelihood. Where masked marginals exist: the direct PLL over the mutated positions and,
    to check against it, the two single-pass measures over the same positions."""
    cols = {}
    for s in OWN:
        wt = {p.stem: LogProbs.load(p) for p in sorted((CACHE / f"{s}_wt").glob("*.npz"))}
        cols[f"pll_single_{s}"] = {k: v.single_pass_pll() for k, v in wt.items()}
        cols[f"loglik_unmasked_{s}"] = {k: v.wildtype_log_likelihood() for k, v in wt.items()}
        for p in sorted((CACHE / f"{s}_masked").glob("*.npz")):
            m = LogProbs.load(p)
            cols.setdefault(f"pll_exact_{s}", {})[p.stem] = m.wildtype_log_likelihood()
            cols.setdefault(f"pll_single_mutated_{s}", {})[p.stem] = wt[p.stem].single_pass_pll(m.positions)
            cols.setdefault(f"loglik_unmasked_mutated_{s}", {})[p.stem] = \
                wt[p.stem].wildtype_log_likelihood(m.positions)
    return pd.DataFrame(cols).reindex(ref.index)


def build_assay_table(ref: pd.DataFrame) -> pd.DataFrame:
    pub = pd.read_csv(PUBLISHED).set_index("DMS ID")
    df = ref[["UniProt_ID", "coarse_selection_type", "taxon", "seq_len", "MSA_Neff_L", "MSA_Neff_L_category",
              "DMS_total_number_mutants", "DMS_number_single_mutants"]].copy()
    for s in SIZES:
        df[f"published_{s}"] = pub[f"ESM2 ({s})"]
    df["published_site_independent"] = pub["Site-Independent"]
    df["published_eve"] = pub["EVE (ensemble)"]
    caches = sorted(p.name for p in CACHE.iterdir() if p.is_dir() and p.name.split("_")[-1] in ("wt", "masked"))
    for name, t in spearman_tables(caches, ref).items():
        df[f"mine_{name}"] = t["spearman"]
        df[f"singles_{name}"] = t["spearman_singles"]
        print(f"  scored {name}: {len(t)} assays", flush=True)
    return df.join(pll_table(ref))


def scaling_table(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("All", None)] + [(f, ("coarse_selection_type", f)) for f in sorted(df["coarse_selection_type"].unique())] \
        + [(t, ("taxon", t)) for t in sorted(df["taxon"].unique())] \
        + [(f"{c} MSA depth", ("MSA_Neff_L_category", c)) for c in ("Low", "Medium", "High")]
    for s in SIZES:
        for source, col in (("published (masked)", f"published_{s}"), ("this project (wild-type)", f"mine_{s}_wt"),
                            ("this project (masked)", f"mine_{s}_masked")):
            if col not in df or df[col].notna().sum() < len(df):
                continue
            for label, cond in groups:
                sub = df if cond is None else df[df[cond[0]] == cond[1]]
                if cond is None:
                    val = aggregate(sub[col], ref)
                elif cond[0] == "coarse_selection_type":
                    val = by_function(df[col], ref)[cond[1]]
                else:
                    val = aggregate(sub[col], ref)
                rows.append({"size": s, "params": PARAMS[s], "source": source, "group": label, "spearman": val,
                             "se": bootstrap_se(df[col], ref, n=300) if cond is None else np.nan})
    return pd.DataFrame(rows)


def posterior_summary(idata, d) -> dict:
    post = idata.posterior
    out = {}

    def summ(x):
        x = np.asarray(x).ravel()
        return {"mean": float(x.mean()), "q05": float(np.quantile(x, 0.05)), "q95": float(np.quantile(x, 0.95)),
                "p_positive": float((x > 0).mean())}

    alpha = post["alpha"].values
    out["alpha"] = summ(alpha)
    out["baseline_spearman"] = summ(np.tanh(alpha))
    for name, coord in (("function_effect", "function"), ("taxon_effect", "taxon")):
        vals = post[name].values
        out[name] = {str(k): summ(vals[..., i]) for i, k in enumerate(post[coord].values)}
        # what each category's typical assay scores, other features at their average
        out[name + "_spearman"] = {str(k): summ(np.tanh(alpha + vals[..., i])) for i, k in enumerate(post[coord].values)}
    g = post["g"].values
    out["g"] = {str(k): summ(g[..., i]) for i, k in enumerate(post["feature"].values)}
    spreads = [k for k in ("tau_function", "tau_taxon", "tau_protein", "sigma") if k in post]
    for k in spreads:
        out[k] = summ(post[k].values)
    out["x_mean"] = dict(zip(d.x_names, map(float, d.x_mean)))
    out["x_sd"] = dict(zip(d.x_names, map(float, d.x_sd)))
    out["n_assays"] = int(len(d.z))
    # the turning point of the quadratic feature, back on its original scale
    for j, name in enumerate(d.x_names):
        if not name.endswith("^2"):
            continue
        i = d.x_names.index(name[:-2])
        peak = -g[..., i] / (2 * g[..., j])
        inside = (peak > -2) & (peak < 2) & (g[..., j] < 0)      # a maximum within the observed range
        v = {"share_with_interior_maximum": float(inside.mean())}
        if inside.any():
            v.update(summ(d.x_mean[i] + d.x_sd[i] * peak[inside]))
        out[f"peak_{name[:-2]}"] = v
    import arviz as az
    summary = az.summary(idata, var_names=["alpha", "g"] + spreads)
    out["max_rhat"] = float(summary["r_hat"].max())
    out["min_ess_bulk"] = float(summary["ess_bulk"].min())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-model", action="store_true", help="tables only, no hierarchical model")
    ap.add_argument("--reuse", action="store_true", help="reuse results/assays.csv instead of rescoring")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    ref = reference()
    if args.reuse and (RESULTS / "assays.csv").exists():
        df = pd.read_csv(RESULTS / "assays.csv", index_col=0)
    else:
        df = build_assay_table(ref)
        df["log_neff_l"] = np.log10(df["MSA_Neff_L"] + 1.0)
        df["log_length"] = np.log10(df["seq_len"])
        df.to_csv(RESULTS / "assays.csv")
        scaling = scaling_table(df, ref)
        scaling.to_csv(RESULTS / "scaling.csv", index=False)
        print(scaling[scaling["group"] == "All"].pivot(index="size", columns="source", values="spearman")
              .reindex(SIZES).round(3).to_string())
    if args.skip_model:
        return

    from plm.hierarchy import design, fit
    df = df.assign(n=df["DMS_total_number_mutants"])
    feats = ["log_neff_l", "log_length", "pll_single_650M"]
    d = design(df, "published_650M", "n", feats, squares=("pll_single_650M",))
    _, idata = fit(d)
    summary = posterior_summary(idata, d)
    (RESULTS / "hierarchy_650M.json").write_text(json.dumps(summary, indent=1))
    protein_eff = idata.posterior["protein_effect"].values.reshape(-1, d.protein.max() + 1).mean(axis=0)
    pd.Series(protein_eff, index=pd.factorize(df["UniProt_ID"], sort=True)[1],
              name="protein_effect").to_csv(RESULTS / "protein_effects_650M.csv")
    report("ESM-2 650M", summary)

    # the same without protein effects, to show how much the group effects rest on repeated proteins
    _, idata_np = fit(d, protein_effects=False)
    (RESULTS / "hierarchy_650M_no_protein.json").write_text(
        json.dumps(posterior_summary(idata_np, d), indent=1))

    # what changes from 650M to 15B: same structure, outcome the difference in Fisher z
    z15, se15 = np.arctanh(df["published_15B"].clip(-0.999, 0.999)), np.sqrt(1.06 / (df["n"] - 3))
    df["delta_15B"] = np.tanh(z15 - np.arctanh(df["published_650M"].clip(-0.999, 0.999)))
    d2 = design(df, "delta_15B", "n", feats, squares=("pll_single_650M",))
    d2.se = np.sqrt(2) * se15.to_numpy()
    _, idata2 = fit(d2)
    summary2 = posterior_summary(idata2, d2)
    (RESULTS / "hierarchy_scale.json").write_text(json.dumps(summary2, indent=1))
    report("650M to 15B", summary2)


def report(title: str, s: dict) -> None:
    print(f"\n{title}: {s['n_assays']} assays, R-hat <= {s['max_rhat']:.3f}, ESS >= {s['min_ess_bulk']:.0f}")
    print(f"  baseline Spearman {s['baseline_spearman']['mean']:+.3f}")
    for key in ("g", "function_effect", "taxon_effect"):
        for name, v in s[key].items():
            print(f"  {name:28s} {v['mean']:+.3f} [{v['q05']:+.3f}, {v['q95']:+.3f}]  p+ {v['p_positive']:.2f}")
    for k, v in s.items():
        if k.startswith("peak_") and "mean" in v:
            print(f"  {k} {v['mean']:+.2f} [{v['q05']:+.2f}, {v['q95']:+.2f}] "
                  f"(interior maximum in {v['share_with_interior_maximum']:.0%} of draws)")


if __name__ == "__main__":
    main()
