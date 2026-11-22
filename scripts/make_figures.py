"""Draw the charts in results/ from the results files (skips any whose inputs are missing).

Usage:  python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
SIZES = ["8M", "35M", "150M", "650M", "3B", "15B"]
OWN = ["8M", "35M", "150M", "650M"]
PARAMS = {"8M": 8e6, "35M": 35e6, "150M": 150e6, "650M": 650e6, "3B": 3e9, "15B": 15e9}
# one hue, light to dark: model size is a magnitude
RAMP = ["#bcd4ee", "#8fb8e2", "#5d97d4", "#2a78d6", "#1b56a0", "#123a6e"]
SIZE_COLOUR = dict(zip(SIZES, RAMP))
SCHEME = {"random": "#2a78d6", "modulo": "#eb6834", "contiguous": "#1baf7a"}
TAXON = {"Human": "#2a78d6", "Eukaryote": "#eb6834", "Prokaryote": "#1baf7a", "Virus": "#6250d6"}
PREDICTOR = {"zero-shot": "#6b6a66", "one-hot": "#eb6834", "augmented": "#2a78d6"}
INK, MUTED, GRID = "#1d1d1b", "#6b6a66", "#e1e0d9"
FUNCTION_NAMES = {"OrganismalFitness": "Organismal fitness"}
FEATURE_NAMES = {"log_neff_l": "Alignment depth", "log_length": "Sequence length",
                 "pll_single_650M": "Wild-type preference", "pll_single_650M^2": "Preference, squared"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
    "lines.linewidth": 2.0,
})


def save(fig, name):
    fig.savefig(RESULTS / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def read(name, index_col=0):
    path = RESULTS / name
    return pd.read_csv(path, index_col=index_col) if path.exists() else None


def reproduction_figure(df):
    """Per-assay accuracy: the one-pass score against ProteinGym's published masked-marginal numbers."""
    have = [x for x in OWN if f"mine_{x}_wt" in df]
    fig, axes = plt.subplots(1, len(have) + 1, figsize=(2.6 * (len(have) + 1), 2.9))
    for ax, size in zip(axes, have):
        x, y = df[f"published_{size}"], df[f"mine_{size}_wt"]
        ax.plot([-0.3, 0.95], [-0.3, 0.95], color=GRID, lw=1.2, zorder=0)
        ax.scatter(x, y, s=8, color=SIZE_COLOUR[size], alpha=0.75, linewidths=0)
        ax.set(title=f"ESM-2 {size}", xlabel="published (masked)", xlim=(-0.3, 0.95), ylim=(-0.3, 0.95))
        ax.set_aspect("equal")
    axes[0].set_ylabel("this project (one pass)")
    ax = axes[-1]
    ax.axvline(0, color=MUTED, lw=1)
    for size in have:
        d = (df[f"mine_{size}_wt"] - df[f"published_{size}"]).dropna()
        ax.hist(d, bins=np.linspace(-0.2, 0.4, 61), histtype="step", lw=1.8, color=SIZE_COLOUR[size],
                label=f"{size}: mean {d.mean():+.3f}, better on {(d > 0).mean():.0%}")
    ax.set(title="One pass − masked, per assay", xlabel="difference in Spearman", ylabel="assays")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("The one-pass score: a little worse on most assays, better on average", x=0.01, ha="left",
                 y=1.04)
    save(fig, "reproduction.png")


def scoring_methods_figure(cmp):
    """Where the two ways of scoring differ: single mutants, and assays with multiple substitutions."""
    sizes = [x for x in OWN if x in set(cmp["size"])]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    ax = axes[0]
    lims = (-0.1, 0.9)
    ax.plot(lims, lims, color=GRID, lw=1.2, zorder=0)
    for size in sizes:
        g = cmp[cmp["size"] == size]
        ax.scatter(g["masked_singles"], g["wt_singles"], s=11, color=SIZE_COLOUR[size], alpha=0.75,
                   linewidths=0, label=f"{size} ({g['wt_singles'].notna().sum()} assays)")
    ax.set(xlim=lims, ylim=lims, xlabel="masked marginal", ylabel="wild-type marginal",
           title="Single mutants only")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    ax.axhline(0, color=MUTED, lw=1)
    for size in sizes:
        g = cmp[cmp["size"] == size]
        ax.scatter(g["share_multiple"], g["wt_all"] - g["masked_all"], s=11, color=SIZE_COLOUR[size],
                   alpha=0.8, linewidths=0, label=size)
    ax.set(xlabel="share of variants with several substitutions", ylabel="wild-type − masked (Spearman)",
           title="All variants")

    ax = axes[2]
    g = cmp[(cmp["size"] == sizes[0]) & (cmp["share_multiple"] > 0.2)]
    x = np.arange(3)
    vals = [g["measured_vs_count"].mean(), g["wt_vs_count"].mean(), g["masked_vs_count"].mean()]
    ax.bar(x, vals, color=[MUTED, SIZE_COLOUR[sizes[0]], "#eb6834"], width=0.6)
    ax.set_xticks(x, ["measured\nfitness", "wild-type\nmarginal", "masked\nmarginal"], fontsize=9)
    ax.set(ylabel="rank correlation with the number\nof substitutions",
           title=f"Why: {sizes[0]}, {len(g)} mixed assays")
    ax.grid(axis="x", visible=False)
    fig.suptitle("The two scores agree on single mutants and differ on multiple ones", x=0.01, ha="left",
                 y=1.03)
    save(fig, "scoring_methods.png")


def scaling_figure(scaling):
    """Aggregate accuracy against model size: overall, by function, by taxon."""
    panels = [("All", [("All", None)]),
              ("By function", [(f, None) for f in ("Stability", "Activity", "Expression", "Binding",
                                                   "OrganismalFitness")]),
              ("By taxon", [(t, None) for t in ("Human", "Eukaryote", "Prokaryote", "Virus")])]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
    pub = scaling[scaling["source"] == "published (masked)"]
    mine = scaling[scaling["source"] == "this project (wild-type)"]
    ax = axes[0]
    for src, label, style in ((pub, "published (masked marginal)", "-"), (mine, "this project (one pass)", "--")):
        d = src[src["group"] == "All"].set_index("size").reindex(SIZES).dropna(subset=["spearman"])
        ax.errorbar([PARAMS[s] for s in d.index], d["spearman"], yerr=d["se"], ls=style, color=INK,
                    marker="o", ms=5, capsize=3, label=label)
    ax.legend(fontsize=8, loc="lower right")
    ax.set(ylabel="Spearman (ProteinGym average)")
    for ax, (title, groups) in zip(axes[1:], panels[1:]):
        for name, _ in groups:
            d = pub[pub["group"] == name].set_index("size").reindex(SIZES).dropna(subset=["spearman"])
            if d.empty:
                continue
            ax.plot([PARAMS[s] for s in d.index], d["spearman"], marker="o", ms=4,
                    label=FUNCTION_NAMES.get(name, name))
        ax.legend(fontsize=8, loc="lower right", ncol=1)
        ax.set_title(title)
    for ax in axes:
        ax.set(xscale="log", xlabel="parameters")
    axes[0].set_title("All assays")
    fig.suptitle("Accuracy against model size: the gains stop at 650M", x=0.01, ha="left", y=1.03)
    save(fig, "scaling.png")


def preference_figure(df):
    """Accuracy against how strongly the model prefers the wild-type sequence."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    ax = axes[0]
    sizes = [x for x in OWN if f"pll_single_{x}" in df]
    big = sizes[-1]
    ax.scatter(df[f"pll_single_{big}"], df[f"published_{big}"], s=9, color="#d6e3f2", linewidths=0, zorder=0)
    for size in sizes:
        x, y = df[f"pll_single_{size}"], df[f"published_{size}"]
        g = np.linspace(x.quantile(0.01), x.quantile(0.99), 100)
        c = np.polyfit(x, y, 2)
        ax.plot(g, np.polyval(c, g), color=SIZE_COLOUR[size], label=f"ESM-2 {size}")
        peak = -c[1] / (2 * c[0])
        if c[0] < 0 and g[0] < peak < g[-1]:
            ax.plot([peak], [np.polyval(c, peak)], "o", color=SIZE_COLOUR[size], ms=5, mec="white", mew=1.2)
    ax.legend(fontsize=8, loc="upper left")
    ax.set(xlabel="wild-type pseudo-log-likelihood", ylabel="Spearman",
           title=f"Fitted curves; points are {big}")

    ax = axes[1]
    if "published_15B" in df:
        x = df["pll_single_650M"]
        d = df["published_15B"] - df["published_650M"]
        ax.axhline(0, color=MUTED, lw=1)
        for taxon, colour in TAXON.items():
            m = df["taxon"] == taxon
            ax.scatter(x[m], d[m], s=11, color=colour, alpha=0.85, linewidths=0, label=taxon)
        g = np.linspace(x.min(), x.max(), 100)
        ax.plot(g, np.polyval(np.polyfit(x, d, 1), g), color=INK, lw=1.5)
        ax.legend(fontsize=8, loc="lower left", ncol=2)
        ax.set(xlabel="650M's wild-type PLL", ylabel="Spearman, 15B - 650M",
               title="Where the biggest model helps")

    ax = axes[2]
    pairs = [(f"pll_exact_{x}", f"pll_single_mutated_{x}", x) for x in OWN
             if f"pll_exact_{x}" in df and df[f"pll_exact_{x}"].notna().any()]
    for exact, single, s in pairs:
        m = df[exact].notna()
        ax.scatter(df.loc[m, exact], df.loc[m, single], s=9, color=SIZE_COLOUR[s], alpha=0.7, linewidths=0,
                   label=f"{s} ({m.sum()} assays)")
    lims = (-3.6, 0.1)
    ax.plot(lims, lims, color=GRID, lw=1.2, zorder=0)
    ax.set(xlabel="PLL, one pass per position", ylabel="PLL from the single pass", xlim=lims, ylim=lims,
           title="The single-pass approximation")
    ax.set_aspect("equal")
    if pairs:
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Preference for the natural sequence: more is better, until it is not", x=0.01, ha="left",
                 y=1.03)
    save(fig, "preference.png")


def hierarchy_figure():
    """Posterior effects from the partially pooled models."""
    files = [("hierarchy_650M.json", "ESM-2 650M's accuracy"), ("hierarchy_scale.json", "650M to 15B")]
    have = [(json.loads((RESULTS / f).read_text()), t) for f, t in files if (RESULTS / f).exists()]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(5.6 * len(have), 4.0), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, (s, title) in zip(axes, have):
        rows, labels, colours = [], [], []
        for key, group, colour in (("function_effect", "function", RAMP[3]),
                                   ("taxon_effect", "taxon", RAMP[1]), ("g", "feature", "#eb6834")):
            for name, v in s[key].items():
                rows.append(v)
                labels.append(FEATURE_NAMES.get(name, FUNCTION_NAMES.get(name, name)))
                colours.append(colour)
        y = np.arange(len(rows))[::-1]
        ax.axvline(0, color=MUTED, lw=1)
        for yi, v, c in zip(y, rows, colours):
            ax.plot([v["q05"], v["q95"]], [yi, yi], color=c, lw=2.5, solid_capstyle="round")
            ax.plot([v["mean"]], [yi], "o", color=c, ms=6)
        ax.set_yticks(y, labels, fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"effect on Fisher z of Spearman   (R-hat at most {s['max_rhat']:.3f}, "
                      f"{s['n_assays']} assays)")
    fig.suptitle("What predicts accuracy, and what predicts the gain from scale", x=0.01, ha="left", y=1.02)
    save(fig, "hierarchy.png")


def fewshot_figure(summary):
    """Learning curves by split scheme."""
    schemes = ["random", "modulo", "contiguous"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True)
    for ax, scheme in zip(axes, schemes):
        d = summary[summary["scheme"] == scheme]
        if d.empty:
            continue
        for model in ("one-hot", "augmented"):
            m = d[d["model"] == model].sort_values("n_train")
            ax.plot(m["n_train"], m["spearman"], marker="o", ms=4, color=PREDICTOR[model], label=model)
        z = d[d["model"] == "augmented"].sort_values("n_train")
        ax.plot(z["n_train"], z["zero_shot"], color=PREDICTOR["zero-shot"], ls="--",
                label="zero-shot (same assays)")
        ax.set(xscale="log", xlabel="measured variants used for training", title=scheme)
    axes[0].set_ylabel("Spearman (ProteinGym average)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("How many measurements it takes to beat the zero-shot score depends on the split", x=0.01,
                 ha="left", y=1.03)
    save(fig, "fewshot.png")


def main():
    RESULTS.mkdir(exist_ok=True)
    df = read("assays.csv")
    scaling, summary = read("scaling.csv", None), read("fewshot_summary.csv", None)
    cmp = read("scoring_methods.csv", None)
    if df is not None:
        reproduction_figure(df)
        preference_figure(df)
    if cmp is not None:
        scoring_methods_figure(cmp)
    if scaling is not None:
        scaling_figure(scaling)
    hierarchy_figure()
    if summary is not None:
        fewshot_figure(summary)


if __name__ == "__main__":
    main()
