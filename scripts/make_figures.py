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
PREDICTOR = {"zero-shot": "#6b6a66", "one-hot": "#eb6834", "augmented": "#2a78d6"}
INK, MUTED, GRID = "#1d1d1b", "#6b6a66", "#e1e0d9"
FUNCTION_NAMES = {"OrganismalFitness": "Organismal fitness"}

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
    """Per-assay accuracy: the cheap scoring against ProteinGym's published masked-marginal numbers."""
    have = [s for s in OWN if f"mine_{s}_wt" in df]
    fig, axes = plt.subplots(1, len(have) + 1, figsize=(3.0 * (len(have) + 1), 3.1))
    for ax, s in zip(axes, have):
        x, y = df[f"published_{s}"], df[f"mine_{s}_wt"]
        ax.plot([-0.3, 0.95], [-0.3, 0.95], color=GRID, lw=1.2, zorder=0)
        ax.scatter(x, y, s=9, color=SIZE_COLOUR[s], alpha=0.75, linewidths=0)
        ax.set(title=f"ESM-2 {s}", xlabel="published (masked)", xlim=(-0.3, 0.95), ylim=(-0.3, 0.95))
        ax.set_aspect("equal")
        ax.text(0.04, 0.93, f"median gain {np.median(y - x):+.3f}", transform=ax.transAxes, fontsize=8,
                color=MUTED)
    axes[0].set_ylabel("this project (wild-type)")
    ax = axes[-1]
    masked = [s for s in OWN if f"mine_{s}_masked" in df and df[f"mine_{s}_masked"].notna().all()]
    for s in masked:
        d = (df[f"mine_{s}_masked"] - df[f"published_{s}"]).abs()
        ax.hist(d, bins=np.linspace(0, 0.002, 21), color=SIZE_COLOUR[s], label=f"{s} masked")
    ax.set(title="Masked marginal: my score vs published", xlabel="absolute difference in Spearman",
           ylabel="assays")
    ax.grid(axis="x", visible=False)
    if masked:
        ax.legend(fontsize=8)
    fig.suptitle("Reproducing ProteinGym, and what the one-pass score costs in accuracy", x=0.01, ha="left",
                 y=1.03)
    save(fig, "reproduction.png")


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
    for s in [x for x in OWN if f"pll_single_{s}" in df]:
        x, y = df[f"pll_single_{s}"], df[f"published_{s}"]
        ax.scatter(x, y, s=7, color=SIZE_COLOUR[s], alpha=0.5, linewidths=0)
        g = np.linspace(x.min(), x.max(), 100)
        ax.plot(g, np.polyval(np.polyfit(x, y, 2), g), color=SIZE_COLOUR[s], label=f"ESM-2 {s}")
    ax.legend(fontsize=8, loc="upper left")
    ax.set(xlabel="wild-type pseudo-log-likelihood", ylabel="Spearman", title="Each model against its own PLL")

    ax = axes[1]
    if "published_15B" in df:
        x = df["pll_single_650M"]
        d = df["published_15B"] - df["published_650M"]
        ax.axhline(0, color=MUTED, lw=1)
        for taxon, colour in zip(("Human", "Eukaryote", "Prokaryote", "Virus"), RAMP[1:]):
            m = df["taxon"] == taxon
            ax.scatter(x[m], d[m], s=9, color=colour, alpha=0.8, linewidths=0, label=taxon)
        g = np.linspace(x.min(), x.max(), 100)
        ax.plot(g, np.polyval(np.polyfit(x, d, 1), g), color=INK, lw=1.5)
        ax.legend(fontsize=8, loc="lower left", ncol=2)
        ax.set(xlabel="650M's wild-type PLL", ylabel="Spearman, 15B - 650M",
               title="Where the biggest model helps")

    ax = axes[2]
    pairs = [(f"pll_exact_{s}", f"pll_single_mutated_{s}", s) for s in OWN
             if f"pll_exact_{s}" in df and df[f"pll_exact_{s}"].notna().any()]
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
                labels.append(FUNCTION_NAMES.get(name, name).replace("log_", "").replace("_", " "))
                colours.append(colour)
        y = np.arange(len(rows))[::-1]
        ax.axvline(0, color=MUTED, lw=1)
        for yi, v, c in zip(y, rows, colours):
            ax.plot([v["q05"], v["q95"]], [yi, yi], color=c, lw=2.5, solid_capstyle="round")
            ax.plot([v["mean"]], [yi], "o", color=c, ms=6)
        ax.set_yticks(y, labels, fontsize=9)
        ax.set(xlabel="effect on Fisher z of Spearman", title=title)
        ax.grid(axis="y", visible=False)
        ax.text(0.98, 0.02, f"R-hat max {s['max_rhat']:.2f}", transform=ax.transAxes, ha="right", fontsize=8,
                color=MUTED)
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
    if df is not None:
        reproduction_figure(df)
        preference_figure(df)
    if scaling is not None:
        scaling_figure(scaling)
    hierarchy_figure()
    if summary is not None:
        fewshot_figure(summary)


if __name__ == "__main__":
    main()
