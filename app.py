"""Zero-shot variant effects with ESM-2: a Streamlit dashboard.

Run locally:   streamlit run app.py
It reads the results written by the scripts in scripts/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Protein language models on ProteinGym", page_icon="🧬", layout="wide")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
METHODOLOGY_FILE = ROOT / "docs" / "METHODOLOGY.md"

SIZES = ["8M", "35M", "150M", "650M", "3B", "15B"]
OWN = ["8M", "35M", "150M", "650M"]
PARAMS = {"8M": 8e6, "35M": 35e6, "150M": 150e6, "650M": 650e6, "3B": 3e9, "15B": 15e9}
RAMP = ["#bcd4ee", "#8fb8e2", "#5d97d4", "#2a78d6", "#1b56a0", "#123a6e"]
SIZE_COLOUR = dict(zip(SIZES, RAMP))
PREDICTOR = {"zero-shot": "#6b6a66", "one-hot": "#eb6834", "augmented": "#2a78d6"}
SCHEME_NAMES = {"random": "Random variants", "modulo": "Every fifth position", "contiguous": "Blocks of sequence"}
FUNCTIONS = ["Activity", "Binding", "Expression", "OrganismalFitness", "Stability"]
TAXA = ["Human", "Eukaryote", "Prokaryote", "Virus"]
PRETTY = {"OrganismalFitness": "Organismal fitness", "log_neff_l": "Alignment depth",
          "log_length": "Sequence length", "pll_single_650M": "Wild-type preference",
          "pll_single_650M^2": "Preference, squared"}
NEUTRAL, MUTED, GRID, INK = "#52514e", "#898781", "#e1e0d9", "#1d1d1b"

st.markdown(f"""
<style>
.block-container {{ padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1320px; }}
.eyebrow {{ color: {NEUTRAL}; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
            margin: 0.4rem 0 0.35rem; }}
.lede {{ color: #3a3936; font-size: 1.02rem; margin: -0.4rem 0 0.6rem; max-width: 62rem; }}
.kpi {{ background: #ffffff; border: 1px solid {GRID}; border-radius: 10px; padding: 0.8rem 1rem 0.85rem;
        height: 100%; }}
.kpi .label {{ color: {NEUTRAL}; font-size: 0.86rem; line-height: 1.3; min-height: 2.2rem; }}
.kpi .value {{ color: {INK}; font-size: 2rem; font-weight: 600; line-height: 1.15; margin: 0.15rem 0 0.3rem;
               font-variant-numeric: tabular-nums; white-space: nowrap; }}
.kpi .note {{ color: {MUTED}; font-size: 0.78rem; line-height: 1.35; }}
@media (max-width: 1100px) {{ .kpi .value {{ font-size: 1.45rem; }} }}
.callout {{ background: #ffffff; border: 1px solid {GRID}; border-left: 4px solid #2a78d6; border-radius: 8px;
            padding: 0.85rem 1.1rem; margin: 0.3rem 0 1rem; }}
.chart-note {{ color: {MUTED}; font-size: 0.8rem; margin: -0.3rem 0 0.4rem; }}
</style>
""", unsafe_allow_html=True)


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="chart-note">{text}</div>', unsafe_allow_html=True)


def callout(text: str) -> None:
    """A bordered note. Markdown bold is translated, since the div is raw HTML."""
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    st.markdown(f'<div class="callout">{html}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, text: str) -> str:
    return (f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
            f'<div class="note">{text}</div></div>')


def legend(items: list[tuple[str, str]]) -> None:
    chips = "".join(f'<span style="margin-right:1.2rem;white-space:nowrap"><span style="display:inline-block;'
                    f'width:12px;height:3px;background:{c};vertical-align:middle;margin-right:0.4rem"></span>'
                    f'{name}</span>' for name, c in items)
    st.markdown(f'<div style="color:{NEUTRAL};font-size:0.86rem;margin:0.1rem 0 0.2rem">{chips}</div>',
                unsafe_allow_html=True)


@st.cache_data
def load_csv(name: str, index_col=None) -> pd.DataFrame | None:
    path = RESULTS / name
    return pd.read_csv(path, index_col=index_col) if path.exists() else None


@st.cache_data
def load_json(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def pretty(name: str) -> str:
    return PRETTY.get(name, name)


assays = load_csv("assays.csv", 0)
scaling = load_csv("scaling.csv")
fewshot = load_csv("fewshot_summary.csv")
h650 = load_json("hierarchy_650M.json")
h650_np = load_json("hierarchy_650M_no_protein.json")
hscale = load_json("hierarchy_scale.json")

# ---------------------------------------------------------------- header
eyebrow("Protein language models · benchmarking · hierarchical models")
st.title("What a protein language model knows about mutations")
st.markdown('<p class="lede">ESM-2 is trained only on natural protein sequences, with no measurements at '
            "all, yet the probabilities it assigns to amino acids predict the effects of mutations. On "
            "ProteinGym's 217 deep mutational scans: how well, how cheaply, on which proteins it fails, "
            "whether a bigger model is better, and how many measurements it takes to beat it.</p>",
            unsafe_allow_html=True)

if scaling is not None:
    allrows = scaling[scaling["group"] == "All"].set_index(["source", "size"])["spearman"]
    pub = allrows.xs("published (masked)")
    mine = allrows.xs("this project (wild-type)")
    cols = st.columns(4)
    cards = [
        kpi("Assays scored, four model sizes", "217 × 4",
            "2.5 million variants, scored in one forward pass per protein instead of one per position"),
        kpi("Best published accuracy (650M)", f"{pub['650M']:.3f}",
            "Spearman, ProteinGym's average over functions. The 15B model manages "
            f"{pub['15B']:.3f}"),
        kpi("The one-pass score, same model", f"{mine['650M']:.3f}",
            "a median of 150× cheaper per assay, and ahead at every size: "
            f"{mine['8M'] - pub['8M']:+.3f} at 8M"),
        kpi("Where accuracy peaks", "PLL ≈ −0.8" if h650 else "—",
            "How strongly the model prefers the natural sequence: more is better, until it is not"),
    ]
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)

st.write("")
tabs = st.tabs(["Reproduction", "Is bigger better?", "Where it fails", "Preference", "Few-shot",
                "Methodology"])

# ---------------------------------------------------------------- reproduction
with tabs[0]:
    st.subheader("Does the cheap score cost anything?")
    st.markdown("A substitution's score is how much more likely the model finds the mutant residue than the "
                "wild type. The probabilities can come from masking each mutated position in turn — one "
                "forward pass per position, which is how ProteinGym scores ESM models — or from a single "
                "pass over the unmasked sequence. The second is hundreds of times cheaper.")
    if assays is None:
        st.info("Run `python scripts/run_analysis.py` first.")
    else:
        have = [s for s in OWN if f"mine_{s}_wt" in assays]
        pick = st.radio("Model size", have, index=len(have) - 1, horizontal=True)
        d = assays.reset_index().rename(columns={"index": "DMS_id"})
        d["measured"] = d["coarse_selection_type"].map(lambda f: PRETTY.get(f, f))
        d["published"] = d[f"published_{pick}"]
        d["mine"] = d[f"mine_{pick}_wt"]
        d["difference"] = d["mine"] - d["published"]
        line = pd.DataFrame({"x": [-0.3, 0.95]})
        base = alt.Chart(d).properties(height=340)
        points = base.mark_circle(size=42, opacity=0.75).encode(
            x=alt.X("published:Q", title="published, masked marginal",
                    scale=alt.Scale(domain=[-0.3, 0.95], nice=False)),
            y=alt.Y("mine:Q", title="this project, one pass",
                    scale=alt.Scale(domain=[-0.3, 0.95], nice=False)),
            color=alt.Color("measured:N", title="measured", scale=alt.Scale(
                domain=[PRETTY.get(f, f) for f in FUNCTIONS],
                range=["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100"])),
            tooltip=[alt.Tooltip("DMS_id:N", title="assay"), alt.Tooltip("published:Q", format=".3f"),
                     alt.Tooltip("mine:Q", format=".3f"), alt.Tooltip("n:Q", title="variants", format=",")]
            if "n" in d else ["DMS_id:N"])
        diag = alt.Chart(line).mark_line(color=GRID).encode(x="x:Q", y="x:Q")
        cols = st.columns([3, 2])
        cols[0].altair_chart(diag + points, use_container_width=True)
        hist = alt.Chart(d).mark_bar(color="#2a78d6").encode(
            x=alt.X("difference:Q", bin=alt.Bin(maxbins=40), title="one pass − masked (Spearman)"),
            y=alt.Y("count():Q", title="assays")).properties(height=340)
        rule = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color=INK).encode(x="x:Q")
        cols[1].altair_chart(hist + rule, use_container_width=True)
        agg = scaling[(scaling["group"] == "All")].pivot(index="size", columns="source", values="spearman")
        better = int((d["difference"] > 0).sum())
        callout(f"At **{pick}** the one-pass score is better on **{better} of {len(d)} assays** but ahead "
                "in the aggregate by "
                f"**{agg.loc[pick, 'this project (wild-type)'] - agg.loc[pick, 'published (masked)']:+.3f}**: "
                "a small loss on most assays and a large gain on a few. The next panel shows which.")

        cmp = load_csv("scoring_methods.csv")
        if cmp is not None:
            st.subheader("Where the two scores part company")
            st.markdown("On the assays scored both ways, separating single mutants from variants carrying "
                        "several substitutions.")
            sizes = [x for x in OWN if x in set(cmp["size"])]
            e = cmp.copy()
            e["gain"] = e["wt_all"] - e["masked_all"]
            left = alt.Chart(e).mark_circle(size=40, opacity=0.75).encode(
                x=alt.X("masked_singles:Q", title="masked marginal", scale=alt.Scale(nice=False)),
                y=alt.Y("wt_singles:Q", title="wild-type marginal", scale=alt.Scale(nice=False)),
                color=alt.Color("size:N", title="ESM-2", scale=alt.Scale(
                    domain=sizes, range=[SIZE_COLOUR[x] for x in sizes])),
                tooltip=["DMS_id:N", "size:N", alt.Tooltip("masked_singles:Q", format=".3f"),
                         alt.Tooltip("wt_singles:Q", format=".3f")]).properties(
                height=320, title="Single mutants only")
            diag = alt.Chart(pd.DataFrame({"x": [-0.1, 0.9]})).mark_line(color=GRID).encode(x="x:Q", y="x:Q")
            right = (alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color=INK).encode(y="y:Q")
                     + alt.Chart(e).mark_circle(size=40, opacity=0.75).encode(
                         x=alt.X("share_multiple:Q", title="share of variants with several substitutions"),
                         y=alt.Y("gain:Q", title="wild-type − masked (Spearman)"),
                         color=alt.Color("size:N", title="ESM-2", scale=alt.Scale(
                             domain=sizes, range=[SIZE_COLOUR[x] for x in sizes])),
                         tooltip=["DMS_id:N", "size:N", alt.Tooltip("share_multiple:Q", format=".2f"),
                                  alt.Tooltip("gain:Q", format="+.3f")])).properties(
                height=320, title="All variants")
            cols = st.columns(2)
            cols[0].altair_chart(diag + left, use_container_width=True)
            cols[1].altair_chart(right, use_container_width=True)
            rows = []
            for size in sizes:
                g = cmp[cmp["size"] == size]
                heavy = g["share_multiple"] > 0.2
                rows.append({"ESM-2": size, "assays": len(g),
                             "singles, wild-type": g["wt_singles"].mean(),
                             "singles, masked": g["masked_singles"].mean(),
                             "with multiple mutants, wild-type": g.loc[heavy, "wt_all"].mean(),
                             "with multiple mutants, masked": g.loc[heavy, "masked_all"].mean(),
                             "measured vs count": g.loc[heavy, "measured_vs_count"].mean(),
                             "wild-type vs count": g.loc[heavy, "wt_vs_count"].mean(),
                             "masked vs count": g.loc[heavy, "masked_vs_count"].mean()})
            t = pd.DataFrame(rows).set_index("ESM-2")
            st.dataframe(t.style.format({c: "{:.3f}" for c in t.columns if c != "assays"}),
                         use_container_width=True)
            callout("The two agree on single mutants at every size. The gap opens on assays with multiple "
                    "substitutions, and closes as the model grows. The last three columns say why: "
                    "measured fitness falls with the number of substitutions, the one-pass score tracks "
                    "that strongly whatever the size, and the masked score tracks it weakly in the "
                    "smallest model and increasingly well in the larger ones. Masking a position leaves a "
                    "small model very unsure of the residue that belongs there, so each substitution "
                    "subtracts little and a sum over several of them barely registers how many there are.")
        masked = [x for x in OWN if f"mine_{x}_masked" in assays and assays[f"mine_{x}_masked"].notna().any()]
        if masked:
            rows = []
            for size in masked:
                diff = (assays[f"mine_{size}_masked"] - assays[f"published_{size}"]).abs().dropna()
                rows.append({"size": size, "assays": len(diff), "largest difference": diff.max(),
                             "mean difference": diff.mean()})
            st.markdown("**The reproduction check.** Scoring the way ProteinGym does reproduces its "
                        "published per-assay numbers:")
            st.dataframe(pd.DataFrame(rows).style.format({"largest difference": "{:.5f}",
                                                          "mean difference": "{:.5f}"}), hide_index=True)

# ---------------------------------------------------------------- scaling
with tabs[1]:
    st.subheader("Accuracy against model size")
    if scaling is None:
        st.info("Run `python scripts/run_analysis.py` first.")
    else:
        groups = st.selectbox("Break down by", ["Nothing", "What was measured", "Taxon", "Alignment depth"])
        key = {"Nothing": ["All"], "What was measured": FUNCTIONS, "Taxon": TAXA,
               "Alignment depth": ["Low MSA depth", "Medium MSA depth", "High MSA depth"]}[groups]
        d = scaling[scaling["group"].isin(key)].copy()
        d["group"] = d["group"].map(lambda g: PRETTY.get(g, g).replace(" MSA depth", " depth"))
        d["parameters"] = d["size"].map(PARAMS)
        if groups == "Nothing":
            enc_colour = alt.Color("source:N", title=None, scale=alt.Scale(
                domain=["published (masked)", "this project (wild-type)"], range=[INK, "#2a78d6"]))
            d = d[d["source"].isin(["published (masked)", "this project (wild-type)"])]
        else:
            d = d[d["source"] == "published (masked)"]
            enc_colour = alt.Color("group:N", title=None, scale=alt.Scale(scheme="tableau10"))
        base = alt.Chart(d).encode(
            x=alt.X("parameters:Q", scale=alt.Scale(type="log"), title="parameters",
                    axis=alt.Axis(format="~s")),
            y=alt.Y("spearman:Q", title="Spearman (ProteinGym average)", scale=alt.Scale(zero=False)),
            color=enc_colour,
            tooltip=["size:N", "group:N", "source:N", alt.Tooltip("spearman:Q", format=".3f")])
        st.altair_chart((base.mark_line(point=True) if groups == "Nothing" else
                         base.mark_line(point=True)).properties(height=400), use_container_width=True)
        wide = scaling[scaling["source"] == "published (masked)"].pivot(
            index="group", columns="size", values="spearman").reindex(columns=SIZES)
        wide.index = [PRETTY.get(g, g) for g in wide.index]
        delta = (wide["15B"] - wide["650M"]).sort_values()
        callout("Accuracy stops improving at 650M: the published 15B model is worse overall "
                f"({wide.loc['All', '15B']:.3f} against {wide.loc['All', '650M']:.3f}). The averages hide "
                f"opposite movements — scale helps most on **{delta.index[-1]}** ({delta.iloc[-1]:+.3f}) and "
                f"hurts most on **{delta.index[0]}** ({delta.iloc[0]:+.3f}).")
        st.dataframe(wide.style.format("{:.3f}"), use_container_width=True)

# ---------------------------------------------------------------- hierarchy
with tabs[2]:
    st.subheader("Which assays does the model get right?")
    if h650 is None:
        st.info("Run `python scripts/run_analysis.py` first.")
    else:
        st.markdown("Per-assay accuracy on the Fisher *z* scale, regressed on what was measured, the taxon, "
                    "the alignment depth, the sequence length and the model's preference for the natural "
                    "sequence, with partial pooling of the function, taxon and protein effects. Bars are 90% "
                    "credible intervals; a coefficient is the change in *z* for one standard deviation of "
                    "the feature.")
        which = st.radio("Outcome", ["ESM-2 650M's accuracy", "The change from 650M to 15B"], horizontal=True)
        s = h650 if which.startswith("ESM-2") else hscale
        rows = []
        for key, group in (("g", "Feature"), ("function_effect", "What was measured"),
                           ("taxon_effect", "Taxon")):
            for name, v in s[key].items():
                rows.append({"group": group, "name": pretty(name), **v})
        d = pd.DataFrame(rows)
        order = list(d["name"])
        base = alt.Chart(d).encode(
            y=alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
            color=alt.Color("group:N", title=None, scale=alt.Scale(
                domain=["Feature", "What was measured", "Taxon"], range=["#eb6834", "#2a78d6", "#8fb8e2"])))
        bars = base.mark_rule(size=3).encode(x=alt.X("q05:Q", title="effect on Fisher z"), x2="q95:Q")
        dots = base.mark_circle(size=90).encode(x="mean:Q", tooltip=[
            "name:N", alt.Tooltip("mean:Q", format="+.3f"), alt.Tooltip("q05:Q", format="+.3f"),
            alt.Tooltip("q95:Q", format="+.3f"), alt.Tooltip("p_positive:Q", title="P(positive)",
                                                             format=".2f")])
        zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color=INK).encode(x="x:Q")
        st.altair_chart((zero + bars + dots).properties(height=380), use_container_width=True)
        spread = {k: s[k]["mean"] for k in ("tau_function", "tau_taxon", "tau_protein", "sigma") if k in s}
        note(f"R-hat at most {s['max_rhat']:.3f}, smallest effective sample size {s['min_ess_bulk']:.0f}. "
             "Spreads: " + ", ".join(f"{k.replace('tau_', '')} {v:.3f}" for k, v in spread.items()))
        if which.startswith("ESM-2"):
            peak = s.get("peak_pll_single_650M", {})
            callout("Two features matter and three do not. A deeper alignment helps "
                    f"(**{s['g']['log_neff_l']['mean']:+.3f}** per standard deviation), and preference for "
                    "the natural sequence has a clear optimum: the squared term is negative in "
                    f"**{1 - s['g']['pll_single_650M^2']['p_positive']:.0%}** of draws, with the maximum at "
                    f"a PLL of **{peak.get('mean', float('nan')):.2f}** "
                    f"({peak.get('q05', float('nan')):.2f} to {peak.get('q95', float('nan')):.2f}). Once "
                    "those are in, no function and no taxon is reliably better or worse than average: "
                    f"the spread across proteins (**{spread.get('tau_protein', 0):.3f}**) is twice the "
                    f"spread across functions (**{spread.get('tau_function', 0):.3f}**). Which protein it "
                    "is matters more than what was measured on it.")
            if h650_np is not None:
                st.markdown("**Does the protein level change the picture?** Taxon is nested inside protein "
                            "and only 14 of 186 proteins have assays of more than one function, so the "
                            "protein effects and the group effects compete for the same variation. Dropping "
                            "the protein level moves variance into the residual but leaves the coefficients "
                            "where they were:")
                comp = pd.DataFrame({
                    "with protein effects": {pretty(k): v["mean"] for k, v in h650["g"].items()},
                    "without": {pretty(k): v["mean"] for k, v in h650_np["g"].items()}})
                comp.loc["— residual spread"] = [h650["sigma"]["mean"], h650_np["sigma"]["mean"]]
                st.dataframe(comp.style.format("{:+.3f}"), use_container_width=False)
        else:
            callout("The only thing that predicts where the 15B model loses to the 650M one is how strongly "
                    "650M already preferred the natural sequence "
                    f"(**{s['g']['pll_single_650M']['mean']:+.3f}** per standard deviation, negative in "
                    f"**{1 - s['g']['pll_single_650M']['p_positive']:.0%}** of draws). The apparent gain on "
                    "viral proteins is mostly this: those are the sequences the smaller model prefers least.")

# ---------------------------------------------------------------- preference
with tabs[3]:
    st.subheader("Preference for the natural sequence")
    st.markdown("The pseudo-log-likelihood (PLL) is the average log probability the model gives the natural "
                "residue at each position, with that position masked. Gordon, Lu and Abbeel (2025) argue it "
                "predicts accuracy with an optimum in the middle: too little preference and the model has "
                "nothing to say, too much and every substitution looks equally bad.")
    if assays is None:
        st.info("Run `python scripts/run_analysis.py` first.")
    else:
        sizes = [s for s in OWN if f"pll_single_{s}" in assays]
        rows = []
        for s in sizes:
            rows.append(pd.DataFrame({"size": s, "pll": assays[f"pll_single_{s}"],
                                      "spearman": assays[f"published_{s}"], "taxon": assays["taxon"],
                                      "measured": assays["coarse_selection_type"]}))
        d = pd.concat(rows).dropna()
        pick = st.multiselect("Model sizes", sizes, default=["8M", "650M"])
        d = d[d["size"].isin(pick)]
        base = alt.Chart(d).encode(
            x=alt.X("pll:Q", title="wild-type pseudo-log-likelihood"),
            y=alt.Y("spearman:Q", title="Spearman"),
            color=alt.Color("size:N", title="ESM-2", scale=alt.Scale(
                domain=[s for s in SIZES if s in pick], range=[SIZE_COLOUR[s] for s in SIZES if s in pick])))
        pts = base.mark_circle(size=40, opacity=0.55).encode(
            tooltip=["size:N", alt.Tooltip("pll:Q", format=".2f"), alt.Tooltip("spearman:Q", format=".3f"),
                     "taxon:N", "measured:N"])
        fit = base.transform_regression("pll", "spearman", method="poly", order=2,
                                        groupby=["size"]).mark_line(size=2.5)
        st.altair_chart((pts + fit).properties(height=380), use_container_width=True)
        if "published_15B" in assays:
            e = pd.DataFrame({"pll": assays["pll_single_650M"],
                              "delta": assays["published_15B"] - assays["published_650M"],
                              "taxon": assays["taxon"], "measured": assays["coarse_selection_type"]}).dropna()
            b2 = alt.Chart(e).encode(x=alt.X("pll:Q", title="650M's wild-type PLL"),
                                     y=alt.Y("delta:Q", title="Spearman, 15B − 650M"))
            st.altair_chart((alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color=INK).encode(y="y:Q")
                             + b2.mark_circle(size=44, opacity=0.8).encode(
                                 color=alt.Color("taxon:N", title=None,
                                                 scale=alt.Scale(domain=TAXA, range=RAMP[1:5])),
                                 tooltip=["taxon:N", "measured:N", alt.Tooltip("pll:Q", format=".2f"),
                                          alt.Tooltip("delta:Q", format="+.3f")])
                             + b2.transform_regression("pll", "delta").mark_line(color=INK, size=2)
                             ).properties(height=340), use_container_width=True)
            r = np.corrcoef(e["pll"], e["delta"])[0, 1]
            callout("The 15B model gains most where 650M liked the natural sequence least, and loses where "
                    f"it liked it most (correlation {r:+.2f}). That is the same mechanism read in reverse: "
                    "scale buys more preference, which helps only until it has gone too far.")
        pairs = [s for s in OWN if f"pll_exact_{s}" in assays and assays[f"pll_exact_{s}"].notna().any()]
        if pairs:
            st.markdown("**The single-pass approximation.** Computing the PLL properly costs one pass per "
                        "position. It can be recovered from the single unmasked pass by inverting ESM-2's "
                        "training mix, which is what is plotted above. Against the direct value, on the "
                        "assays where both exist:")
            rows = []
            for s in pairs:
                m = assays[f"pll_exact_{s}"].notna()
                x, y = assays.loc[m, f"pll_exact_{s}"], assays.loc[m, f"pll_single_mutated_{s}"]
                rows.append({"size": s, "assays": int(m.sum()),
                             "rank correlation": float(pd.Series(x).corr(pd.Series(y), method="spearman")),
                             "mean error": float((y - x).mean())})
            st.dataframe(pd.DataFrame(rows).style.format({"rank correlation": "{:.3f}",
                                                          "mean error": "{:+.3f}"}), hide_index=True)
            note("The approximation is systematically too high, because a residue the model shows at less "
                 "than even odds unmasked is floored. It ranks proteins well enough for the regression, "
                 "which is all it is used for here.")

# ---------------------------------------------------------------- few-shot
with tabs[4]:
    st.subheader("How many measurements does it take to beat the zero-shot score?")
    if fewshot is None:
        st.info("Run `python scripts/run_fewshot.py` first.")
    else:
        st.markdown("Ridge regression on measured variants, against the zero-shot score, under ProteinGym's "
                    "three ways of splitting an assay. **Random** puts variants in folds at random, so a "
                    "test variant's position has usually been measured already with a different "
                    "substitution. **Every fifth position** and **blocks of sequence** hold out whole "
                    "positions, which is the situation of someone choosing where to mutate next.")
        d = fewshot.copy()
        d["split"] = d["scheme"].map(SCHEME_NAMES)
        order = [SCHEME_NAMES[s] for s in ("random", "modulo", "contiguous")]
        zs = d[d["model"] == "augmented"].assign(model="zero-shot", spearman=lambda f: f["zero_shot"])
        long = pd.concat([d, zs], ignore_index=True)
        chart = alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("n_train:Q", scale=alt.Scale(type="log"), title="measured variants used for training"),
            y=alt.Y("spearman:Q", title="Spearman (ProteinGym average)", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", title=None, scale=alt.Scale(
                domain=["zero-shot", "one-hot", "augmented"],
                range=[PREDICTOR["zero-shot"], PREDICTOR["one-hot"], PREDICTOR["augmented"]])),
            strokeDash=alt.StrokeDash("model:N", legend=None, scale=alt.Scale(
                domain=["zero-shot", "one-hot", "augmented"], range=[[5, 3], [1, 0], [1, 0]])),
            tooltip=["split:N", "model:N", "n_train:Q", alt.Tooltip("spearman:Q", format=".3f"),
                     alt.Tooltip("n_assays:Q", title="assays")]).properties(height=330, width=330)
        st.altair_chart(chart.facet(column=alt.Column("split:N", sort=order, title=None), spacing=24),
                        use_container_width=True)
        legend([("one-hot ridge", PREDICTOR["one-hot"]), ("one-hot + zero-shot score", PREDICTOR["augmented"]),
                ("zero-shot alone, same assays", PREDICTOR["zero-shot"])])
        cross = {}
        for scheme in ("random", "modulo", "contiguous"):
            g = d[(d["scheme"] == scheme) & (d["model"] == "augmented")].sort_values("n_train")
            beat = g[g["spearman"] > g["zero_shot"]]
            cross[scheme] = int(beat["n_train"].iloc[0]) if len(beat) else None
        pieces = ", ".join(f"**{SCHEME_NAMES[k].lower()}**: "
                           + (f"{v} variants" if v else "not within 1,536") for k, v in cross.items())
        callout("Where the augmented model first beats the zero-shot score — " + pieces + ". Random splits "
                "flatter the supervised model: it can learn that a position is sensitive from other "
                "substitutions at the same position, which is not what a prospective user has. Each point "
                "compares only the assays large enough to reach that training size in every fold, so the "
                "zero-shot line moves too.")
        st.dataframe(d.pivot_table(index=["split", "n_train"], columns="model",
                                   values="spearman").round(3), use_container_width=True)

# ---------------------------------------------------------------- methodology
with tabs[5]:
    if METHODOLOGY_FILE.exists():
        st.markdown(METHODOLOGY_FILE.read_text())
    else:
        st.info("docs/METHODOLOGY.md is missing.")
