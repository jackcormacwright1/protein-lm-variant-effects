"""Which assays does a protein language model get right? A partially pooled regression of accuracy.

The outcome for assay a is its Spearman correlation r_a, on the Fisher z scale, z_a = atanh(r_a), where
sampling error is roughly normal with variance 1.06 / (n_a - 3) for a Spearman correlation over n_a variants
(Fieller, Hartley and Pearson, 1957). Most assays have thousands of variants, so sampling error is small and
nearly all the variation between assays is real; the model separates the two:

  z_a ~ Normal(mu_a, sqrt(se_a^2 + sigma^2))
  mu_a = alpha + b_function[f_a] + b_taxon[t_a] + g . x_a + u_protein[p_a]

with x_a the standardised continuous features (alignment depth, sequence length, the model's preference for
the wild-type sequence and its square), and partial pooling within each grouping: function and taxon
effects are drawn from normal distributions whose spread is estimated, and so are the protein effects (186
proteins, some with several assays). The protein effect captures what is consistent across one protein's
assays beyond the listed features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def fisher_z(r: pd.Series, n: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    r = np.clip(r.to_numpy(dtype=float), -0.999, 0.999)
    return np.arctanh(r), np.sqrt(1.06 / (n.to_numpy(dtype=float) - 3))


@dataclass
class Design:
    z: np.ndarray
    se: np.ndarray
    function: np.ndarray
    taxon: np.ndarray
    protein: np.ndarray
    X: np.ndarray
    function_names: list
    taxon_names: list
    x_names: list
    x_mean: np.ndarray
    x_sd: np.ndarray
    index: pd.Index


def design(df: pd.DataFrame, outcome: str, n_col: str, continuous: list[str],
           squares: tuple[str, ...] = ()) -> Design:
    """df: one row per assay with the outcome correlation, n, coarse_selection_type, taxon, UniProt_ID and the
    continuous features."""
    z, se = fisher_z(df[outcome], df[n_col])
    f_codes, f_names = pd.factorize(df["coarse_selection_type"], sort=True)
    t_codes, t_names = pd.factorize(df["taxon"], sort=True)
    p_codes, _ = pd.factorize(df["UniProt_ID"], sort=True)
    raw = df[continuous].to_numpy(dtype=float)
    mean, sd = raw.mean(axis=0), raw.std(axis=0)
    X = (raw - mean) / sd
    names = list(continuous)
    for s in squares:
        j = continuous.index(s)
        X = np.c_[X, X[:, j] ** 2]
        names.append(f"{s}^2")
    return Design(z=z, se=se, function=f_codes, taxon=t_codes, protein=p_codes, X=X,
                  function_names=list(f_names), taxon_names=list(t_names), x_names=names, x_mean=mean,
                  x_sd=sd, index=df.index)


def fit(d: Design, draws: int = 2000, tune: int = 2000, chains: int = 4, seed: int = 0,
        protein_effects: bool = True):
    """Sample the model. With protein_effects=False the protein level is dropped, which shows how much of the
    function and taxon effects rests on treating one protein's several assays as independent: taxon is nested
    inside protein (no protein spans two taxa) and only 14 of 186 proteins have assays of more than one
    function, so the protein effects and the group effects compete for the same between-protein variation."""
    import pymc as pm
    coords = {"function": d.function_names, "taxon": d.taxon_names, "feature": d.x_names,
              "protein": np.arange(d.protein.max() + 1), "assay": np.arange(len(d.z))}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", 0.4, 0.5)
        tau_f = pm.HalfNormal("tau_function", 0.3)
        tau_t = pm.HalfNormal("tau_taxon", 0.3)
        sigma = pm.HalfNormal("sigma", 0.3)
        b_f = pm.Normal("b_function", 0, 1, dims="function") * tau_f
        b_t = pm.Normal("b_taxon", 0, 1, dims="taxon") * tau_t
        g = pm.Normal("g", 0, 0.3, dims="feature")
        mu = alpha + b_f[d.function] + b_t[d.taxon] + pm.math.dot(d.X, g)
        if protein_effects:
            tau_p = pm.HalfNormal("tau_protein", 0.3)
            u_p = pm.Normal("u_protein", 0, 1, dims="protein") * tau_p
            mu = mu + u_p[d.protein]
            pm.Deterministic("protein_effect", u_p, dims="protein")
        pm.Deterministic("function_effect", b_f, dims="function")
        pm.Deterministic("taxon_effect", b_t, dims="taxon")
        pm.Normal("z", mu, pm.math.sqrt(d.se ** 2 + sigma ** 2), observed=d.z, dims="assay")
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, target_accept=0.95,
                          progressbar=False)
    return model, idata
