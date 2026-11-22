"""Tests that run without the assay data or model weights (CI): parsing, scoring arithmetic, windows,
ProteinGym's aggregation and the few-shot features."""

import numpy as np
import pandas as pd
import pytest

from plm.data import AMINO_ACIDS, Assay, Mutation, PUBLISHED, REFERENCE, parse, reference
from plm.evaluate import aggregate, by_function
from plm.fewshot import features, learning_curve
from plm.scoring import MAX_RESIDUES, LogProbs, optimal_window, tiled_windows


def test_parse_single_and_multiple_mutants():
    assert parse("A23C") == [Mutation("A", 23, "C")]
    assert parse("A23C:D45E") == [Mutation("A", 23, "C"), Mutation("D", 45, "E")]


def test_scores_are_log_ratios_and_add_over_substitutions():
    seq = "ACD"
    logp = np.log(np.full((3, 20), 0.01))
    logp[0, AMINO_ACIDS.index("A")] = np.log(0.5)
    logp[0, AMINO_ACIDS.index("G")] = np.log(0.05)
    logp[2, AMINO_ACIDS.index("D")] = np.log(0.2)
    logp[2, AMINO_ACIDS.index("E")] = np.log(0.4)
    lp = LogProbs(positions=np.array([1, 2, 3]), logp=logp, sequence=seq)
    s = lp.score(pd.Series(["A1G", "D3E", "A1G:D3E"]))
    assert s[0] == pytest.approx(np.log(0.05 / 0.5))
    assert s[1] == pytest.approx(np.log(2.0))
    assert s[2] == pytest.approx(s[0] + s[1])


def test_windows_cover_long_proteins_and_centre_each_position():
    assert optimal_window(10, 300) == (0, 300)
    assert optimal_window(5, 3000) == (0, MAX_RESIDUES)
    assert optimal_window(2990, 3000) == (3000 - MAX_RESIDUES, 3000)
    s, e = optimal_window(1500, 3000)
    assert e - s == MAX_RESIDUES and s < 1500 < e and abs((s + e) / 2 - 1500) <= 1
    wins = tiled_windows(3000)
    assert wins[0][0] == 0 and wins[-1][1] == 3000
    assert all(e - s == MAX_RESIDUES for s, e in wins)
    covered = np.zeros(3000, bool)
    for s, e in wins:
        covered[s:e] = True
    assert covered.all()


def test_single_pass_pll_inverts_the_training_mix():
    seq = "AC"
    q = np.array([0.6, 0.05])                        # masked probabilities of the wild-type residues
    u = 0.5 * q + 0.5                                # what a mask-consistent model shows unmasked
    logp = np.log(np.full((2, 20), 1e-4))
    logp[0, AMINO_ACIDS.index("A")] = np.log(u[0])
    logp[1, AMINO_ACIDS.index("C")] = np.log(u[1])
    lp = LogProbs(positions=np.array([1, 2]), logp=logp, sequence=seq)
    assert lp.single_pass_pll() == pytest.approx(np.mean(np.log(q)))
    assert lp.wildtype_log_likelihood(np.array([2])) == pytest.approx(np.log(u[1]))


def test_aggregation_reproduces_proteingym_summary():
    ref = reference()
    pub = pd.read_csv(PUBLISHED).set_index("DMS ID")
    summary = pd.read_csv(REFERENCE.parent / "Summary_performance_DMS_substitutions_Spearman.csv")
    summary = summary.set_index("Model_name")
    for model in ("ESM2 (8M)", "ESM2 (650M)", "ESM2 (15B)", "Site-Independent"):
        assert aggregate(pub[model], ref) == pytest.approx(summary.loc[model, "Average_Spearman"], abs=6e-4)
        assert by_function(pub[model], ref)["Stability"] == pytest.approx(
            summary.loc[model, "Function_Stability"], abs=6e-4)


def _toy_assay(n_pos=30, seed=0):
    """A synthetic assay where fitness = position sensitivity x substitution severity + noise."""
    rng = np.random.default_rng(seed)
    seq = "".join(rng.choice(list(AMINO_ACIDS), n_pos))
    sens = rng.gamma(2.0, 1.0, n_pos)
    rows = []
    for p in range(1, n_pos + 1):
        for a in AMINO_ACIDS:
            if a != seq[p - 1]:
                rows.append({"mutant": f"{seq[p - 1]}{p}{a}", "DMS_score": -sens[p - 1] + rng.normal(0, 0.3)})
    v = pd.DataFrame(rows)
    v["DMS_score_bin"] = (v["DMS_score"] > v["DMS_score"].median()).astype(int)
    v["n_mutations"] = 1
    v["fold_random_5"] = rng.integers(0, 5, len(v))
    pos = v["mutant"].str[1:-1].astype(int)
    v["fold_modulo_5"] = (pos - 1) % 5
    v["fold_contiguous_5"] = ((pos - 1) * 5 // n_pos).astype(int)
    return Assay(id="TOY", sequence=seq, variants=v, info=pd.Series(dtype=object)), sens


def test_features_put_each_single_mutant_in_three_columns():
    assay, _ = _toy_assay()
    X = features(assay, assay.variants["mutant"])
    assert X.shape == (len(assay.variants), 30 * 21 + 20)
    assert np.all(np.asarray(X.sum(axis=1)).ravel() == 3)


def test_random_splits_reward_memorising_positions_but_contiguous_splits_do_not():
    assay, sens = _toy_assay(n_pos=40)
    zero_shot = -np.array([sens[int(m[1:-1]) - 1] for m in assay.variants["mutant"]])
    zero_shot = zero_shot + np.random.default_rng(1).normal(0, 1.5, len(zero_shot))   # a noisy prior
    rng = np.random.default_rng(2)
    rand = learning_curve(assay, zero_shot, (400,), "random", rng)
    cont = learning_curve(assay, zero_shot, (400,), "contiguous", rng)
    one_hot = lambda d: d[(d["model"] == "one-hot") & (d["n_train"] == 400)]["spearman"].mean()  # noqa: E731
    assert one_hot(rand) > 0.8          # positions seen in training: their sensitivity is learned
    assert one_hot(cont) < 0.3          # new positions: nothing to go on
    aug = cont[(cont["model"] == "augmented") & (cont["n_train"] == 400)]["spearman"].mean()
    assert aug > one_hot(cont) + 0.2    # the zero-shot prior carries over to new positions
