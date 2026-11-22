# Methodology

## The question

A deep mutational scan (DMS) makes thousands of variants of one protein, each with one or a few amino acids
changed, and measures how well each one works: how stably it folds, how tightly it binds a partner, how fast
the cells carrying it grow. Protein language models are trained only on natural sequences, with no
measurements at all, yet the probabilities they assign to amino acids predict these effects surprisingly
well. That is useful for anyone choosing which variants of a protein to make, or judging whether a human
missense variant is likely to be harmful.

This project asks four questions of one family of these models, ESM-2 (Lin et al., 2023), on ProteinGym's
217 substitution assays (Notin et al., 2023):

1. **Can the published benchmark numbers be reproduced, and is the cheap way of scoring as good as the
   expensive one?**
2. **Which proteins does the model get wrong, and why?** A partially pooled regression of per-assay accuracy
   on what was measured, the organism, how many relatives the protein has, and how strongly the model
   prefers the natural sequence.
3. **Is a bigger model better?** ESM-2 comes in six sizes from 8 million to 15 billion parameters.
4. **How many measurements does it take for a supervised model to beat the zero-shot score, and does the
   answer depend on how the test variants relate to the training ones?**

## Data

ProteinGym v1 substitutions (Notin et al., 2023; MIT licence): 217 DMS assays of 186 proteins, about 2.5
million variants, of which 0.7 million are single substitutions. Every assay's `DMS_score` is oriented so
that higher means fitter. The reference file gives each assay's target sequence, what was selected for
(`coarse_selection_type`: activity, binding, expression, organismal fitness or stability), the organism's
taxon (human, other eukaryote, prokaryote, virus), and the depth of its multiple sequence alignment
(`MSA_Neff_L`: the effective number of homologous sequences per residue, after down-weighting near
duplicates). ProteinGym also publishes every model's Spearman correlation for every assay, which is what the
reproduction is checked against, and five-fold cross-validation splits of the single mutants, used in the
few-shot section.

The assay files and splits (1.4 GB) and model weights (3.2 GB) are not in this repository; `data/reference/` holds
the reference file and the published results.

## Scoring a variant

A masked language model gives, at every position of a sequence, a probability for each amino acid given the
rest of the sequence. The score of a substitution X → Y at position *i* is how much more likely the model
finds the mutant residue than the wild type there:

```
score = log p(Y at i) - log p(X at i)
```

A multiple mutant scores the sum of its substitutions' scores, which assumes their effects add (Meier et al.,
2021). Two ways of getting the probabilities:

| | How | Cost per protein |
|---|---|---|
| **Masked marginal** | mask position *i*, run the model, read the probabilities at *i* | one pass per mutated position |
| **Wild-type marginal** | run the model once on the unmasked wild-type sequence, read every position | one pass |

ProteinGym scores ESM models with the masked marginal. Meier et al. (2021) found the two nearly as good as
each other for ESM-1v; this project checks that for ESM-2 at every size.

The two are compared on the assays scored both ways, separating single from multiple mutants, since only
multiple mutants expose the additivity assumption. How strongly each score tracks the *number* of
substitutions in a variant is recorded alongside, because that is where the two methods part company: the
unmasked model is much more confident in the wild-type residue than the masked one (mean log probability
−0.41 against −2.45 at 8M), so each substitution subtracts a larger amount, and a sum over several of them
depends more strongly on how many there are.

ESM-2 reads at most 1,022 residues. For a longer protein each position is read from a window of that length:
for the masked marginal, the window ProteinGym uses (centred on the position, clipped to the sequence ends);
for the wild-type marginal, the most central of a set of half-overlapping windows that tile the sequence.

## Measuring accuracy

Each assay's accuracy is the Spearman rank correlation between scores and measured fitness over all its
variants. ProteinGym's headline number does not simply average the 217 assays, because some proteins have
several assays and organismal-fitness assays outnumber binding assays six to one. It averages assays within
each protein (per function), then proteins within each of the five functions, then the five functions. Every
aggregate here uses that scheme, and `tests/test_core.py` checks it reproduces ProteinGym's published summary
from the published per-assay numbers. Standard errors come from resampling assays with replacement.

## How much the model prefers the natural sequence

Gordon, Lu and Abbeel (2025) argue that a model's accuracy on a protein depends on how strongly it prefers
that protein's natural sequence: too little and the model has nothing useful to say, too much and it has
effectively memorised the wild type and ranks every substitution as roughly equally bad. The measure is the
pseudo-log-likelihood (PLL): the average over positions of the log probability of the wild-type residue
*with that position masked*.

Computed directly that costs one forward pass per position, which is why it is usually estimated. Their
observation is that it can be recovered from the single unmasked pass, because of how ESM-2 was trained. Of
the residues the model was asked to predict, 80% were replaced by a mask token, 10% by a random residue and
10% left unchanged. A model consistent with that training, shown an unmasked residue, cannot tell whether it
is being asked to predict a residue it has been shown (probability α = 0.1) or one that was randomly
replaced (β = 0.1); so

```
p(x_i | all residues shown) = α/(α+β) p(x_i | x_i masked) + β/(α+β) · 1
```

which rearranges to

```
p(x_i | x_i masked) = (α+β)/α · p(x_i | shown) - β/α      = 2 p(x_i | shown) - 1   for ESM-2
```

floored at 0.001 before taking the log, since the right-hand side can go negative. The wild-type marginal
stores every position, so this PLL comes free with the scores; where masked marginals exist, the direct PLL
over the same positions checks it.

For ESM-2 8M, on the 198 proteins where both are available, the single-pass PLL correlates with the direct
one at Spearman 0.84 across proteins, and per residue the recovered log probabilities correlate with the
direct ones at Pearson 0.80 on average. The approximation is systematically too high (mean PLL −1.70 against
−2.45) and is worst where the model is least sure: residues it shows at probability below 0.5 unmasked are
floored, and those are 13% of residues. It preserves the ranking of proteins well enough for the regression
below, which is what it is used for here, but it is not a substitute for the direct PLL.

## Which assays does the model get right?

Per-assay accuracy is modelled on the Fisher z scale, z = atanh(r), where the sampling error of a Spearman
correlation over n variants is roughly normal with variance 1.06/(n − 3) (Fieller, Hartley and Pearson,
1957). Most assays have thousands of variants, so sampling error is small and nearly all the variation
between assays is real. The model separates the two:

```
z_a ~ Normal(mu_a, sqrt(se_a^2 + sigma^2))
mu_a = alpha + b_function[f_a] + b_taxon[t_a] + g . x_a + u_protein[p_a]
```

with x the standardised continuous features — log alignment depth, log sequence length, the PLL and its
square — and partial pooling within each grouping: the function, taxon and protein effects are each drawn
from a normal distribution whose spread is estimated from the data. Partial pooling matters because the
groups are very unequal (13 binding assays against 77 of organismal fitness, and 186 proteins for 217
assays): a group with few assays is pulled towards the overall mean in proportion to how little it has to
say. The protein effect captures what is consistent across one protein's assays beyond the listed features,
so that a protein measured five times does not count as five independent pieces of evidence. The squared PLL
term is what would show the inverted U. Priors are weakly informative (alpha ~ Normal(0.4, 0.5), each
spread ~ HalfNormal(0.3), g ~ Normal(0, 0.3)); sampled with PyMC's NUTS, four chains, and R-hat and
effective sample size reported with the results.

The same model is fitted a second time with the outcome the *change* in accuracy from 650M to 15B, to ask
where scale helps and where it hurts.

## How many measurements does it take to beat the zero-shot score?

Zero-shot scoring needs no measurements at all. The practical question for someone with a protein in hand is
how much measured data it takes before fitting a model to those measurements does better — and that answer
depends entirely on which variants they will want to predict.

ProteinGym supplies three five-fold splits of each assay's single mutants, which differ in how the test
variants relate to the training ones:

| Split | How folds are assigned | A test variant's position |
|---|---|---|
| **random** | variants at random | usually seen in training, with a different substitution |
| **modulo** | positions in rotation (1, 6, 11, … in one fold) | new, but its neighbours were seen |
| **contiguous** | the sequence cut into five blocks | new, and so are its neighbours |

Three predictors are compared on the held-out fold, following Hsu et al. (2022):

- **zero-shot** — the language model's score (ESM-2 650M, wild-type marginal), no training;
- **one-hot** — ridge regression on indicators of the position, of the (position, amino acid) pair and of the
  mutant amino acid, which for single mutants is the information in a one-hot encoding of the whole mutated
  sequence;
- **augmented** — the same features plus the zero-shot score, scaled up thirtyfold so the ridge penalty
  barely shrinks it.

Training sets of 24 to 1,536 variants are drawn at random from the four training folds, and the ridge
penalty is chosen within the training set by leave-one-out cross-validation over nine values. A larger
training set is only possible in a larger assay, so each point on a learning curve compares the supervised
models with the zero-shot score on exactly the assays big enough to reach that training size in every fold,
and the number of assays is reported alongside.

## Diagnostics and caveats

The hierarchical models are sampled with four chains of 2,000 draws after 2,000 tuning steps at
`target_accept = 0.99`; R-hat is at most 1.009 and the smallest bulk effective sample size 374 for the
accuracy model, with no divergences. The variant without protein effects has a handful of divergences, which
is the usual funnel between the group spreads and is why the main model is the one reported.

Three caveats are worth stating plainly.

**Taxon is nested inside protein.** No protein in ProteinGym spans two taxa, and only 14 of 186 have assays
of more than one function, so the protein effects and the group effects compete for the same between-protein
variation. The protein level is included because 24 proteins have several assays, which would otherwise be
counted as independent evidence; dropping it moves variance from the protein spread into the residual
(0.153 to 0.205) and leaves the coefficients where they were.

**The scaling comparison uses published numbers for 3B and 15B.** Those two sizes were not run here — the
weights are 11 GB and 60 GB. The comparison of 650M with 15B therefore mixes this project's reproduction
with ProteinGym's published figures, which is sound because the reproduction matches to within 0.0005, but
it is not a fresh measurement.

**The masked-marginal comparison is a subset.** Scoring the expensive way costs a forward pass per mutated
position, so it was run at 8M for 198 of the 217 assays, at 150M for those up to 250 residues and at 650M
for those up to 100. Short proteins in ProteinGym are mostly stability assays, so the larger-model
comparisons are not a random sample of the benchmark.

## References

- Fieller, E. C., Hartley, H. O. and Pearson, E. S. (1957). Tests for rank correlation coefficients I.
  *Biometrika* 44, 470–481.
- Gordon, C., Lu, A. X. and Abbeel, P. (2025). Protein language model fitness is a matter of preference.
  *International Conference on Learning Representations*.
- Hsu, C., Nisonoff, H., Fannjiang, C. and Listgarten, J. (2022). Learning protein fitness models from
  evolutionary and assay-labeled data. *Nature Biotechnology* 40, 1114–1122.
- Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language
  model. *Science* 379, 1123–1130.
- Meier, J. et al. (2021). Language models enable zero-shot prediction of the effects of mutations on protein
  function. *Advances in Neural Information Processing Systems* 34.
- Notin, P. et al. (2023). ProteinGym: large-scale benchmarks for protein fitness prediction and design.
  *Advances in Neural Information Processing Systems* 36.
