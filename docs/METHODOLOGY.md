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
