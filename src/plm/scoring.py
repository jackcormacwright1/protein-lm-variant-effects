"""Zero-shot variant scores from ESM-2.

A masked protein language model gives, for every position in a sequence, a probability for each of the 20
amino acids given the rest of the sequence. The score of a substitution X -> Y at position i is the log of how
much more likely the model finds Y than X there: log p(Y at i) - log p(X at i). A multiple mutant scores the
sum of its substitutions' scores, which assumes their effects add (Meier et al., 2021).

Two ways of getting the probabilities (Meier et al., 2021):

  masked marginal     mask position i and read the model's probabilities there: one forward pass per mutated
                      position. This is how ProteinGym scores ESM models.
  wild-type marginal  read the probabilities at every position from one pass over the unmasked wild-type
                      sequence. One pass per protein instead of one per position, so hundreds of times cheaper.

ESM-2 reads at most 1,022 residues. For a longer protein each position is read from a window of that length
placed so the position is as central as possible: for the masked marginal its own window, as ProteinGym does;
for the wild-type marginal, the most central of a set of half-overlapping windows.

The model's preference for the wild-type sequence itself is measured by its pseudo-log-likelihood: the average,
over positions, of the log probability of the wild-type residue with that position masked. For the larger
models this is estimated from a random sample of positions ("pll" strategy) rather than every position.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import AA_INDEX, AMINO_ACIDS, Assay, parse

MAX_RESIDUES = 1022
MODELS = {"8M": "esm2_t6_8M_UR50D", "35M": "esm2_t12_35M_UR50D", "150M": "esm2_t30_150M_UR50D",
          "650M": "esm2_t33_650M_UR50D"}


@dataclass
class LogProbs:
    """Log probabilities of the 20 amino acids at each scored position of one protein."""
    positions: np.ndarray    # 1-based
    logp: np.ndarray         # (positions, 20), columns in AMINO_ACIDS order
    sequence: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, positions=self.positions, logp=self.logp.astype(np.float32),
                            sequence=np.array(self.sequence))

    @classmethod
    def load(cls, path: Path) -> "LogProbs":
        z = np.load(path)
        return cls(positions=z["positions"], logp=z["logp"], sequence=str(z["sequence"]))

    def score(self, mutants: pd.Series) -> np.ndarray:
        """Sum over each variant's substitutions of log p(mutant) - log p(wild type)."""
        row = {p: i for i, p in enumerate(self.positions)}
        out = np.empty(len(mutants))
        for k, mut in enumerate(mutants):
            s = 0.0
            for m in parse(mut):
                lp = self.logp[row[m.pos]]
                s += lp[AA_INDEX[m.mt]] - lp[AA_INDEX[m.wt]]
            out[k] = s
        return out

    def wildtype_log_likelihood(self) -> float:
        """Mean log probability of the wild-type residue over the scored positions."""
        idx = [AA_INDEX[self.sequence[p - 1]] for p in self.positions]
        return float(np.mean(self.logp[np.arange(len(idx)), idx]))


def optimal_window(pos0: int, length: int, window: int = MAX_RESIDUES) -> tuple[int, int]:
    """ProteinGym's window for a 0-based position: as central as possible, clipped to the sequence."""
    half = window // 2
    if length <= window:
        return 0, length
    if pos0 < half:
        return 0, window
    if pos0 >= length - half:
        return length - window, length
    return pos0 - half, pos0 + half


def tiled_windows(length: int, window: int = MAX_RESIDUES) -> list[tuple[int, int]]:
    if length <= window:
        return [(0, length)]
    starts = sorted(set(list(range(0, length - window, window // 2)) + [length - window]))
    return [(s, s + window) for s in starts]


class Scorer:
    def __init__(self, weights: Path, threads: int | None = None):
        import esm
        import torch
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        data = torch.load(weights, map_location="cpu", weights_only=False)
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_core(Path(weights).stem, data, None)
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.aa_tokens = [self.alphabet.get_idx(a) for a in AMINO_ACIDS]
        self.mask = self.alphabet.mask_idx

    def _tokens(self, seq: str):
        a = self.alphabet
        return [a.cls_idx] + [a.get_idx(c) for c in seq] + [a.eos_idx]

    def _forward(self, batch: list[list[int]]) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            logits = self.model(torch.tensor(batch))["logits"]
            logp = torch.log_softmax(logits.float(), dim=-1)
        return logp[:, :, self.aa_tokens].numpy()

    def wildtype_marginal(self, seq: str, positions: np.ndarray | None = None) -> LogProbs:
        L = len(seq)
        positions = np.arange(1, L + 1) if positions is None else np.asarray(positions)
        wins = tiled_windows(L)
        logp_win = {w: self._forward([self._tokens(seq[w[0]:w[1]])])[0] for w in wins}
        out = np.empty((len(positions), 20))
        for k, p in enumerate(positions):
            i = p - 1
            w = max((w for w in wins if w[0] <= i < w[1]), key=lambda w: min(i - w[0], w[1] - 1 - i))
            out[k] = logp_win[w][1 + i - w[0]]
        return LogProbs(positions=positions, logp=out, sequence=seq)

    def masked_marginal(self, seq: str, positions: np.ndarray, token_budget: int = 16384) -> LogProbs:
        L = len(seq)
        width = min(L, MAX_RESIDUES) + 2
        batch_size = max(1, token_budget // width)
        out = np.empty((len(positions), 20))
        items = []
        for k, p in enumerate(positions):
            s, e = optimal_window(p - 1, L)
            toks = self._tokens(seq[s:e])
            toks[1 + (p - 1 - s)] = self.mask
            items.append((k, toks, 1 + (p - 1 - s)))
        for b in range(0, len(items), batch_size):
            chunk = items[b:b + batch_size]
            lp = self._forward([t for _, t, _ in chunk])
            for j, (k, _, idx) in enumerate(chunk):
                out[k] = lp[j, idx]
        return LogProbs(positions=np.asarray(positions), logp=out, sequence=seq)

    def score_assay(self, assay: Assay, strategy: str, n_sample: int = 40) -> tuple[LogProbs, float]:
        t0 = time.time()
        if strategy == "wt":
            lp = self.wildtype_marginal(assay.sequence, assay.positions)
        elif strategy == "masked":
            lp = self.masked_marginal(assay.sequence, assay.positions)
        elif strategy == "pll":
            lp = self.masked_marginal(assay.sequence, sample_positions(assay.id, len(assay.sequence), n_sample))
        else:
            raise ValueError(strategy)
        return lp, time.time() - t0


def sample_positions(key: str, length: int, n: int) -> np.ndarray:
    """A reproducible random sample of n positions (1-based; all of them if the protein is shorter)."""
    if length <= n:
        return np.arange(1, length + 1)
    seed = sum(ord(c) * 31 ** i for i, c in enumerate(key)) % (2 ** 32)
    return np.sort(np.random.default_rng(seed).choice(np.arange(1, length + 1), size=n, replace=False))
