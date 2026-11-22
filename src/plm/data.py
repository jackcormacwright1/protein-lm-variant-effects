"""Loading ProteinGym's deep mutational scanning (DMS) substitution assays.

ProteinGym (Notin et al., 2023) collects 217 assays, each measuring the effect of many amino-acid substitutions
in one protein. Every assay file has one row per variant: `mutant` (for example "A23C", or "A23C:D45E" for a
double mutant, positions counted from 1 in the assay's target sequence) and `DMS_score`, oriented so that
higher means fitter. The reference file describes each assay: its target sequence, what was selected for
(`coarse_selection_type`: activity, binding, expression, organismal fitness or stability), the organism's
taxon, and how many homologous sequences exist (`MSA_Neff_L`, the effective number of sequences in the
alignment per residue).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AMINO_ACIDS)}

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ASSAYS = DATA / "DMS_ProteinGym_substitutions"
FOLDS = DATA / "cv_folds_singles_substitutions"
REFERENCE = DATA / "reference" / "DMS_substitutions.csv"
PUBLISHED = DATA / "reference" / "DMS_substitutions_Spearman_DMS_level.csv"


def reference(path: Path = REFERENCE) -> pd.DataFrame:
    return pd.read_csv(path).set_index("DMS_id")


@dataclass(frozen=True)
class Mutation:
    wt: str
    pos: int      # 1-based position in the target sequence
    mt: str


def parse(mutant: str) -> list[Mutation]:
    """'A23C:D45E' -> [Mutation('A', 23, 'C'), Mutation('D', 45, 'E')]."""
    out = []
    for m in mutant.split(":"):
        out.append(Mutation(m[0], int(m[1:-1]), m[-1]))
    return out


@dataclass
class Assay:
    id: str
    sequence: str
    variants: pd.DataFrame      # mutant, DMS_score, DMS_score_bin, n_mutations
    info: pd.Series

    @property
    def positions(self) -> np.ndarray:
        """Every mutated position (1-based), sorted."""
        pos = {m.pos for mut in self.variants["mutant"] for m in parse(mut)}
        return np.array(sorted(pos))

    @property
    def singles(self) -> pd.DataFrame:
        return self.variants[self.variants["n_mutations"] == 1]


def load(dms_id: str, ref: pd.DataFrame | None = None, folds: bool = False) -> Assay:
    """One assay. With folds=True, only the single mutants, with ProteinGym's cross-validation folds."""
    ref = ref if ref is not None else reference()
    info = ref.loc[dms_id]
    path = (FOLDS if folds else ASSAYS) / info["DMS_filename"]
    cols = ["mutant", "DMS_score", "DMS_score_bin"]
    if folds:
        cols += ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
    v = pd.read_csv(path, usecols=cols)
    v["n_mutations"] = v["mutant"].str.count(":") + 1
    seq = info["target_seq"]
    for mut in v["mutant"].head(50):
        for m in parse(mut):
            if seq[m.pos - 1] != m.wt:
                raise ValueError(f"{dms_id}: {m} does not match the target sequence")
    return Assay(id=dms_id, sequence=seq, variants=v.reset_index(drop=True), info=info)
