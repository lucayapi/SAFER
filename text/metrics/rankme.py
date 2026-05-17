"""RankMe et énergie PCA (C1, C10)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from scgm_text.metrics import pca_energy_c1_c10, rankme_effective_rank

__all__ = ["rankme_effective_rank", "pca_energy_c1_c10"]
