"""Compare Sinkhorn sample priors uniform vs macro_balanced on BTP labels."""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scgm_text.data_metadata import ID2LABEL
from scgm_text.data_metadata import load_filtered_metadata
from scgm_text.sinkhorn_estep import build_sinkhorn_marginals, macro_masses_in_b

N_CLASSES = 4


def _print_masses(title: str, masses: dict[int, float]) -> None:
    print(title)
    for m in sorted(masses.keys()):
        name = ID2LABEL.get(m, str(m))
        print(f"  {name}: {masses[m]:.3f}")
    print()


def main() -> None:
    data_csv = os.path.join(ROOT, "dataset", "data_btp.csv")
    if not os.path.isfile(data_csv):
        raise FileNotFoundError(f"Missing {data_csv}")

    meta = load_filtered_metadata(data_csv)
    labels = torch.tensor(meta["label_id"].to_numpy(), dtype=torch.long)
    n = labels.numel()
    n_latents = 32

    _, b_uniform = build_sinkhorn_marginals(
        labels, n_latents=n_latents, n_classes=N_CLASSES, sample_prior="uniform"
    )
    _, b_balanced = build_sinkhorn_marginals(
        labels, n_latents=n_latents, n_classes=N_CLASSES, sample_prior="macro_balanced"
    )

    _print_masses("Uniform prior:", macro_masses_in_b(b_uniform, labels, N_CLASSES))
    _print_masses("Macro-balanced prior:", macro_masses_in_b(b_balanced, labels, N_CLASSES))

    c_present = len(torch.unique(labels))
    target = 1.0 / c_present
    for m in torch.unique(labels).tolist():
        mass = float(b_balanced[labels == m].sum())
        assert abs(mass - target) < 1e-5, f"macro {m}: mass={mass}, target={target}"

    print(
        f"[OK] n={n}, c_present={c_present}, "
        f"b_uniform.sum={float(b_uniform.sum()):.4f}, b_balanced.sum={float(b_balanced.sum()):.4f}"
    )


if __name__ == "__main__":
    main()
