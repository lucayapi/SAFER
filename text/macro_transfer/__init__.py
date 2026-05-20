"""Transfert macro-guidé et découverte de topics intra-macro sur corpus cible."""

from macro_transfer.gating import apply_macro_gating
from macro_transfer.pipeline import run_macro_transfer_discovery
from macro_transfer.transfer_eval import evaluate_transfer_classification

__all__ = [
    "apply_macro_gating",
    "evaluate_transfer_classification",
    "run_macro_transfer_discovery",
]
