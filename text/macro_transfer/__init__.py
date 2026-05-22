"""Transfert macro TPN et découverte de topics intra-macro sur corpus cible."""

from macro_transfer.tpn_gating import build_gating_frame
from macro_transfer.tpn_pipeline import run_tpn_macro_transfer_discovery
from macro_transfer.tpn_eval import evaluate_tpn_transfer

__all__ = [
    "build_gating_frame",
    "evaluate_tpn_transfer",
    "run_tpn_macro_transfer_discovery",
]
