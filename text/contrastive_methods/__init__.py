"""Méthodes contrastives natives (batch triplet, SoftTriple, SupCon)."""

__all__ = ["run_contrastive_method"]


def run_contrastive_method(method_name: str, argv=None):
    from contrastive_methods.train import run_contrastive_method as _run

    return _run(method_name, argv)
