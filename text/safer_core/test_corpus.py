"""Résolution des corpus de test via configs/test_corpora.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from safer_core.io import load_yaml
from safer_core.paths import CONFIG_DIR, RESULTS_ROOT, TEXT_ROOT, resolve_repo_path

REGISTRY_PATH = CONFIG_DIR / "test_corpora.yaml"


@dataclass(frozen=True)
class TestCorpusSpec:
    """Chemins et métadonnées d'un corpus de test."""

    id: str
    display_name: str
    data_csv: Path
    emb_csv: Path

    def data_csv_str(self) -> str:
        return str(self.data_csv)

    def emb_csv_str(self) -> str:
        return str(self.emb_csv)


def _registry_path(path: Optional[Path | str] = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("SAFER_TEST_CORPORA_YAML")
    if env:
        return resolve_repo_path(env, anchor=TEXT_ROOT)
    return REGISTRY_PATH


def load_test_corpora_registry(
    registry_path: Optional[Path | str] = None,
    *,
    anchor: Optional[Path] = None,
) -> Dict[str, Any]:
    """Charge le YAML registre (dict avec ``default`` et ``corpora``)."""
    root = anchor or TEXT_ROOT
    path = _registry_path(registry_path)
    if not path.is_file():
        raise FileNotFoundError(f"Registre corpus test introuvable : {path}")
    return load_yaml(path)


def list_test_corpus_ids(registry_path: Optional[Path | str] = None) -> List[str]:
    reg = load_test_corpora_registry(registry_path)
    corpora = reg.get("corpora") or {}
    if not isinstance(corpora, dict):
        return []
    return sorted(corpora.keys())


def default_test_corpus_id(registry_path: Optional[Path | str] = None) -> str:
    reg = load_test_corpora_registry(registry_path)
    default = reg.get("default")
    if default:
        return str(default)
    ids = list_test_corpus_ids(registry_path)
    if not ids:
        raise ValueError("Registre test_corpora.yaml vide")
    return ids[0]


def resolve_test_corpus(
    corpus_id: Optional[str] = None,
    *,
    registry_path: Optional[Path | str] = None,
    anchor: Optional[Path] = None,
    require_files: bool = False,
) -> TestCorpusSpec:
    """
    Résout un identifiant de corpus en chemins absolus sous ``text/``.

    ``corpus_id`` : id explicite, sinon ``TEST_CORPUS`` env, sinon ``default`` du registre.
    """
    root = anchor or TEXT_ROOT
    reg = load_test_corpora_registry(registry_path, anchor=root)
    cid = corpus_id or os.environ.get("TEST_CORPUS") or reg.get("default")
    if not cid:
        raise ValueError("corpus_id requis (ou default / TEST_CORPUS dans le registre)")
    cid = str(cid)
    corpora = reg.get("corpora") or {}
    if cid not in corpora:
        known = ", ".join(sorted(corpora.keys()))
        raise KeyError(f"Corpus test inconnu : {cid!r}. Connus : {known or '(aucun)'}")
    entry = corpora[cid]
    if not isinstance(entry, dict):
        raise ValueError(f"Entrée invalide pour corpus {cid}")
    data_rel = entry.get("data_csv")
    emb_rel = entry.get("emb_csv")
    if not data_rel:
        raise ValueError(f"corpus {cid} : data_csv manquant dans le registre")
    data_csv = resolve_repo_path(data_rel, repo_root=root)
    emb_csv = resolve_repo_path(emb_rel, repo_root=root) if emb_rel else data_csv.parent / "missing_emb.csv"
    display = str(entry.get("display_name") or cid)
    if require_files:
        if not data_csv.is_file():
            raise FileNotFoundError(f"Corpus {cid} : data_csv absent : {data_csv}")
        if emb_rel and not emb_csv.is_file():
            raise FileNotFoundError(f"Corpus {cid} : emb_csv absent : {emb_csv}")
    return TestCorpusSpec(id=cid, display_name=display, data_csv=data_csv, emb_csv=emb_csv)


def conventional_test_paths(corpus_id: str, *, anchor: Optional[Path] = None) -> tuple[str, str]:
    """Chemins relatifs par convention ``data_<id>`` / ``Qwen3-Embedding-0.6B_<id>``."""
    cid = str(corpus_id)
    return (
        f"dataset/test/data_{cid}.csv",
        f"embeddings/test/Qwen3-Embedding-0.6B_{cid}.csv",
    )


def output_test_root(*, anchor: Optional[Path] = None) -> Path:
    """Racine des sorties corpus de test (hors ``output/<method>/`` BTP)."""
    root = anchor or TEXT_ROOT
    return (root / "output_test").resolve()


def resultats_test_root(*, anchor: Optional[Path] = None) -> Path:
    """Alias rétrocompat → :func:`output_test_root`."""
    return output_test_root(anchor=anchor)


def method_test_results_dir(method: str, corpus_id: Optional[str] = None, *, anchor: Optional[Path] = None) -> Path:
    """Métriques / embeddings test par méthode : ``output_test/<corpus>/<method>/``."""
    cid = corpus_id or os.environ.get("TEST_CORPUS") or default_test_corpus_id()
    return (output_test_root(anchor=anchor) / cid / method).resolve()


def macro_transfer_output_dir(method: str, corpus_id: Optional[str] = None, *, anchor: Optional[Path] = None) -> Path:
    """Pipeline macro_transfer : ``output_test/<corpus>/macro_transfer/<method>/``."""
    cid = corpus_id or os.environ.get("TEST_CORPUS") or default_test_corpus_id()
    return (output_test_root(anchor=anchor) / cid / "macro_transfer" / method).resolve()


def bn_staging_dir(corpus_id: Optional[str] = None, *, anchor: Optional[Path] = None) -> Path:
    """Staging BN sur corpus test : ``output_test/<corpus>/bn_staging/``."""
    cid = corpus_id or os.environ.get("TEST_CORPUS") or default_test_corpus_id()
    return (output_test_root(anchor=anchor) / cid / "bn_staging").resolve()


def target_discovery_dir(method: str, corpus_id: str, *, anchor: Optional[Path] = None) -> Path:
    """Alias rétrocompat → :func:`macro_transfer_output_dir`."""
    return macro_transfer_output_dir(method, corpus_id, anchor=anchor)


def raw_embedding_test_dir(corpus_id: str, *, anchor: Optional[Path] = None) -> Path:
    root = anchor or TEXT_ROOT
    return (output_test_root(anchor=root) / corpus_id / "raw_embedding").resolve()


def resolve_test_paths_from_config(
    config: Dict[str, Any],
    *,
    corpus_id: Optional[str] = None,
    anchor: Optional[Path] = None,
) -> tuple[TestCorpusSpec, str, str]:
    """
    Fusionne config YAML et registre : overrides explicites ``data_csv`` / ``emb_csv`` prioritaires.
    Retourne (spec, data_csv_str relatif ou absolu résolu, emb_csv_str).
    """
    root = anchor or TEXT_ROOT
    cid = corpus_id or config.get("corpus") or config.get("test_corpus")
    spec = resolve_test_corpus(cid, anchor=root)
    data = config.get("data_csv") or config.get("test_data_csv")
    emb = config.get("emb_csv") or config.get("test_emb_csv")
    data_path = resolve_repo_path(data, repo_root=root) if data else spec.data_csv
    emb_path = resolve_repo_path(emb, repo_root=root) if emb else spec.emb_csv
    if data or emb:
        spec = TestCorpusSpec(
            id=spec.id,
            display_name=spec.display_name,
            data_csv=data_path,
            emb_csv=emb_path,
        )
    return spec, str(data_path), str(emb_path)
