"""Résolution des corpus de test via configs/test_corpora.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from safer_core.io import load_yaml
from safer_core.paths import CONFIG_DIR, RESULTS_ROOT, TEXT_ROOT, resolve_repo_path

REGISTRY_PATH = CONFIG_DIR / "test_corpora.yaml"
DEFAULT_BACKBONE_ID = "Qwen3-Embedding-0.6B"


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
        return resolve_repo_path(env, repo_root=TEXT_ROOT)
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
    require_emb_csv: Optional[bool] = None,
) -> TestCorpusSpec:
    """
    Résout un identifiant de corpus en chemins absolus sous ``text/``.

    ``corpus_id`` : id explicite, sinon ``TEST_CORPUS`` env, sinon ``default`` du registre.
    ``require_files`` : vérifie ``data_csv`` (et ``emb_csv`` sauf si ``require_emb_csv=False``).
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
    emb_csv = _resolve_test_emb_csv(emb_rel, cid, anchor=root)
    display = str(entry.get("display_name") or cid)
    check_emb = require_emb_csv if require_emb_csv is not None else require_files
    if require_files:
        if not data_csv.is_file():
            raise FileNotFoundError(f"Corpus {cid} : data_csv absent : {data_csv}")
    if check_emb and emb_rel and not emb_csv.is_file():
        raise FileNotFoundError(f"Corpus {cid} : emb_csv absent : {emb_csv}")
    return TestCorpusSpec(id=cid, display_name=display, data_csv=data_csv, emb_csv=emb_csv)


def backbone_short_id(backbone_name: str) -> str:
    """Identifiant court HF pour nommer les CSV d'embeddings (ex. ``Qwen3-Embedding-0.6B``)."""
    return str(backbone_name).strip().split("/")[-1]


def conventional_test_paths(
    corpus_id: str,
    *,
    backbone_id: str = DEFAULT_BACKBONE_ID,
    anchor: Optional[Path] = None,
) -> tuple[str, str]:
    """Chemins relatifs par convention ``dataset/data_<id>.csv`` / ``embeddings/<backbone>_<id>.csv``."""
    cid = str(corpus_id)
    bid = backbone_id or DEFAULT_BACKBONE_ID
    return (
        f"dataset/data_{cid}.csv",
        f"embeddings/{bid}_{cid}.csv",
    )


def _resolve_test_emb_csv(
    emb_rel: Optional[str],
    corpus_id: str,
    *,
    anchor: Path,
    backbone_id: str = DEFAULT_BACKBONE_ID,
) -> Path:
    """Résout le CSV d'embeddings (chemins plats + rétrocompat ``embeddings/test/``)."""
    cid = str(corpus_id)
    bid = backbone_id or DEFAULT_BACKBONE_ID
    if emb_rel:
        explicit = resolve_repo_path(str(emb_rel), repo_root=anchor).resolve()
        if explicit.is_file():
            return explicit
    rel_candidates = []
    if emb_rel:
        rel_candidates.append(str(emb_rel))
    rel_candidates.extend(
        [
            f"embeddings/{bid}_{cid}.csv",
            f"embeddings/{bid}__{cid}.csv",
            f"embeddings/test/{bid}_{cid}.csv",
            f"embeddings/test/{bid}__{cid}.csv",
        ]
    )
    seen: set[Path] = set()
    ordered: List[Path] = []
    for rel in rel_candidates:
        p = resolve_repo_path(rel, repo_root=anchor).resolve()
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    for p in ordered:
        if p.is_file():
            return p
    if emb_rel:
        return resolve_repo_path(str(emb_rel), repo_root=anchor).resolve()
    return ordered[0] if ordered else anchor / "embeddings" / f"{bid}_{cid}.csv"


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


def method_btp_results_dir(method: str, *, anchor: Optional[Path] = None) -> Path:
    """Sorties BTP (fit final) par méthode contrastive : ``output/<method>/``."""
    root = anchor or TEXT_ROOT
    return (root / "output" / method).resolve()


def resolve_projected_embeddings_paths(
    method: str,
    stem: str,
    *,
    anchor: Optional[Path] = None,
) -> Optional[tuple[Path, Path]]:
    """
    Résout ``projected_<stem>.npy`` + métadonnées sous ``output/<method>/embeddings/``.

    Repli legacy SCGM (``projected_embeddings.npy``) et ancien layout
    ``output_test/<corpus>/<method>/embeddings/``.
    """
    emb_dir = method_btp_results_dir(method, anchor=anchor) / "embeddings"
    pairs: List[tuple[str, str]] = [
        (f"projected_{stem}.npy", f"projected_{stem}_metadata.csv"),
    ]
    if stem == "btp":
        pairs.append(("projected_embeddings.npy", "metadata_with_predictions.csv"))
    for npy_name, meta_name in pairs:
        npy = emb_dir / npy_name
        meta = emb_dir / meta_name
        if npy.is_file() and meta.is_file():
            return npy.resolve(), meta.resolve()

    if stem != "btp":
        legacy_emb = method_test_results_dir(method, stem, anchor=anchor) / "embeddings"
        for npy_name, meta_name in (
            (f"projected_{stem}.npy", f"projected_{stem}_metadata.csv"),
            ("projected_embeddings_test.npy", "test_metadata.csv"),
        ):
            npy = legacy_emb / npy_name
            meta = legacy_emb / meta_name
            if npy.is_file() and meta.is_file():
                return npy.resolve(), meta.resolve()
    return None


def resolve_contrastive_embeddings_csv(
    method: str,
    corpus: Literal["btp", "test"],
    *,
    corpus_id: Optional[str] = None,
    anchor: Optional[Path] = None,
) -> Path:
    """
    Chemin attendu de ``final_embeddings_*.csv`` pour les notebooks de lecture.

    - ``btp`` : ``output/<method>/embeddings/final_embeddings_btp.csv`` (repli ``final_embeddings.csv``)
    - ``test`` : ``output_test/<corpus>/<method>/embeddings/final_embeddings_test.csv``
    """
    if corpus == "test":
        return (
            method_test_results_dir(method, corpus_id, anchor=anchor)
            / "embeddings"
            / "final_embeddings_test.csv"
        ).resolve()
    method_dir = method_btp_results_dir(method, anchor=anchor)
    for name in ("final_embeddings_btp.csv", "final_embeddings.csv"):
        candidate = method_dir / "embeddings" / name
        if candidate.is_file():
            return candidate.resolve()
    return (method_dir / "embeddings" / "final_embeddings_btp.csv").resolve()


def macro_transfer_output_dir(method: str, corpus_id: Optional[str] = None, *, anchor: Optional[Path] = None) -> Path:
    """Pipeline macro_transfer : ``output_test/<corpus>/macro_transfer/<method>/``."""
    cid = corpus_id or os.environ.get("TEST_CORPUS") or default_test_corpus_id()
    return (output_test_root(anchor=anchor) / cid / "macro_transfer" / method).resolve()


def bn_results_dir(corpus_id: Optional[str] = None, *, anchor: Optional[Path] = None) -> Path:
    """Sorties BN sur corpus test : ``output_test/<corpus>/bn_results/``."""
    cid = corpus_id or os.environ.get("TEST_CORPUS") or default_test_corpus_id()
    return (output_test_root(anchor=anchor) / cid / "bn_results").resolve()


def bn_staging_dir(corpus_id: Optional[str] = None, *, anchor: Optional[Path] = None) -> Path:
    """Alias rétrocompat → :func:`bn_results_dir`."""
    return bn_results_dir(corpus_id, anchor=anchor)


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
