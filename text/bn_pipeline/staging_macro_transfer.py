"""Staging des exports macro_transfer pour le pipeline BN (corpus test)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from macro_transfer.constants import MACRO_NAMES
from macro_transfer.fsp_config import FSP_METHOD_ALIASES, normalize_fsp_base_method
from safer_core.paths import find_repo_root, resolve_repo_path
from safer_core.test_corpus import bn_results_dir, macro_transfer_output_dir

TOPICS_SUBDIR = "topics_bertopic"

# Fichier métadonnées transfert FSP (sous transfer/).
_FSP_METADATA = "target_macro_predictions.csv"

METADATA_BY_METHOD: dict[str, str] = {
    "frozen_source_prototypes": _FSP_METADATA,
    "frozen_source_prototypes/scgm": _FSP_METADATA,
    "frozen_source_prototypes/raw": _FSP_METADATA,
    "frozen_source_prototypes/scgm_text": _FSP_METADATA,
    "frozen_source_prototypes/raw_embedding": _FSP_METADATA,
    "frozen_source_prototypes/softtriple": _FSP_METADATA,
    "frozen_source_prototypes/softtriple_native": _FSP_METADATA,
    "frozen_source_prototypes/supcon": _FSP_METADATA,
    "frozen_source_prototypes/batch_triplet": _FSP_METADATA,
    "supervised_macro_ft": _FSP_METADATA,
}

BN_STAGING_SUBDIR_BY_METHOD: dict[str, str] = {
    "frozen_source_prototypes": "frozen_source_prototypes",
    "frozen_source_prototypes/scgm": "frozen_source_prototypes_scgm",
    "frozen_source_prototypes/raw": "frozen_source_prototypes_raw",
    "frozen_source_prototypes/scgm_text": "frozen_source_prototypes_scgm_text",
    "frozen_source_prototypes/raw_embedding": "frozen_source_prototypes_raw_embedding",
    "frozen_source_prototypes/softtriple": "frozen_source_prototypes_softtriple",
    "frozen_source_prototypes/softtriple_native": "frozen_source_prototypes_softtriple_native",
    "frozen_source_prototypes/supcon": "frozen_source_prototypes_supcon",
    "frozen_source_prototypes/batch_triplet": "frozen_source_prototypes_batch_triplet",
    "supervised_macro_ft": "supervised_macro_ft",
}

# Seul pt_y_given_z est copié tel quel (matrice n_z × 4). pt_z / pt_y viennent du CSV (même n lignes).
_COPY_EMBEDDINGS = (("embeddings/prob_y_z.npy", "pt_y_given_z.npy"),)


def macro_transfer_root(
    method: str = "scgm_text",
    corpus_id: Optional[str] = None,
    *,
    repo_root: Optional[Path] = None,
) -> Path:
    root = repo_root or find_repo_root()
    return macro_transfer_output_dir(method, corpus_id, anchor=root)


def topics_subdir() -> str:
    """Sous-dossier topics intra-macro (BERTopic)."""
    return TOPICS_SUBDIR


def _macro_prob_columns(meta: pd.DataFrame) -> list[str]:
    cols = [f"p_{m}" for m in MACRO_NAMES]
    return [c for c in cols if c in meta.columns]


def write_bn_compat_arrays(exports: Path, meta: pd.DataFrame) -> None:
    """
    Fichiers attendus par ``load_metadata_for_bn`` (legacy noms) dérivés du transfert macro.

    - ``pt_y_target.npy`` : colonnes ``p_A0`` … ``p_C``
    - ``pt_z_target.npy`` : même matrice (pseudo-composantes = macros)
    - ``pt_y_given_z.npy`` : identité 4×4 si absent du run SCGM
    - ``z_assignments_target.csv`` : stub aligné unité (``z_hat`` dérivé de ``m_hat``)
    """
    prob_cols = _macro_prob_columns(meta)
    if len(prob_cols) != len(MACRO_NAMES):
        raise ValueError(
            f"Colonnes p_macro manquantes dans les métadonnées transfert "
            f"(attendu {list(MACRO_NAMES)}, trouvé {prob_cols})."
        )
    p_macro = meta[prob_cols].to_numpy(dtype=np.float64)
    n = len(meta)
    np.save(exports / "pt_y_target.npy", p_macro)
    np.save(exports / "pt_z_target.npy", p_macro)

    pyz_path = exports / "pt_y_given_z.npy"
    if not pyz_path.is_file():
        np.save(pyz_path, np.eye(len(MACRO_NAMES), dtype=np.float64))

    label2z = {name: i for i, name in enumerate(MACRO_NAMES)}
    z_rows: dict = {}
    for col in ("accident_id", "fact_id", "doc_id", "pred_label", "m_hat", "q_conf"):
        if col in meta.columns:
            z_rows[col] = meta[col]
    zdf = pd.DataFrame(z_rows)
    if "doc_id" not in zdf.columns:
        zdf["doc_id"] = np.arange(len(meta), dtype=np.int64)
    if "m_hat" in zdf.columns:
        zdf["z_hat"] = zdf["m_hat"].astype(str).map(label2z).fillna(0).astype(int)
    else:
        zdf["z_hat"] = np.argmax(p_macro, axis=1)
    zdf["max_prob_z"] = (
        zdf["q_conf"].astype(float) if "q_conf" in zdf.columns else np.max(p_macro, axis=1)
    )
    zdf.to_csv(exports / "z_assignments_target.csv", index=False)


def _normalize_fsp_metadata_for_bn(meta: pd.DataFrame) -> pd.DataFrame:
    """Normalise schéma FSP vers schéma BN attendu (m_hat, q_conf, p_*)."""
    out = meta.copy()
    if "m_hat" not in out.columns and "pred_macro" in out.columns:
        out["m_hat"] = out["pred_macro"].astype(str)
    if "q_conf" not in out.columns and "confidence" in out.columns:
        out["q_conf"] = pd.to_numeric(out["confidence"], errors="coerce")
    for m in MACRO_NAMES:
        src = f"prob_{m}"
        dst = f"p_{m}"
        if dst not in out.columns and src in out.columns:
            out[dst] = pd.to_numeric(out[src], errors="coerce")
    return out


def _normalize_staging_method(method: str) -> str:
    m = str(method).strip()
    if m.startswith("frozen_source_prototypes/"):
        suffix = m.split("/", 1)[1]
        canonical = normalize_fsp_base_method(suffix)
        return f"frozen_source_prototypes/{canonical}"
    return m


def _fsp_legacy_method_keys(method: str) -> list[str]:
    """Chemins rétrocompat pour anciens runs (scgm, raw)."""
    if not method.startswith("frozen_source_prototypes/"):
        return []
    suffix = method.split("/", 1)[1]
    legacy = {v: k for k, v in FSP_METHOD_ALIASES.items()}
    old = legacy.get(suffix)
    if old:
        return [f"frozen_source_prototypes/{old}"]
    return []


def resolve_transfer_metadata_path(mt: Path, method: str) -> Path:
    """Chemin CSV métadonnées macro pour une méthode macro_transfer."""
    norm = _normalize_staging_method(method)
    name = METADATA_BY_METHOD.get(norm, METADATA_BY_METHOD.get(method, _FSP_METADATA))
    return mt / "transfer" / name


def stage_bn_exports_from_macro_transfer(
    method: str = "scgm_text",
    corpus_id: Optional[str] = None,
    *,
    output_dir: Optional[str | Path] = None,
    metadata_filename: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> Path:
    """
    Copie les artefacts macro_transfer vers ``bn_results/staging/bn_exports/``.

    Entrées attendues sous ``output_test/<corpus>/macro_transfer/frozen_source_prototypes/<base_method>/`` :
    ``transfer/target_macro_predictions.csv``, ``topics_bertopic/assignments.csv``,
    ``topics_bertopic/themes_by_macro.csv``, ``embeddings/prob_*.npy``.
    """
    root = repo_root or find_repo_root()
    norm_method = _normalize_staging_method(method)
    mt = macro_transfer_root(norm_method, corpus_id, repo_root=root)
    if not (mt / "transfer").is_dir():
        for legacy in _fsp_legacy_method_keys(norm_method):
            alt = macro_transfer_root(legacy, corpus_id, repo_root=root)
            if (alt / "transfer").is_dir():
                mt = alt
                break
    if output_dir is not None:
        out_root = resolve_repo_path(output_dir, root)
    else:
        out_root = bn_results_dir(corpus_id, anchor=root)
        sub = BN_STAGING_SUBDIR_BY_METHOD.get(norm_method) or BN_STAGING_SUBDIR_BY_METHOD.get(method)
        if sub:
            out_root = out_root / sub
    exports = out_root / "staging" / "bn_exports"
    exports.mkdir(parents=True, exist_ok=True)

    meta_name = metadata_filename or METADATA_BY_METHOD.get(norm_method, _FSP_METADATA)
    transfer_meta = mt / "transfer" / meta_name
    if not transfer_meta.is_file():
        raise FileNotFoundError(f"{meta_name} manquant : {transfer_meta}")
    meta_src_df = pd.read_csv(transfer_meta, low_memory=False)
    meta_src_df = _normalize_fsp_metadata_for_bn(meta_src_df)
    meta_src_df.to_csv(exports / "metadata_with_predictions.csv", index=False)
    topics_dir = mt / topics_subdir()
    assign_src = topics_dir / "assignments.csv"
    themes_src = topics_dir / "themes_by_macro.csv"
    if not assign_src.is_file():
        raise FileNotFoundError(f"assignments.csv manquant : {assign_src}")
    shutil.copy2(assign_src, exports / "macro_topic_assignments.csv")
    if themes_src.is_file():
        shutil.copy2(themes_src, exports / "themes_by_macro.csv")

    emb_dir = mt / "embeddings"
    for rel_src, dst_name in _COPY_EMBEDDINGS:
        src = emb_dir / Path(rel_src).name
        if src.is_file():
            shutil.copy2(src, exports / dst_name)

    meta = pd.read_csv(exports / "metadata_with_predictions.csv", low_memory=False)
    write_bn_compat_arrays(exports, meta)

    manifest = {
        "method": method,
        "corpus_id": corpus_id,
        "macro_transfer_dir": str(mt),
        "topic_source": "bertopic",
        "bn_exports": str(exports),
    }
    (exports / "staging_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return exports
