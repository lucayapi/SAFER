"""Génère notebooks/02_scgm_text_results.ipynb (lecture seule, résultats SCGM BTP + test)."""
from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
OUT = NOTEBOOKS / "02_scgm_text_results.ipynb"

OBJECTIVE_MD = """## 1. Objectif de l'expérience

- Source = BTP (segments de récits d'accidents).
- Labels macro = A0, A1, B, C (superclasses SCGM).
- SCGM apprend des ancres macro `mu_y` et des centres latents `mu_z`.
- Les composantes latentes servent à explorer des motifs intra-macro non observés.
- `pred_subtype` n'est pas un label expert : diagnostic exploratoire uniquement.
"""

IMPORTS_SOURCE = """from pathlib import Path
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
import yaml

from IPython.display import display

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
"""


def cell_from_source(source: str, cell_type: str = "code", cell_id: str | None = None) -> dict:
    src_lines = source.splitlines(keepends=True)
    if not src_lines:
        src_lines = ["\n"]
    c: dict = {"cell_type": cell_type, "metadata": {}, "source": src_lines}
    if cell_id:
        c["id"] = cell_id
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def get_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def replace_cell_by_prefix(cells: list, prefix: str, new_source: str, cell_type: str = "code") -> bool:
    for c in cells:
        if c["cell_type"] == cell_type and get_source(c).startswith(prefix):
            c["source"] = [line + "\n" for line in new_source.strip().split("\n")]
            c["source"][-1] = c["source"][-1].rstrip("\n") + "\n"
            return True
    return False


PARAMS_SOURCE = """# Parameters — lecture seule (entraînement via scripts/ ou jobs/)
import os
import sys
from pathlib import Path


def _find_text_root(start: Path) -> Path:
    here = start.resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "safer_core" / "paths.py").is_file():
            return candidate
        nested = candidate / "text"
        if (nested / "safer_core" / "paths.py").is_file():
            return nested
    raise FileNotFoundError(
        "Racine text/ introuvable (safer_core/paths.py). "
        "Lancez Jupyter depuis text/ ou SAFER/."
    )


REPO_ROOT = _find_text_root(Path.cwd())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

OUTPUT_DIR = "output/scgm_text"
CHECKPOINT_PATH = None  # None → OUTPUT_DIR/checkpoints/best_model.pt

DATA_CSV = "dataset/data_btp.csv"
TEST_CORPORA = ["metallurgie", "caou", "nicollin"]  # configs/test_corpora.yaml
TEST_CORPUS = TEST_CORPORA[0]  # défaut affichage / raw viz

from safer_core.paths import TEXT_ROOT
from safer_core.test_corpus import raw_embedding_test_dir, resolve_test_corpus

_test_spec = resolve_test_corpus(TEST_CORPUS)
DATA_TEST_CSV = str(_test_spec.data_csv.relative_to(TEXT_ROOT)).replace("\\\\", "/")
EMB_CSV = "embeddings/Qwen3-Embedding-0.6B_btp.csv"
EMB_TEST_CSV = str(_test_spec.emb_csv.relative_to(TEXT_ROOT)).replace("\\\\", "/")
METRICS_BTP = "metrics/metrics_classification_btp.csv"
METRICS_TEST = "metrics/all_test_corpora_metrics.csv"
METRICS_CROSS = "metrics/cross_domain_generalization.csv"
METRICS_RAW = "output/raw_embedding/metrics/metrics_classification_btp.csv"
METRICS_RAW_TEST = "output/raw_embedding/metrics/all_test_corpora_metrics.csv"
KFOLD_SUMMARY = "metrics/kfold_summary.csv"
KFOLD_PER_FOLD = "metrics/kfold_per_fold.csv"
FOLDS_DIR = "folds"
AUTO_EXPORT_TEST_IF_MISSING = True  # tente save_scgm_projected_corpus si checkpoint + emb test OK
TUNING_GRID = "tuning/grid_summary.csv"
LABEL_COL = "pred_label"
PRED_OK_COL = "pred_ok"
GROUP_COL = "accident_id"
SEED = 42
VAL_RATIO = 0.1
BATCH_SIZE = 512  # export / évaluation

TSNE_SAMPLE_SIZE = 8000
RAW_EMBEDDING_UMAP_MAX_POINTS = 12000
"""

SETUP_SOURCE = """# REPO_ROOT / sys.path déjà configurés dans la cellule Parameters
REPO_ROOT = Path(REPO_ROOT).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from scgm_text.utils_io import create_doc_id_if_missing, ensure_dir, get_dim_columns, load_json, save_json, set_seed

_output = Path(OUTPUT_DIR)
OUTPUT_PATH = _output.resolve() if _output.is_absolute() else (REPO_ROOT / _output).resolve()
CHECKPOINTS_DIR = OUTPUT_PATH / "checkpoints"
_checkpoint = Path(CHECKPOINT_PATH) if CHECKPOINT_PATH else CHECKPOINTS_DIR / "best_model.pt"
CHECKPOINT_PATH = _checkpoint.resolve() if _checkpoint.is_absolute() else (REPO_ROOT / _checkpoint).resolve()
EXPORTS_DIR = OUTPUT_PATH / "embeddings"
EVAL_DIR = OUTPUT_PATH / "metrics"
FIGURES_DIR = OUTPUT_PATH / "figures"
TABLES_DIR = OUTPUT_PATH / "tables"
for folder in [OUTPUT_PATH, CHECKPOINTS_DIR, EXPORTS_DIR, EVAL_DIR, FIGURES_DIR, TABLES_DIR]:
    ensure_dir(str(folder))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

run_config: dict = {}
_cfg_json = OUTPUT_PATH / "configs" / "config.json"
if _cfg_json.is_file():
    run_config = load_json(str(_cfg_json))


def save_fig(name: str) -> Path:
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    return path


def display_df_for_paper(df: pd.DataFrame, name: str) -> Path:
    path = TABLES_DIR / name
    df.to_csv(path, index=False)
    display(df)
    return path


CLASSIFICATION_DISPLAY_COLS = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "ba_ood_avg",
    "ba_ood_worst",
    "cv_balanced_accuracy",
    "cv_balanced_accuracy_std",
]


def _slim_classification_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CLASSIFICATION_DISPLAY_COLS if c in df.columns]
    if "method" in df.columns:
        cols = ["method"] + cols
    if "corpus" in df.columns and "corpus" not in cols:
        cols = ["corpus"] + cols
    return df[cols] if cols else df


def _display_classification_metrics(df: pd.DataFrame, title: str) -> None:
    print(title)
    display(_slim_classification_df(df))


def _corpus_metrics_comparison(raw_df, scgm_df, corpus_label: str) -> pd.DataFrame:
    rows = []
    if raw_df is not None:
        r = _slim_classification_df(raw_df).copy()
        r.insert(0, "représentation", "Embedding brut")
        rows.append(r)
    if scgm_df is not None:
        s = _slim_classification_df(scgm_df).copy()
        s.insert(0, "représentation", "SCGM projeté")
        rows.append(s)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        print(f"=== {corpus_label} — brut vs SCGM (classification) ===")
        display(out)
        slug = corpus_label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        display_df_for_paper(out, f"paper_comparison_{slug}.csv")
    return out


def run_cli(cmd, stream=True):
    print(" ".join(cmd))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if stream:
        completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, cmd)
        return completed
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            cmd,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def _read_training_logs(output_path: Path) -> pd.DataFrame:
    for candidate in (
        output_path / "metrics" / "train_log.csv",
        output_path / "logs.csv",
    ):
        if candidate.exists():
            return pd.read_csv(candidate)
    raise FileNotFoundError(f"Aucun journal trouvé sous {output_path}")


def show_training_progress(output_path=OUTPUT_PATH):
    try:
        logs_df = _read_training_logs(output_path)
    except FileNotFoundError as exc:
        print(exc)
        return
    display(logs_df.tail(5))
    if len(logs_df) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    loss_cols = [c for c in ["train_loss", "loss_macro", "loss_latent"] if c in logs_df.columns]
    if loss_cols:
        logs_df.plot(x="epoch", y=loss_cols, ax=axes[0])
    val_cols = [
        c
        for c in [
            "val_eta2_macro_balanced",
            "val_eta2_weighted",
            "val_macro_f1",
            "val_balanced_acc",
            "val_acc",
        ]
        if c in logs_df.columns
    ]
    if val_cols:
        logs_df.plot(x="epoch", y=val_cols, ax=axes[1])
    plt.show()


print(f"REPO_ROOT={REPO_ROOT}")
print(f"OUTPUT_DIR={OUTPUT_PATH}")
print(f"CHECKPOINT={CHECKPOINT_PATH}")
print(f"DEVICE={device}")
if run_config:
    print(
        "Run config:",
        run_config.get("fidelity_mode"),
        "| best epoch:",
        run_config.get("best_checkpoint_epoch"),
        "| metric:",
        run_config.get("best_checkpoint_metric"),
    )
"""

NOTEBOOK_TOC_MD = """## Sommaire

1. Paramètres  
2. Setup (helpers)  
3. **Chargement des artefacts**  
4. Validation **K-fold** (in-domain)  
5. **Corpus BTP** — classification + projections avant LR  
6. **Corpus test OOD** — métallurgie + caoutchouc
"""

LOAD_MD = """## 3. Chargement des résultats

Tous les fichiers sont lus ici. Les sections suivantes utilisent les variables en mémoire (pas d'export subprocess).

```bash
sbatch jobs/train_scgm_text.sh
sbatch jobs/export_raw_embeddings_eval.sh
```
"""

LOAD_ARTIFACTS_SOURCE = """def _artifact_status(path: Path) -> str:
    return "OK" if path.is_file() else "absent"


def _load_csv_optional(path: Path):
    return pd.read_csv(path) if path.is_file() else None


def _load_npy_optional(path: Path):
    return np.load(path) if path.is_file() else None


def _path_display(path: Path) -> str:
    \"\"\"Chemin relatif à REPO_ROOT (Windows-safe : resolve avant relative_to).\"\"\"
    p = path.expanduser().resolve()
    root = REPO_ROOT.resolve()
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p).replace("\\\\", "/")


def resolve_scgm_checkpoint_path() -> Path:
    \"\"\"best_model.pt, sinon last_model.pt (fit final interrompu avant sélection).\"\"\"
    for name in ("best_model.pt", "last_model.pt"):
        candidate = (CHECKPOINTS_DIR / name).resolve()
        if candidate.is_file():
            if name != "best_model.pt":
                print(
                    f"[checkpoint] Utilisation de {name} "
                    "(best_model.pt absent — relancer le fit final 100 % BTP si besoin)."
                )
            return candidate
    return Path(CHECKPOINT_PATH).resolve()


def require_scgm_artifacts(*, kfold_summary_present: bool = False) -> None:
    global CHECKPOINT_PATH
    CHECKPOINT_PATH = resolve_scgm_checkpoint_path()
    missing: list[str] = []
    if not CHECKPOINT_PATH.is_file():
        if kfold_summary_present:
            print(
                "Mode lecture K-fold uniquement (pas de checkpoint fit final). "
                "Relancer sbatch jobs/train_scgm_text.sh pour le fit 100 % BTP "
                "si vous avez besoin des projections / corpus BTP."
            )
        else:
            missing.append(_path_display(CHECKPOINTS_DIR / "best_model.pt"))
    has_train_log = (OUTPUT_PATH / "metrics" / "train_log.csv").is_file() or (
        OUTPUT_PATH / "logs.csv"
    ).is_file()
    if not has_train_log and not kfold_summary_present:
        missing.append("metrics/train_log.csv ou logs.csv")
    if missing:
        raise FileNotFoundError(
            "Artefacts minimaux manquants — sbatch jobs/train_scgm_text.sh\\n"
            f"Manquant : {missing}"
        )


_kfold_summary_early = _load_csv_optional((OUTPUT_PATH / KFOLD_SUMMARY).resolve())


def _resolve_btp_projected_paths() -> tuple[Path, Path]:
    for npy_name, meta_name in (
        ("projected_btp.npy", "projected_btp_metadata.csv"),
        ("projected_embeddings.npy", "metadata_with_predictions.csv"),
    ):
        npy = (EXPORTS_DIR / npy_name).resolve()
        meta = (EXPORTS_DIR / meta_name).resolve()
        if npy.is_file() and meta.is_file():
            return npy, meta
    return (EXPORTS_DIR / "projected_btp.npy").resolve(), (EXPORTS_DIR / "projected_btp_metadata.csv").resolve()


def _resolve_ood_projected_paths(corpus_id: str) -> tuple[Path, Path] | None:
    from safer_core.test_corpus import method_test_results_dir, resolve_projected_embeddings_paths

    pair = resolve_projected_embeddings_paths("scgm_text", corpus_id, anchor=REPO_ROOT)
    if pair is not None:
        return pair
    legacy_emb = method_test_results_dir("scgm_text", corpus_id, anchor=REPO_ROOT) / "embeddings"
    for npy_name, meta_name in (
        (f"projected_{corpus_id}.npy", f"projected_{corpus_id}_metadata.csv"),
        ("projected_embeddings_test.npy", "test_metadata.csv"),
    ):
        npy = legacy_emb / npy_name
        meta = legacy_emb / meta_name
        if npy.is_file() and meta.is_file():
            return npy.resolve(), meta.resolve()
    return None


_btp_npy, _btp_meta = _resolve_btp_projected_paths()

PATHS = {
    "checkpoint": Path(CHECKPOINT_PATH).resolve(),
    "kfold_summary": (OUTPUT_PATH / KFOLD_SUMMARY).resolve(),
    "kfold_per_fold": (OUTPUT_PATH / KFOLD_PER_FOLD).resolve(),
    "metrics_btp": (OUTPUT_PATH / METRICS_BTP).resolve(),
    "metrics_test": (OUTPUT_PATH / METRICS_TEST).resolve(),
    "metrics_cross": (OUTPUT_PATH / METRICS_CROSS).resolve(),
    "metrics_raw": (REPO_ROOT / METRICS_RAW).resolve(),
    "metrics_raw_test": (REPO_ROOT / METRICS_RAW_TEST).resolve(),
    "projected_btp": _btp_npy,
    "meta_btp": _btp_meta,
    "raw_embeddings": (EXPORTS_DIR / "raw_embeddings.npy").resolve(),
}

for _cid in TEST_CORPORA:
    _pair = _resolve_ood_projected_paths(_cid)
    if _pair is not None:
        PATHS[f"projected_{_cid}"] = _pair[0]
        PATHS[f"meta_{_cid}"] = _pair[1]

require_scgm_artifacts(kfold_summary_present=_kfold_summary_early is not None)

logs = _load_csv_optional(OUTPUT_PATH / "metrics" / "train_log.csv")
if logs is None:
    logs = _load_csv_optional(OUTPUT_PATH / "logs.csv")

kfold_summary = _kfold_summary_early if _kfold_summary_early is not None else _load_csv_optional(PATHS["kfold_summary"])
kfold_per_fold = _load_csv_optional(PATHS["kfold_per_fold"])
metrics_btp = _load_csv_optional(PATHS["metrics_btp"])
metrics_test = _load_csv_optional(PATHS["metrics_test"])
metrics_cross = _load_csv_optional(PATHS["metrics_cross"])
metrics_raw = _load_csv_optional(PATHS["metrics_raw"])
metrics_raw_test = _load_csv_optional(PATHS["metrics_raw_test"])
projected_btp = _load_npy_optional(PATHS["projected_btp"])
meta_btp = _load_csv_optional(PATHS["meta_btp"])
raw_embeddings = _load_npy_optional(PATHS["raw_embeddings"])
themes_btp = _load_csv_optional((OUTPUT_PATH / "topics" / "themes_by_z.csv").resolve())

projected_ood: dict[str, np.ndarray] = {}
meta_ood: dict[str, pd.DataFrame] = {}
for _cid in TEST_CORPORA:
    _pn = PATHS.get(f"projected_{_cid}")
    _mn = PATHS.get(f"meta_{_cid}")
    if _pn is not None and _mn is not None:
        _proj = _load_npy_optional(_pn)
        _meta = _load_csv_optional(_mn)
        if _proj is not None and _meta is not None:
            projected_ood[_cid] = _proj
            meta_ood[_cid] = _meta

projected_test = projected_ood.get(TEST_CORPUS)
meta_test = meta_ood.get(TEST_CORPUS)


def _meta_has_z(meta_df) -> bool:
    return meta_df is not None and "z_hat" in meta_df.columns


inventory_rows = [
    {"artifact": k, "path": _path_display(v), "status": _artifact_status(v)}
    for k, v in PATHS.items()
]
inventory_rows.append(
    {"artifact": "logs", "path": "train_log", "status": "OK" if logs is not None else "absent"}
)
display(pd.DataFrame(inventory_rows).sort_values("artifact"))

for hint, cond in (
    ("sbatch jobs/train_scgm_text.sh (projections BTP)", projected_btp is None),
    (
        f"train_scgm_text.py (projections OOD : projected_<corpus>.npy)",
        len(projected_ood) < len(TEST_CORPORA),
    ),
    (f"sbatch jobs/export_raw_embeddings_eval.sh (métriques raw)", metrics_raw_test is None),
):
    if cond:
        print("→", hint)
"""

KFOLD_MD = """## 4. Validation K-fold (in-domain)

Validation croisée sur le **BTP** (groupes `accident_id`). Distinct du corpus **test** (§6).
"""

KFOLD_TABLES_SOURCE = """_kfold_mu_sigma_labels = (
    ("val_balanced_accuracy", "balanced accuracy (val)"),
    ("eta2_macro_balanced_perc", "η² macro balanced (%)"),
    ("train_wall_time_sec", "Temps entraînement fold (s)"),
)

if kfold_summary is not None:
    print("=== K-fold — résumé μ±σ ===")
    display(kfold_summary)
    if len(kfold_summary) == 1:
        row = kfold_summary.iloc[0]
        for key, label in _kfold_mu_sigma_labels:
            mean_col = f"mean_{key}"
            std_col = f"std_{key}"
            if mean_col in row.index:
                m = float(row[mean_col])
                s = float(row.get(std_col, 0.0))
                print(f"  {label} : {m:.4g} ± {s:.4g}")
        if "final_fit_wall_time_sec" in row.index and pd.notna(row["final_fit_wall_time_sec"]):
            print(f"  Fit final 100 % BTP : {float(row['final_fit_wall_time_sec']):.1f} s")
else:
    print(f"(absent) {KFOLD_SUMMARY}")

if kfold_per_fold is not None:
    print("\\n=== K-fold — par fold ===")
    display(kfold_per_fold)
else:
    print(f"(absent) {KFOLD_PER_FOLD}")
"""

BTP_MD = """## 5. Corpus BTP (train / modèle final)

Fit final **100 % BTP** après K-fold ; checkpoint `checkpoints/best_model.pt`.
"""

BTP_CONFIG_SOURCE = """if run_config:
    display(
        pd.Series(
            {
                "fidelity_mode": run_config.get("fidelity_mode"),
                "best_checkpoint_epoch": run_config.get("best_checkpoint_epoch"),
                "best_checkpoint_metric": run_config.get("best_checkpoint_metric"),
                "best_checkpoint_score": run_config.get("best_checkpoint_score"),
            }
        )
    )

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
summary_ckpt = {
    "input_dim": checkpoint.get("input_dim"),
    "train_size": len(checkpoint.get("train_idx", [])),
    "val_size": len(checkpoint.get("val_idx", [])),
}
display(pd.Series(summary_ckpt))
if checkpoint.get("args"):
    display(pd.json_normalize(checkpoint["args"]))
"""

BTP_METRICS_SOURCE = """if metrics_btp is not None:
    _display_classification_metrics(metrics_btp, "=== Classification BTP (modèle final, 100 % train) ===")
else:
    print(f"(absent) {METRICS_BTP}")

if metrics_cross is not None:
    _display_classification_metrics(metrics_cross, "=== Généralisation cross-domain ===")
else:
    print(f"(absent) {METRICS_CROSS}")
"""

TEST_MD = """## 6. Corpus test OOD (`TEST_CORPORA`)

Évaluation **hors distribution** : métriques sous `output/scgm_text/metrics/` et projections `embeddings/projected_<corpus>.npy` (avant classification LR).
"""

TEST_METRICS_SOURCE = """if metrics_test is not None:
    _display_classification_metrics(metrics_test, "=== Classification SCGM — tous corpus test ===")
else:
    print(f"(absent) {METRICS_TEST}")
    print("  → train_scgm_text.py (eval finale)")

for _cid in TEST_CORPORA:
    _spec = resolve_test_corpus(_cid)
    _mcsv = (OUTPUT_PATH / "metrics" / f"metrics_classification_test_{_cid}.csv").resolve()
    if _mcsv.is_file():
        _display_classification_metrics(
            pd.read_csv(_mcsv),
            f"=== Classification test — {_spec.display_name} ({_cid}) ===",
        )

if metrics_raw_test is not None:
    _display_classification_metrics(metrics_raw_test, "=== Embedding brut — tous corpus test ===")
else:
    print(f"(absent) {METRICS_RAW_TEST}")
    print("  → sbatch jobs/export_raw_embeddings.sh")
"""

TEST_RAW_VIZ_MD = """### 6b bis. Embedding brut test — PCA / t-SNE / UMAP

Vecteurs encodeur Qwen du corpus test, couleur = étape de la chaîne accidentelle.
"""

TEST_RAW_VIZ_CODE = """from macro_transfer.notebook_viz import plot_test_corpus_raw_embeddings

_raw_emb = plot_test_corpus_raw_embeddings(
    TEST_CORPUS,
    fig_dir=FIGURES_DIR,
    anchor=REPO_ROOT,
    label_col=LABEL_COL,
    max_points=RAW_EMBEDDING_UMAP_MAX_POINTS,
    seed=SEED,
    prefix="raw_test_embedding",
    show=True,
    display_metrics=False,
)
if _raw_emb.missing:
    print("Embedding brut test — fichiers manquants :", ", ".join(_raw_emb.missing))
elif _raw_emb.pca_tsne_path:
    print(
        "Figures embedding brut :",
        _raw_emb.pca_tsne_path,
        _raw_emb.tsne_per_macro_path,
        _raw_emb.umap_png_path,
    )
"""

TEST_VIZ_SOURCE = """from scgm_text.notebook_viz import (
    display_plotly_html,
    plot_btp_test_umap_pair,
    plot_corpus_projections,
    plot_corpus_umap,
)

for _cid in TEST_CORPORA:
    _spec = resolve_test_corpus(_cid)
    _proj = projected_ood.get(_cid)
    _meta = meta_ood.get(_cid)
    if _proj is None or _meta is None:
        print(f"(absent) projections test {_cid} — train_scgm_text.py")
        continue
    if len(_meta) != len(_proj):
        print(f"Attention {_cid} : meta ({len(_meta)}) vs projections ({len(_proj)})")
        continue
    if LABEL_COL not in _meta.columns:
        print(f"Colonne {LABEL_COL} absente de projected_{_cid}_metadata")
        continue
    _slug = _cid.replace(" ", "_")
    plot_corpus_projections(
        _proj,
        _meta,
        LABEL_COL,
        corpus_name=f"{_spec.display_name} (SCGM projeté, avant classif. LR)",
        save_fig=save_fig,
        figures_dir=FIGURES_DIR,
        max_points=TSNE_SAMPLE_SIZE,
        seed=SEED,
        png_name=f"10_test_{_slug}_scgm_pca_tsne.png",
        show_macro_centroids=True,
        show_z_centroids=_meta_has_z(_meta),
        themes_z=themes_btp,
    )
    plot_corpus_umap(
        _proj,
        _meta,
        LABEL_COL,
        corpus_name=_spec.display_name,
        save_fig=save_fig,
        figures_dir=FIGURES_DIR,
        max_points=RAW_EMBEDDING_UMAP_MAX_POINTS,
        seed=SEED,
        png_name=f"10_test_{_slug}_umap.png",
        html_name=f"10_test_{_slug}_umap_interactive.html",
    )
    display_plotly_html(FIGURES_DIR / f"10_test_{_slug}_umap_interactive.html")

if projected_btp is not None and meta_btp is not None and projected_test is not None and meta_test is not None:
    plot_btp_test_umap_pair(
        projected_btp,
        meta_btp,
        projected_test,
        meta_test,
        LABEL_COL,
        save_fig=save_fig,
        figures_dir=FIGURES_DIR,
        max_points=min(TSNE_SAMPLE_SIZE, RAW_EMBEDDING_UMAP_MAX_POINTS),
        seed=SEED,
    )
"""

BTP_LOGS_MD = """### 5b. Courbes d'entraînement (BTP)
"""

LOGS_SOURCE = """if logs is None:
    print("(absent) metrics/train_log.csv — pas de courbes")
else:
    display(logs.tail())

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
loss_cols = [c for c in ["train_loss", "loss_macro", "loss_latent"] if c in logs.columns]
if loss_cols:
    logs.plot(x="epoch", y=loss_cols, ax=axes[0, 0])
val_cols = [
    c
    for c in [
        "val_eta2_macro_balanced",
        "val_eta2_weighted",
        "val_acc",
        "val_macro_f1",
        "val_balanced_acc",
    ]
    if c in logs.columns
]
if val_cols:
    logs.plot(x="epoch", y=val_cols, ax=axes[0, 1])
    axes[0, 1].set_title("Validation (η² ou classif)")
eta_cols = [c for c in ["val_eta2_macro_balanced", "val_eta2_weighted", "val_eta2_macro_balanced_perc"] if c in logs.columns]
if eta_cols:
    logs.plot(x="epoch", y=eta_cols, ax=axes[1, 0], marker="o", markersize=3)
    axes[1, 0].set_title("Eta² macro (validation)")
axes[1, 1].axis("off")
save_fig("04_training_curves.png")
display_df_for_paper(logs, "training_logs.csv")

from scgm_text.notebook_viz import plot_training_geometry_curves

plot_training_geometry_curves(logs, save_fig=save_fig)
"""

BTP_PROJECTION_MD = """### 5e. PCA / t-SNE — embeddings projetés BTP (avant classification LR)

PCA + t-SNE sur un sous-échantillon (`TSNE_SAMPLE_SIZE`). Couleur = macro (`pred_label`).

- Centroïdes macro (`X`) et composantes latentes `z` (`*`) si `z_hat` présent dans les métadonnées
- Statique : `FIGURES_DIR/05_projection_macro.png`
- Interactif Plotly : `05_projection_pca_interactive.html`, `05_projection_tsne_interactive.html`

Cartes UMAP / topics BERTopic : notebook **06_macro_transfer_topics**.
"""

PROJECTION_CODE = """from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scgm_text.notebook_viz import (
    display_plotly_html,
    plot_projection_matplotlib,
    plot_projection_plotly,
    sample_projection_indices,
)

if projected_btp is None or meta_btp is None:
    print("(absent) projections BTP")
else:
    idx = sample_projection_indices(meta_btp, LABEL_COL, max_points=TSNE_SAMPLE_SIZE, seed=SEED)
    sample_df = meta_btp.loc[idx]
    sample_x = projected_btp[idx]
    pca_xy = PCA(n_components=2, random_state=SEED).fit_transform(sample_x)
    tsne_xy = TSNE(n_components=2, random_state=SEED, init="pca", learning_rate="auto").fit_transform(sample_x)
    plot_projection_matplotlib(
        pca_xy,
        tsne_xy,
        sample_df,
        LABEL_COL,
        save_fig=save_fig,
        pca_title="PCA 2D — BTP (SCGM projeté)",
        tsne_title="t-SNE 2D — BTP (SCGM projeté)",
        show_macro_centroids=True,
        show_z_centroids=_meta_has_z(meta_btp),
        themes_z=themes_btp,
    )
    plot_projection_plotly(pca_xy, tsne_xy, sample_df, LABEL_COL, figures_dir=FIGURES_DIR)
    display_plotly_html(FIGURES_DIR / "05_projection_pca_interactive.html")
    display_plotly_html(FIGURES_DIR / "05_projection_tsne_interactive.html")
"""

BTP_TSNE_PER_MACRO_MD = """### 5e bis. t-SNE par macro — BTP (SCGM projeté)

Grille 2×2 : t-SNE recalculé séparément sur chaque macro (structure intra-rôle).
"""

BTP_TSNE_PER_MACRO_CODE = """from scgm_text.notebook_viz import plot_tsne_per_macro_grid, sample_projection_indices

if projected_btp is None or meta_btp is None:
    print("(absent) projections BTP pour t-SNE par macro")
else:
    idx = sample_projection_indices(meta_btp, LABEL_COL, max_points=TSNE_SAMPLE_SIZE, seed=SEED)
    sample_df = meta_btp.loc[idx]
    sample_x = projected_btp[idx]
    p_pm = plot_tsne_per_macro_grid(
        sample_x,
        sample_df[LABEL_COL].astype(str).to_numpy(),
        corpus_name="BTP (SCGM projeté)",
        save_fig=save_fig,
        png_name="05_btp_scgm_tsne_per_macro.png",
        seed=SEED,
    )
    if p_pm is not None:
        print(p_pm)
"""

EVAL_RAW_PROJ_MD = """### 5g. Embedding brut BTP

Tableau classification sur les vecteurs **encodeur** (`metrics_classification_btp.csv` de `export_raw_embeddings.py`).

PCA + t-SNE (`RAW_EMBEDDING_UMAP_MAX_POINTS`) sur `raw_embeddings.npy` / `EMB_CSV`, couleur = macro. Figure : `09_raw_embedding_pca_tsne.png`.
"""

EVAL_GEOMETRY_CODE = """from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scgm_text.notebook_viz import plot_projection_matplotlib, sample_projection_indices

if metrics_raw is not None:
    _display_classification_metrics(metrics_raw, "=== Embedding brut (encodeur Qwen) ===")
else:
    print(f"(absent) {METRICS_RAW}")
    print("  → python scripts/export_raw_embeddings.py")
    if metrics_btp is not None:
        print("  (metrics_btp = SCGM projeté — non affiché ici)")

# --- PCA / t-SNE embedding brut BTP ---
if meta_btp is None:
    print("(absent) meta_btp pour carte embedding brut")
elif raw_embeddings is not None:
    raw_emb = raw_embeddings
elif Path(REPO_ROOT / EMB_CSV).is_file():
    from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings
    slim = meta_btp.drop(columns=[c for c in meta_btp.columns if c.startswith("dim_")], errors="ignore")
    merged, dim_cols = merge_metadata_with_embeddings(slim, str(REPO_ROOT / EMB_CSV))
    raw_emb = merged[dim_cols].to_numpy(dtype=np.float64)
else:
    raw_emb = None

if raw_emb is not None:
    idx_raw = sample_projection_indices(
        meta_btp, LABEL_COL, max_points=RAW_EMBEDDING_UMAP_MAX_POINTS, seed=SEED
    )
    sample_raw_df = meta_btp.loc[idx_raw]
    sample_raw_x = raw_emb[idx_raw]
    pca_raw_xy = PCA(n_components=2, random_state=SEED).fit_transform(sample_raw_x)
    tsne_raw_xy = TSNE(n_components=2, random_state=SEED, init="pca", learning_rate="auto").fit_transform(
        sample_raw_x
    )
    p_raw = plot_projection_matplotlib(
        pca_raw_xy,
        tsne_raw_xy,
        sample_raw_df,
        LABEL_COL,
        save_fig=save_fig,
        png_name="09_raw_embedding_pca_tsne.png",
        pca_title="PCA 2D — BTP embedding brut",
        tsne_title="t-SNE 2D — BTP embedding brut",
        show_macro_centroids=True,
        show_z_centroids=False,
    )
    print(p_raw)
"""

RAW_BTP_TSNE_PER_MACRO_MD = """### 5g bis. t-SNE par macro — BTP embedding brut
"""

RAW_BTP_TSNE_PER_MACRO_CODE = """from scgm_text.notebook_viz import plot_tsne_per_macro_grid, sample_projection_indices

if raw_emb is not None and meta_btp is not None:
    idx_raw = sample_projection_indices(
        meta_btp, LABEL_COL, max_points=RAW_EMBEDDING_UMAP_MAX_POINTS, seed=SEED
    )
    sample_raw_df = meta_btp.loc[idx_raw]
    sample_raw_x = raw_emb[idx_raw]
    p_raw_pm = plot_tsne_per_macro_grid(
        sample_raw_x,
        sample_raw_df[LABEL_COL].astype(str).to_numpy(),
        corpus_name="BTP embedding brut",
        save_fig=save_fig,
        png_name="09_raw_embedding_tsne_per_macro.png",
        seed=SEED,
    )
    if p_raw_pm is not None:
        print(p_raw_pm)
else:
    print("(absent) embedding brut BTP pour t-SNE par macro")
"""

TEST_TSNE_PER_MACRO_MD = """### 6c bis. t-SNE par macro — corpus test OOD (SCGM projeté, avant classif. LR)
"""

TEST_TSNE_PER_MACRO_CODE = """from scgm_text.notebook_viz import plot_tsne_per_macro_grid, sample_projection_indices

for _cid in TEST_CORPORA:
    _spec = resolve_test_corpus(_cid)
    _proj = projected_ood.get(_cid)
    _meta = meta_ood.get(_cid)
    if _proj is None or _meta is None:
        print(f"(absent) t-SNE par macro — {_cid}")
        continue
    if LABEL_COL not in _meta.columns:
        print(f"Colonne {LABEL_COL} absente de projected_{_cid}_metadata")
        continue
    idx = sample_projection_indices(_meta, LABEL_COL, max_points=TSNE_SAMPLE_SIZE, seed=SEED)
    sample_df = _meta.loc[idx]
    sample_x = _proj[idx]
    _slug = _cid.replace(" ", "_")
    p_test_pm = plot_tsne_per_macro_grid(
        sample_x,
        sample_df[LABEL_COL].astype(str).to_numpy(),
        corpus_name=f"{_spec.display_name} (SCGM projeté)",
        save_fig=save_fig,
        png_name=f"10_test_{_slug}_scgm_tsne_per_macro.png",
        seed=SEED,
    )
    if p_test_pm is not None:
        print(f"{_cid}:", p_test_pm)
"""

SUMMARY_TABLES_CODE = """_corpus_metrics_comparison(metrics_raw, metrics_btp, "Train (BTP)")

if metrics_test is not None and not metrics_test.empty:
    for _cid in TEST_CORPORA:
        _spec = resolve_test_corpus(_cid)
        _scgm_row = metrics_test[metrics_test.get("corpus", pd.Series(dtype=str)).astype(str) == _cid]
        _raw_row = (
            metrics_raw_test[metrics_raw_test.get("corpus", pd.Series(dtype=str)).astype(str) == _cid]
            if metrics_raw_test is not None
            else None
        )
        if _scgm_row is not None and not _scgm_row.empty:
            _corpus_metrics_comparison(_raw_row, _scgm_row, f"Test ({_spec.display_name})")

notebook_summary = {
    "output_dir": str(OUTPUT_PATH),
    "exports_dir": str(EXPORTS_DIR),
    "evaluation_dir": str(EVAL_DIR),
    "figures_dir": str(FIGURES_DIR),
    "tables_dir": str(TABLES_DIR),
    "best_checkpoint_epoch": run_config.get("best_checkpoint_epoch"),
    "device": str(device),
    "figure_files": sorted(p.name for p in FIGURES_DIR.glob("*.png")),
    "table_files": sorted(p.name for p in TABLES_DIR.glob("*.csv")),
}
save_json(notebook_summary, OUTPUT_PATH / "notebook_summary.json")
notebook_summary
"""


def main() -> None:
    cells: list[dict] = [
        cell_from_source(
            "# 02 — SCGM Text (lecture seule)\n\n"
            "Analyse des sorties sous `output/scgm_text/`. "
            "Entraînement BTP : `train_scgm_text.sh` ; raw BTP+test : `export_raw_embeddings_eval.sh`. "
            "Métriques classification LR + t-SNE sur `embeddings/projected_*.npy` (avant classif.).\n",
            "markdown",
            "nb_title",
        ),
        cell_from_source(NOTEBOOK_TOC_MD, "markdown", "nb_toc"),
        cell_from_source(OBJECTIVE_MD, "markdown"),
        cell_from_source(
            "## 2. Imports\n\nRégler `OUTPUT_DIR` dans la cellule **Parameters**.\n",
            "markdown",
        ),
        cell_from_source(IMPORTS_SOURCE, cell_id="imports"),
        cell_from_source(PARAMS_SOURCE, cell_id="91307aa9"),
        cell_from_source(SETUP_SOURCE, cell_id="c308cd48"),
        cell_from_source(LOAD_MD, "markdown", "load_md"),
        cell_from_source(LOAD_ARTIFACTS_SOURCE, cell_id="load01"),
        cell_from_source(KFOLD_MD, "markdown", "kfold_md"),
        cell_from_source(KFOLD_TABLES_SOURCE, cell_id="kfold_tables"),
        cell_from_source(BTP_MD, "markdown", "btp_md"),
        cell_from_source(BTP_CONFIG_SOURCE, cell_id="btp_config"),
        cell_from_source(BTP_LOGS_MD, "markdown"),
        cell_from_source(LOGS_SOURCE, cell_id="386ed2ac"),
        cell_from_source("### 5c. Classification BTP\n", "markdown"),
        cell_from_source(BTP_METRICS_SOURCE, cell_id="btp_metrics"),
        cell_from_source(BTP_PROJECTION_MD, "markdown"),
        cell_from_source(PROJECTION_CODE, cell_id="proj01"),
        cell_from_source(BTP_TSNE_PER_MACRO_MD, "markdown"),
        cell_from_source(BTP_TSNE_PER_MACRO_CODE, cell_id="btp_tsne_per_macro"),
        cell_from_source(EVAL_RAW_PROJ_MD, "markdown"),
        cell_from_source(EVAL_GEOMETRY_CODE, cell_id="eval_geom"),
        cell_from_source(RAW_BTP_TSNE_PER_MACRO_MD, "markdown"),
        cell_from_source(RAW_BTP_TSNE_PER_MACRO_CODE, cell_id="raw_btp_tsne_per_macro"),
        cell_from_source(TEST_MD, "markdown", "test_md"),
        cell_from_source(TEST_METRICS_SOURCE, cell_id="test_metrics"),
        cell_from_source(TEST_RAW_VIZ_MD, "markdown", "test_raw_viz_md"),
        cell_from_source(TEST_RAW_VIZ_CODE, cell_id="test_raw_viz"),
        cell_from_source(
            "### 6c. Projections 2D test OOD (SCGM projeté, avant classif. LR)\n\n"
            "PCA / t-SNE + UMAP macro pour chaque corpus dans `TEST_CORPORA`. Topics BERTopic : notebook **06**.\n",
            "markdown",
        ),
        cell_from_source(TEST_VIZ_SOURCE, cell_id="test_viz"),
        cell_from_source(TEST_TSNE_PER_MACRO_MD, "markdown"),
        cell_from_source(TEST_TSNE_PER_MACRO_CODE, cell_id="test_tsne_per_macro"),
        cell_from_source("## Synthèse — comparaison classification train / test\n", "markdown"),
        cell_from_source(SUMMARY_TABLES_CODE, cell_id="summary_tables"),
    ]
    cells[5]["metadata"] = {"tags": ["parameters"]}

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "cells": cells,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
