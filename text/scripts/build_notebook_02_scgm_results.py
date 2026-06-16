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
TEST_CORPUS = "metallurgie"  # configs/test_corpora.yaml

from safer_core.paths import TEXT_ROOT
from safer_core.test_corpus import method_test_results_dir, raw_embedding_test_dir, resolve_test_corpus

_test_spec = resolve_test_corpus(TEST_CORPUS)
TEST_OUTPUT = method_test_results_dir("scgm_text", TEST_CORPUS)
TEST_OUTPUT_REL = str(TEST_OUTPUT.relative_to(TEXT_ROOT)).replace("\\\\", "/")
DATA_TEST_CSV = str(_test_spec.data_csv.relative_to(TEXT_ROOT)).replace("\\\\", "/")
EMB_CSV = "embeddings/Qwen3-Embedding-0.6B_btp.csv"
EMB_TEST_CSV = str(_test_spec.emb_csv.relative_to(TEXT_ROOT)).replace("\\\\", "/")
METRICS_BTP = "metrics/metrics_geometry_btp.csv"
METRICS_TEST = f"{TEST_OUTPUT_REL}/metrics/metrics_geometry_test.csv"
METRICS_RAW = "output/raw_embedding/metrics/metrics_geometry.csv"
METRICS_RAW_TEST = str(raw_embedding_test_dir(TEST_CORPUS).relative_to(TEXT_ROOT)).replace("\\\\", "/") + "/metrics/metrics_geometry.csv"
KFOLD_SUMMARY = "metrics/kfold_summary.csv"
KFOLD_PER_FOLD = "metrics/kfold_per_fold.csv"
FOLDS_DIR = "folds"
TEST_PROJ_NPY = f"{TEST_OUTPUT_REL}/embeddings/projected_embeddings_test.npy"
TEST_META_CSV = f"{TEST_OUTPUT_REL}/embeddings/test_metadata.csv"
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


GEOM_DISPLAY_COLS = [
    "eta2_macro_balanced",
    "eta2_macro_balanced_perc",
    "eta2_weighted",
]


def _slim_geom_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in GEOM_DISPLAY_COLS if c in df.columns]
    if "method" in df.columns:
        cols = ["method"] + cols
    return df[cols] if cols else df


def _display_geom_metrics(df: pd.DataFrame, title: str) -> None:
    print(title)
    display(_slim_geom_df(df))


def _corpus_metrics_comparison(raw_df, scgm_df, corpus_label: str) -> pd.DataFrame:
    rows = []
    if raw_df is not None:
        r = _slim_geom_df(raw_df).copy()
        r.insert(0, "représentation", "Embedding brut")
        rows.append(r)
    if scgm_df is not None:
        s = _slim_geom_df(scgm_df).copy()
        s.insert(0, "représentation", "SCGM projeté")
        rows.append(s)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        print(f"=== {corpus_label} — brut vs SCGM ===")
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
5. **Corpus BTP** — train / modèle final  
6. **Corpus test** — métallurgie
"""

LOAD_MD = """## 3. Chargement des résultats

Tous les fichiers sont lus ici. Les sections suivantes utilisent les variables en mémoire (pas d'export subprocess).

```bash
sbatch jobs/train_scgm_text.sh
sbatch jobs/export_raw_geometry.sh
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

PATHS = {
    "checkpoint": Path(CHECKPOINT_PATH).resolve(),
    "kfold_summary": (OUTPUT_PATH / KFOLD_SUMMARY).resolve(),
    "kfold_per_fold": (OUTPUT_PATH / KFOLD_PER_FOLD).resolve(),
    "metrics_btp": (OUTPUT_PATH / METRICS_BTP).resolve(),
    "metrics_test": (REPO_ROOT / METRICS_TEST).resolve(),
    "metrics_raw": (REPO_ROOT / METRICS_RAW).resolve(),
    "metrics_raw_test": (REPO_ROOT / METRICS_RAW_TEST).resolve(),
    "projected_btp": (EXPORTS_DIR / "projected_embeddings.npy").resolve(),
    "meta_btp": (EXPORTS_DIR / "metadata_with_predictions.csv").resolve(),
    "projected_test": (REPO_ROOT / TEST_PROJ_NPY).resolve(),
    "meta_test": (REPO_ROOT / TEST_META_CSV).resolve(),
    "raw_embeddings": (EXPORTS_DIR / "raw_embeddings.npy").resolve(),
}

require_scgm_artifacts(kfold_summary_present=_kfold_summary_early is not None)

logs = _load_csv_optional(OUTPUT_PATH / "metrics" / "train_log.csv")
if logs is None:
    logs = _load_csv_optional(OUTPUT_PATH / "logs.csv")

kfold_summary = _kfold_summary_early if _kfold_summary_early is not None else _load_csv_optional(PATHS["kfold_summary"])
kfold_per_fold = _load_csv_optional(PATHS["kfold_per_fold"])
metrics_btp = _load_csv_optional(PATHS["metrics_btp"])
metrics_test = _load_csv_optional(PATHS["metrics_test"])
metrics_raw = _load_csv_optional(PATHS["metrics_raw"])
metrics_raw_test = _load_csv_optional(PATHS["metrics_raw_test"])
projected_btp = _load_npy_optional(PATHS["projected_btp"])
meta_btp = _load_csv_optional(PATHS["meta_btp"])
projected_test = _load_npy_optional(PATHS["projected_test"])
meta_test = _load_csv_optional(PATHS["meta_test"])
raw_embeddings = _load_npy_optional(PATHS["raw_embeddings"])
themes_btp = _load_csv_optional((OUTPUT_PATH / "topics" / "themes_by_z.csv").resolve())


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
    (f"train_scgm + TEST_CORPUS ou export_scgm_test_projections.py", projected_test is None),
    (f"sbatch jobs/export_raw_geometry.sh (métriques raw test)", metrics_raw_test is None),
    (f"BASE_METHOD=scgm_text CORPUS={TEST_CORPUS} bash jobs/run_frozen_source_prototypes.sh (topics test)", False),
):
    if cond:
        print("→", hint)
"""

KFOLD_MD = """## 4. Validation K-fold (in-domain)

Validation croisée sur le **BTP** (groupes `accident_id`). Distinct du corpus **test** (§6).
"""

KFOLD_TABLES_SOURCE = """_kfold_mu_sigma_labels = (
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
Topics / OpenAI sur corpus test : voir `jobs/run_frozen_source_prototypes.sh` et notebook 06.
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
    _display_geom_metrics(metrics_btp, "=== Géométrie BTP (modèle final, 100 % train) ===")
else:
    print(f"(absent) {METRICS_BTP}")
"""

TEST_MD = """## 6. Corpus test (`TEST_CORPUS`)

Évaluation **hors distribution** : métriques et projections sous `output_test/<TEST_CORPUS>/scgm_text/`.  
**Topics intra-macro (BERTopic, OpenAI)** : `output_test/<TEST_CORPUS>/macro_transfer/` — notebook **06**.
"""

TEST_METRICS_SOURCE = """if metrics_test is not None:
    _display_geom_metrics(metrics_test, "=== Géométrie SCGM — test métallurgie ===")
else:
    print(f"(absent) {METRICS_TEST}")
    print("  → train_scgm (eval test) ou eval_scgm_test_metrics.py")

if metrics_raw_test is not None:
    _display_geom_metrics(metrics_raw_test, "=== Embedding brut — test métallurgie ===")
else:
    print(f"(absent) {METRICS_RAW_TEST}")
    print("  → sbatch jobs/export_raw_geometry.sh ou export_raw_embeddings.py --corpus TEST_CORPUS")
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

if projected_test is None or meta_test is None:
    print(f"(absent) projections test — train_scgm ou export_scgm_test_projections.py")
elif len(meta_test) != len(projected_test):
    print(f"Attention : meta ({len(meta_test)}) vs projections ({len(projected_test)})")
elif LABEL_COL not in meta_test.columns:
    print(f"Colonne {LABEL_COL} absente de test_metadata")
else:
    plot_corpus_projections(
        projected_test,
        meta_test,
        LABEL_COL,
        corpus_name="Test métallurgie (SCGM projeté)",
        save_fig=save_fig,
        figures_dir=FIGURES_DIR,
        max_points=TSNE_SAMPLE_SIZE,
        seed=SEED,
        png_name="10_test_scgm_pca_tsne.png",
        show_macro_centroids=True,
        show_z_centroids=_meta_has_z(meta_test),
        themes_z=themes_btp,
    )
    plot_corpus_umap(
        projected_test,
        meta_test,
        LABEL_COL,
        corpus_name="Test métallurgie",
        save_fig=save_fig,
        figures_dir=FIGURES_DIR,
        max_points=RAW_EMBEDDING_UMAP_MAX_POINTS,
        seed=SEED,
    )
    display_plotly_html(FIGURES_DIR / "10_test_umap_interactive.html")
    if projected_btp is not None and meta_btp is not None:
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

BTP_PROJECTION_MD = """### 5e. PCA / t-SNE — embeddings projetés BTP

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

Tableau géométrie sur les vecteurs **encodeur** (`metrics_geometry.csv` de `export_raw_embeddings.py`) : η², δ_macro.

PCA + t-SNE (`RAW_EMBEDDING_UMAP_MAX_POINTS`) sur `raw_embeddings.npy` / `EMB_CSV`, couleur = macro. Figure : `09_raw_embedding_pca_tsne.png`.
"""

EVAL_GEOMETRY_CODE = """from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scgm_text.notebook_viz import plot_projection_matplotlib, sample_projection_indices

if metrics_raw is not None:
    _display_geom_metrics(metrics_raw, "=== Embedding brut (encodeur Qwen) ===")
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

TEST_TSNE_PER_MACRO_MD = """### 6b ter. t-SNE par macro — test métallurgie (SCGM projeté)
"""

TEST_TSNE_PER_MACRO_CODE = """from scgm_text.notebook_viz import plot_tsne_per_macro_grid, sample_projection_indices

if projected_test is None or meta_test is None:
    print("(absent) projections test pour t-SNE par macro")
elif LABEL_COL not in meta_test.columns:
    print(f"Colonne {LABEL_COL} absente de test_metadata")
else:
    idx = sample_projection_indices(meta_test, LABEL_COL, max_points=TSNE_SAMPLE_SIZE, seed=SEED)
    sample_df = meta_test.loc[idx]
    sample_x = projected_test[idx]
    p_test_pm = plot_tsne_per_macro_grid(
        sample_x,
        sample_df[LABEL_COL].astype(str).to_numpy(),
        corpus_name="Test métallurgie (SCGM projeté)",
        save_fig=save_fig,
        png_name="10_test_scgm_tsne_per_macro.png",
        seed=SEED,
    )
    if p_test_pm is not None:
        print(p_test_pm)
"""

SUMMARY_TABLES_CODE = """_corpus_metrics_comparison(metrics_raw, metrics_btp, "Train (BTP)")
_corpus_metrics_comparison(metrics_raw_test, metrics_test, "Test (métallurgie)")

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
            "Entraînement BTP : `train_scgm_text.sh` ; raw BTP+test : `export_raw_geometry.sh`. "
            "Test : `output_test/` ; topics : `run_frozen_source_prototypes.sh` + notebook 06.\n",
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
        cell_from_source("### 5c. Géométrie BTP\n", "markdown"),
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
            "### 6c. Projections 2D test (SCGM projeté)\n\n"
            "PCA / t-SNE + UMAP macro. Topics BERTopic : notebook **06**.\n",
            "markdown",
        ),
        cell_from_source(TEST_VIZ_SOURCE, cell_id="test_viz"),
        cell_from_source(TEST_TSNE_PER_MACRO_MD, "markdown"),
        cell_from_source(TEST_TSNE_PER_MACRO_CODE, cell_id="test_tsne_per_macro"),
        cell_from_source("## Synthèse — comparaison géométrie train / test\n", "markdown"),
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
