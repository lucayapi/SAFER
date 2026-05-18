"""Rebuild 01_scgm_text_experiment.ipynb from 01_draft + extras (OpenAI, DataMap, optimizer)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
DRAFT = NOTEBOOKS / "01_draft.ipynb"
OUT = NOTEBOOKS / "01_scgm_text_experiment.ipynb"


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


def strip_outputs(nb: dict) -> dict:
    out = copy.deepcopy(nb)
    for c in out["cells"]:
        if c["cell_type"] == "code":
            c["execution_count"] = None
            c["outputs"] = []
    return out


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
OUTPUT_DIR = "resultats/scgm_text"
CHECKPOINT_PATH = None  # None → OUTPUT_DIR/checkpoints/best_model.pt

DATA_CSV = "dataset/data_btp.csv"
DATA_TEST_CSV = "dataset/test/data_metallurgie.csv"
EMB_CSV = "embeddings/Qwen3-Embedding-0.6B_btp.csv"
EMB_TEST_CSV = "embeddings/test/Qwen3-Embedding-0.6B_metallurgie.csv"
METRICS_BTP = "metrics/metrics_geometry_btp.csv"
METRICS_TEST = "metrics/metrics_geometry_test.csv"
METRICS_RAW = "resultats/raw_embedding/metrics/metrics_geometry.csv"
METRICS_RAW_TEST = "resultats/raw_embedding_test/metrics/metrics_geometry.csv"
KFOLD_SUMMARY = "metrics/kfold_summary.csv"
KFOLD_PER_FOLD = "metrics/kfold_per_fold.csv"
FOLDS_DIR = "folds"
TEST_PROJ_NPY = "embeddings/projected_embeddings_test.npy"
TEST_META_CSV = "embeddings/test_metadata.csv"
AUTO_EXPORT_TEST_IF_MISSING = True  # tente save_scgm_projected_corpus si checkpoint + emb test OK
TUNING_GRID = "tuning/grid_summary.csv"
LABEL_COL = "pred_label"
PRED_OK_COL = "pred_ok"
GROUP_COL = "accident_id"
SEED = 42
VAL_RATIO = 0.1
BATCH_SIZE = 512  # export / évaluation

TSNE_SAMPLE_SIZE = 8000
DATAMAP_MAX_POINTS = 12000
RAW_EMBEDDING_UMAP_MAX_POINTS = 12000
DATAMAP_SEED = 42
DATAMAP_LABEL_MODE = "theme_summary"  # theme_summary | macro_z (theme_summary exige themes_by_z_openai.csv)
DATAMAP_SHOW_MACRO_CENTROIDS = True
"""

SETUP_SOURCE = """def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "scgm_text").is_dir() and (candidate / "scripts").is_dir():
            return candidate
        if (candidate / "text" / "scgm_text").is_dir() and (candidate / "text" / "scripts").is_dir():
            return candidate / "text"
    raise FileNotFoundError("Impossible de localiser la racine text/ (scgm_text/ + scripts/).")


REPO_ROOT = find_repo_root(Path.cwd())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from scgm_text.dataset_text_embeddings import (
    ID2LABEL,
    LABEL2ID,
    TextEmbeddingDataset,
    split_by_group,
)
from scgm_text.utils_io import create_doc_id_if_missing, ensure_dir, get_dim_columns, load_json, save_json, set_seed

_output = Path(OUTPUT_DIR)
OUTPUT_PATH = _output.resolve() if _output.is_absolute() else (REPO_ROOT / _output).resolve()
CHECKPOINTS_DIR = OUTPUT_PATH / "checkpoints"
_checkpoint = Path(CHECKPOINT_PATH) if CHECKPOINT_PATH else CHECKPOINTS_DIR / "best_model.pt"
CHECKPOINT_PATH = _checkpoint.resolve() if _checkpoint.is_absolute() else (REPO_ROOT / _checkpoint).resolve()
EXPORTS_DIR = OUTPUT_PATH / "embeddings"
TOPICS_DIR = OUTPUT_PATH / "topics"
EVAL_DIR = OUTPUT_PATH / "metrics"
FIGURES_DIR = OUTPUT_PATH / "figures"
TABLES_DIR = OUTPUT_PATH / "tables"
for folder in [OUTPUT_PATH, CHECKPOINTS_DIR, EXPORTS_DIR, TOPICS_DIR, EVAL_DIR, FIGURES_DIR, TABLES_DIR]:
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
    "delta_macro_pct",
    "eta2_weighted",
    "rankme_global",
    "c1_global",
    "c10_global",
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
sbatch jobs/postprocess_scgm_text.sh
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


def require_scgm_artifacts() -> None:
    missing: list[str] = []
    if not Path(CHECKPOINT_PATH).resolve().is_file():
        missing.append(_path_display(Path(CHECKPOINT_PATH)))
    if not (OUTPUT_PATH / "metrics" / "train_log.csv").is_file() and not (OUTPUT_PATH / "logs.csv").is_file():
        missing.append("metrics/train_log.csv ou logs.csv")
    if missing:
        raise FileNotFoundError(
            "Artefacts minimaux manquants — sbatch jobs/train_scgm_text.sh\\n"
            f"Manquant : {missing}"
        )


PATHS = {
    "checkpoint": Path(CHECKPOINT_PATH).resolve(),
    "kfold_summary": (OUTPUT_PATH / KFOLD_SUMMARY).resolve(),
    "kfold_per_fold": (OUTPUT_PATH / KFOLD_PER_FOLD).resolve(),
    "metrics_btp": (OUTPUT_PATH / METRICS_BTP).resolve(),
    "metrics_test": (OUTPUT_PATH / METRICS_TEST).resolve(),
    "metrics_raw": (REPO_ROOT / METRICS_RAW).resolve(),
    "metrics_raw_test": (REPO_ROOT / METRICS_RAW_TEST).resolve(),
    "projected_btp": (EXPORTS_DIR / "projected_embeddings.npy").resolve(),
    "meta_btp": (EXPORTS_DIR / "metadata_with_predictions.csv").resolve(),
    "projected_test": (OUTPUT_PATH / TEST_PROJ_NPY).resolve(),
    "meta_test": (OUTPUT_PATH / TEST_META_CSV).resolve(),
    "themes_z": (TOPICS_DIR / "themes_by_z.csv").resolve(),
    "themes_openai": (TOPICS_DIR / "themes_by_z_openai.csv").resolve(),
    "themes_macro": (TOPICS_DIR / "themes_by_macro_z.csv").resolve(),
    "raw_embeddings": (EXPORTS_DIR / "raw_embeddings.npy").resolve(),
}

require_scgm_artifacts()

logs = _load_csv_optional(OUTPUT_PATH / "metrics" / "train_log.csv")
if logs is None:
    logs = _load_csv_optional(OUTPUT_PATH / "logs.csv")

kfold_summary = _load_csv_optional(PATHS["kfold_summary"])
kfold_per_fold = _load_csv_optional(PATHS["kfold_per_fold"])
metrics_btp = _load_csv_optional(PATHS["metrics_btp"])
metrics_test = _load_csv_optional(PATHS["metrics_test"])
metrics_raw = _load_csv_optional(PATHS["metrics_raw"])
metrics_raw_test = _load_csv_optional(PATHS["metrics_raw_test"])
projected_btp = _load_npy_optional(PATHS["projected_btp"])
meta_btp = _load_csv_optional(PATHS["meta_btp"])
projected_test = _load_npy_optional(PATHS["projected_test"])
meta_test = _load_csv_optional(PATHS["meta_test"])
themes_z = _load_csv_optional(PATHS["themes_z"])
themes_openai = _load_csv_optional(PATHS["themes_openai"])
themes_macro = _load_csv_optional(PATHS["themes_macro"])
raw_embeddings = _load_npy_optional(PATHS["raw_embeddings"])

if themes_z is None and (EXPORTS_DIR / "themes_by_z.csv").is_file():
    themes_z = pd.read_csv(EXPORTS_DIR / "themes_by_z.csv")
if themes_macro is None and (EXPORTS_DIR / "themes_by_macro_z.csv").is_file():
    themes_macro = pd.read_csv(EXPORTS_DIR / "themes_by_macro_z.csv")

inventory_rows = [
    {"artifact": k, "path": _path_display(v), "status": _artifact_status(v)}
    for k, v in PATHS.items()
]
inventory_rows.append(
    {"artifact": "logs", "path": "train_log", "status": "OK" if logs is not None else "absent"}
)
display(pd.DataFrame(inventory_rows).sort_values("artifact"))

for hint, cond in (
    ("sbatch jobs/postprocess_scgm_text.sh", projected_btp is None or themes_z is None),
    (f"export_scgm_test_projections.py --output_dir {OUTPUT_DIR}", projected_test is None),
    ("SKIP_OPENAI=0 bash jobs/enrich_scgm_themes_openai.sh", themes_openai is None),
):
    if cond:
        print("→", hint)
"""

KFOLD_MD = """## 4. Validation K-fold (in-domain)

Validation croisée sur le **BTP** (groupes `accident_id`). Distinct du corpus **test** (§6).
"""

KFOLD_TABLES_SOURCE = """if kfold_summary is not None:
    print("=== K-fold — résumé μ±σ ===")
    display(kfold_summary)
    if "mean_delta_macro_pct" in kfold_summary.columns and len(kfold_summary) == 1:
        m = float(kfold_summary["mean_delta_macro_pct"].iloc[0])
        s = float(kfold_summary.get("std_delta_macro_pct", pd.Series([0])).iloc[0])
        print(f"  δ_macro val : {m:.2f} ± {s:.2f} %")
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
Visualisations et topics sur les segments d'entraînement uniquement.
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

TEST_MD = """## 6. Corpus test — métallurgie

Évaluation **hors distribution** : mêmes embeddings Qwen figés + tête SCGM entraînée sur BTP.  
Distinct du K-fold (§4) et des viz BTP (§5).
"""

TEST_METRICS_SOURCE = """if metrics_test is not None:
    _display_geom_metrics(metrics_test, "=== Géométrie SCGM — test métallurgie ===")
else:
    print(f"(absent) {METRICS_TEST}")
    print("  → sbatch jobs/postprocess_scgm_text.sh (étape métriques SCGM test)")

if metrics_raw_test is not None:
    _display_geom_metrics(metrics_raw_test, "=== Embedding brut — test métallurgie ===")
else:
    print(f"(absent) {METRICS_RAW_TEST}")
    print("  → postprocess (étape métriques raw test) ou export_raw_embeddings.py + raw_embedding_test.yaml")
"""

TEST_RAW_VIZ_MD = """### 6b bis. Embedding brut test — PCA / t-SNE

PCA + t-SNE sur les vecteurs encodeur Qwen (`EMB_TEST_CSV`), couleur = macro. Centroïdes macro affichés.
"""

TEST_RAW_VIZ_CODE = """from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scgm_text.notebook_viz import plot_projection_matplotlib, sample_projection_indices

if meta_test is None:
    print("(absent) meta_test pour carte embedding brut test")
elif not Path(REPO_ROOT / EMB_TEST_CSV).is_file():
    print(f"(absent) {EMB_TEST_CSV}")
else:
    from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings

    slim = meta_test.drop(columns=[c for c in meta_test.columns if c.startswith("dim_")], errors="ignore")
    merged_test, dim_cols = merge_metadata_with_embeddings(slim, str(REPO_ROOT / EMB_TEST_CSV))
    raw_test = merged_test[dim_cols].to_numpy(dtype=np.float64)
    idx_rt = sample_projection_indices(
        meta_test, LABEL_COL, max_points=RAW_EMBEDDING_UMAP_MAX_POINTS, seed=SEED
    )
    sample_rt_df = meta_test.loc[idx_rt]
    sample_rt_x = raw_test[idx_rt]
    pca_rt = PCA(n_components=2, random_state=SEED).fit_transform(sample_rt_x)
    tsne_rt = TSNE(n_components=2, random_state=SEED, init="pca", learning_rate="auto").fit_transform(
        sample_rt_x
    )
    plot_projection_matplotlib(
        pca_rt,
        tsne_rt,
        sample_rt_df,
        LABEL_COL,
        save_fig=save_fig,
        png_name="10_raw_test_pca_tsne.png",
        pca_title="PCA 2D — test embedding brut",
        tsne_title="t-SNE 2D — test embedding brut",
        show_macro_centroids=True,
        show_z_centroids=False,
    )
"""

TEST_VIZ_SOURCE = """from scgm_text.notebook_viz import (
    display_plotly_html,
    plot_btp_test_umap_pair,
    plot_corpus_projections,
    plot_corpus_umap,
)

if projected_test is None or meta_test is None:
    print(f"(absent) projections test — voir §3 (postprocess / export_scgm_test_projections)")
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
        show_z_centroids=True,
        themes_z=themes_z,
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
geom_cols = [c for c in ["rankme_global", "c1_global", "c10_global"] if c in logs.columns]
if geom_cols:
    logs.plot(x="epoch", y=geom_cols, ax=axes[1, 0], marker="o", markersize=3)
    axes[1, 0].set_title("RankMe / C1 / C10 (global)")
axes[1, 1].axis("off")
save_fig("04_training_curves.png")
display_df_for_paper(logs, "training_logs.csv")

from scgm_text.notebook_viz import plot_training_geometry_curves

plot_training_geometry_curves(logs, save_fig=save_fig)
"""

THEMES_SOURCE = """from scgm_text.notebook_viz import (
    plot_topics_distribution_by_macro,
    plot_topics_n_units_by_z,
)

if themes_z is None:
    print("(absent) themes_by_z — sbatch jobs/postprocess_scgm_text.sh")
else:
    topics_tbl = themes_z.copy()
    if themes_openai is not None and "theme_summary" in themes_openai.columns:
        topics_tbl = topics_tbl.merge(
            themes_openai[["z_id", "theme_summary"]],
            on="z_id",
            how="left",
        )
    else:
        topics_tbl["theme_summary"] = pd.NA
        print("→ SKIP_OPENAI=0 bash jobs/enrich_scgm_themes_openai.sh (libellés topics)")

    topics_tbl = topics_tbl.sort_values("n_units", ascending=False)
    print("=== Topics par composante z ===")
    display(topics_tbl)
    display_df_for_paper(topics_tbl, "topics_by_z_with_openai.csv")

    plot_topics_distribution_by_macro(topics_tbl, save_fig=save_fig)
    plot_topics_n_units_by_z(topics_tbl, save_fig=save_fig)
"""

BTP_PROJECTION_MD = """### 5e. PCA / t-SNE — embeddings projetés BTP
"""

SECTION12_MD = """### 5e. PCA / t-SNE — embeddings projetés BTP

PCA + t-SNE sur un sous-échantillon (`TSNE_SAMPLE_SIZE`). Couleur = macro (`pred_label`).

- Statique : `FIGURES_DIR/05_projection_macro.png`
- Interactif Plotly : `05_projection_pca_interactive.html`, `05_projection_tsne_interactive.html`
"""

DATAMAP_MD = """### 5f. Carte 2D BTP (UMAP + DataMapPlot)

Sous-échantillon (`DATAMAP_MAX_POINTS`). UMAP sur `projected_embeddings.npy`.

Libellés OpenAI (`theme_summary`) : produits **hors notebook** avec
`bash jobs/enrich_scgm_themes_openai.sh` sur le **login** (accès Internet), après export SCGM.
Fichier attendu : `topics/themes_by_z_openai.csv`. Sinon `DATAMAP_LABEL_MODE = "macro_z"`.

`DATAMAP_SHOW_MACRO_CENTROIDS` : marqueurs `P` = moyenne UMAP par macro (A0–C).

- Statique : `datamap_segments.png`
- Interactif Plotly : `datamap_segments_interactive.html`
"""

DATAMAP_CODE = """from umap import UMAP

from scgm_text.notebook_viz import (
    display_plotly_html,
    macro_umap_centroids,
    plot_umap_datamap_static,
    plot_umap_plotly,
    resolve_datamap_labels,
)

if projected_btp is None or meta_btp is None:
    print("(absent) projected_btp / meta_btp — postprocess_scgm_text.sh")
else:
    projected_all = projected_btp
    meta_all = meta_btp
    if len(meta_all) != len(projected_all):
        raise ValueError(f"Alignement meta/embeddings : {len(meta_all)} vs {len(projected_all)}")
    n = min(DATAMAP_MAX_POINTS, len(projected_all))
    rng = np.random.default_rng(DATAMAP_SEED)
    idx = rng.choice(len(projected_all), size=n, replace=False)
    X = projected_all[idx]
    lab = meta_all.iloc[idx].copy()

    labels, label_kind = resolve_datamap_labels(
        lab,
        label_col=LABEL_COL,
        label_mode=DATAMAP_LABEL_MODE,
        themes_openai_path=PATHS["themes_openai"],
    )
    if label_kind == "theme_summary" and themes_openai is not None:
        _to = themes_openai
        _z2s = dict(zip(_to["z_id"].astype(int), _to["theme_summary"].astype(str)))
        lab["hover_theme"] = lab["z_hat"].map(lambda z: _z2s.get(int(z), f"z={int(z)}"))
    else:
        lab["hover_theme"] = lab[LABEL_COL].astype(str) + "|z=" + lab["z_hat"].astype(str)

    reducer = UMAP(
        n_components=2,
        random_state=DATAMAP_SEED,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
    )
    coords = reducer.fit_transform(X)

    centroids = None
    if DATAMAP_SHOW_MACRO_CENTROIDS:
        centroids = macro_umap_centroids(coords, lab[LABEL_COL].astype(str).to_numpy())

    fig, _ax = plot_umap_datamap_static(
        coords,
        labels,
        title="Segments BTP — embedding SCGM (normalisé)",
        label_font_size=8,
        macro_centroids=centroids,
    )
    out_png = FIGURES_DIR / "datamap_segments.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(out_png)

    fig_pl = plot_umap_plotly(
        coords,
        lab,
        label_col=LABEL_COL,
        hover_label="hover_theme",
        title="UMAP segments BTP (interactif)",
        out_html=FIGURES_DIR / "datamap_segments_interactive.html",
    )
    display_plotly_html(FIGURES_DIR / "datamap_segments_interactive.html")
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
        show_z_centroids=True,
        themes_z=themes_z,
    )
    plot_projection_plotly(pca_xy, tsne_xy, sample_df, LABEL_COL, figures_dir=FIGURES_DIR)
    display_plotly_html(FIGURES_DIR / "05_projection_pca_interactive.html")
    display_plotly_html(FIGURES_DIR / "05_projection_tsne_interactive.html")
"""

EVAL_RAW_PROJ_MD = """### 5g. Embedding brut BTP

Tableau géométrie sur les vecteurs **encodeur** (`metrics_geometry.csv` de `export_raw_embeddings.py`) : η², δ_macro, RankMe, C1, C10.

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
    draft = strip_outputs(json.loads(DRAFT.read_text(encoding="utf-8")))
    objective_md = draft["cells"][1]["source"]
    imports_code = draft["cells"][3]["source"]

    cells: list[dict] = [
        cell_from_source(
            "# 01 — SCGM Text (lecture seule)\n\n"
            "Analyse des sorties sous `resultats/scgm_text/`. "
            "Entraînement et export : `train_scgm_text.sh` puis `postprocess_scgm_text.sh`.\n",
            "markdown",
            "nb_title",
        ),
        cell_from_source(NOTEBOOK_TOC_MD, "markdown", "nb_toc"),
        cell_from_source("".join(objective_md), "markdown"),
        cell_from_source(
            "## 2. Imports\n\nRégler `OUTPUT_DIR` dans la cellule **Parameters**.\n",
            "markdown",
        ),
        {"cell_type": "code", "metadata": {}, "source": imports_code, "execution_count": None, "outputs": []},
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
        cell_from_source("### 5d. Topics par composante z\n", "markdown"),
        cell_from_source(THEMES_SOURCE, cell_id="themes01"),
        cell_from_source(BTP_PROJECTION_MD, "markdown"),
        cell_from_source(PROJECTION_CODE, cell_id="proj01"),
        cell_from_source(DATAMAP_MD, "markdown", cell_id="datamap_md"),
        cell_from_source(DATAMAP_CODE, cell_id="datamap01"),
        cell_from_source(EVAL_RAW_PROJ_MD, "markdown"),
        cell_from_source(EVAL_GEOMETRY_CODE, cell_id="eval_geom"),
        cell_from_source(TEST_MD, "markdown", "test_md"),
        cell_from_source(TEST_METRICS_SOURCE, cell_id="test_metrics"),
        cell_from_source(TEST_RAW_VIZ_MD, "markdown", "test_raw_viz_md"),
        cell_from_source(TEST_RAW_VIZ_CODE, cell_id="test_raw_viz"),
        cell_from_source("### 6c. Projections 2D test (SCGM projeté)\n", "markdown"),
        cell_from_source(TEST_VIZ_SOURCE, cell_id="test_viz"),
        cell_from_source("## Synthèse — comparaison géométrie train / test\n", "markdown"),
        cell_from_source(SUMMARY_TABLES_CODE, cell_id="summary_tables"),
    ]
    cells[5]["metadata"] = {"tags": ["parameters"]}

    nb = draft
    nb["cells"] = cells
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
