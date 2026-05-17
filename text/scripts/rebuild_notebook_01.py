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
SKIP_EXPORT_IF_PRESENT = True  # True : export seulement si projected_embeddings.npy absent

DATA_CSV = "dataset/data_btp.csv"
EMB_CSV = "embeddings/Qwen3-Embedding-0.6B_btp.csv"
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

OUTPUT_PATH = Path(OUTPUT_DIR)
CHECKPOINTS_DIR = OUTPUT_PATH / "checkpoints"
CHECKPOINT_PATH = Path(CHECKPOINT_PATH) if CHECKPOINT_PATH else CHECKPOINTS_DIR / "best_model.pt"
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


def require_scgm_artifacts(
    output_path: Path = OUTPUT_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> None:
    missing: list[str] = []
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        missing.append(str(ckpt))
    try:
        _read_training_logs(output_path)
    except FileNotFoundError:
        missing.append("metrics/train_log.csv ou logs.csv")
    if missing:
        raise FileNotFoundError(
            "Artefacts SCGM manquants — entraînez hors notebook, par ex.:\\n"
            "  cd text/jobs && sbatch train_scgm_text.sh\\n"
            "  python scripts/train_scgm_text.py --config configs/scgm_text_strict_fidelity.yaml "
            "--scgm_strict_mode --output_dir resultats/scgm_text\\n"
            f"Manquant : {missing}"
        )


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

SECTION8_MD = """## 8. Résultats d'entraînement (lecture seule)

**Pas d'entraînement dans ce notebook.** Produire d'abord les artefacts sous `OUTPUT_DIR` :

```bash
cd text/jobs && sbatch train_scgm_text.sh
# ou
python scripts/train_scgm_text.py \\
  --config configs/scgm_text_strict_fidelity.yaml \\
  --scgm_strict_mode \\
  --output_dir resultats/scgm_text
```

Puis export (si `embeddings/projected_embeddings.npy` absent) :

```bash
python scripts/export_scgm_text_outputs.py \\
  --checkpoint resultats/scgm_text/checkpoints/best_model.pt \\
  --output_dir resultats/scgm_text
```

Attendu : `checkpoints/best_model.pt`, `metrics/train_log.csv` (ou `logs.csv`), puis exports sous `embeddings/` et `topics/`.

Libellés OpenAI pour la carte DataMap : `bash jobs/enrich_scgm_themes_openai.sh` sur le **login** (pas dans ce notebook).
"""

RESULTS_SOURCE = """require_scgm_artifacts()

if run_config:
    display(
        pd.Series(
            {
                "fidelity_mode": run_config.get("fidelity_mode"),
                "best_checkpoint_epoch": run_config.get("best_checkpoint_epoch"),
                "best_checkpoint_metric": run_config.get("best_checkpoint_metric"),
                "best_checkpoint_score": run_config.get("best_checkpoint_score"),
                "output_dir": run_config.get("output_dir", str(OUTPUT_PATH)),
            }
        )
    )

show_training_progress()
"""

LOGS_SOURCE = """logs = _read_training_logs(OUTPUT_PATH)
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

EXPORT_SOURCE = """def _verify_export_layout() -> None:
    required = {
        "projected_embeddings.npy": EXPORTS_DIR / "projected_embeddings.npy",
        "metadata_with_predictions.csv": EXPORTS_DIR / "metadata_with_predictions.csv",
        "themes_by_z.csv": TOPICS_DIR / "themes_by_z.csv",
    }
    missing = {k: str(v) for k, v in required.items() if not v.is_file()}
    if not missing:
        return
    nested = EXPORTS_DIR / "embeddings" / "projected_embeddings.npy"
    if nested.is_file():
        raise FileNotFoundError(
            "Export écrit sous embeddings/embeddings/ (mauvais --output_dir). "
            f"Relancez avec --output_dir {OUTPUT_PATH} (racine du run, pas le sous-dossier embeddings/)."
        )
    raise FileNotFoundError(f"Export incomplet — fichiers manquants : {missing}")


_proj = EXPORTS_DIR / "projected_embeddings.npy"
if SKIP_EXPORT_IF_PRESENT and _proj.is_file():
    print(f"Export ignoré (déjà présent) : {_proj}")
else:
    export_cmd = [
        sys.executable,
        "scripts/export_scgm_text_outputs.py",
        "--data_csv", DATA_CSV,
        "--emb_csv", EMB_CSV,
        "--checkpoint", str(CHECKPOINT_PATH),
        "--output_dir", str(OUTPUT_PATH),
        "--label_col", LABEL_COL,
        "--pred_ok_col", PRED_OK_COL,
        "--group_col", GROUP_COL,
        "--batch_size", str(BATCH_SIZE),
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
    ]
    run_cli(export_cmd)
    for sub in ("embeddings", "topics", "assignments"):
        folder = OUTPUT_PATH / sub
        if folder.is_dir():
            names = sorted(p.name for p in folder.iterdir() if p.is_file())
            print(f"{sub}/ ({len(names)} fichiers):", names[:12], "..." if len(names) > 12 else "")

_verify_export_layout()
print("Export OK —", _proj)
"""

THEMES_SOURCE = """def _topics_csv(name: str) -> Path:
    \"\"\"themes_by_z*.csv : topics/ (export SCGM) ; repli embeddings/ (anciens runs).\"\"\"
    for base in (TOPICS_DIR, EXPORTS_DIR):
        path = base / name
        if path.is_file():
            if base != TOPICS_DIR:
                print(f"WARN: {name} sous {base} — attendu {TOPICS_DIR}")
            return path
    raise FileNotFoundError(
        f"{name} introuvable sous {TOPICS_DIR} ni {EXPORTS_DIR}. "
        "Relancer l'export : python scripts/export_scgm_text_outputs.py "
        f"--checkpoint {CHECKPOINT_PATH} --output_dir {OUTPUT_PATH}"
    )


themes_z = pd.read_csv(_topics_csv("themes_by_z.csv"))
themes_macro = pd.read_csv(_topics_csv("themes_by_macro_z.csv"))
display(themes_macro)
top_components = themes_z.sort_values("n_units", ascending=False).head(15)
display_df_for_paper(top_components, "top_themes_by_z.csv")
"""

CHECKPOINT_SOURCE = """checkpoint = torch.load(
    CHECKPOINT_PATH, map_location="cpu", weights_only=False
)
config = load_json(str(OUTPUT_PATH / "configs" / "config.json"))
summary_ckpt = {
    "input_dim": checkpoint.get("input_dim"),
    "label2id": checkpoint.get("label2id"),
    "train_size": len(checkpoint.get("train_idx", [])),
    "val_size": len(checkpoint.get("val_idx", [])),
    "args": checkpoint.get("args", {}),
}
display(pd.json_normalize(summary_ckpt["args"]))
pd.Series({
    "input_dim": summary_ckpt["input_dim"],
    "train_size": summary_ckpt["train_size"],
    "val_size": summary_ckpt["val_size"],
})
"""

SECTION12_MD = """## 12. Visualisation des embeddings projetés

PCA + t-SNE sur un sous-échantillon (`TSNE_SAMPLE_SIZE`). Couleur = macro (`pred_label`).

- Statique : `FIGURES_DIR/05_projection_macro.png`
- Interactif Plotly : `05_projection_pca_interactive.html`, `05_projection_tsne_interactive.html`
"""

DATAMAP_MD = """## 11 bis — Carte 2D des segments (UMAP + DataMapPlot)

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

emb_path = EXPORTS_DIR / "projected_embeddings.npy"
meta_path = EXPORTS_DIR / "metadata_with_predictions.csv"
themes_openai_path = TOPICS_DIR / "themes_by_z_openai.csv"

if not emb_path.exists() or not meta_path.exists():
    print("Embeddings projetés ou métadonnées manquants ; exécuter l'export.")
else:
    projected_all = np.load(emb_path)
    meta_all = pd.read_csv(meta_path)
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
        themes_openai_path=themes_openai_path,
    )
    if label_kind == "theme_summary":
        _to = pd.read_csv(themes_openai_path)
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

projected = np.load(EXPORTS_DIR / "projected_embeddings.npy")
meta_pred = pd.read_csv(EXPORTS_DIR / "metadata_with_predictions.csv")

idx = sample_projection_indices(meta_pred, LABEL_COL, max_points=TSNE_SAMPLE_SIZE, seed=SEED)
sample_df = meta_pred.loc[idx]
sample_x = projected[idx]

pca = PCA(n_components=2, random_state=SEED)
pca_xy = pca.fit_transform(sample_x)
tsne = TSNE(n_components=2, random_state=SEED, init="pca", learning_rate="auto")
tsne_xy = tsne.fit_transform(sample_x)

plot_projection_matplotlib(pca_xy, tsne_xy, sample_df, LABEL_COL, save_fig=save_fig)
_, pca_html = plot_projection_plotly(pca_xy, tsne_xy, sample_df, LABEL_COL, figures_dir=FIGURES_DIR)
print(pca_html)
display_plotly_html(FIGURES_DIR / "05_projection_pca_interactive.html")
display_plotly_html(FIGURES_DIR / "05_projection_tsne_interactive.html")
"""

EVAL_RAW_PROJ_MD = """### Carte 2D — embedding brut (PCA + t-SNE statiques, couleur = macro)

Sous-échantillon (`RAW_EMBEDDING_UMAP_MAX_POINTS`) sur les vecteurs **encodeur** (`EMB_CSV` / `raw_embeddings.npy`), colorés par `pred_label` (A0–C). Figure statique : `09_raw_embedding_pca_tsne.png`.
"""

EVAL_GEOMETRY_CODE = """eval_cmd = [
    sys.executable,
    "scripts/evaluate_scgm_text.py",
    "--exports_dir", str(EXPORTS_DIR),
    "--output_dir", str(EVAL_DIR),
    "--label_col", LABEL_COL,
    "--emb_csv", EMB_CSV,
]
run_cli(eval_cmd)

import importlib
import scgm_text.notebook_viz as _notebook_viz
importlib.reload(_notebook_viz)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scgm_text.notebook_viz import (
    display_plotly_html,
    plot_evaluation_geometry_dashboard,
    plot_projection_matplotlib,
    sample_projection_indices,
)

metrics_table = pd.read_csv(EVAL_DIR / "metrics_table.csv")
display(metrics_table)

html_paths = plot_evaluation_geometry_dashboard(
    metrics_table,
    figures_dir=FIGURES_DIR,
    save_fig=save_fig,
)
for p in html_paths:
    print(p)
display_plotly_html(FIGURES_DIR / "09_eta2_macro_interactive.html")
display_plotly_html(FIGURES_DIR / "09_rankme_c1_c10_global_interactive.html")

# --- PCA / t-SNE 2D embedding brut (macro, statique) ---
meta_eval = pd.read_csv(EXPORTS_DIR / "metadata_with_predictions.csv")
raw_npy = EXPORTS_DIR / "raw_embeddings.npy"
if raw_npy.is_file():
    raw_emb = np.load(raw_npy)
elif Path(EMB_CSV).is_file():
    from scgm_text.dataset_text_embeddings import merge_metadata_with_embeddings
    slim = meta_eval.drop(columns=[c for c in meta_eval.columns if c.startswith("dim_")], errors="ignore")
    merged, dim_cols = merge_metadata_with_embeddings(slim, EMB_CSV)
    raw_emb = merged[dim_cols].to_numpy(dtype=np.float64)
else:
    raise FileNotFoundError("raw_embeddings.npy ou EMB_CSV requis pour la carte embedding brut.")

idx_raw = sample_projection_indices(
    meta_eval, LABEL_COL, max_points=RAW_EMBEDDING_UMAP_MAX_POINTS, seed=SEED
)
sample_raw_df = meta_eval.loc[idx_raw]
sample_raw_x = raw_emb[idx_raw]

pca_raw = PCA(n_components=2, random_state=SEED)
pca_raw_xy = pca_raw.fit_transform(sample_raw_x)
tsne_raw = TSNE(n_components=2, random_state=SEED, init="pca", learning_rate="auto")
tsne_raw_xy = tsne_raw.fit_transform(sample_raw_x)

p_raw = plot_projection_matplotlib(
    pca_raw_xy,
    tsne_raw_xy,
    sample_raw_df,
    LABEL_COL,
    save_fig=save_fig,
    png_name="09_raw_embedding_pca_tsne.png",
    pca_title="PCA 2D — embedding brut (macro)",
    tsne_title="t-SNE 2D — embedding brut (macro)",
)
print(p_raw)
"""

PAPER_TABLES_CODE = """display_df_for_paper(metrics_table, "paper_metrics_summary.csv")
display_df_for_paper(themes_macro, "paper_themes_by_macro.csv")

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
    nb = strip_outputs(json.loads(DRAFT.read_text(encoding="utf-8")))
    cells = nb["cells"]

    cells[2]["source"] = [
        "## 2. Imports et configuration\n",
        "\n",
        "Notebook **lecture seule** : régler `OUTPUT_DIR` et `CHECKPOINT_PATH` dans la cellule paramètres. "
        "L'entraînement se fait via `scripts/train_scgm_text.py` ou `jobs/train_scgm_text.sh`.\n",
    ]
    cells[4] = cell_from_source(PARAMS_SOURCE, cell_id="91307aa9")
    cells[4]["metadata"] = {"tags": ["parameters"]}
    cells[5] = cell_from_source(SETUP_SOURCE, cell_id="c308cd48")
    cells[19] = cell_from_source(SECTION8_MD, cell_type="markdown")
    cells[20] = cell_from_source(RESULTS_SOURCE, cell_id="5144d005")
    cells[22] = cell_from_source(LOGS_SOURCE, cell_id="386ed2ac")
    cells[24] = cell_from_source(CHECKPOINT_SOURCE, cell_id="c33b3818")

    # DataMap après export (index 27)
    insert_at = 27
    extras = [
        cell_from_source(DATAMAP_MD, cell_type="markdown", cell_id="a429416f"),
        cell_from_source(DATAMAP_CODE, cell_id="2e816d8a"),
    ]
    cells[insert_at:insert_at] = extras

    # Section 12 markdown + projection / évaluation
    for c in cells:
        if get_source(c).startswith("## 12. Visualisation"):
            c["source"] = [line + "\n" for line in SECTION12_MD.strip().split("\n")]
            c["source"][-1] = c["source"][-1].rstrip("\n") + "\n"
            break

    replace_cell_by_prefix(cells, "export_cmd = [", EXPORT_SOURCE)
    replace_cell_by_prefix(cells, "themes_z = pd.read_csv", THEMES_SOURCE)
    replace_cell_by_prefix(cells, "def _topics_csv", THEMES_SOURCE)
    replace_cell_by_prefix(cells, "from sklearn.decomposition import PCA", PROJECTION_CODE)
    replace_cell_by_prefix(cells, "eval_cmd = [", EVAL_GEOMETRY_CODE)
    # Markdown UMAP embedding brut (juste avant la cellule eval)
    for i, c in enumerate(cells):
        if get_source(c).startswith("eval_cmd = ["):
            cells.insert(i, cell_from_source(EVAL_RAW_PROJ_MD, cell_type="markdown"))
            break
    replace_cell_by_prefix(cells, "confusion = pd.read_csv", PAPER_TABLES_CODE)

    # Drop empty trailing cells from draft
    while cells and not get_source(cells[-1]).strip():
        cells.pop()

    for c in cells:
        if c.get("cell_type") == "markdown":
            src = get_source(c)
            if "SCGM Text BTP" in src:
                c["source"] = [
                    ln.replace("SCGM Text BTP Experiment", "SCGM Text — analyse (lecture seule)")
                    .replace("SCGM Text Experiment", "SCGM Text — analyse (lecture seule)")
                    .replace("BTP Experiment", "analyse (lecture seule)")
                    for ln in c["source"]
                ]
                if "lecture seule" not in "".join(c["source"]):
                    c["source"].insert(
                        1,
                        "\n**Analyse et visualisation** des sorties sous `resultats/scgm_text/` — pas d'entraînement ici.\n",
                    )
                break

    nb["cells"] = cells
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
