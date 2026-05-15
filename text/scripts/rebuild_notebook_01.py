"""Rebuild 01_scgm_text_btp_experiment.ipynb from 01_draft + extras (OpenAI, DataMap, optimizer)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
DRAFT = NOTEBOOKS / "01_draft.ipynb"
OUT = NOTEBOOKS / "01_scgm_text_btp_experiment.ipynb"


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


PARAMS_SOURCE = """# Parameters
DATA_CSV = "dataset/data_btp.csv"
EMB_CSV = "embeddings/Qwen3-Embedding-0.6B_btp.csv"
OUTPUT_DIR = "runs/scgm_text_qwen06_notebook"
LABEL_COL = "pred_label"
PRED_OK_COL = "pred_ok"
GROUP_COL = "accident_id"
SEED = 42
VAL_RATIO = 0.1
RUN_FULL_TRAINING = True
EPOCHS_FAST = 10
EPOCHS_FULL = 50
BATCH_SIZE = 512
HIDDIM = 128
PROJECTION = "identity"  # identity | linear | mlp (ignoré si preset strict/pragmatic)
N_SUBCLASS = 32
TAU = 0.1
ALPHA = 0.5
LMD = 25
N_ITER_ESTEP = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
TSNE_SAMPLE_SIZE = 8000
DATAMAP_MAX_POINTS = 12000
DATAMAP_SEED = 42
DATAMAP_LABEL_MODE = "theme_summary"  # theme_summary | macro_z
N_OPENAI_EXAMPLE_TEXTS = 5
DATAMAP_SHOW_MACRO_CENTROIDS = True

# --- Optimiseur / fidélité SCGM-G ---
# pragmatic : AdamW, scheduler none, projection linear (défaut texte)
# strict    : SGD + momentum + cosine, projection mlp (proche SCGM-G officiel)
# custom    : OPTIMIZER, SCHEDULER, PROJECTION, LR ci-dessous
TRAINING_PRESET = "pragmatic"  # pragmatic | strict | custom
OPTIMIZER = "adamw"  # adamw | sgd (custom uniquement)
SCHEDULER = "none"  # none | cosine (custom uniquement)
MOMENTUM = 0.9
NUM_CYCLES = 10
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
from scgm_text.fidelity import describe_fidelity_mode
from scgm_text.utils_io import create_doc_id_if_missing, ensure_dir, get_dim_columns, load_json, set_seed

OUTPUT_PATH = Path(OUTPUT_DIR)
EXPORTS_DIR = OUTPUT_PATH / "exports"
EVAL_DIR = OUTPUT_PATH / "evaluation"
FIGURES_DIR = OUTPUT_PATH / "figures"
TABLES_DIR = OUTPUT_PATH / "tables"
for folder in [OUTPUT_PATH, EXPORTS_DIR, EVAL_DIR, FIGURES_DIR, TABLES_DIR]:
    ensure_dir(str(folder))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = EPOCHS_FULL if RUN_FULL_TRAINING else EPOCHS_FAST
set_seed(SEED)


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
    val_cols = [c for c in ["val_macro_f1", "val_balanced_acc", "val_acc"] if c in logs_df.columns]
    if val_cols:
        logs_df.plot(x="epoch", y=val_cols, ax=axes[1])
    plt.show()


print(f"REPO_ROOT={REPO_ROOT}")
print(f"OUTPUT_DIR={OUTPUT_PATH}")
print(f"DEVICE={device} | EPOCHS={epochs} | TRAINING_PRESET={TRAINING_PRESET}")
"""

SECTION8_MD = """## 8. Entraînement SCGM-G texte depuis le notebook

- **MODE RAPIDE** : `RUN_FULL_TRAINING = False` (10 epochs par défaut).
- **MODE COMPLET** : `RUN_FULL_TRAINING = True` (`EPOCHS_FULL` epochs).
- **Optimiseur** : `TRAINING_PRESET` en tête du notebook :
  - `pragmatic` — AdamW, pas de scheduler, projection `linear`.
  - `strict` — SGD + momentum + scheduler cosine, projection `mlp`.
  - `custom` — `OPTIMIZER`, `SCHEDULER`, `PROJECTION`, `LR` dans les paramètres.
- Entraînement via `run_training()` (même logique que `scripts/train_scgm_text.py`).
- Journaux : `metrics/train_log.csv` ou `logs.csv`, puis `show_training_progress()`.
"""

TRAIN_SOURCE = """import argparse
import importlib.util

train_script_path = REPO_ROOT / "scripts" / "train_scgm_text.py"
train_module_name = "scgm_train_text"
sys.modules.pop(train_module_name, None)
train_spec = importlib.util.spec_from_file_location(train_module_name, train_script_path)
scgm_train_text = importlib.util.module_from_spec(train_spec)
sys.modules[train_module_name] = scgm_train_text
train_spec.loader.exec_module(scgm_train_text)

preset = str(TRAINING_PRESET).strip().lower()
train_args = argparse.Namespace(
    config=None,
    run_name=None,
    data_csv=DATA_CSV,
    emb_csv=EMB_CSV,
    output_dir=str(OUTPUT_PATH),
    label_col=LABEL_COL,
    pred_ok_col=PRED_OK_COL,
    group_col=GROUP_COL,
    batch_size=BATCH_SIZE,
    epochs=epochs,
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    optimizer=OPTIMIZER,
    scheduler=SCHEDULER,
    num_cycles=NUM_CYCLES,
    hiddim=HIDDIM,
    n_class=4,
    n_subclass=N_SUBCLASS,
    tau=TAU,
    alpha=ALPHA,
    lmd=LMD,
    n_iter_estep=N_ITER_ESTEP,
    val_ratio=VAL_RATIO,
    seed=SEED,
    device="cuda" if torch.cuda.is_available() else "cpu",
    projection=PROJECTION,
    with_mlp=None,
    scgm_strict_mode=False,
    text_pragmatic_mode=False,
    use_self_distillation=False,
    kd_t=4.0,
    beta=1.0,
    beta1=None,
    beta2=None,
    beta3=None,
    teacher_mode="none",
    ema_decay=0.999,
    resume_from_checkpoint=None,
    num_workers=0,
    smoke_epochs=None,
)

if preset == "strict":
    train_args.scgm_strict_mode = True
elif preset == "pragmatic":
    train_args.text_pragmatic_mode = True
elif preset != "custom":
    raise ValueError(f"TRAINING_PRESET inconnu : {TRAINING_PRESET!r} (pragmatic | strict | custom)")

scgm_train_text.finalize_args(train_args)
print(describe_fidelity_mode(train_args))
print(
    f"optimizer={train_args.optimizer} scheduler={train_args.scheduler} "
    f"projection={train_args.projection} lr={train_args.lr}"
)
scgm_train_text.run_training(train_args)
show_training_progress()
"""

LOGS_SOURCE = """logs = _read_training_logs(OUTPUT_PATH)
display(logs.tail())

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
loss_cols = [c for c in ["train_loss", "loss_macro", "loss_latent"] if c in logs.columns]
if loss_cols:
    logs.plot(x="epoch", y=loss_cols, ax=axes[0, 0])
val_cols = [c for c in ["val_acc", "val_macro_f1", "val_balanced_acc"] if c in logs.columns]
if val_cols:
    logs.plot(x="epoch", y=val_cols, ax=axes[0, 1])
geom_cols = [c for c in ["rankme_global", "c1_global", "c10_global"] if c in logs.columns]
if geom_cols:
    logs.plot(x="epoch", y=geom_cols, ax=axes[1, 0])
axes[1, 1].axis("off")
save_fig("04_training_curves.png")
display_df_for_paper(logs, "training_logs.csv")
"""

CHECKPOINT_SOURCE = """checkpoint = torch.load(
    OUTPUT_PATH / "best_model.pt", map_location="cpu", weights_only=False
)
config = load_json(str(OUTPUT_PATH / "config.json"))
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
Figure : `FIGURES_DIR/05_projection_macro.png`.
"""

OPENAI_MD = """## 11 bis — Thèmes latents étiquetés par OpenAI (optionnel)

Nécessite `OPENAI_API_KEY` (optionnellement `OPENAI_BASE_URL`). Définir dans `.env` à la racine ou `scgm_text/.env` ; la cellule appelle `load_openai_dotenv()`. Ne pas versionner la clé.
"""

OPENAI_CODE = """import importlib

import scgm_text.openai_theme_labels as _openai_theme_labels

importlib.reload(_openai_theme_labels)
enrich_themes_by_z_openai = _openai_theme_labels.enrich_themes_by_z_openai
load_openai_dotenv = _openai_theme_labels.load_openai_dotenv

load_openai_dotenv()

themes_in = EXPORTS_DIR / "themes_by_z.csv"
themes_out = EXPORTS_DIR / "themes_by_z_openai.csv"
if not themes_in.exists():
    print(f"Fichier manquant : {themes_in} (exécuter l'export d'abord).")
elif not os.environ.get("OPENAI_API_KEY"):
    print("OPENAI_API_KEY non défini : enrichissement OpenAI ignoré.")
else:
    enrich_themes_by_z_openai(
        themes_in,
        themes_out,
        n_example_texts=N_OPENAI_EXAMPLE_TEXTS,
    )
    print(f"Écrit : {themes_out}")
"""

DATAMAP_MD = """## 11 ter — Carte 2D des segments (UMAP + DataMapPlot)

Sous-échantillon (`DATAMAP_MAX_POINTS`). UMAP sur `projected_embeddings.npy`.
Si `DATAMAP_LABEL_MODE == "theme_summary"` et `themes_by_z_openai.csv` existe, libellés = `theme_summary` via `z_hat`.
`DATAMAP_SHOW_MACRO_CENTROIDS` : marqueurs `P` = moyenne UMAP par macro (A0–C).
"""

DATAMAP_CODE = """import datamapplot as dmp
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from umap import UMAP

emb_path = EXPORTS_DIR / "projected_embeddings.npy"
meta_path = EXPORTS_DIR / "metadata_with_predictions.csv"
themes_openai_path = EXPORTS_DIR / "themes_by_z_openai.csv"

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
    lab = meta_all.iloc[idx]

    _mode = str(DATAMAP_LABEL_MODE).strip().lower()
    labels = None
    if _mode == "theme_summary" and themes_openai_path.is_file():
        _to = pd.read_csv(themes_openai_path)
        if "theme_summary" in _to.columns and "z_id" in _to.columns:
            _z2s = dict(zip(_to["z_id"].astype(int), _to["theme_summary"].astype(str)))
            labels = (
                lab["z_hat"]
                .map(lambda z: _z2s.get(int(z), f"z={int(z)}"))
                .to_numpy(dtype=object)
            )
        else:
            print("themes_by_z_openai.csv : colonnes theme_summary ou z_id absentes ; repli macro_z.")
    if labels is None:
        if _mode == "theme_summary" and not themes_openai_path.is_file():
            print(
                f"Fichier absent : {themes_openai_path} — exécuter la cellule OpenAI (11 bis) "
                "ou définir DATAMAP_LABEL_MODE='macro_z'. Repli macro|z."
            )
        labels = (lab[LABEL_COL].astype(str) + "|z=" + lab["z_hat"].astype(str)).to_numpy(dtype=object)

    reducer = UMAP(
        n_components=2,
        random_state=DATAMAP_SEED,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
    )
    coords = reducer.fit_transform(X)
    fig, ax = dmp.create_plot(
        coords,
        labels,
        title="Segments BTP — embedding SCGM (normalisé)",
        label_font_size=8,
    )
    if DATAMAP_SHOW_MACRO_CENTROIDS:
        macros_order = ["A0", "A1", "B", "C"]
        lab_mac = lab[LABEL_COL].astype(str).to_numpy()
        pal_hex = [mcolors.to_hex(c) for c in sns.color_palette("Set2", 4)]
        macro_to_color = dict(zip(macros_order, pal_hex))
        cx, cy, names = [], [], []
        for m in macros_order:
            mask = lab_mac == m
            if not np.any(mask):
                continue
            mu = coords[mask].mean(axis=0)
            cx.append(float(mu[0]))
            cy.append(float(mu[1]))
            names.append(m)
        if names:
            ax.scatter(
                cx,
                cy,
                s=240,
                c=[macro_to_color[m] for m in names],
                marker="P",
                edgecolors="#111111",
                linewidths=1.2,
                zorder=100,
            )
            for xi, yi, m in zip(cx, cy, names):
                ax.annotate(
                    m,
                    (xi, yi),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    color="#111111",
                    zorder=101,
                )
            leg_handles = [
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker="P",
                    color="w",
                    markerfacecolor=macro_to_color[m],
                    markeredgecolor="#111111",
                    markersize=11,
                    label=f"{m} — centroïde macro (moyenne UMAP)",
                )
                for m in names
            ]
            ax.legend(
                handles=leg_handles,
                loc="lower left",
                frameon=True,
                title="Macro (pred_label)",
                fontsize=8,
                title_fontsize=9,
            )
    out_png = FIGURES_DIR / "datamap_segments.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(out_png)
"""


def main() -> None:
    nb = strip_outputs(json.loads(DRAFT.read_text(encoding="utf-8")))
    cells = nb["cells"]

    cells[2]["source"] = [
        "## 2. Imports et configuration\n",
        "\n",
        "Régler **`TRAINING_PRESET`** (`pragmatic` | `strict` | `custom`) dans la cellule paramètres.\n",
    ]
    cells[4] = cell_from_source(PARAMS_SOURCE, cell_id="91307aa9")
    cells[4]["metadata"] = {"tags": ["parameters"]}
    cells[5] = cell_from_source(SETUP_SOURCE, cell_id="c308cd48")
    cells[19] = cell_from_source(SECTION8_MD, cell_type="markdown")
    cells[20] = cell_from_source(TRAIN_SOURCE, cell_id="5144d005")
    cells[22] = cell_from_source(LOGS_SOURCE, cell_id="386ed2ac")
    cells[24] = cell_from_source(CHECKPOINT_SOURCE, cell_id="c33b3818")

    # Insert 11 bis / 11 ter after export (index 27)
    insert_at = 27
    extras = [
        cell_from_source(OPENAI_MD, cell_type="markdown", cell_id="eb38d538"),
        cell_from_source(OPENAI_CODE, cell_id="3d4702b8"),
        cell_from_source(DATAMAP_MD, cell_type="markdown", cell_id="a429416f"),
        cell_from_source(DATAMAP_CODE, cell_id="2e816d8a"),
    ]
    cells[insert_at:insert_at] = extras

    # Section 12 markdown (was index 27, now 31)
    for c in cells:
        if get_source(c).startswith("## 12. Visualisation"):
            c["source"] = [line + "\n" for line in SECTION12_MD.strip().split("\n")]
            c["source"][-1] = c["source"][-1].rstrip("\n") + "\n"
            break

    # Drop empty trailing cells from draft
    while cells and not get_source(cells[-1]).strip():
        cells.pop()

    nb["cells"] = cells
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
