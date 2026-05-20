"""Génère notebooks/06_macro_transfer_topics.ipynb (lecture seule, corpus test configurable)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "06_macro_transfer_topics.ipynb"


def md(text: str) -> dict:
    src = [line + "\n" for line in text.strip().split("\n")]
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def py(text: str) -> dict:
    lines = text.strip().split("\n")
    src = [ln + "\n" for ln in lines]
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": src,
    }


def main() -> None:
    cells = [
        md(
            r"""
# 06 — Transfert macro-guidé + topics intra-macro (corpus test)

Encodeur source : checkpoints BTP sous `output/scgm_text/` ou `output/softtriple/` (job `METHOD=scgm_text|softtriple|both`).

Comparaison **SCGM** vs **SoftTriple** sur le corpus test (`TEST_CORPUS`, registre `configs/test_corpora.yaml`) :
- Phase 1 : \(p(m|u)\), métriques de classification (`transfer/`)
- Phase 2 : topics **BERTopic** vs **GMM** par macro
- Phase 3 : libellés OpenAI optionnels (`openai/`)

**Prérequis** : `CORPUS=<id> bash jobs/run_macro_transfer.sh` pour les deux méthodes.
"""
        ),
        py(
            r"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from safer_core.test_corpus import resolve_test_corpus, target_discovery_dir

# --- Parameters (modifier ici ou via papermill) ---
TEST_CORPUS = "metallurgie"

TEXT_ROOT = Path.cwd()
if not (TEXT_ROOT / "macro_transfer").is_dir():
    TEXT_ROOT = Path.cwd().parent

_spec = resolve_test_corpus(TEST_CORPUS, anchor=TEXT_ROOT)
SCGM_DIR = target_discovery_dir("scgm_text", _spec.id, anchor=TEXT_ROOT)
SOFT_DIR = target_discovery_dir("softtriple", _spec.id, anchor=TEXT_ROOT)
FIG_DIR = TEXT_ROOT / "notebooks" / "figures" / f"06_macro_transfer_{_spec.id}"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Corpus test : {_spec.display_name} ({_spec.id})")
print("SCGM :", SCGM_DIR)
print("SoftTriple :", SOFT_DIR)

sns.set_theme(style="whitegrid")
"""
        ),
        md("## § Transfert macro"),
        py(
            r"""
def load_transfer(root: Path, label: str):
    tdir = root / "transfer"
    metrics_path = tdir / "transfer_metrics.json"
    meta_path = tdir / "metadata_with_macro_probs.csv"
    if not metrics_path.is_file():
        print(f"[{label}] absent:", metrics_path)
        return None, None
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    meta = pd.read_csv(meta_path) if meta_path.is_file() else None
    return metrics, meta

rows = []
for label, root in (("SCGM", SCGM_DIR), ("SoftTriple", SOFT_DIR)):
    m, _ = load_transfer(root, label)
    if m:
        rows.append({
            "method": label,
            "n_eval": m.get("n_eval"),
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "balanced_accuracy": m.get("balanced_accuracy"),
            "mean_q_conf": m.get("mean_q_conf"),
        })
transfer_df = pd.DataFrame(rows)
display(transfer_df)
"""
        ),
        py(
            r"""
for label, root in (("SCGM", SCGM_DIR), ("SoftTriple", SOFT_DIR)):
    m, meta = load_transfer(root, label)
    if meta is None or "pred_label" not in meta.columns:
        continue
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(meta["q_conf"].dropna(), bins=30, color="steelblue", edgecolor="white")
    ax.axvline(0.5, color="crimson", ls="--", label="seuil 0.5")
    ax.set_title(f"Confiance macro — {label} ({TEST_CORPUS})")
    ax.set_xlabel("q_conf = max_m p(m|u)")
    ax.legend()
    fig.savefig(FIG_DIR / f"hist_q_conf_{label.lower()}.png", dpi=120, bbox_inches="tight")
    plt.show()
"""
        ),
        md("## § Topics BERTopic vs GMM"),
        py(
            r"""
def load_topics(root: Path, sub: str):
    p = root / sub / "themes_by_macro.csv"
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()

def load_openai(root: Path, name: str):
    p = root / "openai" / name
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()

for method_label, root in (("SCGM", SCGM_DIR), ("SoftTriple", SOFT_DIR)):
    comp_path = root / "summary/comparison_bertopic_vs_gmm.csv"
    if comp_path.is_file():
        print("===", method_label, "===")
        display(pd.read_csv(comp_path))
    for sub, tag in (("topics_bertopic", "BERTopic"), ("topics_gmm", "GMM")):
        th = load_topics(root, sub)
        if len(th):
            print(method_label, tag, "— effectifs par macro")
            display(th.groupby("macro")["n_units"].sum().reset_index())
"""
        ),
        py(
            r"""
for method_label, root in (("SCGM", SCGM_DIR), ("SoftTriple", SOFT_DIR)):
    for fname, tag in (
        ("themes_bertopic_openai.csv", "BERTopic"),
        ("themes_gmm_openai.csv", "GMM"),
    ):
        oai = load_openai(root, fname)
        if len(oai) and "theme_summary" in oai.columns:
            print(f"\n{method_label} — {tag} (OpenAI)")
            cols = ["macro", "topic_id", "n_units", "theme_summary"]
            display(oai[cols].head(12))
"""
        ),
        md("## § Qualité (support, topics vides)"),
        py(
            r"""
quality_rows = []
for method_label, root in (("SCGM", SCGM_DIR), ("SoftTriple", SOFT_DIR)):
    for sub, tag in (("topics_bertopic", "BERTopic"), ("topics_gmm", "GMM")):
        th = load_topics(root, sub)
        if not len(th):
            continue
        quality_rows.append({
            "method": method_label,
            "topic_algo": tag,
            "n_topics": th["topic_id"].nunique(),
            "n_empty": int((th["n_units"] == 0).sum()),
            "min_support": int(th["n_units"].min()),
            "median_support": float(th["n_units"].median()),
        })
display(pd.DataFrame(quality_rows))
"""
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("OK:", NB_PATH)


if __name__ == "__main__":
    main()
