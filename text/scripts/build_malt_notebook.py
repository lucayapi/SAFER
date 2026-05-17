import json
import textwrap
from pathlib import Path


def md(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(True),
    }


def code(source: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": textwrap.dedent(source).strip().splitlines(True),
    }


cells = [
    md(
        """
        # 02 — MALT-EM Transfer: BTP -> Métallurgie

        Transfert macro-ancré avec boucle EM type SCGM (E-step Sinkhorn full-dataset, M-step à q fixé).

        **Lecture seule** : entraînement via `python scripts/train_malt_target.py` (ou job SLURM), puis analyse ici.
        """
    ),
    code(
        """
        # Parameters (Papermill : surcharger ces variables)
        SOURCE_CHECKPOINT = "resultats/scgm_text/checkpoints/best_model.pt"
        SOURCE_CONFIG = "resultats/scgm_text/configs/config.json"
        TARGET_DATA_CSV = "dataset/data_mettalurgie.csv"
        TARGET_DATA_CSV_ALT = "dataset/data_metallurgie.csv"
        TARGET_EMB_CSV = "embeddings/Qwen3-Embedding-0.6B_mettalurgie.csv"
        TARGET_EMB_CSV_ALT = "embeddings/Qwen3-Embedding-0.6B_metallurgie.csv"
        OUTPUT_DIR = "resultats/malt"
        SKIP_EXPORT_IF_PRESENT = True
        BATCH_SIZE = 512
        N_SUBCLASS = 32
        SEED = 42
        N_ITER_ESTEP = 5
        SINKHORN_LMD = 25.0
        EM_Q_MODE = "hard"
        BETA_ANCHOR = 1.0
        BETA_DIV = 0.1
        BETA_MACRO = 0.5
        """
    ),
    md(
        """
        ## 1. Objectif

        Source BTP, cible métallurgie. Macro cible = responsabilités souples `p0(y|x)` (pas de label dur).
        E-step : Sinkhorn sur tout le train ; M-step : `L_EM` avec q fixé + régularisations.
        """
    ),
    md(
        """
        ## 2. Imports et paramètres

        Les chemins et hyperparamètres d’exécution sont dans la **cellule Parameters** ci-dessus.

        Cellule suivante : importations Python, racine du dépôt `REPO_ROOT`, dossiers de sortie, résolution des CSV cible / embeddings.
        """
    ),
    code(
        """
        import importlib.util
        import sys
        from pathlib import Path

        import argparse
        import json

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import torch

        REPO_ROOT = Path.cwd()
        if not (REPO_ROOT / "malt_text").exists():
            REPO_ROOT = REPO_ROOT.parent
        sys.path.insert(0, str(REPO_ROOT))

        from malt_text.utils import resolve_existing_path, resolve_target_embedding_csv
        from scgm_text.utils_io import ensure_dir, load_json

        OUTPUT_PATH = REPO_ROOT / OUTPUT_DIR
        EXPORTS_DIR = OUTPUT_PATH / "exports"
        FIGURES_DIR = OUTPUT_PATH / "figures"
        EVAL_DIR = OUTPUT_PATH / "evaluation"
        for path in (OUTPUT_PATH, EXPORTS_DIR, FIGURES_DIR, EVAL_DIR):
            ensure_dir(str(path))

        # Chemins absolus (notebook souvent lancé hors racine du repo)
        resolved_data = resolve_existing_path(
            str(REPO_ROOT / TARGET_DATA_CSV),
            str(REPO_ROOT / TARGET_DATA_CSV_ALT),
            "target data CSV",
        )
        resolved_emb = resolve_target_embedding_csv(
            str(REPO_ROOT / TARGET_EMB_CSV),
            str(REPO_ROOT / TARGET_EMB_CSV_ALT),
            str(REPO_ROOT / "embeddings"),
        )
        print("data:", resolved_data)
        print("embeddings:", resolved_emb)
        """
    ),
    md("## 3. Vérification source"),
    code(
        """
        ckpt_path = REPO_ROOT / SOURCE_CHECKPOINT
        if not ckpt_path.exists():
            alt = REPO_ROOT / "resultats/scgm_text/checkpoints/best_model.pt"
            ckpt_path = alt if alt.exists() else ckpt_path
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print({k: checkpoint.get(k) for k in ["input_dim", "label2id"]})
        """
    ),
    md("## 4. Chargement cible"),
    code(
        """
        target_df = pd.read_csv(resolved_data)
        print(target_df.shape)
        print(target_df.columns.tolist())
        if "pred_label" in target_df.columns:
            display(target_df["pred_label"].value_counts())
        """
    ),
    md("## 5. Projection source de la cible"),
    code(
        """
        from malt_text.malt_dataset import MALTTargetDataset, build_target_dataloader
        from malt_text.malt_transfer import compute_p0_target
        from malt_text.utils import load_source_scgm, select_device

        device = select_device("cuda" if torch.cuda.is_available() else "cpu")
        source_model, _, _, input_dim = load_source_scgm(str(ckpt_path), device)
        dataset = MALTTargetDataset(resolved_data, resolved_emb, expected_input_dim=input_dim)
        loader = build_target_dataloader(dataset, BATCH_SIZE, False)
        projected_source, p0 = compute_p0_target(source_model, loader, device, 0.1)
        pd.Series(p0.argmax(1)).value_counts()
        """
    ),
    md(
        """
        ## 6. Résultats MALT-EM (lecture seule)

        Entraînement **hors notebook** :

        ```bash
        python scripts/train_malt_target.py --config configs/malt_btp_to_mettalurgie_qwen06.yaml
        ```

        Attendu : `resultats/malt/best_model.pt` et `resultats/malt/metrics/train_log.csv`.
        """
    ),
    code(
        """
        best_ckpt = OUTPUT_PATH / "best_model.pt"
        log_path = OUTPUT_PATH / "metrics" / "train_log.csv"
        if not log_path.is_file():
            log_path = OUTPUT_PATH / "logs.csv"
        missing = []
        if not best_ckpt.is_file():
            missing.append(str(best_ckpt))
        if not log_path.is_file():
            missing.append("metrics/train_log.csv ou logs.csv")
        if missing:
            raise FileNotFoundError(
                "Artefacts MALT manquants. Lancez d'abord :\\n"
                "  python scripts/train_malt_target.py --config configs/malt_btp_to_mettalurgie_qwen06.yaml\\n"
                f"Manquant : {missing}"
            )
        print("Checkpoint:", best_ckpt)
        print("Journal:", log_path)
        """
    ),
    md("## 7. Courbes d'entraînement"),
    code(
        """
        log_path = OUTPUT_PATH / "metrics" / "train_log.csv"
        if not log_path.is_file():
            log_path = OUTPUT_PATH / "logs.csv"
        logs = pd.read_csv(log_path)
        ycols = [c for c in ["loss_total", "loss_em", "loss_z", "loss_yz", "loss_anchor", "loss_div", "loss_macro"] if c in logs.columns]
        logs.plot(x="epoch", y=ycols)
        plt.savefig(FIGURES_DIR / "malt_em_losses.png", dpi=150)
        em_cols = [c for c in logs.columns if c.startswith("estep_")]
        if em_cols:
            logs.plot(x="epoch", y=em_cols)
            plt.savefig(FIGURES_DIR / "malt_em_diagnostics.png", dpi=150)
        plt.show()
        """
    ),
    md("## 8. Export MALT"),
    code(
        """
        _meta = EXPORTS_DIR / "metadata_with_malt_predictions.csv"
        if SKIP_EXPORT_IF_PRESENT and _meta.is_file():
            print(f"Export ignoré (déjà présent) : {_meta}")
        else:
            import subprocess

            export_cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "export_malt_outputs.py"),
                "--checkpoint",
                str(OUTPUT_PATH / "best_model.pt"),
                "--source_checkpoint",
                str(ckpt_path),
                "--target_data_csv",
                TARGET_DATA_CSV,
                "--target_emb_csv",
                TARGET_EMB_CSV,
                "--output_dir",
                str(EXPORTS_DIR),
                "--batch_size",
                str(BATCH_SIZE),
                "--device",
                "cuda" if torch.cuda.is_available() else "cpu",
            ]
            print(" ".join(export_cmd))
            subprocess.run(export_cmd, cwd=str(REPO_ROOT), check=True)
        sorted(p.name for p in EXPORTS_DIR.iterdir())
        """
    ),
    md(
        """
        ## 9. Évaluation du transfert

        Tableau principal **`metrics_table.csv`** : **eta2_macro_balanced** (principal) et **eta2_weighted** (secondaire), inertie macro sur distance euclidienne au carré (0–1), plus RankMe, C1 et C10. Lignes : Embedding brut (si dispo), MALT_source, MALT_adapted.
        """
    ),
    code(
        """
        eval_script = REPO_ROOT / "scripts" / "evaluate_malt_transfer.py"
        spec = importlib.util.spec_from_file_location("malt_eval", eval_script)
        malt_eval = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(malt_eval)
        malt_eval.run_evaluate(
            argparse.Namespace(
                exports_dir=str(EXPORTS_DIR),
                output_dir=str(EVAL_DIR),
                label_col="pred_label",
            )
        )
        display(pd.read_csv(EVAL_DIR / "metrics_table.csv"))
        with open(EVAL_DIR / "metrics_summary.json", encoding="utf-8") as handle:
            malt_eval_summary = json.load(handle)
        print("change_rate:", malt_eval_summary["malt_diagnostics"].get("change_rate"))
        """
    ),
    md("## 10. Visualisation avant/après adaptation"),
    code(
        """
        from sklearn.decomposition import PCA
        from malt_text.malt_visualization import save_pca_scatter

        meta = pd.read_csv(EXPORTS_DIR / "metadata_with_malt_predictions.csv")
        projected_source = np.load(EXPORTS_DIR / "target_projected_source.npy")
        projected_adapted = np.load(EXPORTS_DIR / "target_projected_adapted.npy")
        meta = pd.read_csv(EXPORTS_DIR / "metadata_with_malt_predictions.csv")
        pca = PCA(n_components=2, random_state=SEED)
        source_xy = pca.fit_transform(projected_source)
        adapted_xy = pca.fit_transform(projected_adapted)
        save_pca_scatter(source_xy, meta["p0_macro_name"].to_numpy(), FIGURES_DIR / "pca_source_projected_by_p0.png", "PCA source-projected by p0")
        save_pca_scatter(adapted_xy, meta["pt_macro_name"].to_numpy(), FIGURES_DIR / "pca_adapted_by_pt.png", "PCA adapted by pt")
        save_pca_scatter(adapted_xy, meta["z_hat"].to_numpy(), FIGURES_DIR / "pca_adapted_by_z.png", "PCA adapted by z")
        """
    ),
    md("## 11. Analyse des ancres macro"),
    code(
        """
        mu_source = np.load(EXPORTS_DIR / "mu_y_source.npy")
        mu_target = np.load(EXPORTS_DIR / "mu_y_target.npy")
        from malt_text.malt_metrics import anchor_drift_metrics
        from malt_text.malt_visualization import save_heatmap

        drift = anchor_drift_metrics(mu_source, mu_target)
        pd.Series(drift)
        cosine = (mu_source / np.linalg.norm(mu_source, axis=1, keepdims=True)) @ (
            mu_target / np.linalg.norm(mu_target, axis=1, keepdims=True)
        ).T
        save_heatmap(cosine, ["A0", "A1", "B", "C"], ["A0", "A1", "B", "C"], FIGURES_DIR / "anchor_similarity.png", "Anchor cosine similarity")
        """
    ),
    md("## 12. Analyse des motifs latents globaux"),
    code(
        """
        meta = pd.read_csv(EXPORTS_DIR / "metadata_with_malt_predictions.csv")
        prob_y_z = np.load(EXPORTS_DIR / "pt_y_given_z.npy")
        z_sizes = meta["z_hat"].value_counts().sort_index()
        z_sizes.plot(kind="bar", figsize=(10, 4), title="Taille des clusters z")
        plt.savefig(FIGURES_DIR / "z_cluster_sizes.png", dpi=150)
        plt.show()
        sns.heatmap(prob_y_z, annot=False, cmap="viridis")
        plt.title("p(y|z)")
        plt.savefig(FIGURES_DIR / "py_given_z.png", dpi=150)
        plt.show()
        """
    ),
    md("## 13. Thèmes par z"),
    code(
        """
        themes = pd.read_csv(EXPORTS_DIR / "themes_by_z_malt.csv")

        def display_malt_theme(z_id: int):
            row = themes.loc[themes["z_id"] == z_id].iloc[0]
            display(row)

        display_malt_theme(int(themes.sort_values("n_units", ascending=False).iloc[0]["z_id"]))
        """
    ),
    md("## 14. Analyse des changements p0 -> pt"),
    code(
        """
        meta = pd.read_csv(EXPORTS_DIR / "metadata_with_malt_predictions.csv")
        changed = meta[meta["p0_macro_name"] != meta["pt_macro_name"]].copy()
        changed.head(20).to_csv(OUTPUT_PATH / "macro_changed_examples.csv", index=False)
        transition = pd.crosstab(meta["p0_macro_name"], meta["pt_macro_name"])
        sns.heatmap(transition, annot=True, fmt="d", cmap="Blues")
        plt.savefig(FIGURES_DIR / "macro_transition_heatmap.png", dpi=150)
        plt.show()
        """
    ),
    md("## 15. Comparaison source-only vs MALT"),
    code(
        """
        p0 = np.load(EXPORTS_DIR / "p0_y_target.npy")
        pt = np.load(EXPORTS_DIR / "pt_y_target.npy")
        comparison = pd.DataFrame(
            [
                {"method": "source_only_p0", "mean_entropy": float(-(p0 * np.log(np.clip(p0, 1e-12, 1))).sum(1).mean())},
                {"method": "malt_adapted_pt", "mean_entropy": float(-(pt * np.log(np.clip(pt, 1e-12, 1))).sum(1).mean())},
            ]
        )
        comparison.to_csv(OUTPUT_PATH / "transfer_comparison_table.csv", index=False)
        comparison
        """
    ),
    md("## 16. Rapport automatique"),
    code(
        """
        report_path = OUTPUT_PATH / "experiment_report_malt.md"
        report_path.write_text(
            f"# Rapport MALT\\n\\n- source: {ckpt_path}\\n- target: {resolved_data}\\n- embeddings: {resolved_emb}\\n- K global: {N_SUBCLASS}\\n",
            encoding="utf-8",
        )
        report_path
        """
    ),
    md("## 17. Conclusion\n\nLe transfert montre l'adaptation des ancres macro et la réémergence locale des motifs latents. Les labels cible restent diagnostiques."),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "cells": cells,
}

output = Path(__file__).resolve().parents[1] / "notebooks" / "02_malt_btp_to_mettalurgie_transfer.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {output} with {len(cells)} cells")
