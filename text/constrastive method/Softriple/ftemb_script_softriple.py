import os
import json
import argparse
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

import ftemb_softtriple


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_float(x: Any, default: float = float("-inf")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    v = str(v).strip().lower()
    if v in {"true", "1", "yes", "y", "oui", "on"}:
        return True
    if v in {"false", "0", "no", "n", "non", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Valeur booléenne invalide: {v}. Utilise true/false, 1/0, yes/no."
    )


def combo_name(
    lr: float,
    bs: int,
    ep: int,
    centers_per_class: int,
    gamma: float,
    la: float,
    delta: float,
    tau: float,
) -> str:
    lr_txt = f"{lr:.0e}".replace("+0", "").replace("+", "")
    gamma_txt = str(gamma).replace(".", "p")
    la_txt = str(la).replace(".", "p")
    delta_txt = str(delta).replace(".", "p")
    tau_txt = str(tau).replace(".", "p")
    return (
        f"lr_{lr_txt}"
        f"__bs_{bs}"
        f"__ep_{ep}"
        f"__k_{centers_per_class}"
        f"__gamma_{gamma_txt}"
        f"__lambda_{la_txt}"
        f"__delta_{delta_txt}"
        f"__tau_{tau_txt}"
    )


def build_default_output_root(target: str) -> str:
    return f"models_research/grid_search_softtriple_qwen_06_{target}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search + final training pour fine-tuning SoftTriple."
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/Qwen_Qwen3-8B__v1__snapshot.csv",
        help="Chemin du CSV d'entrée."
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["pred_label", "pred_subtype"],
        default="pred_label",
        help="Colonne cible à utiliser pour l'entraînement et la sélection."
    )
    parser.add_argument(
        "--grid_output_root",
        type=str,
        default=None,
        help="Dossier racine de sortie. Si absent, construit automatiquement selon le target."
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Nom du modèle HF de base."
    )

    parser.add_argument(
        "--use_contextual_prompt_with_summary",
        type=str2bool,
        default=False,
        help="Active ou non le mode prompt+récit+phrase."
    )
    parser.add_argument(
        "--summary_col",
        type=str,
        default="accident_summary",
        help="Nom de la colonne résumé."
    )
    parser.add_argument(
        "--contextual_prompt_template",
        type=str,
        default=(
            "Tâche: encoder une unité factuelle d'accident avec son contexte narratif.\n"
            "Récit: {accident_summary}\n"
            "Phrase: {sentence}"
        ),
        help="Template utilisé si use_contextual_prompt_with_summary=True."
    )

    parser.add_argument(
        "--use_fixed_instruction_prefix",
        type=str2bool,
        default=False,
        help="Active ou non le préfixe d'instruction fixe."
    )
    parser.add_argument(
        "--fixed_instruction_prefix",
        type=str,
        default=(
            "Represent this occupational accident factual unit according to its "
            "prevention-relevant role in the accident scenario."
        ),
        help="Préfixe d'instruction fixe à ajouter avant le texte encodé."
    )

    # =====================================================
    # Paramètres data / training
    # =====================================================
    parser.add_argument("--text_col", type=str, default="sentence")
    parser.add_argument("--group_col", type=str, default="accident_id")
    parser.add_argument("--unit_id_col", type=str, default="doc_id")

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--min_count_per_class_in_train", type=int, default=4)

    parser.add_argument("--batch_size_eval", type=int, default=64)
    parser.add_argument("--batch_size_encode", type=int, default=128)

    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=False)

    parser.add_argument("--normalize_for_eval", type=str2bool, default=True)
    parser.add_argument(
        "--singleton_policy",
        type=str,
        choices=["zero", "drop"],
        default="zero",
    )
    parser.add_argument(
        "--max_eval_triplets",
        type=int,
        default=3000,
        help="Conservé pour compatibilité API. Non utilisé ici."
    )

    parser.add_argument(
        "--softtriple_normalize_train_embeddings",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--softtriple_normalize_centers",
        type=str2bool,
        default=True,
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Ex: cuda, cpu. Si absent, détection auto."
    )

    parser.add_argument(
        "--run_final_training_on_full_data",
        type=str2bool,
        default=True,
        help="Lance ou non le training final avec la meilleure config."
    )
    parser.add_argument(
        "--hf_cache_folder",
        type=str,
        default="./hf_cache",
    )
    parser.add_argument(
        "--export_embeddings_root",
        type=str,
        default="fnembeddings_grid_softtriple",
        help=(
            "Dossier où exporter les embeddings fold par fold, "
            "avec colonnes SoftTriple : softtriple_pred_macro, "
            "softtriple_pred_center, softtriple_pred_center_score."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # =====================================================
    # PARAMÈTRES GÉNÉRAUX
    # =====================================================
    INPUT_CSV = args.input_csv
    TARGET = args.target
    GRID_OUTPUT_ROOT = args.grid_output_root or build_default_output_root(TARGET)

    BASE_MODEL_NAME = args.base_model_name

    TEXT_COL = args.text_col
    LABEL_COL = TARGET
    GROUP_COL = args.group_col
    UNIT_ID_COL = args.unit_id_col

    SUMMARY_COL = args.summary_col
    USE_CONTEXTUAL_PROMPT_WITH_SUMMARY = args.use_contextual_prompt_with_summary
    CONTEXTUAL_PROMPT_TEMPLATE = args.contextual_prompt_template

    USE_FIXED_INSTRUCTION_PREFIX = args.use_fixed_instruction_prefix
    FIXED_INSTRUCTION_PREFIX = args.fixed_instruction_prefix
    FIXED_INSTRUCTION_PREFIX_TO_USE: Optional[str] = (
        FIXED_INSTRUCTION_PREFIX if USE_FIXED_INSTRUCTION_PREFIX else None
    )

    EVAL_LABEL_COLS = ["pred_label", "pred_subtype"]
    SELECTION_LABEL_COL = TARGET

    HF_CACHE_FOLDER = args.hf_cache_folder

    # =====================================================
    # K-FOLD SIMPLE
    # =====================================================
    N_SPLITS = args.n_splits
    MIN_COUNT_PER_CLASS_IN_TRAIN = args.min_count_per_class_in_train

    BATCH_SIZE_EVAL = args.batch_size_eval
    BATCH_SIZE_ENCODE = args.batch_size_encode

    WARMUP_RATIO = args.warmup_ratio
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    GRADIENT_CHECKPOINTING = args.gradient_checkpointing

    NORMALIZE_FOR_EVAL = args.normalize_for_eval
    SINGLETON_POLICY = args.singleton_policy
    MAX_EVAL_TRIPLETS = args.max_eval_triplets

    SOFTTRIPLE_NORMALIZE_TRAIN_EMBEDDINGS = args.softtriple_normalize_train_embeddings
    SOFTTRIPLE_NORMALIZE_CENTERS = args.softtriple_normalize_centers

    SEED = args.seed
    DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    RUN_FINAL_TRAINING_ON_FULL_DATA = args.run_final_training_on_full_data
    EXPORT_EMBEDDINGS_ROOT = args.export_embeddings_root

    # =====================================================
    # GRILLE D'HYPERPARAMÈTRES
    # Première exploration simple et raisonnable
    # Recommandée surtout pour TARGET=pred_label
    # =====================================================
    LEARNING_RATE_GRID = [1e-5]
    BATCH_SIZE_TRAIN_GRID = [16]
    NUM_TRAIN_EPOCHS_GRID = [2]
    
    SOFTTRIPLE_CENTERS_PER_CLASS_GRID = [3, 5]
    SOFTTRIPLE_GAMMA_GRID = [0.10]
    SOFTTRIPLE_LAMBDA_GRID = [10.0]
    SOFTTRIPLE_DELTA_GRID = [0.01]
    SOFTTRIPLE_TAU_GRID = [0.0, 0.01]

    print(f"[INFO] torch.cuda.is_available()={torch.cuda.is_available()} -> device={DEVICE}")
    print(f"[INFO] INPUT_CSV = {INPUT_CSV}")
    print(f"[INFO] TARGET = {TARGET}")
    print(f"[INFO] LABEL_COL = {LABEL_COL}")
    print(f"[INFO] SELECTION_LABEL_COL = {SELECTION_LABEL_COL}")
    print(f"[INFO] GRID_OUTPUT_ROOT = {GRID_OUTPUT_ROOT}")
    print(f"[INFO] BASE_MODEL_NAME = {BASE_MODEL_NAME}")
    print(
        "[INFO] mode texte = "
        + (
            "prompt+récit+phrase"
            if USE_CONTEXTUAL_PROMPT_WITH_SUMMARY
            else "phrase seule"
        )
    )
    print(f"[INFO] USE_CONTEXTUAL_PROMPT_WITH_SUMMARY = {USE_CONTEXTUAL_PROMPT_WITH_SUMMARY}")
    print(f"[INFO] USE_FIXED_INSTRUCTION_PREFIX = {USE_FIXED_INSTRUCTION_PREFIX}")
    if USE_FIXED_INSTRUCTION_PREFIX:
        print(f"[INFO] FIXED_INSTRUCTION_PREFIX = {FIXED_INSTRUCTION_PREFIX}")

    print("[INFO] Grille SoftTriple utilisée :")
    print(f"  LEARNING_RATE_GRID = {LEARNING_RATE_GRID}")
    print(f"  BATCH_SIZE_TRAIN_GRID = {BATCH_SIZE_TRAIN_GRID}")
    print(f"  NUM_TRAIN_EPOCHS_GRID = {NUM_TRAIN_EPOCHS_GRID}")
    print(f"  SOFTTRIPLE_CENTERS_PER_CLASS_GRID = {SOFTTRIPLE_CENTERS_PER_CLASS_GRID}")
    print(f"  SOFTTRIPLE_GAMMA_GRID = {SOFTTRIPLE_GAMMA_GRID}")
    print(f"  SOFTTRIPLE_LAMBDA_GRID = {SOFTTRIPLE_LAMBDA_GRID}")
    print(f"  SOFTTRIPLE_DELTA_GRID = {SOFTTRIPLE_DELTA_GRID}")
    print(f"  SOFTTRIPLE_TAU_GRID = {SOFTTRIPLE_TAU_GRID}")

    ensure_dir(GRID_OUTPUT_ROOT)
    ensure_dir(EXPORT_EMBEDDINGS_ROOT)

    all_results: List[Dict[str, Any]] = []

    all_combos = list(product(
        LEARNING_RATE_GRID,
        BATCH_SIZE_TRAIN_GRID,
        NUM_TRAIN_EPOCHS_GRID,
        SOFTTRIPLE_CENTERS_PER_CLASS_GRID,
        SOFTTRIPLE_GAMMA_GRID,
        SOFTTRIPLE_LAMBDA_GRID,
        SOFTTRIPLE_DELTA_GRID,
        SOFTTRIPLE_TAU_GRID,
    ))

    print(f"[INFO] Nombre total de combinaisons = {len(all_combos)}")

    # =====================================================
    # BOUCLE DE GRID SEARCH
    # =====================================================
    for i, (
        learning_rate,
        batch_size_train,
        num_train_epochs,
        centers_per_class,
        gamma,
        la,
        delta,
        tau,
    ) in enumerate(all_combos, start=1):
        combo_id = combo_name(
            learning_rate,
            batch_size_train,
            num_train_epochs,
            centers_per_class,
            gamma,
            la,
            delta,
            tau,
        )
        combo_output_root = os.path.join(GRID_OUTPUT_ROOT, combo_id)
        combo_embeddings_csv = os.path.join(
            EXPORT_EMBEDDINGS_ROOT,
            f"embeddings__{TARGET}__{combo_id}.csv"
        )

        print("\n" + "=" * 100)
        print(f"[GRID] Combinaison {i}/{len(all_combos)}")
        print(
            f"[GRID] learning_rate={learning_rate} | "
            f"batch_size_train={batch_size_train} | "
            f"num_train_epochs={num_train_epochs} | "
            f"centers_per_class={centers_per_class} | "
            f"gamma={gamma} | lambda={la} | delta={delta} | tau={tau}"
        )
        print("=" * 100)

        results = ftemb_softtriple.finetune_softtriple_embedder_research_kfold_simple(
            input_csv=INPUT_CSV,
            output_root=combo_output_root,
            base_model_name=BASE_MODEL_NAME,
            text_col=TEXT_COL,
            label_col=LABEL_COL,
            group_col=GROUP_COL,
            unit_id_col=UNIT_ID_COL,
            summary_col=SUMMARY_COL,
            use_contextual_prompt_with_summary=USE_CONTEXTUAL_PROMPT_WITH_SUMMARY,
            contextual_prompt_template=CONTEXTUAL_PROMPT_TEMPLATE,
            fixed_instruction_prefix=FIXED_INSTRUCTION_PREFIX_TO_USE,
            eval_label_cols=EVAL_LABEL_COLS,
            selection_label_col=SELECTION_LABEL_COL,
            hf_cache_folder=HF_CACHE_FOLDER,
            n_splits=N_SPLITS,
            min_count_per_class_in_train=MIN_COUNT_PER_CLASS_IN_TRAIN,
            batch_size_train=batch_size_train,
            batch_size_eval=BATCH_SIZE_EVAL,
            batch_size_encode=BATCH_SIZE_ENCODE,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            warmup_ratio=WARMUP_RATIO,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            normalize_for_eval=NORMALIZE_FOR_EVAL,
            singleton_policy=SINGLETON_POLICY,
            max_eval_triplets=MAX_EVAL_TRIPLETS,
            softtriple_centers_per_class=centers_per_class,
            softtriple_gamma=gamma,
            softtriple_lambda=la,
            softtriple_delta=delta,
            softtriple_tau=tau,
            softtriple_normalize_train_embeddings=SOFTTRIPLE_NORMALIZE_TRAIN_EMBEDDINGS,
            softtriple_normalize_centers=SOFTTRIPLE_NORMALIZE_CENTERS,
            seed=SEED,
            device=DEVICE,
            export_full_embeddings_csv=combo_embeddings_csv,
        )

        agg = results["aggregate_metrics"]
        mean_test_delta_ratio = results["selection_score_mean_test_delta_ratio"]

        pred_label_delta_mean = (
            agg.get("test_separation", {})
               .get("pred_label", {})
               .get("delta_ratio", {})
               .get("mean", None)
        )
        pred_label_delta_std = (
            agg.get("test_separation", {})
               .get("pred_label", {})
               .get("delta_ratio", {})
               .get("std", None)
        )

        pred_subtype_delta_mean = (
            agg.get("test_separation", {})
               .get("pred_subtype", {})
               .get("delta_ratio", {})
               .get("mean", None)
        )
        pred_subtype_delta_std = (
            agg.get("test_separation", {})
               .get("pred_subtype", {})
               .get("delta_ratio", {})
               .get("std", None)
        )

        row = {
            "combo_id": combo_id,
            "learning_rate": learning_rate,
            "batch_size_train": batch_size_train,
            "num_train_epochs": num_train_epochs,
            "softtriple_centers_per_class": centers_per_class,
            "softtriple_gamma": gamma,
            "softtriple_lambda": la,
            "softtriple_delta": delta,
            "softtriple_tau": tau,
            "target": TARGET,
            "label_col": LABEL_COL,
            "selection_label_col": SELECTION_LABEL_COL,
            "base_model_name": BASE_MODEL_NAME,
            "use_contextual_prompt_with_summary": USE_CONTEXTUAL_PROMPT_WITH_SUMMARY,
            "summary_col": SUMMARY_COL,
            "use_fixed_instruction_prefix": USE_FIXED_INSTRUCTION_PREFIX,
            "fixed_instruction_prefix": FIXED_INSTRUCTION_PREFIX_TO_USE,
            "selection_metric_name": f"mean test delta_ratio on {SELECTION_LABEL_COL}",
            "selection_score_mean_test_delta_ratio": mean_test_delta_ratio,
            "pred_label_delta_ratio_mean": pred_label_delta_mean,
            "pred_label_delta_ratio_std": pred_label_delta_std,
            "pred_subtype_delta_ratio_mean": pred_subtype_delta_mean,
            "pred_subtype_delta_ratio_std": pred_subtype_delta_std,
            "output_root": combo_output_root,
            "aggregate_metrics_json": results["aggregate_metrics_json"],
            "fold_results_json": results["fold_results_json"],
        }

        all_results.append(row)

        print("[GRID] Résumé combinaison :")
        print(json.dumps(row, ensure_ascii=False, indent=2))

    # =====================================================
    # RÉSULTATS GLOBAUX
    # =====================================================
    summary_df = pd.DataFrame(all_results)
    if len(summary_df) == 0:
        raise ValueError("Aucun résultat de grid search n'a été produit.")

    summary_df["selection_score_mean_test_delta_ratio"] = pd.to_numeric(
        summary_df["selection_score_mean_test_delta_ratio"],
        errors="coerce",
    )

    summary_df = summary_df.sort_values(
        by="selection_score_mean_test_delta_ratio",
        ascending=False
    ).reset_index(drop=True)

    summary_csv = os.path.join(GRID_OUTPUT_ROOT, "grid_search_summary.csv")
    summary_json = os.path.join(GRID_OUTPUT_ROOT, "grid_search_summary.json")

    summary_df.to_csv(summary_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("[GRID] TOP COMBINAISONS")
    print("=" * 100)
    print(summary_df.head(10).to_string(index=False))

    best_row = summary_df.iloc[0].to_dict()

    best_config = {
        "learning_rate": float(best_row["learning_rate"]),
        "batch_size_train": int(best_row["batch_size_train"]),
        "num_train_epochs": int(best_row["num_train_epochs"]),
        "softtriple_centers_per_class": int(best_row["softtriple_centers_per_class"]),
        "softtriple_gamma": float(best_row["softtriple_gamma"]),
        "softtriple_lambda": float(best_row["softtriple_lambda"]),
        "softtriple_delta": float(best_row["softtriple_delta"]),
        "softtriple_tau": float(best_row["softtriple_tau"]),
        "selection_score_mean_test_delta_ratio": safe_float(
            best_row["selection_score_mean_test_delta_ratio"],
            default=float("-inf")
        ),
        "selection_metric_name": best_row["selection_metric_name"],
        "combo_id": best_row["combo_id"],
        "base_model_name": BASE_MODEL_NAME,
        "target": TARGET,
        "label_col": LABEL_COL,
        "selection_label_col": SELECTION_LABEL_COL,
        "train_loss_name": "SoftTripleLoss",
        "use_contextual_prompt_with_summary": USE_CONTEXTUAL_PROMPT_WITH_SUMMARY,
        "summary_col": SUMMARY_COL,
        "contextual_prompt_template": (
            CONTEXTUAL_PROMPT_TEMPLATE if USE_CONTEXTUAL_PROMPT_WITH_SUMMARY else None
        ),
        "use_fixed_instruction_prefix": USE_FIXED_INSTRUCTION_PREFIX,
        "fixed_instruction_prefix": FIXED_INSTRUCTION_PREFIX_TO_USE,
    }

    best_config_json = os.path.join(GRID_OUTPUT_ROOT, "best_config.json")
    with open(best_config_json, "w", encoding="utf-8") as f:
        json.dump(best_config, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("[GRID] MEILLEURE CONFIGURATION")
    print("=" * 100)
    print(json.dumps(best_config, ensure_ascii=False, indent=2))

    # =====================================================
    # ENTRAÎNEMENT FINAL SUR TOUT LE DATASET
    # =====================================================
    if RUN_FINAL_TRAINING_ON_FULL_DATA:
        final_output_root = os.path.join(GRID_OUTPUT_ROOT, "best_model_full_data")

        print("\n" + "=" * 100)
        print("[FINAL] Entraînement final SoftTriple sur tout le dataset")
        print("=" * 100)

        final_results = ftemb_softtriple.train_final_softtriple_model_on_full_data(
            input_csv=INPUT_CSV,
            output_root=final_output_root,
            base_model_name=BASE_MODEL_NAME,
            text_col=TEXT_COL,
            label_col=LABEL_COL,
            group_col=GROUP_COL,
            unit_id_col=UNIT_ID_COL,
            summary_col=SUMMARY_COL,
            use_contextual_prompt_with_summary=USE_CONTEXTUAL_PROMPT_WITH_SUMMARY,
            contextual_prompt_template=CONTEXTUAL_PROMPT_TEMPLATE,
            fixed_instruction_prefix=FIXED_INSTRUCTION_PREFIX_TO_USE,
            hf_cache_folder=HF_CACHE_FOLDER,
            min_count_per_class_in_train=MIN_COUNT_PER_CLASS_IN_TRAIN,
            batch_size_train=int(best_row["batch_size_train"]),
            num_train_epochs=int(best_row["num_train_epochs"]),
            learning_rate=float(best_row["learning_rate"]),
            warmup_ratio=WARMUP_RATIO,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            softtriple_centers_per_class=int(best_row["softtriple_centers_per_class"]),
            softtriple_gamma=float(best_row["softtriple_gamma"]),
            softtriple_lambda=float(best_row["softtriple_lambda"]),
            softtriple_delta=float(best_row["softtriple_delta"]),
            softtriple_tau=float(best_row["softtriple_tau"]),
            softtriple_normalize_train_embeddings=SOFTTRIPLE_NORMALIZE_TRAIN_EMBEDDINGS,
            softtriple_normalize_centers=SOFTTRIPLE_NORMALIZE_CENTERS,
            seed=SEED,
            device=DEVICE,
        )

        final_results_json = os.path.join(GRID_OUTPUT_ROOT, "final_full_data_training.json")
        with open(final_results_json, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        print("[FINAL] Modèle final SoftTriple entraîné :")
        print(json.dumps(final_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()