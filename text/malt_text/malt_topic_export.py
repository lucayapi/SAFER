import os
from typing import Sequence

import numpy as np
import pandas as pd

from scgm_text.dataset_text_embeddings import ID2LABEL
from scgm_text.topic_export import _top_sentences_by_distance, _top_words_for_texts
from malt_text.malt_metrics import entropy


def _top_values(series: pd.Series, top_k: int = 5) -> str:
    if series is None or series.empty:
        return ""
    counts = series.astype(str).value_counts().head(top_k)
    return " | ".join(f"{name}:{count}" for name, count in counts.items())


def export_malt_topic_tables(
    metadata_df: pd.DataFrame,
    projected_embeddings: np.ndarray,
    nu: np.ndarray,
    z_hat: np.ndarray,
    prob_y_z: np.ndarray,
    p0: np.ndarray,
    pt: np.ndarray,
    output_dir: str,
    sentence_col: str = "sentence",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for z_id in range(nu.shape[0]):
        mask = z_hat == z_id
        pyz = prob_y_z[z_id]
        dominant_macro = ID2LABEL[int(np.argmax(pyz))]
        row = {
            "z_id": z_id,
            "dominant_macro": dominant_macro,
            "n_units": int(mask.sum()),
            "p_A0_given_z": float(pyz[0]),
            "p_A1_given_z": float(pyz[1]),
            "p_B_given_z": float(pyz[2]),
            "p_C_given_z": float(pyz[3]),
            "entropy_y_given_z": float(entropy(pyz.reshape(1, -1))[0]),
            "top_words": "",
            "top_sentences": "",
            "top_equipment": "",
            "top_company_code": "",
            "mean_pt_confidence": float("nan"),
            "mean_p0_confidence": float("nan"),
        }
        if np.any(mask):
            subset = metadata_df.loc[mask]
            sentences = subset[sentence_col].astype(str).tolist()
            row["top_words"] = _top_words_for_texts(sentences)
            row["top_sentences"] = _top_sentences_by_distance(
                sentences,
                projected_embeddings[mask],
                nu[z_id],
            )
            if "equipment_involved" in subset.columns:
                row["top_equipment"] = _top_values(subset["equipment_involved"])
            if "company_code" in subset.columns:
                row["top_company_code"] = _top_values(subset["company_code"])
            row["mean_pt_confidence"] = float(pt[mask].max(axis=1).mean())
            row["mean_p0_confidence"] = float(p0[mask].max(axis=1).mean())
        rows.append(row)

    themes_by_z = pd.DataFrame(rows)
    themes_by_z.to_csv(os.path.join(output_dir, "themes_by_z_malt.csv"), index=False)

    macro_rows = []
    for macro_name in ["A0", "A1", "B", "C"]:
        subset = themes_by_z[themes_by_z["dominant_macro"] == macro_name].sort_values("n_units", ascending=False)
        macro_rows.append(
            {
                "macro": macro_name,
                "macro_name": macro_name,
                "n_components": int(len(subset)),
                "n_units": int(subset["n_units"].sum()),
                "top_words": " | ".join(subset["top_words"].head(5).tolist()),
            }
        )
    pd.DataFrame(macro_rows).to_csv(os.path.join(output_dir, "themes_by_macro_malt.csv"), index=False)
