import re
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

FRENCH_STOPWORDS = {
    "a", "ai", "aie", "aient", "aies", "ait", "as", "au", "aux", "avec", "ce", "ces", "cette",
    "d", "dans", "de", "des", "du", "elle", "en", "es", "est", "et", "eu", "eue", "eues", "eus",
    "eut", "eux", "il", "ils", "je", "la", "le", "les", "leur", "leurs", "lui", "ma", "mais",
    "me", "mes", "moi", "mon", "ne", "nos", "notre", "nous", "on", "ou", "par", "pas", "pour",
    "que", "qui", "sa", "se", "ses", "son", "sont", "sur", "ta", "te", "tes", "toi", "ton", "tu",
    "un", "une", "vos", "votre", "vous", "y",
}


def clean_french_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zàâäæçéèêëîïôœùûüÿñ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _top_words_for_texts(texts: Sequence[str], top_k: int = 12) -> str:
    cleaned = [clean_french_text(text) for text in texts if isinstance(text, str) and text.strip()]
    if not cleaned:
        return ""
    vectorizer = TfidfVectorizer(
        stop_words=list(FRENCH_STOPWORDS),
        token_pattern=r"(?u)\b[a-zàâäæçéèêëîïôœùûüÿñ0-9]{2,}\b",
        max_features=5000,
    )
    try:
        matrix = vectorizer.fit_transform(cleaned)
    except ValueError:
        return ""
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    terms = np.asarray(vectorizer.get_feature_names_out())
    top_idx = np.argsort(scores)[::-1][:top_k]
    return " ".join(terms[top_idx].tolist())


def _as_numpy(array: np.ndarray) -> np.ndarray:
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _top_sentences_by_distance(
    sentences: Sequence[str],
    projected_embeddings: np.ndarray,
    mu_z: np.ndarray,
    top_k: int = 5,
) -> str:
    if len(sentences) == 0:
        return ""
    mu_z = _as_numpy(mu_z)
    distances = np.linalg.norm(projected_embeddings - mu_z.reshape(1, -1), axis=1)
    order = np.argsort(distances)[:top_k]
    return " || ".join(str(sentences[idx]) for idx in order)


def export_topic_tables(
    metadata_df: pd.DataFrame,
    projected_embeddings: np.ndarray,
    mu_z: np.ndarray,
    z_hat: np.ndarray,
    output_dir: str,
    sentence_col: str = "sentence",
    label_col: str = "pred_label",
) -> None:
    mu_z = _as_numpy(mu_z)
    rows = []
    for z_id in range(mu_z.shape[0]):
        mask = z_hat == z_id
        if not np.any(mask):
            rows.append(
                {
                    "z_id": z_id,
                    "dominant_macro": "",
                    "n_units": 0,
                    "top_words": "",
                    "top_sentences": "",
                }
            )
            continue
        subset = metadata_df.loc[mask]
        dominant_macro = subset[label_col].astype(str).value_counts().idxmax()
        sentences = subset[sentence_col].astype(str).tolist()
        top_words = _top_words_for_texts(sentences)
        top_sentences = _top_sentences_by_distance(
            sentences,
            projected_embeddings[mask],
            mu_z[z_id],
        )
        rows.append(
            {
                "z_id": z_id,
                "dominant_macro": dominant_macro,
                "n_units": int(mask.sum()),
                "top_words": top_words,
                "top_sentences": top_sentences,
            }
        )

    themes_by_z = pd.DataFrame(rows)
    themes_by_z.to_csv(f"{output_dir}/themes_by_z.csv", index=False)

    macro_rows = []
    for macro in sorted(themes_by_z["dominant_macro"].dropna().unique()):
        if macro == "":
            continue
        subset = themes_by_z[themes_by_z["dominant_macro"] == macro].sort_values("n_units", ascending=False)
        macro_rows.append(
            {
                "macro": macro,
                "macro_name": macro,
                "n_components": int(len(subset)),
                "n_units": int(subset["n_units"].sum()),
                "top_words": " | ".join(subset["top_words"].head(5).tolist()),
            }
        )
    pd.DataFrame(macro_rows).to_csv(f"{output_dir}/themes_by_macro_z.csv", index=False)
