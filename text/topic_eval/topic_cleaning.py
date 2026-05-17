"""Nettoyage des top words (stopwords FR + métier) sans altérer les phrases sources."""

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Iterable, List, Set

_FRENCH_STOP_FALLBACK: Set[str] = {
    "a", "ai", "aie", "aient", "aies", "ait", "as", "au", "aux", "avec", "ce", "ces", "cette",
    "d", "dans", "de", "des", "du", "elle", "en", "es", "est", "et", "eu", "eue", "eues", "eus",
    "eut", "eux", "il", "ils", "je", "la", "le", "les", "leur", "leurs", "lui", "ma", "mais",
    "me", "mes", "moi", "mon", "ne", "nos", "notre", "nous", "on", "ou", "par", "pas", "pour",
    "que", "qui", "sa", "se", "ses", "son", "sont", "sur", "ta", "te", "tes", "toi", "ton", "tu",
    "un", "une", "vos", "votre", "vous", "y",
}

_nltk_fr: Set[str] | None = None


def _get_nltk_french_stopwords() -> Set[str]:
    global _nltk_fr
    if _nltk_fr is not None:
        return _nltk_fr
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            _nltk_fr = set(stopwords.words("french"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            _nltk_fr = set(stopwords.words("french"))
    except Exception:
        _nltk_fr = set()
    return _nltk_fr


def load_domain_stopwords(path: str | Path) -> Set[str]:
    p = Path(path)
    if not p.is_file():
        default = Path(__file__).resolve().parents[1] / "resultats/comparisons/topics_legacy/stopwords_domain.txt"
        if default.is_file():
            p = default
        else:
            raise FileNotFoundError(f"Fichier stopwords métier introuvable : {path}")
    words: Set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        w = line.strip()
        if w and not w.startswith("#"):
            words.add(normalize_token(w))
    return words


def normalize_token(token: str) -> str:
    return token.strip().lower()


_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "«»—–…")


def preprocess_for_ctfidf(text: str) -> str:
    """Normalise un texte pour vectorisation / c-TF-IDF (minuscules, sans ponctuation forte)."""
    if not isinstance(text, str):
        return ""
    t = text.lower().translate(_PUNCT_TABLE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _is_valid_token(tok: str, fr_stop: Set[str], domain: Set[str]) -> bool:
    if len(tok) < 3:
        return False
    if tok.isdigit():
        return False
    if re.fullmatch(r"\d+", tok):
        return False
    if tok in fr_stop or tok in domain:
        return False
    return True


def clean_top_words(words: Iterable[str], stopwords_domain: Set[str]) -> List[str]:
    """Filtre une liste de tokens / mots déjà segmentés."""
    fr = _get_nltk_french_stopwords() | _FRENCH_STOP_FALLBACK
    out: List[str] = []
    seen: Set[str] = set()
    for w in words:
        tok = normalize_token(str(w))
        if not tok:
            continue
        if not _is_valid_token(tok, fr, stopwords_domain):
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def tokenize_for_coherence(text: str, stopwords_domain: Set[str]) -> List[str]:
    """Tokenisation simple alignée sur les règles de nettoyage (hors corpus complet)."""
    base = preprocess_for_ctfidf(text)
    raw = re.findall(r"[a-zàâäæçéèêëîïôœùûüÿñ]{2,}", base)
    return clean_top_words(raw, stopwords_domain)
