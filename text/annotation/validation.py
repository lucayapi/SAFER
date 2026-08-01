"""Validation des réponses JSON du modèle d'annotation."""

from __future__ import annotations

import difflib
import json
import re
from typing import Any, Dict, Optional

from annotation.prompts import is_v13_prompt
from annotation.prompts.v10_macro_labels_independent_outcomes import LABELS, MENTION_VALUES
from annotation.prompts.v13_two_pass_ambiguity_context import (
    ALTERNATIVE_LABELS,
    AMBIGUITY_TYPES,
    PASS_MODES,
)

_MENTION_ALIASES = {
    "NOT_MOUNDED": "NOT_MENTIONED",
    "NOT_MENTIONNED": "NOT_MENTIONED",
    "NOT_MENTONED": "NOT_MENTIONED",
    "NOT_MENTION": "NOT_MENTIONED",
    "NOTMENTIONED": "NOT_MENTIONED",
    "NOT_MENTIONE": "NOT_MENTIONED",
    "Y": "YES",
    "N": "NO",
}

_CONTEXT_USED_RE = re.compile(r"Contexte utilisé\s*:\s*(oui|non)", re.IGNORECASE)

V13_PRED_FIELDS = (
    "pred_ambiguous",
    "pred_context_needed",
    "pred_alternative_label",
    "pred_ambiguity_type",
    "pred_ambiguity_reason",
)


def extract_context_used(justification: str) -> bool:
    """Extrait si le résumé global a été utilisé pour désambiguïser (v10/v12)."""
    match = _CONTEXT_USED_RE.search(str(justification))
    if not match:
        raise ValueError("La justification doit indiquer si le contexte a été utilisé.")
    return match.group(1).lower() == "oui"


def normalize_mention_value(value: Any, *, field_name: str) -> str:
    """Normalise YES/NO/NOT_MENTIONED en tolérant les fautes de frappe du modèle."""
    if value is None:
        raise ValueError(f"Valeur manquante pour {field_name}.")
    raw = str(value).strip().upper().replace(" ", "_").replace("-", "_")
    if raw in MENTION_VALUES:
        return raw
    if raw in _MENTION_ALIASES:
        return _MENTION_ALIASES[raw]
    close = difflib.get_close_matches(raw, MENTION_VALUES, n=1, cutoff=0.82)
    if close:
        return close[0]
    raise ValueError(f"Valeur invalide pour {field_name} : {value}")


def repair_justification(text: str) -> str:
    """Complète une justification presque valide avec les champs obligatoires manquants."""
    justification = str(text).strip()
    if not justification:
        raise ValueError("Justification vide ou invalide.")
    if not re.search(r"Contexte utilisé\s*:\s*(oui|non)", justification, re.IGNORECASE):
        justification = f"Contexte utilisé: non. {justification}"
    if not re.search(r"Indice principal\s*:", justification, re.IGNORECASE):
        justification = f"Indice principal: « non précisé ». {justification}"
    return justification.strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL,
    ).strip()
    think_block = r"<" + "think" + r">.*?</" + "think" + r">"
    text = re.sub(think_block, "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("Impossible d'extraire un JSON valide depuis la réponse du modèle.")


def _normalize_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError(f"Valeur manquante pour {field_name}.")
    raw = str(value).strip().lower()
    if raw in {"true", "1", "yes", "oui"}:
        return True
    if raw in {"false", "0", "no", "non"}:
        return False
    raise ValueError(f"Valeur booléenne invalide pour {field_name} : {value}")


def _validate_core_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    label = parsed.get("label")
    injury_mentioned = normalize_mention_value(
        parsed.get("injury_mentioned"), field_name="injury_mentioned"
    )
    hospitalized = normalize_mention_value(parsed.get("hospitalized"), field_name="hospitalized")
    fatal = normalize_mention_value(parsed.get("fatal"), field_name="fatal")
    confidence = float(parsed.get("confidence"))
    justification = repair_justification(parsed.get("justification", ""))

    if label not in LABELS:
        raise ValueError(f"Label invalide : {label}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence hors bornes : {confidence}")
    if not re.search(r"Contexte utilisé\s*:\s*(oui|non)", justification, re.IGNORECASE):
        raise ValueError("La justification doit indiquer si le contexte a été utilisé.")
    if not re.search(r"Indice principal\s*:", justification, re.IGNORECASE):
        raise ValueError("La justification doit contenir l'indice principal.")

    return {
        "label": label,
        "injury_mentioned": injury_mentioned,
        "hospitalized": hospitalized,
        "fatal": fatal,
        "confidence": confidence,
        "justification": justification,
    }


def validate_prediction_v10(parsed: Dict[str, Any]) -> Dict[str, Any]:
    core = _validate_core_fields(parsed)
    context_used = extract_context_used(core["justification"])
    return {
        "pred_label": core["label"],
        "pred_injury_mentioned": core["injury_mentioned"],
        "pred_hospitalized": core["hospitalized"],
        "pred_fatal": core["fatal"],
        "pred_confidence": core["confidence"],
        "pred_justification": core["justification"],
        "pred_context_used": context_used,
        "pred_ok": True,
    }


def validate_prediction_v13(
    parsed: Dict[str, Any],
    *,
    pass_mode: str = "pass1",
) -> Dict[str, Any]:
    if pass_mode not in PASS_MODES:
        raise ValueError(f"pass_mode invalide : {pass_mode}")

    core = _validate_core_fields(parsed)
    ambiguous = _normalize_bool(parsed.get("ambiguous"), field_name="ambiguous")
    context_needed = _normalize_bool(parsed.get("context_needed"), field_name="context_needed")
    context_used = _normalize_bool(parsed.get("context_used"), field_name="context_used")
    alternative_label = str(parsed.get("alternative_label", "")).strip().upper()
    ambiguity_type = str(parsed.get("ambiguity_type", "")).strip().upper()
    ambiguity_reason = str(parsed.get("ambiguity_reason", "")).strip()

    if alternative_label not in ALTERNATIVE_LABELS:
        raise ValueError(f"alternative_label invalide : {alternative_label}")
    if ambiguity_type not in AMBIGUITY_TYPES:
        raise ValueError(f"ambiguity_type invalide : {ambiguity_type}")

    if pass_mode == "pass1" and context_used:
        raise ValueError("context_used doit être false en passe 1.")

    if not ambiguous:
        # Harmonisation : ambiguous=false implique aucun signal d'ambiguïté.
        # Certains modèles renvoient encore ambiguity_type=INSUFFICIENT_INFORMATION
        # avec ambiguous=false ; on normalise plutôt que de rejeter le label.
        if alternative_label != "NONE":
            raise ValueError("alternative_label doit être NONE si ambiguous=false.")
        ambiguity_type = "NONE"
        ambiguity_reason = ""
    else:
        if alternative_label == "NONE":
            raise ValueError("alternative_label requis si ambiguous=true.")
        if alternative_label == core["label"]:
            raise ValueError("alternative_label doit différer de label si ambiguous=true.")
        if not ambiguity_reason:
            raise ValueError("ambiguity_reason requis si ambiguous=true.")

    justification_context = extract_context_used(core["justification"])
    if context_used != justification_context:
        raise ValueError(
            "context_used JSON et « Contexte utilisé » dans la justification doivent être cohérents."
        )

    return {
        "pred_label": core["label"],
        "pred_injury_mentioned": core["injury_mentioned"],
        "pred_hospitalized": core["hospitalized"],
        "pred_fatal": core["fatal"],
        "pred_confidence": core["confidence"],
        "pred_justification": core["justification"],
        "pred_context_used": context_used,
        "pred_ambiguous": ambiguous,
        "pred_context_needed": context_needed,
        "pred_alternative_label": alternative_label,
        "pred_ambiguity_type": ambiguity_type,
        "pred_ambiguity_reason": ambiguity_reason,
        "pred_ok": True,
    }


def validate_prediction(
    parsed: Dict[str, Any],
    *,
    prompt_version: Optional[str] = None,
    pass_mode: str = "pass1",
) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("La sortie parsée n'est pas un objet JSON.")

    use_v13 = False
    if prompt_version is not None and is_v13_prompt(prompt_version):
        use_v13 = True
    elif "ambiguous" in parsed or "context_needed" in parsed:
        use_v13 = True

    if use_v13:
        return validate_prediction_v13(parsed, pass_mode=pass_mode)
    return validate_prediction_v10(parsed)
