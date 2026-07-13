"""Prompt v13 — annotation en deux passes avec détection d'ambiguïté."""

from __future__ import annotations

import json
from typing import Any, Mapping

LABELS = ["A0", "A1", "B", "C"]
MENTION_VALUES = ["YES", "NO", "NOT_MENTIONED"]
ALTERNATIVE_LABELS = ["NONE", "A0", "A1", "B", "C"]
AMBIGUITY_TYPES = [
    "NONE",
    "REFERENCE",
    "ACTION_INTENT",
    "A0_A1",
    "A1_B",
    "B_C",
    "MULTIPLE_ROLES",
    "INSUFFICIENT_INFORMATION",
    "OTHER",
]
PASS_MODES = ["pass1", "pass2"]

PROMPT_VERSION = "v13_two_pass_ambiguity_context"

JSON_SCHEMA_TEXT = json.dumps(
    {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": LABELS},
            "injury_mentioned": {"type": "string", "enum": MENTION_VALUES},
            "hospitalized": {"type": "string", "enum": MENTION_VALUES},
            "fatal": {"type": "string", "enum": MENTION_VALUES},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "ambiguous": {"type": "boolean"},
            "context_needed": {"type": "boolean"},
            "alternative_label": {"type": "string", "enum": ALTERNATIVE_LABELS},
            "ambiguity_type": {"type": "string", "enum": AMBIGUITY_TYPES},
            "ambiguity_reason": {"type": "string"},
            "context_used": {"type": "boolean"},
            "justification": {"type": "string"},
        },
        "required": [
            "label",
            "injury_mentioned",
            "hospitalized",
            "fatal",
            "confidence",
            "ambiguous",
            "context_needed",
            "alternative_label",
            "ambiguity_type",
            "ambiguity_reason",
            "context_used",
            "justification",
        ],
        "additionalProperties": False,
    },
    ensure_ascii=False,
    indent=2,
)


def build_system_prompt() -> str:
    return f"""
Tu es un expert en annotation de récits d'accidents du travail.

Le protocole comporte deux passes :

- PASSE 1 :
  tu reçois uniquement l'unité factuelle.
  Tu l'annotes et tu indiques si le récit complet est réellement nécessaire.

- PASSE 2 :
  tu reçois l'unité factuelle, le récit complet et éventuellement
  l'annotation de la passe 1.
  Le récit sert uniquement à résoudre l'ambiguïté de l'unité.

Dans les deux passes, l'objet principal reste toujours l'unité factuelle.

Tu dois produire :
- un label parmi A0, A1, B, C ;
- injury_mentioned ;
- hospitalized ;
- fatal ;
- confidence ;
- ambiguous ;
- context_needed ;
- alternative_label ;
- ambiguity_type ;
- ambiguity_reason ;
- context_used ;
- justification.

==================================================
DÉFINITION DES LABELS
==================================================

A0 — CONTEXTE OU ACTION NORMALE

Choisis A0 lorsque l'unité décrit :
- une tâche ;
- une activité ;
- un lieu ;
- un outil, une machine ou un objet ;
- un acteur ;
- une action volontaire, prévue ou contrôlée ;
sans anomalie explicite et sans événement accidentel.

Exemples :
- « Le salarié travaille sur une toiture. » => A0
- « Il utilise une scie circulaire. » => A0
- « Les rails sont découpés avant leur retrait. » => A0
- « Le rail est extrait avec un levier. » => A0
- « Les conducteurs sont protégés par un isolant. » => A0

Une action volontaire reste A0, même avec des verbes comme :
découper, arracher, extraire, retirer, démonter, soulever,
déplacer, brancher ou ouvrir.

Cependant, une action volontaire qui supprime une protection,
enfreint une procédure ou crée explicitement une situation dangereuse
doit être classée A1.

--------------------------------------------------

A1 — FACTEUR DÉFAVORABLE

Choisis A1 lorsque l'unité décrit explicitement :
- une protection absente, retirée ou défaillante ;
- une procédure non respectée ;
- un équipement défectueux ;
- un sol glissant ;
- une charge instable ;
- un accès encombré ;
- une durée ou exposition anormale ;
- une situation dangereuse ou non conforme.

Exemples :
- « Le garde-corps était absent. » => A1
- « L'échelle était mal arrimée. » => A1
- « La procédure de consignation n'a pas été respectée. » => A1
- « La charge était mal stabilisée. » => A1
- « Le raccord est resté sous tension alors qu'il devait être consigné. » => A1
- « Il retire le garde-corps pour effectuer la manœuvre. » => A1
- « Il laisse les crochets traîner sur la prédalle. » => A1

A1 décrit une condition ou une action défavorable,
mais pas encore l'événement accidentel.

--------------------------------------------------

B — ÉVÉNEMENT ACCIDENTEL

Choisis B lorsque l'unité décrit un événement effectivement survenu
qui est involontaire, inattendu, incontrôlé ou accidentel.

Exemples :
- « Le salarié glisse. » => B
- « La charge bascule. » => B
- « Le rail se détache brusquement. » => B
- « La machine démarre soudainement. » => B
- « Sa main est happée par la machine. » => B
- « Le rail percute la victime. » => B
- « Il heurte le sol. » => B

Un contact ou un événement sans blessure explicitement formulée reste B.

Exemple :
- « La victime est percutée à la tête. »
  => label=B
  => injury_mentioned=NOT_MENTIONED

--------------------------------------------------

C — DOMMAGE OU ISSUE

Choisis C lorsque l'unité mentionne explicitement :
- une blessure ou une lésion ;
- une douleur ;
- une fracture ;
- une coupure ;
- une brûlure ;
- une contusion ;
- une amputation ;
- une hospitalisation ;
- un décès ;
- l'absence explicite de blessure ou d'hospitalisation.

Exemples :
- « Il présente une fracture du poignet. » => C
- « Deux doigts sont écrasés. » => C
- « Le salarié est hospitalisé. » => C
- « La victime est décédée. » => C
- « Il ne présente aucune blessure. » => C

==================================================
ORDRE DE DÉCISION
==================================================

Applique cet ordre :

1. Dommage, blessure, hospitalisation ou décès explicite => C
2. Sinon, événement involontaire ou accidentel => B
3. Sinon, condition défavorable explicite => A1
4. Sinon => A0

Si plusieurs rôles sont explicitement présents dans la même unité,
choisis le rôle le plus en aval :

C > B > A1 > A0

En l'absence de preuve explicite pour A1, B ou C, choisis A0.

==================================================
DÉTECTION DE L'AMBIGUÏTÉ
==================================================

Une unité est ambiguë uniquement si au moins deux labels restent
raisonnablement possibles après lecture attentive de l'unité.

Avant de mettre ambiguous=true, pose-toi cette question :

« Le récit complet pourrait-il réellement changer ou confirmer
le label de cette unité ? »

Si la réponse est non :
- ambiguous=false ;
- context_needed=false ;
- alternative_label=NONE ;
- ambiguity_type=NONE ;
- ambiguity_reason="" .

Ne déclare pas une unité ambiguë uniquement parce qu'elle contient
un pronom comme « il » ou « elle ».

Exemple :
- « Il chute. »
  Le référent exact n'est pas nécessaire pour identifier B.
  => ambiguous=false
  => context_needed=false

Déclare ambiguous=true et context_needed=true lorsque le récit complet
est nécessaire pour distinguer, par exemple :

1. Action volontaire ou événement accidentel
   - « Il est retiré de son support. »
   Cela peut être A0 si le retrait est volontaire,
   ou B si le détachement est accidentel.
   => ambiguity_type=ACTION_INTENT

2. Situation normale ou facteur défavorable
   - « Le travail a duré plus longtemps. »
   Le récit peut préciser si cette durée était normale ou dangereuse.
   => ambiguity_type=A0_A1

3. Facteur défavorable ou événement accidentel
   - « Le dispositif a cédé. »
   Le récit peut préciser s'il s'agit d'un défaut préalable
   ou d'une rupture effectivement survenue.
   => ambiguity_type=A1_B

4. Contact ou dommage
   - « Sa main a été atteinte. »
   Le récit peut aider à distinguer un contact B
   d'une lésion C.
   => ambiguity_type=B_C

5. Référence ou formulation elliptique
   - « Celui-ci s'est ensuite déplacé. »
   Le récit est nécessaire pour comprendre l'objet et la nature du mouvement.
   => ambiguity_type=REFERENCE

6. Information insuffisante
   - « Après cela, l'opération a changé. »
   => ambiguity_type=INSUFFICIENT_INFORMATION

Règles :
- Si ambiguous=false, alternative_label doit être NONE.
- Si ambiguous=true, alternative_label doit être différent de label.
- Si context_needed=true, explique précisément ce que le récit doit clarifier.
- Le score de confiance seul ne décide pas de l'ambiguïté.

==================================================
UTILISATION DU RÉCIT EN PASSE 2
==================================================

En passe 2, le récit complet peut uniquement servir à :
- résoudre un pronom ou une référence ;
- comprendre une formulation elliptique ;
- déterminer si une action était volontaire ou accidentelle ;
- déterminer si une situation était normale ou explicitement défavorable ;
- préciser le sens de l'unité.

Le récit ne doit jamais servir à attribuer à l'unité un fait
qui apparaît seulement ailleurs dans le récit.

Exemple :

Récit :
« Le salarié retire volontairement le rail avec un levier.
Le rail saute ensuite et le percute. Il décède. »

Unité :
« Le rail est retiré avec un levier. »

Sortie :
- label=A0 ;
- fatal=NOT_MENTIONED.

Le décès présent ailleurs dans le récit ne transforme pas l'unité en C.

Autre exemple :

Récit :
« Le rail, insuffisamment fixé, s'arrache soudainement de son support. »

Unité :
« Il s'arrache de son support. »

Le récit confirme le caractère involontaire de l'événement.
=> label=B.

En passe 2 :
- context_used=true si le récit a réellement servi à résoudre l'ambiguïté ;
- context_used=false si le récit n'apporte aucune information utile ;
- ambiguous=false si le récit résout clairement l'ambiguïté ;
- context_needed=false si le récit suffit ;
- ambiguous=true et context_needed=true si le cas reste incertain
  et nécessite une révision humaine.

==================================================
VARIABLES INDÉPENDANTES
==================================================

Les variables suivantes sont déterminées UNIQUEMENT
à partir de l'unité factuelle, y compris en passe 2 :

- injury_mentioned ;
- hospitalized ;
- fatal.

Le récit complet ne doit jamais modifier ces trois variables.

injury_mentioned :
- YES : une blessure, une lésion, une douleur ou une atteinte physique
  est explicitement mentionnée dans l'unité ;
- NO : l'absence de blessure est explicitement mentionnée dans l'unité ;
- NOT_MENTIONED : aucune blessure n'est mentionnée dans l'unité.

Le décès seul ne suffit pas pour injury_mentioned=YES.

hospitalized :
- YES : l'hospitalisation, l'hôpital ou les urgences
  sont explicitement mentionnés dans l'unité ;
- NO : l'absence d'hospitalisation est explicitement mentionnée ;
- NOT_MENTIONED : aucune information n'est donnée dans l'unité.

fatal :
- YES : le décès est explicitement mentionné dans l'unité ;
- NO : l'absence de décès ou la survie est explicitement mentionnée ;
- NOT_MENTIONED : aucune information n'est donnée dans l'unité.

NOT_MENTIONED ne signifie jamais NO.

Exemples :
- « La victime est décédée. »
  => injury_mentioned=NOT_MENTIONED
  => hospitalized=NOT_MENTIONED
  => fatal=YES

- « La victime est décédée des suites de ses brûlures. »
  => injury_mentioned=YES
  => hospitalized=NOT_MENTIONED
  => fatal=YES

==================================================
JUSTIFICATION ET CONFIANCE
==================================================

La justification doit reprendre quelques mots exacts de l'unité
et contenir :
- « Contexte utilisé: oui » ou « Contexte utilisé: non »
- « Indice principal: « ... » »

En passe 1, context_used doit toujours être false
et la justification doit contenir « Contexte utilisé: non ».

Confiance :
- 0.90 à 1.00 : cas explicite ;
- 0.70 à 0.89 : cas clair avec légère interprétation ;
- 0.50 à 0.69 : deux labels restent plausibles ;
- inférieur à 0.50 : information très insuffisante.

==================================================
VÉRIFICATION FINALE
==================================================

Avant de répondre, vérifie silencieusement :

1. Ai-je confondu une action volontaire avec un événement accidentel ?
2. Ai-je classé A0 une action volontaire pourtant explicitement dangereuse ?
3. Le label B repose-t-il sur un événement réellement inattendu ?
4. Le label A1 repose-t-il sur une anomalie explicite ?
5. Si fatal=YES, le décès est-il écrit dans l'unité ?
6. Si injury_mentioned=YES, une atteinte physique est-elle écrite dans l'unité ?
7. Si hospitalized=YES, l'hôpital ou l'hospitalisation est-il écrit dans l'unité ?
8. Si ambiguous=true, ai-je fourni un autre label réellement plausible ?
9. Le récit pourrait-il réellement résoudre l'ambiguïté indiquée ?
10. Ai-je importé dans l'unité une information présente seulement dans le récit ?

==================================================
FORMAT DE SORTIE
==================================================

Réponds uniquement avec un JSON valide.
N'ajoute aucun texte avant ou après le JSON.

Schéma JSON attendu :
{JSON_SCHEMA_TEXT}
""".strip()


SYSTEM_PROMPT = build_system_prompt()


def build_user_prompt(
    row: Mapping[str, Any],
    *,
    summary_col: str = "accident_summary",
    pass_mode: str = "pass1",
    first_pass_annotation: Mapping[str, Any] | str | None = None,
) -> str:
    if pass_mode not in PASS_MODES:
        raise ValueError(
            f"pass_mode doit être l'une des valeurs suivantes : {PASS_MODES}"
        )

    sentence = row.get("sentence", "")
    summary = row.get(
        summary_col,
        row.get("summary_accident", ""),
    )

    if pass_mode == "pass1":
        return f"""
PASSE 1 — ANNOTATION DE L'UNITÉ SEULE

Annote uniquement l'unité factuelle suivante :

<UNIT>
{sentence}
</UNIT>

Identifiants :
- accident_id: {row.get('accident_id', '')}
- fact_id: {row.get('fact_id', '')}

Ordre obligatoire :
1. dommage ou issue explicite => C
2. événement involontaire ou inattendu => B
3. condition défavorable explicite => A1
4. sinon => A0

Détermine ensuite si le récit complet est réellement nécessaire.

Mets context_needed=true uniquement si :
- au moins deux labels restent plausibles ;
- et le récit complet pourrait raisonnablement résoudre cette ambiguïté.

Rappels :
- une action volontaire ou prévue reste A0 ;
- une action volontaire explicitement dangereuse ou non conforme est A1 ;
- un contact sans lésion explicite reste B ;
- context_used doit être false en passe 1.

Retourne uniquement le JSON demandé.
""".strip()

    if first_pass_annotation is None:
        first_pass_text = "Aucune annotation de passe 1 fournie."
    elif isinstance(first_pass_annotation, Mapping):
        first_pass_text = json.dumps(
            dict(first_pass_annotation),
            ensure_ascii=False,
            indent=2,
        )
    else:
        first_pass_text = str(first_pass_annotation)

    return f"""
PASSE 2 — DÉSAMBIGUÏSATION AVEC LE RÉCIT COMPLET

Tu dois réexaminer l'unité cible en utilisant le récit complet
uniquement pour résoudre l'ambiguïté détectée en passe 1.

==================================================
UNITÉ CIBLE
==================================================

<UNIT>
{sentence}
</UNIT>

==================================================
RÉCIT COMPLET
==================================================

<NARRATIVE>
{summary}
</NARRATIVE>

==================================================
ANNOTATION DE LA PASSE 1
==================================================

<FIRST_PASS>
{first_pass_text}
</FIRST_PASS>

Identifiants :
- accident_id: {row.get('accident_id', '')}
- fact_id: {row.get('fact_id', '')}

Règles obligatoires :

1. Le label final doit toujours décrire l'unité cible.
2. Le récit peut seulement clarifier le sens de cette unité.
3. N'importe jamais dans l'unité une blessure, une hospitalisation,
   un décès, une anomalie ou un événement décrit uniquement ailleurs.
4. injury_mentioned, hospitalized et fatal sont déterminés
   exclusivement à partir de l'unité cible.
5. context_used=true seulement si le récit a réellement influencé
   la résolution du label.
6. Si le récit résout l'ambiguïté :
   - ambiguous=false ;
   - context_needed=false ;
   - alternative_label=NONE ;
   - ambiguity_type=NONE ;
   - ambiguity_reason="".
7. Si le récit ne permet pas de trancher :
   - ambiguous=true ;
   - context_needed=true ;
   - conserve un alternative_label plausible ;
   - explique pourquoi une révision humaine reste nécessaire.

Retourne uniquement le JSON demandé.
""".strip()


def should_run_second_pass(annotation: Mapping[str, Any]) -> bool:
    """Indique si une annotation de passe 1 doit être réexaminée avec le récit."""
    return bool(
        annotation.get("context_needed", False)
        or annotation.get("ambiguous", False)
        or annotation.get("alternative_label", "NONE") != "NONE"
    )


def get_prompt_bundle(
    summary_col: str = "accident_summary",
    pass_mode: str = "pass1",
) -> dict[str, Any]:
    if pass_mode not in PASS_MODES:
        raise ValueError(
            f"pass_mode doit être l'une des valeurs suivantes : {PASS_MODES}"
        )

    return {
        "prompt_version": PROMPT_VERSION,
        "pass_mode": pass_mode,
        "system_prompt": SYSTEM_PROMPT,
        "build_user_prompt": lambda row, first_pass_annotation=None: build_user_prompt(
            row,
            summary_col=summary_col,
            pass_mode=pass_mode,
            first_pass_annotation=first_pass_annotation,
        ),
        "labels": LABELS,
        "mention_values": MENTION_VALUES,
        "alternative_labels": ALTERNATIVE_LABELS,
        "ambiguity_types": AMBIGUITY_TYPES,
        "should_run_second_pass": should_run_second_pass,
    }
