"""Rapport Markdown synthétique."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def write_bn_malt_report(
    path: Path,
    n_accidents: int,
    n_topics_selected: int,
    params: dict,
    diagnostics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    figure_paths: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    diag = diagnostics_df.to_string(index=False)
    comp = comparison_df.to_string(index=False)
    lines = [
        f"# Rapport BN MALT ({date.today().isoformat()})",
        "",
        f"- Accidents : **{n_accidents}**",
        f"- Motifs Z retenus : **{n_topics_selected}**",
        f"- Seuil confiance : {params.get('CONFIDENCE_THRESHOLD')}",
        f"- Support minimal topic : {params.get('MIN_TOPIC_ACCIDENT_SUPPORT')}",
        "",
        "## Diagnostics",
        "",
        "```",
        diag,
        "```",
        "",
        "## Comparaison de structures",
        "",
        "```",
        comp,
        "```",
        "",
        "## Figures",
        "",
    ]
    for p in figure_paths:
        lines.append(f"- `{p}`")
    lines.extend(
        [
            "",
            "## Limites méthodologiques",
            "",
            "- **X = 0** : motif non mentionné / non extrait dans le récit, pas nécessairement absence factuelle.",
            "- **Arcs du BN** : dépendances conditionnelles apprises sous contraintes, **pas** preuve de causalité.",
            "- Les variables dérivent de **MALT** et héritent de ses incertitudes.",
            "- Validation **métier** indispensable avant interprétation préventionnelle.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
