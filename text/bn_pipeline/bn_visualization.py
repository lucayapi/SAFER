"""Visualisation des graphes BN (matplotlib + Plotly + Pyvis) avec cartes CPD."""

from __future__ import annotations

import math
import re
import textwrap
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

BBox = Tuple[float, float, float, float]  # x0, y0, width, height

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except ImportError:  # tests / environnements minimaux
    sns = None  # type: ignore[assignment]

from bn_pipeline.bn_structure import MACRO_ONTOLOGY_FR
from safer_core.display_labels import CHAIN_CAUSAL_STRUCTURE, CHAIN_LEGEND

MACRO_FILL = {
    "A0": "#D6EAF8",
    "A1": "#FFF2CC",
    "B": "#FAD7A0",
    "C": "#F5B7B1",
    "Severity": "#E8DAEF",
    "Severity_high": "#E8DAEF",
    "SEVERITY": "#E8DAEF",
}

MACRO_DRAW = {
    "A0": "#2E6F9E",
    "A1": "#B9770E",
    "B": "#D35400",
    "C": "#922B21",
    "Severity": "#6C3483",
    "Severity_high": "#6C3483",
    "SEVERITY": "#6C3483",
}

# Alias rétrocompat (légendes, marqueurs, graphes cercles)
MACRO_COLOR = dict(MACRO_DRAW)
MACRO_LAYER_FILL = dict(MACRO_FILL)

# Ancres (x, y) pour le DAG accidentologique — article / lecture gauche→droite, haut→bas.
_MACRO_LAYOUT_ANCHOR: Dict[str, Tuple[float, float]] = {
    "A0": (-1.35, 1.05),
    "A1": (1.35, 1.05),
    "B": (0.0, 0.0),
    "C": (0.0, -1.1),
    "Severity": (0.0, -2.15),
}


def _strip_accents(text: str) -> str:
    nf = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nf if not unicodedata.combining(c))


def _slug_theme(text: str, max_len: int = 32) -> str:
    s = _strip_accents(str(text).strip())
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s-]+", "_", s).strip("_")
    if not s:
        return "motif"
    return s[:max_len]


def _macro_topic_from_node(node: str) -> Optional[tuple[str, int]]:
    """Parse ``macro_topic_A0_03`` → (macro, topic_id)."""
    n = str(node)
    if not n.startswith("macro_topic_"):
        return None
    parts = n.split("_")
    if len(parts) < 4:
        return None
    try:
        return parts[2], int(parts[3])
    except ValueError:
        return None


def _z_id_from_node(node: str) -> Optional[int]:
    n = str(node)
    if not n.startswith("Z_"):
        return None
    parts = n.split("_")
    if len(parts) >= 3:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _macro_from_node(node: str, variable_macro_map: Optional[Dict[str, str]] = None) -> str:
    n = str(node)
    if variable_macro_map and n in variable_macro_map:
        return str(variable_macro_map[n])
    if n.startswith("M_"):
        return n.replace("M_", "")
    mt = _macro_topic_from_node(n)
    if mt is not None:
        return mt[0]
    if n.startswith("Z_"):
        parts = n.split("_")
        if len(parts) >= 3:
            return parts[2]
    if "Severity" in n:
        return "Severity"
    return "A0"


_OPENAI_THEMES_BASENAME = "themes_by_z_openai.csv"


def resolve_openai_themes_path(
    scgm_exports_dir: Path,
    staging_dir: Optional[Path] = None,
    explicit_path: Optional[Path] = None,
) -> Path:
    """Retourne le premier ``themes_by_z_openai.csv`` trouvé (jamais ``themes_by_z.csv``)."""
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.is_file() and p.name == _OPENAI_THEMES_BASENAME:
            return p
        if p.is_file():
            raise ValueError(
                f"Fichier thèmes invalide pour les libellés BN : {p.name!r}. "
                f"Utilisez uniquement {_OPENAI_THEMES_BASENAME!r} (sortie cellule OpenAI, notebook 01)."
            )
    candidates: List[Path] = []
    if staging_dir is not None:
        candidates.append(Path(staging_dir) / "bn_exports" / _OPENAI_THEMES_BASENAME)
        candidates.append(Path(staging_dir) / _OPENAI_THEMES_BASENAME)
    scgm = Path(scgm_exports_dir)
    candidates.append(scgm / _OPENAI_THEMES_BASENAME)
    for path in candidates:
        if path.is_file():
            return path
    searched = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "Fichier themes_by_z_openai.csv introuvable pour les libellés du réseau bayésien.\n"
        f"Chemins testés :\n{searched}\n\n"
        "Exécutez la cellule OpenAI (notebook 01, section 11 bis) pour produire "
        "output/scgm_text/topics/themes_by_z_openai.csv avec la colonne theme_summary."
    )


def load_openai_themes_for_bn(
    scgm_exports_dir: Path,
    staging_dir: Optional[Path] = None,
    explicit_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Charge ``themes_by_z_openai.csv`` uniquement (colonne ``theme_summary`` requise).
    N'utilise jamais ``themes_by_z.csv`` (top_words TF-IDF) ni ``theme_keywords``.
    """
    path = resolve_openai_themes_path(scgm_exports_dir, staging_dir, explicit_path)
    df = pd.read_csv(path)
    if "z_id" not in df.columns:
        raise ValueError(f"{path} : colonne z_id manquante.")
    if "theme_summary" not in df.columns:
        raise ValueError(
            f"{path} : colonne theme_summary manquante. "
            "Relancez l'enrichissement OpenAI (notebook 01, cellule 11 bis)."
        )
    sub = df.copy()
    sub["z_id"] = pd.to_numeric(sub["z_id"], errors="coerce")
    sub = sub.dropna(subset=["z_id"])
    sub["theme_summary"] = sub["theme_summary"].astype(str).str.strip()
    sub = sub[sub["theme_summary"].astype(bool)]
    if sub.empty:
        raise ValueError(
            f"{path} : aucune ligne avec theme_summary non vide. "
            "Vérifiez l'enrichissement OpenAI."
        )
    return sub.drop_duplicates(subset=["z_id"], keep="first")


def _macro_topic_to_theme_summary(themes_df: Optional[pd.DataFrame]) -> Dict[tuple[str, int], str]:
    df = themes_df if themes_df is not None else pd.DataFrame()
    if df.empty or "macro" not in df.columns or "topic_id" not in df.columns:
        return {}
    if "theme_summary" in df.columns:
        summary_col = "theme_summary"
    elif "theme_label" in df.columns:
        summary_col = "theme_label"
    else:
        summary_col = "top_words"
    if summary_col not in df.columns:
        return {}
    out: Dict[tuple[str, int], str] = {}
    for _, row in df.iterrows():
        key = (str(row["macro"]), int(row["topic_id"]))
        text = str(row.get(summary_col, "")).strip()
        if text:
            out[key] = text
    return out


def join_theme_summary_to_selected_variables(
    selected_variables_df: pd.DataFrame,
    themes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Ajoute ``theme_summary`` via ``z_id`` ou ``macro`` + ``topic_id``."""
    if selected_variables_df.empty:
        return selected_variables_df
    out = selected_variables_df.copy()
    if "macro" in out.columns and "topic_id" in out.columns:
        mt_map = _macro_topic_to_theme_summary(themes_df)
        out["theme_summary"] = [
            mt_map.get((str(r["macro"]), int(r["topic_id"])), "")
            for _, r in out.iterrows()
        ]
        return out
    if "z_id" not in out.columns:
        return out
    zmap = _z_id_to_theme_summary(themes_df)
    out["theme_summary"] = out["z_id"].map(lambda z: zmap.get(int(z), ""))
    return out


def _z_id_to_theme_summary(themes_df: Optional[pd.DataFrame]) -> Dict[int, str]:
    """Index ``z_id`` → ``theme_summary`` (OpenAI uniquement)."""
    df = themes_df if themes_df is not None else pd.DataFrame()
    if df.empty or "z_id" not in df.columns or "theme_summary" not in df.columns:
        return {}
    sub = df.copy()
    sub["z_id"] = pd.to_numeric(sub["z_id"], errors="coerce")
    sub = sub.dropna(subset=["z_id"])
    sub["z_id"] = sub["z_id"].astype(int)
    out: Dict[int, str] = {}
    for z_id, summary in sub.set_index("z_id")["theme_summary"].astype(str).items():
        s = str(summary).strip()
        if s:
            out[int(z_id)] = s
    return out


def build_node_summary_label(
    node: str,
    themes_df: Optional[pd.DataFrame] = None,
    variable_macro_map: Optional[Dict[str, str]] = None,
    max_len: int = 80,
) -> str:
    """
    Libellé affiché sur le graphe BN : ``theme_summary`` OpenAI (libellé court FR) pour les ``Z_*``.
    """
    n = str(node)
    macro = _macro_from_node(n, variable_macro_map)
    if n.startswith("M_"):
        return f"{macro} — agrégat (chaîne)"[:max_len]
    if "Severity" in n:
        return "Gravité élevée"[:max_len]

    mt = _macro_topic_from_node(n)
    if mt is not None:
        mt_map = _macro_topic_to_theme_summary(themes_df)
        if mt in mt_map:
            return mt_map[mt][:max_len]
        return f"{macro} — topic {mt[1]}"[:max_len]

    z_id = _z_id_from_node(n)
    z_to_summary = _z_id_to_theme_summary(themes_df)
    if z_id is not None and z_id in z_to_summary:
        return z_to_summary[z_id][:max_len]
    if z_id is not None:
        return f"{macro} — motif z={z_id}"[:max_len]
    return n[:max_len]


def build_node_short_title(
    node: str,
    themes_df: Optional[pd.DataFrame] = None,
    variable_macro_map: Optional[Dict[str, str]] = None,
    max_len: int = 80,
) -> str:
    """Alias : libellé graphe = ``theme_summary`` OpenAI (voir ``build_node_summary_label``)."""
    return build_node_summary_label(node, themes_df, variable_macro_map, max_len=max_len)


def build_topic_node_label_map(
    nodes: Iterable[str],
    themes_df: Optional[pd.DataFrame] = None,
    wrap_width: int = 36,
    variable_macro_map: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Libellés (tooltips) : ``theme_summary`` OpenAI, retour à la ligne si besoin."""
    out: Dict[str, str] = {}
    for raw in nodes:
        label = build_node_summary_label(raw, themes_df, variable_macro_map)
        out[raw] = textwrap.fill(label, width=wrap_width, break_long_words=False)
    return out


def build_short_title_map(
    nodes: Iterable[str],
    themes_df: Optional[pd.DataFrame] = None,
    variable_macro_map: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Cartes / graphe : une entrée par nœud → ``theme_summary`` OpenAI."""
    return {
        str(n): build_node_summary_label(n, themes_df, variable_macro_map)
        for n in nodes
    }


def format_prob_bar(p: float, width: int = 10) -> str:
    p = float(np.clip(p, 0.0, 1.0))
    filled = int(round(p * width))
    filled = min(width, max(0, filled))
    return f"{'█' * filled}{'░' * (width - filled)}  {100 * p:.1f}%"


def cpd_binary_marginal(model: Any, node: str) -> List[Tuple[int, float]]:
    """
    P(X=0), P(X=1) pour un nœud binaire.
    Racine : marginale du CPD ; avec parents : moyenne uniforme sur les états parents.
    """
    cpd = None
    for c in model.get_cpds():
        if c.variable == node:
            cpd = c
            break
    if cpd is None:
        return [(0, 0.5), (1, 0.5)]

    vals = np.asarray(cpd.values, dtype=float)
    if vals.ndim == 1:
        probs = vals.flatten()
    else:
        # première dimension = états du nœud
        probs = vals.mean(axis=tuple(range(1, vals.ndim)))

    probs = np.asarray(probs, dtype=float).flatten()
    if len(probs) < 2:
        probs = np.array([1.0 - float(probs[0]), float(probs[0])]) if len(probs) == 1 else np.array([0.5, 0.5])
    elif len(probs) > 2:
        probs = probs[:2]

    s = float(probs.sum())
    if s <= 0:
        probs = np.array([0.5, 0.5])
    else:
        probs = probs / s

    return [(0, float(probs[0])), (1, float(probs[1]))]


def format_node_card(title: str, probs: List[Tuple[int, float]], bar_width: int = 10) -> str:
    lines = [title]
    for _state, p in probs:
        lines.append(format_prob_bar(p, width=bar_width))
    return "\n".join(lines)


def build_node_cards_for_model(
    model: Any,
    short_title_map: Dict[str, str],
) -> Dict[str, str]:
    cards: Dict[str, str] = {}
    for node in model.nodes():
        title = short_title_map.get(str(node), str(node))
        probs = cpd_binary_marginal(model, str(node))
        cards[str(node)] = format_node_card(title, probs)
    return cards


def export_node_marginals_csv(
    model: Any,
    short_title_map: Dict[str, str],
    output_path: Path,
) -> pd.DataFrame:
    rows = []
    for node in model.nodes():
        title = short_title_map.get(str(node), str(node))
        for state, p in cpd_binary_marginal(model, str(node)):
            rows.append(
                {
                    "node": str(node),
                    "short_title": title,
                    "state": int(state),
                    "probability": float(p),
                }
            )
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def display_node_card(
    model: Any,
    node: str,
    short_title_map: Optional[Dict[str, str]] = None,
) -> str:
    """Retourne et affiche la carte d’un nœud (usage notebook)."""
    title = (short_title_map or {}).get(str(node), str(node))
    card = format_node_card(title, cpd_binary_marginal(model, str(node)))
    print(card)
    return card


def _macro_of_node(node: str, variable_macro_map: Dict[str, str]) -> str:
    return _macro_from_node(node, variable_macro_map)


def _layout_accident_causality(
    nodes: list[str],
    variable_macro_map: Dict[str, str],
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Dispose les nœuds par couche A0/A1 (haut) → B → C (bas), comme le DAG imposé."""
    from collections import defaultdict

    rng = np.random.default_rng(seed)
    by_macro: Dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        by_macro[_macro_of_node(n, variable_macro_map)].append(n)

    pos: Dict[str, Tuple[float, float]] = {}
    for macro, group in by_macro.items():
        anchor = _MACRO_LAYOUT_ANCHOR.get(macro, (0.0, 0.0))
        group = sorted(group)
        n = len(group)
        for i, node in enumerate(group):
            spread = 0.42 * (i - (n - 1) / 2.0) if n > 1 else 0.0
            jitter = 0.04 * rng.standard_normal(2)
            pos[node] = (anchor[0] + spread + float(jitter[0]), anchor[1] + float(jitter[1]))
    return pos


def _layout_by_macro(
    nodes: list[str],
    variable_macro_map: Dict[str, str],
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    return _layout_accident_causality(nodes, variable_macro_map, seed=seed)


# Colonnes gauche → droite (style article pgmpy).
_MACRO_COL_X: Dict[str, float] = {
    "A0": 0.0,
    "A1": 1.0,
    "B": 2.5,
    "C": 4.0,
    "Severity": 5.0,
}
_MACRO_STACK_ORDER = ("A0", "A1", "B", "C", "Severity")


def _wrap_node_title(title: str, width: int = 22) -> str:
    """Retour à la ligne pour affichage à l'intérieur d'un nœud BN."""
    clean = re.sub(r"\s+", " ", str(title).strip())
    if not clean:
        return ""
    return textwrap.fill(clean, width=width, break_long_words=False, break_on_hyphens=False)


def _node_box_height_for_title(
    title: str,
    *,
    wrap_width: int = 22,
    line_height: float = 0.22,
    pad: float = 0.28,
    min_height: float = 0.72,
) -> float:
    n_lines = max(1, len(_wrap_node_title(title, wrap_width).splitlines()))
    return max(min_height, pad + n_lines * line_height)


def _layout_bn_columns_lr(
    nodes: list[str],
    variable_macro_map: Dict[str, str],
    *,
    col_gap: float = 2.4,
    row_gap: float = 1.4,
    box_width: float = 1.05,
    box_height: float = 0.92,
    box_heights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, BBox]]:
    """
    Disposition par colonnes macro (A0, A1 | B | C), empilement vertical par topic.
    Retourne centres (x, y) et boîtes (x0, y0, w, h) pour flèches bord à bord.
    """
    from collections import defaultdict

    by_macro: Dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        by_macro[_macro_of_node(n, variable_macro_map)].append(n)

    pos: Dict[str, Tuple[float, float]] = {}
    bboxes: Dict[str, BBox] = {}

    for macro in _MACRO_STACK_ORDER:
        group = sorted(by_macro.get(macro, []))
        if not group:
            continue
        cx = _MACRO_COL_X.get(macro, 2.0) * col_gap
        heights = [float((box_heights or {}).get(node, box_height)) for node in group]
        total_h = sum(heights) + row_gap * max(0, len(group) - 1)
        y_cursor = total_h / 2.0
        for node, h in zip(group, heights):
            cy = y_cursor - h / 2.0
            x0 = cx - box_width / 2.0
            y0 = cy - h / 2.0
            pos[node] = (cx, cy)
            bboxes[node] = (x0, y0, box_width, h)
            y_cursor = y0 - row_gap

    return pos, bboxes


def _bbox_anchor_lr(bbox: BBox, side: str) -> Tuple[float, float]:
    """Point d'ancrage sur le bord d'une boîte (left/right + centre vertical)."""
    x0, y0, w, h = bbox
    cy = y0 + h / 2.0
    if side == "right":
        return (x0 + w, cy)
    if side == "left":
        return (x0, cy)
    return (x0 + w / 2.0, y0 + h / 2.0)


def _draw_cpd_node_box(
    ax: plt.Axes,
    bbox: BBox,
    title: str,
    *,
    macro: str = "A0",
    wrap_width: int = 22,
) -> None:
    """Boîte de nœud BN avec titre centré (retour à la ligne si besoin)."""
    from matplotlib.patches import FancyBboxPatch

    x0, y0, w, h = bbox
    fill_color = MACRO_FILL.get(macro, "#E5E5E5")
    draw_color = MACRO_DRAW.get(macro, "#444444")
    patch = FancyBboxPatch(
        (x0, y0),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=fill_color,
        edgecolor=draw_color,
        linewidth=1.4,
        zorder=5,
    )
    ax.add_patch(patch)

    wrapped = _wrap_node_title(title, width=wrap_width)
    ax.text(
        x0 + w / 2.0,
        y0 + h / 2.0,
        wrapped,
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=draw_color,
        zorder=6,
        linespacing=1.15,
    )


def _draw_macro_column_bands_lr(
    ax: plt.Axes,
    variable_macro_map: Dict[str, str],
    nodes: list[str],
    *,
    col_gap: float = 2.4,
    row_gap: float = 1.4,
    box_width: float = 1.05,
) -> None:
    """Bandes verticales par colonne macro."""
    from matplotlib.patches import Rectangle

    from collections import defaultdict

    by_macro: Dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        by_macro[_macro_of_node(n, variable_macro_map)].append(n)

    for macro in _MACRO_STACK_ORDER:
        group = by_macro.get(macro, [])
        if not group:
            continue
        cx = _MACRO_COL_X.get(macro, 2.0) * col_gap
        n = len(group)
        y_top = (n - 1) / 2.0 * row_gap + box_width
        y_bot = -(n - 1) / 2.0 * row_gap - box_width
        half_w = box_width * 0.65 + 0.35
        rect = Rectangle(
            (cx - half_w, y_bot),
            2 * half_w,
            y_top - y_bot,
            facecolor=MACRO_LAYER_FILL.get(macro, "#f5f5f5"),
            edgecolor="none",
            alpha=0.5,
            zorder=0,
        )
        ax.add_patch(rect)


def extract_subgraph_for_slide(
    model: Any,
    path_nodes: Sequence[str],
    variable_macro_map: Dict[str, str],
    *,
    mode: str = "scenario_path",
) -> tuple[list[str], list[tuple[str, str]]]:
    """Sous-graphe compact pour slide : nœuds du chemin et arcs consécutifs."""
    del variable_macro_map, mode
    nodes = [str(n) for n in path_nodes if str(n)]
    if not nodes:
        return [], []

    model_edges = {tuple(map(str, edge)) for edge in model.edges()}
    edges: list[tuple[str, str]] = []
    for left, right in zip(nodes, nodes[1:]):
        if (left, right) in model_edges:
            edges.append((left, right))
        else:
            edges.append((left, right))
    return nodes, edges


def plot_bn_graph_cpd_boxes(
    model: Any,
    variable_macro_map: Dict[str, str],
    output_path: Path,
    title: str = "",
    short_title_map: Optional[Dict[str, str]] = None,
    themes_df: Optional[pd.DataFrame] = None,
    *,
    nodes_subset: Optional[Sequence[str]] = None,
    edges_subset: Optional[Sequence[tuple[str, str]]] = None,
    figsize: Optional[tuple[float, float]] = None,
    col_gap: float = 2.4,
    row_gap: float = 1.4,
    box_width: float = 1.75,
    box_height: float = 0.92,
    title_wrap_width: int = 22,
) -> None:
    """
    Graphe BN style article : nœuds boîtes avec titre, colonnes A0/A1 → B → C, flèches bord à bord.
    """
    from matplotlib.patches import FancyArrowPatch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    g = nx.DiGraph()
    if nodes_subset is not None:
        nodes = [str(n) for n in nodes_subset]
        if edges_subset is not None:
            g.add_edges_from(edges_subset)
        else:
            node_set = set(nodes)
            g.add_edges_from(
                (u, v) for u, v in model.edges() if str(u) in node_set and str(v) in node_set
            )
        for node in nodes:
            if node not in g:
                g.add_node(node)
    else:
        g.add_edges_from(model.edges())
        nodes = list(g.nodes())
    if not nodes:
        return

    if short_title_map is None:
        short_title_map = build_short_title_map(nodes, themes_df, variable_macro_map)

    box_heights = {
        n: _node_box_height_for_title(
            short_title_map.get(str(n), str(n)),
            wrap_width=title_wrap_width,
        )
        for n in nodes
    }

    pos, bboxes = _layout_bn_columns_lr(
        nodes,
        variable_macro_map,
        col_gap=col_gap,
        row_gap=row_gap,
        box_width=box_width,
        box_height=box_height,
        box_heights=box_heights,
    )

    from collections import defaultdict

    by_macro: Dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        by_macro[_macro_of_node(n, variable_macro_map)].append(n)

    max_col_h = box_height
    for group in by_macro.values():
        heights = [box_heights.get(node, box_height) for node in group]
        total = sum(heights) + row_gap * max(0, len(heights) - 1)
        max_col_h = max(max_col_h, total)

    n_cols = sum(1 for m in _MACRO_STACK_ORDER if by_macro.get(m))
    fig_w = max(14.0, 2.5 + n_cols * 3.5)
    fig_h = max(6.0, 1.4 + max_col_h * 1.05)
    if figsize is not None:
        fig_w, fig_h = figsize

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("white")

    for n in nodes:
        macro = _macro_of_node(n, variable_macro_map)
        title_n = short_title_map.get(str(n), str(n))
        _draw_cpd_node_box(ax, bboxes[n], title_n, macro=macro, wrap_width=title_wrap_width)

    for u, v in g.edges():
        if u not in bboxes or v not in bboxes:
            continue
        x0, y0 = _bbox_anchor_lr(bboxes[u], "right")
        x1, y1 = _bbox_anchor_lr(bboxes[v], "left")
        mu = _macro_of_node(u, variable_macro_map)
        mv = _macro_of_node(v, variable_macro_map)
        rad = 0.0
        if mu in ("A0", "A1") and mv == "B":
            rad = 0.15 if mu == "A0" else -0.12
        elif mu == "A0" and mv == "A1":
            rad = 0.08
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.5,
                color="#3b6ea8",
                alpha=0.85,
                connectionstyle=f"arc3,rad={rad}",
                zorder=3,
            )
        )

    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    pad_x = box_width + col_gap * 0.35
    pad_y = max(box_heights.values(), default=box_height) + row_gap * 0.45
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=18)

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=MACRO_COLOR[m],
            markeredgecolor="#333333",
            markersize=10,
            label=f"{MACRO_ONTOLOGY_FR.get(m, (m, ''))[0]}",
        )
        for m in ("A0", "A1", "B", "C")
        if any(_macro_of_node(n, variable_macro_map) == m for n in nodes)
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=4,
            frameon=True,
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_macro_layer_bands(
    ax: plt.Axes,
    variable_macro_map: Dict[str, str],
    nodes: list[str],
) -> None:
    """Bandes de fond par couche macro (lisibilité article)."""
    from matplotlib.patches import FancyBboxPatch

    present = {_macro_of_node(n, variable_macro_map) for n in nodes}
    bands = [
        ("A0", "A1", -2.0, 2.0, 0.55, 1.55),
        ("B", "B", -1.05, 1.05, -0.55, 0.55),
        ("C", "C", -1.05, 1.05, -1.55, -0.65),
    ]
    for m0, m1, x0, x1, y0, y1 in bands:
        if m0 not in present and m1 not in present:
            continue
        color = MACRO_LAYER_FILL.get(m0, "#f5f5f5")
        patch = FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,rounding_size=0.08,pad=0.02",
            facecolor=color,
            edgecolor="none",
            alpha=0.55,
            zorder=0,
        )
        ax.add_patch(patch)


def plot_macro_causality_schematic(
    output_path: Path,
    *,
    include_severity: bool = False,
    title: str = CHAIN_CAUSAL_STRUCTURE,
) -> None:
    """
    Schéma fixe A0→A1, A0→B, A1→B, B→C — flux gauche→droite (cohérent avec boîtes CPD topics).
    """
    from matplotlib.patches import FancyArrowPatch, Rectangle

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    node_r = 0.26
    pos = {
        "A0": (0.0, 0.0),
        "A1": (2.5, 0.0),
        "B": (5.2, 0.0),
        "C": (7.8, 0.0),
    }
    edges = [("A0", "A1"), ("A0", "B"), ("A1", "B"), ("B", "C")]
    if include_severity:
        pos["S"] = (10.4, 0.0)
        edges.append(("C", "S"))

    fig_w = 11.5 if not include_severity else 13.5
    fig, ax = plt.subplots(figsize=(fig_w, 4.2), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(-1.2, max(p[0] for p in pos.values()) + 1.4)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")

    for key, (x, _y) in pos.items():
        macro = "Severity" if key == "S" else key
        rect = Rectangle(
            (x - 0.72, -1.05),
            1.44,
            2.1,
            facecolor=MACRO_LAYER_FILL.get(macro, "#f5f5f5"),
            edgecolor="none",
            alpha=0.55,
            zorder=0,
        )
        ax.add_patch(rect)

    for u, v in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        start = (x0 + node_r, y0)
        end = (x1 - node_r, y1)
        rad = 0.0
        if u == "A0" and v == "B":
            rad = 0.12
        elif u == "A1" and v == "B":
            rad = -0.1
        elif u == "A0" and v == "A1":
            rad = 0.05
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=2.0,
                color="#444444",
                connectionstyle=f"arc3,rad={rad}",
                zorder=2,
            )
        )

    for key, (x, y) in pos.items():
        macro = "Severity" if key == "S" else key
        color = MACRO_COLOR.get(macro, "#888888")
        title_n, subtitle = MACRO_ONTOLOGY_FR.get(macro, (macro, ""))
        if key == "S":
            title_n, subtitle = "Gravité", "Issue sévère"
        circle = plt.Circle((x, y), node_r, facecolor=color, edgecolor="#1a1a1a", linewidth=1.2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y + 0.02, macro, ha="center", va="center", fontsize=13, fontweight="bold", color="white", zorder=4)
        ax.text(x, y - 0.42, title_n.split("—", 1)[-1].strip(), ha="center", va="top", fontsize=9.5, color="#222222", zorder=4)
        ax.text(
            x,
            y - 0.68,
            subtitle,
            ha="center",
            va="top",
            fontsize=8,
            color="#555555",
            style="italic",
            wrap=True,
            zorder=4,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    fig.text(
        0.5,
        0.02,
        "Arcs autorisés : A0→A1 · A0→B · A1→B · B→C"
        + (" · C→gravité" if include_severity else "")
        + "  —  pas de lien direct A0/A1→C",
        ha="center",
        fontsize=9,
        color="#666666",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_bn_graph(
    model: Any,
    variable_macro_map: Dict[str, str],
    output_path: Path,
    title: str = "",
    node_label_map: Optional[Dict[str, str]] = None,
    short_title_map: Optional[Dict[str, str]] = None,
    themes_df: Optional[pd.DataFrame] = None,
    node_size_scale: float = 420.0,
    show_cpd_cards: bool = True,
    card_offset: Tuple[float, float] = (0.0, -78.0),
    bar_width: int = 10,
) -> None:
    """
    Graphe BN : cercles colorés seuls ; cartes CPD (titre + barres) en annotation décalée.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if sns is not None:
        sns.set_theme(style="white", context="paper", font_scale=1.05)
    g = nx.DiGraph()
    g.add_edges_from(model.edges())
    nodes = list(g.nodes())
    pos = _layout_accident_causality(nodes, variable_macro_map)

    if short_title_map is None:
        short_title_map = build_short_title_map(nodes, themes_df, variable_macro_map)
    cards = build_node_cards_for_model(model, short_title_map) if show_cpd_cards else {}

    colors = [MACRO_COLOR.get(_macro_of_node(n, variable_macro_map), "#888888") for n in nodes]
    sizes = [node_size_scale * (1.0 + 0.03 * min(g.degree(n), 4)) for n in nodes]

    fig_h = 12 if show_cpd_cards else 10
    fig = plt.figure(figsize=(18, fig_h), facecolor="white")
    ax = fig.add_subplot(111, facecolor="white")
    _draw_macro_layer_bands(ax, variable_macro_map, nodes)

    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    margin = 0.9
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin - (1.2 if show_cpd_cards else 0.3), max(ys) + margin)
    ax.axis("off")

    from matplotlib.patches import FancyArrowPatch

    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        rad = 0.06
        mu = _macro_of_node(u, variable_macro_map)
        mv = _macro_of_node(v, variable_macro_map)
        if mu == "A0" and mv == "B":
            rad = 0.12
        elif mu == "A1" and mv == "B":
            rad = -0.1
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.35,
                color="#5a5a5a",
                alpha=0.75,
                connectionstyle=f"arc3,rad={rad}",
                zorder=2,
            )
        )

    nx.draw_networkx_nodes(
        g, pos, node_color=colors, node_size=sizes, alpha=0.96, edgecolors="#1a1a1a", linewidths=1.0, ax=ax
    )

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=MACRO_COLOR[m],
            markersize=10,
            label=f"{m} — {MACRO_ONTOLOGY_FR.get(m, (m, ''))[1][:40]}",
        )
        for m in ("A0", "A1", "B", "C")
        if any(_macro_of_node(n, variable_macro_map) == m for n in nodes)
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02 if show_cpd_cards else 0.02),
            ncol=2,
            frameon=True,
            fontsize=8,
            title=CHAIN_LEGEND,
            title_fontsize=9,
        )

    if show_cpd_cards:
        for i, n in enumerate(nodes):
            card = cards.get(str(n), short_title_map.get(str(n), str(n)))
            card = textwrap.fill(card, width=38, break_long_words=False)
            ox, oy = card_offset
            oy_adj = oy - (i % 3) * 6
            ax.annotate(
                card,
                xy=pos[n],
                xytext=(ox, oy_adj),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7.5,
                family="monospace",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "#fafafa",
                    "edgecolor": "#bbbbbb",
                    "linewidth": 0.8,
                    "alpha": 0.97,
                },
                zorder=10,
            )
    elif node_label_map:
        for n in nodes:
            lbl = node_label_map.get(n, str(n))
            ax.annotate(
                lbl,
                xy=pos[n],
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.9},
            )

    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, pad=14)
    ymin = min(y for _, y in pos.values()) if pos else -0.5
    ax.set_ylim(ymin - 1.8, max(y for _, y in pos.values()) + 0.6 if pos else 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_node_cards_png(
    model: Any,
    short_title_map: Dict[str, str],
    output_dir: Path,
    *,
    max_single: int = 15,
) -> List[Path]:
    """Exporte les cartes CPD : une PNG par nœud ou grille si nombreux."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = build_node_cards_for_model(model, short_title_map)
    nodes = list(model.nodes())
    saved: List[Path] = []

    if len(nodes) <= max_single:
        for node in nodes:
            card = cards[str(node)]
            fig, ax = plt.subplots(figsize=(4.5, 2.2), facecolor="white")
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                card,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                family="monospace",
            )
            safe = "".join(c if c.isalnum() or c == "_" else "_" for c in str(node))
            path = output_dir / f"node_{safe}.png"
            fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            saved.append(path)
        return saved

    ncols = 3
    nrows = int(math.ceil(len(nodes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.5 * nrows), facecolor="white")
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, node in zip(axes_flat, nodes):
        ax.axis("off")
        ax.text(0.5, 0.5, cards[str(node)], transform=ax.transAxes, ha="center", va="center", fontsize=7, family="monospace")
    for ax in axes_flat[len(nodes) :]:
        ax.axis("off")
    path = output_dir / "node_cards_grid.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)
    return saved


def plot_adjacency_heatmap(
    model: Any,
    variable_order: list[str],
    output_path: Path,
    title: str = "",
    node_label_map: Optional[Dict[str, str]] = None,
    *,
    themes_df: Optional[pd.DataFrame] = None,
    variable_macro_map: Optional[Dict[str, str]] = None,
    tick_label_max_len: int = 48,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nodes = [n for n in variable_order if n in model.nodes()]
    n = len(nodes)
    adj = np.zeros((n, n))
    idx = {nodes[i]: i for i in range(n)}
    for u, v in model.edges():
        if u in idx and v in idx:
            adj[idx[u], idx[v]] = 1.0

    if node_label_map is None and themes_df is not None:
        node_label_map = build_short_title_map(nodes, themes_df, variable_macro_map)

    tick_labels = []
    for node in nodes:
        if node_label_map and node in node_label_map:
            lbl = str(node_label_map[node]).replace("\n", " ")
        elif node_label_map and str(node) in node_label_map:
            lbl = str(node_label_map[str(node)]).replace("\n", " ")
        else:
            lbl = str(node)
        if len(lbl) > tick_label_max_len:
            lbl = lbl[: tick_label_max_len - 1] + "…"
        tick_labels.append(lbl)

    fig_w = max(10, min(24, 0.45 * n + 6))
    plt.figure(figsize=(fig_w, fig_w * 0.85))
    if sns is not None:
        sns.heatmap(adj, xticklabels=tick_labels, yticklabels=tick_labels, cmap="Blues", cbar=False)
    else:
        plt.imshow(adj.values, cmap="Blues", aspect="auto")
        plt.xticks(range(n), tick_labels, rotation=45, ha="right", fontsize=8)
        plt.yticks(range(n), tick_labels, fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.title(title or "Matrice d'adjacence")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def try_plotly_interactive(
    model: Any,
    output_html: Path,
    *,
    node_label_map: Optional[Dict[str, str]] = None,
    short_title_map: Optional[Dict[str, str]] = None,
    variable_macro_map: Optional[Dict[str, str]] = None,
    themes_df: Optional[pd.DataFrame] = None,
    title: str = "Réseau bayésien — exploration interactive",
) -> bool:
    try:
        import plotly.graph_objects as go

        nodes = list(model.nodes())
        if short_title_map is None:
            short_title_map = build_short_title_map(nodes, themes_df, variable_macro_map)
        cards = build_node_cards_for_model(model, short_title_map)
        pos = _layout_by_macro(nodes, variable_macro_map or {})

        edge_x, edge_y = [], []
        for u, v in model.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        hover = []
        for n in nodes:
            long_lbl = (node_label_map or {}).get(n, "")
            card = cards.get(str(n), str(n))
            hover.append(f"<b>{short_title_map.get(n, n)}</b><br>{card.replace(chr(10), '<br>')}")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1.2, color="#888"),
                hoverinfo="none",
            )
        )
        colors = [MACRO_COLOR.get(_macro_from_node(n, variable_macro_map), "#888") for n in nodes]
        fig.add_trace(
            go.Scatter(
                x=[pos[n][0] for n in nodes],
                y=[pos[n][1] for n in nodes],
                mode="markers",
                marker=dict(size=14, color=colors, line=dict(width=1, color="#222")),
                text=[short_title_map.get(n, n) for n in nodes],
                hovertext=hover,
                hoverinfo="text",
            )
        )
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            showlegend=False,
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_html), include_plotlyjs="cdn")
        return True
    except Exception:
        return False


def try_pyvis_bn_graph(
    model: Any,
    output_html: Path,
    *,
    short_title_map: Optional[Dict[str, str]] = None,
    variable_macro_map: Optional[Dict[str, str]] = None,
    themes_df: Optional[pd.DataFrame] = None,
    title: str = "Réseau bayésien — Pyvis",
    height: str = "750px",
    width: str = "100%",
) -> bool:
    try:
        from pyvis.network import Network

        nodes = list(model.nodes())
        if short_title_map is None:
            short_title_map = build_short_title_map(nodes, themes_df, variable_macro_map)
        cards = build_node_cards_for_model(model, short_title_map)

        net = Network(height=height, width=width, directed=True, bgcolor="#ffffff", font_color="#222222")
        net.barnes_hut(gravity=-12000, central_gravity=0.2, spring_length=180)

        layout_px = {"A0": -400, "A1": 130, "B": 0, "C": 400, "Severity": 520}
        for i, n in enumerate(nodes):
            macro = _macro_from_node(n, variable_macro_map)
            x = layout_px.get(macro, 0) + (i % 5) * 14
            y = {"A0": -80, "A1": -80, "B": 40, "C": 160, "Severity": 260}.get(macro, 0) + (i % 3) * 8
            color = MACRO_COLOR.get(macro, "#888888")
            card_html = cards[str(n)].replace("\n", "<br>")
            label = short_title_map.get(str(n), str(n))
            net.add_node(
                n,
                label=label,
                title=f"<pre style='font-family:monospace;font-size:11px'>{card_html}</pre>",
                color=color,
                x=float(x),
                y=float(y),
                physics=False,
                shape="dot",
                size=18,
            )

        for u, v in model.edges():
            net.add_edge(u, v, arrows="to")

        output_html.parent.mkdir(parents=True, exist_ok=True)
        net.set_options(
            """
        var options = {
          "nodes": {"font": {"size": 12, "face": "monospace"}},
          "edges": {"smooth": {"type": "continuous"}},
          "physics": {"enabled": false}
        }
        """
        )
        net.save_graph(str(output_html))
        return output_html.is_file()
    except Exception:
        return False
