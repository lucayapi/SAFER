"""Visualisation des graphes BN (matplotlib + Plotly + Pyvis) avec cartes CPD."""

from __future__ import annotations

import math
import re
import textwrap
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

MACRO_COLOR = {
    "A0": "#4C78A8",
    "A1": "#F58518",
    "B": "#54A24B",
    "C": "#E45756",
    "Severity": "#B279A2",
    "Severity_high": "#B279A2",
    "SEVERITY": "#B279A2",
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
        candidates.append(Path(staging_dir) / "malt_like_exports" / _OPENAI_THEMES_BASENAME)
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
        "resultats/scgm_text/topics/themes_by_z_openai.csv avec la colonne theme_summary."
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


def join_theme_summary_to_selected_variables(
    selected_variables_df: pd.DataFrame,
    themes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Ajoute ``theme_summary`` à ``selected_bn_variables.csv`` via ``z_id``."""
    if selected_variables_df.empty or "z_id" not in selected_variables_df.columns:
        return selected_variables_df
    zmap = _z_id_to_theme_summary(themes_df)
    out = selected_variables_df.copy()
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
    Libellé affiché sur le graphe BN : ``theme_summary`` OpenAI (6–10 mots) pour les ``Z_*``.
    """
    n = str(node)
    macro = _macro_from_node(n, variable_macro_map)
    if n.startswith("M_"):
        return f"Macro {macro} (agrégat)"[:max_len]
    if "Severity" in n:
        return "Gravité élevée"[:max_len]

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
    for state, p in probs:
        lines.append(f"{state}  {format_prob_bar(p, width=bar_width)}")
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


def _layout_by_macro(
    nodes: list[str],
    variable_macro_map: Dict[str, str],
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    x_slot = {"A0": 0.0, "A1": 1.0, "B": 2.0, "C": 3.0}
    rng = np.random.default_rng(seed)
    pos: Dict[str, Tuple[float, float]] = {}
    for n in nodes:
        macro = _macro_of_node(n, variable_macro_map)
        bx = 3.2 if macro == "Severity" else float(x_slot.get(macro, 2.0))
        pos[n] = (
            bx + 0.08 * rng.standard_normal(),
            0.25 * rng.standard_normal(),
        )
    return pos


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
    sns.set_theme(style="white", context="notebook")
    g = nx.DiGraph()
    g.add_edges_from(model.edges())
    nodes = list(g.nodes())
    pos = _layout_by_macro(nodes, variable_macro_map)

    if short_title_map is None:
        short_title_map = build_short_title_map(nodes, themes_df, variable_macro_map)
    cards = build_node_cards_for_model(model, short_title_map) if show_cpd_cards else {}

    colors = [MACRO_COLOR.get(_macro_of_node(n, variable_macro_map), "#888888") for n in nodes]
    sizes = [node_size_scale * (1.0 + 0.04 * g.degree(n)) for n in nodes]

    fig_h = 11 if show_cpd_cards else 9
    fig = plt.figure(figsize=(16, fig_h), facecolor="white")
    ax = fig.add_subplot(111, facecolor="white")

    nx.draw_networkx_edges(
        g, pos, arrows=True, arrowsize=14, width=1.0, alpha=0.5, edge_color="#666666", ax=ax
    )
    nx.draw_networkx_nodes(
        g, pos, node_color=colors, node_size=sizes, alpha=0.95, edgecolors="#222222", linewidths=0.8, ax=ax
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
    sns.heatmap(adj, xticklabels=tick_labels, yticklabels=tick_labels, cmap="Blues", cbar=False)
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

        x_slot = {"A0": -400, "A1": -130, "B": 140, "C": 400}
        for i, n in enumerate(nodes):
            macro = _macro_from_node(n, variable_macro_map)
            x = x_slot.get(macro, 0) + (i % 5) * 12
            y = (i // 5) * 90 - 120
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
