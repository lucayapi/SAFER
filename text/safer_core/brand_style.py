"""
Charte graphique SAFER (palette et typo Inter).

Seules les couleurs listées ici sont autorisées dans les figures du projet.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

# —— Couleurs héros / accents ——
HERO_BLUE = "#44a6f7"
DEEP_BLUE = "#2a32a5"
ACCENT_BLUE = "#1893f8"
ACCENT_RED = "#ef3a5d"
ACCENT_ORANGE = "#ff751f"

# —— Boutons (UI ; matplotlib : barres d’action si besoin) ——
BUTTON_BLUE = "#0052FF"
BUTTON_TEXT = "#ffffff"

# —— Fonds ——
WHITE = "#ffffff"
BG_SURFACE = "#F7F7F7"

# —— Texte ——
TEXT_TITLE = "#0A0A0A"
TEXT_BODY = "#4A4A4A"
TEXT_MUTED = "#7A7A7A"

# —— Liens / actions ——
ACTION = DEEP_BLUE

# Bordures & traits (dérivés du texte titre, pas de gris hors charte)
EDGE = TEXT_TITLE
GRID = BG_SURFACE

# Macros BTP (4 classes)
MACRO_COLORS: Dict[str, str] = {
    "A0": HERO_BLUE,
    "A1": DEEP_BLUE,
    "B": ACCENT_BLUE,
    "C": ACCENT_RED,
}

# Cinq méthodes d’embedding (notebook 01)
METHOD_DISPLAY_COLORS: Dict[str, str] = {
    "Embedding brut": TEXT_MUTED,
    "Batch Triplet": HERO_BLUE,
    "SupCon": ACCENT_BLUE,
    "SoftTriple": ACCENT_RED,
    "SCGM": DEEP_BLUE,
}

SERIES_PALETTE: List[str] = [
    HERO_BLUE,
    DEEP_BLUE,
    ACCENT_BLUE,
    ACCENT_RED,
    ACCENT_ORANGE,
]

ALLOWED_HEX: frozenset[str] = frozenset(
    {
        HERO_BLUE,
        DEEP_BLUE,
        ACCENT_BLUE,
        ACCENT_RED,
        ACCENT_ORANGE,
        BUTTON_BLUE,
        BUTTON_TEXT,
        WHITE,
        BG_SURFACE,
        TEXT_TITLE,
        TEXT_BODY,
        TEXT_MUTED,
        ACTION,
    }
)

# Typo (desktop par défaut ; mobile via scale_font_sizes)
FONT_FAMILY = "Inter"
FONT_SIZES = {
    "h1": 60.0,
    "h2": 34.0,
    "h3": 22.0,
    "body": 17.0,
    "caption": 15.0,
}
FONT_WEIGHTS = {
    "h1": 700,
    "h2": 600,
    "h3": 600,
    "body": 400,
    "caption": 400,
}

SPACING = {
    "section_px": 100.0,
    "block_px": 40.0,
    "text_px": 14.0,
}


def assert_brand_color(hex_color: str) -> str:
    """Valide qu’une couleur fait partie de la charte (insensible à la casse)."""
    key = str(hex_color).strip().lower()
    allowed = {c.lower() for c in ALLOWED_HEX}
    if key not in allowed:
        raise ValueError(f"Couleur hors charte : {hex_color!r}. Autorisées : {sorted(ALLOWED_HEX)}")
    return key if key.startswith("#") else f"#{key}"


def macro_color_map(macros_order: Sequence[str] = ("A0", "A1", "B", "C")) -> Dict[str, str]:
    """Couleur par macro (repli : TEXT_MUTED)."""
    out: Dict[str, str] = {}
    for i, m in enumerate(macros_order):
        out[str(m)] = MACRO_COLORS.get(str(m), SERIES_PALETTE[i % len(SERIES_PALETTE)])
    return out


def method_color_map(method_labels: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i, label in enumerate(method_labels):
        out[str(label)] = METHOD_DISPLAY_COLORS.get(
            str(label), SERIES_PALETTE[i % len(SERIES_PALETTE)]
        )
    return out


def series_colors(n: int) -> List[str]:
    return [SERIES_PALETTE[i % len(SERIES_PALETTE)] for i in range(max(0, n))]


def apply_matplotlib_brand(*, mobile: bool = False) -> None:
    """Applique Inter + palette aux rcParams matplotlib."""
    import matplotlib.pyplot as plt

    scale = 0.58 if mobile else 1.0
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                FONT_FAMILY,
                "Inter",
                "DejaVu Sans",
                "Arial",
                "Helvetica",
            ],
            "font.size": FONT_SIZES["body"] * scale,
            "axes.titlesize": FONT_SIZES["h2"] * scale,
            "axes.titleweight": FONT_WEIGHTS["h2"],
            "axes.labelsize": FONT_SIZES["h3"] * scale,
            "axes.labelweight": FONT_WEIGHTS["h3"],
            "axes.labelcolor": TEXT_BODY,
            "axes.edgecolor": EDGE,
            "axes.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "text.color": TEXT_BODY,
            "grid.color": GRID,
            "grid.alpha": 1.0,
            "legend.frameon": True,
            "legend.facecolor": WHITE,
            "legend.edgecolor": GRID,
        }
    )


def style_axes(
    ax,
    *,
    title: Optional[str] = None,
    title_level: str = "h2",
) -> None:
    """Style axes selon la charte (fond blanc, grille surface)."""
    ax.set_facecolor(WHITE)
    if title:
        size = FONT_SIZES.get(title_level, FONT_SIZES["h2"])
        weight = FONT_WEIGHTS.get(title_level, 600)
        ax.set_title(title, fontsize=size, fontweight=weight, color=TEXT_TITLE)
    ax.tick_params(colors=TEXT_MUTED)
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-", alpha=1.0)
    ax.set_axisbelow(True)


def style_figure(fig) -> None:
    fig.patch.set_facecolor(WHITE)


def bar_kwargs(*, action: bool = False) -> Dict[str, object]:
    """kwargs barres : action → bleu bouton #0052FF."""
    return {
        "color": BUTTON_BLUE if action else DEEP_BLUE,
        "edgecolor": EDGE,
        "linewidth": 0.6,
    }


def plotly_qualitative_colors(n: int) -> List[str]:
    return series_colors(n)


def rgb_css(hex_color: str) -> str:
    """#RRGGBB → rgb(r,g,b) pour Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgb({r},{g},{b})"


def macro_palette_plotly() -> Dict[str, str]:
    return {k: rgb_css(v) for k, v in MACRO_COLORS.items()}


def matplotlib_heatmap_cmap():
    """Colormap heatmap : uniquement couleurs charte (blanc → surface → bleus)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "safer_heat",
        [WHITE, BG_SURFACE, HERO_BLUE, DEEP_BLUE],
        N=256,
    )
