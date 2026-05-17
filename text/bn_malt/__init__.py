"""BN à partir des sorties MALT / SCGM : agrégation, structure, apprentissage."""

from .aggregate_malt_variables import (
    create_accident_topic_matrix,
    export_aggregate_outputs,
)
from .bn_diagnostics import compare_structure_rows, run_model_diagnostics
from .bn_inference import run_bn_queries
from .bn_learning import export_cpds_to_dir, fit_bn_parameters, save_bn_pickle, try_write_bif
from .bn_structure import export_edge_tables, learn_macro_constrained_structure, learn_unconstrained_structure, macro_chain_model
from .bn_visualization import (
    build_node_cards_for_model,
    build_node_short_title,
    build_node_summary_label,
    build_short_title_map,
    build_topic_node_label_map,
    join_theme_summary_to_selected_variables,
    load_openai_themes_for_bn,
    resolve_openai_themes_path,
    cpd_binary_marginal,
    display_node_card,
    export_node_cards_png,
    export_node_marginals_csv,
    format_node_card,
    format_prob_bar,
    plot_adjacency_heatmap,
    plot_bn_graph,
    try_plotly_interactive,
    try_pyvis_bn_graph,
)
from .scenario_mining import export_scenarios, extract_typical_scenarios
from .utils import (
    ensure_output_dirs,
    find_repo_root,
    load_metadata_for_bn,
    require_bn_malt_files,
    resolve_repo_path,
)

__all__ = [
    "require_bn_malt_files",
    "load_metadata_for_bn",
    "find_repo_root",
    "resolve_repo_path",
    "ensure_output_dirs",
    "create_accident_topic_matrix",
    "export_aggregate_outputs",
    "learn_macro_constrained_structure",
    "learn_unconstrained_structure",
    "macro_chain_model",
    "export_edge_tables",
    "fit_bn_parameters",
    "save_bn_pickle",
    "export_cpds_to_dir",
    "try_write_bif",
    "run_bn_queries",
    "plot_bn_graph",
    "build_topic_node_label_map",
    "build_node_short_title",
    "build_node_summary_label",
    "build_short_title_map",
    "build_node_cards_for_model",
    "cpd_binary_marginal",
    "format_prob_bar",
    "format_node_card",
    "display_node_card",
    "export_node_marginals_csv",
    "export_node_cards_png",
    "plot_adjacency_heatmap",
    "try_plotly_interactive",
    "try_pyvis_bn_graph",
    "load_openai_themes_for_bn",
    "resolve_openai_themes_path",
    "join_theme_summary_to_selected_variables",
    "extract_typical_scenarios",
    "export_scenarios",
    "run_model_diagnostics",
    "compare_structure_rows",
]
