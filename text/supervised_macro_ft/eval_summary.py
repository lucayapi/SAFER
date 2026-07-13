"""Re-exports classification_eval (compat supervised_macro_ft)."""

from safer_core.classification_eval import (  # noqa: F401
    CLASSIFICATION_METRIC_KEYS,
    build_all_test_corpora_metrics_table,
    resolve_test_corpora,
    summarize_ood_classification,
)
