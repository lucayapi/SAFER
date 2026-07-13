"""Pipeline d'annotation d'unités factuelles via API OpenAI."""

from annotation.config import AnnotationConfig
from annotation.runner import classify_dataframe_with_cache

__all__ = ["AnnotationConfig", "classify_dataframe_with_cache"]
