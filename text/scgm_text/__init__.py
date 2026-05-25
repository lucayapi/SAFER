"""SCGM-G end-to-end text training (strict fidelity, SGD)."""

from scgm_text.dataset_text_raw import TextRawDataset, split_by_group
from scgm_text.fidelity import apply_scgm_strict_defaults, describe_fidelity_mode, flatten_config_yaml
from scgm_text.scgm_text_model import SCGMTextModel

__all__ = [
    "TextRawDataset",
    "split_by_group",
    "SCGMTextModel",
    "apply_scgm_strict_defaults",
    "describe_fidelity_mode",
    "flatten_config_yaml",
]
