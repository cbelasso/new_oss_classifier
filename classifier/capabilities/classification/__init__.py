"""
Classification detection capability.
"""

from .base import ClassificationCapability
from .bfs import BFSClassificationCapability
from .bundled import BundledClassificationCapability
from .models import (
    BatchClassificationResult,
    BundledClassificationResult,
    ClassificationOutput,
    SingleClassificationResult,
)
from .prompts import bundled_classification_prompt, standard_classification_prompt

__all__ = [
    "ClassificationCapability",
    "ClassificationOutput",
    "SingleClassificationResult",
    "standard_classification_prompt",
    "BFSClassificationCapability",
    "BatchClassificationResult",
    "BundledClassificationCapability",
    "bundled_classification_prompt",
    "BundledClassificationResult",
]
