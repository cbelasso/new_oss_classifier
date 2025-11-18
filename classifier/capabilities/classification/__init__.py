"""
Classification detection capability.
"""

from .capability import ClassificationCapability
from .models import ClassificationOutput, SingleClassificationResult
from .prompts import standard_classification_prompt

__all__ = [
    "ClassificationCapability",
    "ClassificationOutput",
    "SingleClassificationResult",
    "standard_classification_prompt",
]
