"""
Recommendations detection capability.
"""

from .capability import RecommendationsCapability
from .models import RecommendationsOutput, RecommendationSpan
from .prompts import recommendations_detection_prompt

__all__ = [
    "RecommendationsCapability",
    "RecommendationsOutput",
    "RecommendationSpan",
    "recommendations_detection_prompt",
]
