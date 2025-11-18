"""
trend detection capability.
"""

from .capability import TrendCapability
from .models import TrendsOutput, TrendSpan
from .prompts import trends_detection_prompt

__all__ = [
    "TrendCapability",
    "TrendSpan",
    "TrendsOutput",
    "trends_detection_prompt",
]
