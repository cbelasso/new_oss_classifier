"""
Stem-level analysis capabilities.

These capabilities analyze complete classification paths (stems) rather than
individual nodes or raw text. They provide deeper insights by considering
the full hierarchical context of classifications.

Available stem capabilities:
- StemPolarityCapability: Sentiment analysis for complete stems
- SubStemPolarityCapability: Sentiment analysis for all sub-stems
- StemRecommendationsCapability: Recommendation analysis for stems
- StemTrendCapability: Temporal trend analysis for stems
"""

from .base import StemCapability
from .polarity import StemPolarityCapability, StemPolarityOutput, StemPolarityResult
from .recommendations import (
    StemRecommendationsCapability,
    StemRecommendationsOutput,
    StemRecommendationType,
)
from .sub_polarity import SubStemPolarityCapability
from .trend import StemTrendCapability, StemTrendsOutput, StemTrendSpan

__all__ = [
    # Base
    "StemCapability",
    # Polarity
    "StemPolarityCapability",
    "StemPolarityOutput",
    "StemPolarityResult",
    # Sub-stem polarity
    "SubStemPolarityCapability",
    # Recommendations
    "StemRecommendationsCapability",
    "StemRecommendationsOutput",
    "StemRecommendationType",
    # Trend
    "StemTrendCapability",
    "StemTrendSpan",
    "StemTrendsOutput",
]
