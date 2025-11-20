"""
Capabilities system for extensible text classification.

This package provides a plugin-like architecture for adding new features
to the classification system. Each capability is a self-contained module
that can declare dependencies and be composed with other capabilities.
"""

from .alerts import AlertsCapability, AlertsOutput, AlertSpan
from .base import Capability
from .classification import (
    BatchClassificationResult,
    BFSClassificationCapability,
    BundledClassificationCapability,
    BundledClassificationResult,
    ClassificationCapability,
    ClassificationOutput,
    SingleClassificationResult,
    bundled_classification_prompt,
    standard_classification_prompt,
)
from .recommendations import (
    RecommendationsCapability,
    RecommendationsOutput,
    RecommendationSpan,
)
from .registry import CapabilityRegistry, create_default_registry
from .stem import (
    StemCapability,
    StemPolarityCapability,
    StemPolarityOutput,
    StemPolarityResult,
    StemRecommendationsCapability,
    StemRecommendationsOutput,
    StemRecommendationType,
    StemTrendCapability,
    StemTrendsOutput,
    StemTrendSpan,
    SubStemPolarityCapability,
)
from .trend import TrendCapability, TrendsOutput, TrendSpan

__all__ = [
    # Base
    "Capability",
    # Registry
    "CapabilityRegistry",
    "create_default_registry",
    # Classification
    "ClassificationCapability",
    "BFSClassificationCapability",
    "BundledClassificationCapability",
    "ClassificationOutput",
    "SingleClassificationResult",
    "BatchClassificationResult",
    "BundledClassificationResult",
    "standard_classification_prompt",
    "bundled_classification_prompt",
    # Recommendations
    "RecommendationsCapability",
    "RecommendationsOutput",
    "RecommendationSpan",
    # Alerts
    "AlertsCapability",
    "AlertsOutput",
    "AlertSpan",
    # Trend (global)
    "TrendCapability",
    "TrendsOutput",
    "TrendSpan",
    # Stem capabilities
    "StemCapability",
    "StemPolarityCapability",
    "StemPolarityOutput",
    "StemPolarityResult",
    "SubStemPolarityCapability",
    "StemRecommendationsCapability",
    "StemRecommendationsOutput",
    "StemRecommendationType",
    "StemTrendCapability",
    "StemTrendsOutput",
    "StemTrendSpan",
]
