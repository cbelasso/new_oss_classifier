"""
Stem-specific trend analysis capability.

Analyzes temporal change patterns for each complete classification stem.
"""

from typing import Any, Dict, Type

from pydantic import BaseModel

from ..base import StemCapability
from .models import StemTrendsOutput
from .prompts import stem_trends_prompt


class StemTrendCapability(StemCapability):
    """
    Evaluates temporal change patterns for complete classification stems.

    For each complete stem (path that reaches a leaf node), determines whether
    the text describes temporal changes about that specific topic - such as
    improvements, declines, stability, or fluctuations over time.

    Can identify multiple distinct trends for the same topic path if the comment
    discusses multiple aspects changing over time.

    Requires classification results to identify complete stems.
    """

    @property
    def name(self) -> str:
        return "stem_trend"

    @property
    def schema(self) -> Type[BaseModel]:
        return StemTrendsOutput

    def get_stem_prompt_fn(self):
        return stem_trends_prompt

    def _extract_stem_result(self, result_dict: Dict[str, Any]) -> Any:
        """Extract list of trends (or empty list if no trends)."""
        if result_dict and result_dict.get("has_trends"):
            return result_dict.get("trends", [])
        return []

    def get_result_key(self) -> str:
        """Store under 'stem_trends' key."""
        return "stem_trends"
