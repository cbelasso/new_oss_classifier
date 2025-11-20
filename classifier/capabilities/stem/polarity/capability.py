"""
Stem polarity analysis capability.

Analyzes the sentiment/polarity of text towards complete classification stems.
"""

from typing import Any, Dict, Type

from pydantic import BaseModel

from ..base import StemCapability
from .models import StemPolarityOutput
from .prompts import stem_polarity_prompt


class StemPolarityCapability(StemCapability):
    """
    Evaluates polarity (sentiment) for complete classification stems.

    For each complete stem, determines whether the text expresses positive,
    negative, neutral, or mixed sentiment towards that specific topic.

    Requires classification results to identify complete stems.
    """

    @property
    def name(self) -> str:
        return "stem_polarity"

    @property
    def schema(self) -> Type[BaseModel]:
        return StemPolarityOutput

    def get_stem_prompt_fn(self):
        return stem_polarity_prompt

    def _extract_stem_result(self, result_dict: Dict[str, Any]) -> Any:
        """Extract just the polarity_result, or None if no polarity."""
        if result_dict and result_dict.get("has_polarity"):
            return result_dict.get("polarity_result", {})
        return None
