"""
Stem recommendation analysis capability.

Analyzes what types of recommendations apply to complete classification stems.
"""

from typing import Any, Dict, Type

from pydantic import BaseModel

from ..base import StemCapability
from .models import StemRecommendationsOutput
from .prompts import stem_recommendations_prompt

# Mapping from internal types to output format
STEM_RECOMMENDATION_TYPE_MAPPING = {
    "start": "Recommendation - start",
    "stop": "Recommendation - stop",
    "do_more": "Recommendation - do more",
    "do_less": "Recommendation - do less",
    "continue": "Recommendation - continue",
    "change": "Recommendation - change",
}


class StemRecommendationsCapability(StemCapability):
    """
    Evaluates recommendation types for complete classification stems.

    For each complete stem (path that reaches a leaf node), determines what
    types of recommendations (start/stop/do more/do less/continue/change)
    the text suggests about that specific topic.

    Requires classification results to identify complete stems.
    """

    @property
    def name(self) -> str:
        return "stem_recommendations"

    @property
    def schema(self) -> Type[BaseModel]:
        return StemRecommendationsOutput

    def get_stem_prompt_fn(self):
        return stem_recommendations_prompt

    def _extract_stem_result(self, result_dict: Dict[str, Any]) -> Any:
        """Extract and map recommendation types to output format."""
        if result_dict and result_dict.get("has_recommendations"):
            mapped_recommendations = []
            for rec in result_dict.get("recommendations", []):
                mapped_rec = {
                    "recommendation_type": STEM_RECOMMENDATION_TYPE_MAPPING.get(
                        rec["recommendation_type"], rec["recommendation_type"]
                    ),
                    "excerpt": rec["excerpt"],
                    "reasoning": rec.get("reasoning", ""),
                    "paraphrased_recommendation": rec.get("paraphrased_recommendation", ""),
                }
                mapped_recommendations.append(mapped_rec)

            return {
                "has_recommendations": True,
                "recommendations": mapped_recommendations,
            }
        else:
            return {
                "has_recommendations": False,
                "recommendations": [],
            }
