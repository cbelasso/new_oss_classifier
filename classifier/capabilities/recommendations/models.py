from typing import List, Literal

from pydantic import BaseModel


class RecommendationSpan(BaseModel):
    excerpt: str
    reasoning: str = ""
    paraphrased_recommendation: str = ""
    qualifier: Literal[
        "add_or_increase",
        "reduce_or_remove",
        "introduce_or_start",
        "eliminate_or_stop",
        "modify_or_improve",
        "maintain_or_continue",
        "unspecified_or_general",
    ] = "unspecified_or_general"


class RecommendationsOutput(BaseModel):
    has_recommendations: bool
    recommendations: List[RecommendationSpan] = []
