from typing import List, Literal

from pydantic import BaseModel


class StemRecommendationType(BaseModel):
    recommendation_type: Literal["start", "stop", "do_more", "do_less", "continue", "change"]
    excerpt: str
    reasoning: str = ""
    paraphrased_recommendation: str = ""


class StemRecommendationsOutput(BaseModel):
    has_recommendations: bool
    recommendations: List[StemRecommendationType] = []
