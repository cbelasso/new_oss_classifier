from typing import List, Literal

from pydantic import BaseModel


class StemTrendSpan(BaseModel):
    """
    Individual trend detected for a specific classification stem.

    Attributes:
        excerpt: The exact text span from the comment containing the trend
        reasoning: One sentence explaining why this qualifies as a trend for this topic path
        subject: What aspect of the topic is changing (aligned with the topic path)
        direction: Type of change (increasing, decreasing, improving, deteriorating, stable, fluctuating)
        valence: Whether the trend is viewed positively or negatively
        confidence: Confidence score from 1-5 (1=very uncertain, 5=very certain)
    """

    excerpt: str
    reasoning: str = ""
    subject: str = ""
    direction: Literal[
        "increasing",
        "decreasing",
        "improving",
        "deteriorating",
        "stable_positive",
        "stable_negative",
        "fluctuating",
    ]
    valence: Literal["positive", "negative", "neutral", "mixed"] = "neutral"
    confidence: int


class StemTrendsOutput(BaseModel):
    """
    Output for stem-specific trend analysis capability.

    Analyzes temporal change patterns for a specific classification stem.
    Can contain multiple trends if the comment discusses multiple aspects of the topic changing over time.
    """

    has_trends: bool
    trends: List[StemTrendSpan] = []
