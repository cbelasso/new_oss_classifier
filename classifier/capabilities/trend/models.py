from typing import List, Literal

from pydantic import BaseModel


class TrendSpan(BaseModel):
    """
    Individual trend detected in text.

    Attributes:
        excerpt: The exact text span from the comment containing the trend
        reasoning: One sentence explaining why this qualifies as a trend
        subject: What aspect is changing (e.g., "manager support", "training quality")
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


class TrendsOutput(BaseModel):
    """
    Output for global trend detection capability.

    Detects temporal change patterns across the entire text.
    Can contain multiple trends if the comment discusses multiple aspects changing over time.
    """

    has_trends: bool
    trends: List[TrendSpan] = []
