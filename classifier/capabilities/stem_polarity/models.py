from typing import Literal, Optional

from pydantic import BaseModel


class StemPolarityResult(BaseModel):
    polarity: Literal["Positive", "Negative", "Neutral", "Mixed"]
    confidence: int  # 1-5
    reasoning: str = ""
    excerpt: str = ""


class StemPolarityOutput(BaseModel):
    has_polarity: bool
    polarity_result: Optional[StemPolarityResult] = None
