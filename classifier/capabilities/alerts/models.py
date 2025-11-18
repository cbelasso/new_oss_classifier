from typing import List, Literal

from pydantic import BaseModel


class AlertSpan(BaseModel):
    excerpt: str
    reasoning: str = ""
    alert_type: Literal[
        "discrimination",
        "sexual_harassment",
        "workplace_violence",
        "safety_concern",
        "ethical_violation",
        "hostile_environment",
        "retaliation",
        "bullying",
        "substance_abuse",
        "mental_health_crisis",
        "data_breach",
        "fraud",
        "other_serious_concern",
    ]
    severity: Literal["low", "moderate", "high", "critical"] = "moderate"


class AlertsOutput(BaseModel):
    has_alerts: bool
    alerts: List[AlertSpan] = []
