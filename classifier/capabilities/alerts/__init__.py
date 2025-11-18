"""
Alert detection capability.

Detects serious workplace concerns that require immediate attention.
"""

from .capability import AlertsCapability
from .models import AlertsOutput, AlertSpan
from .prompts import alert_detection_prompt

__all__ = [
    "AlertsCapability",
    "AlertsOutput",
    "AlertSpan",
    "alert_detection_prompt",
]
