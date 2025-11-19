"""
Base class for hierarchical classification strategies.

All classification capabilities work with hierarchies and produce classification paths,
but they differ in HOW they traverse/prompt/process.
"""

from abc import abstractmethod
from typing import Any, Dict, List

from ..base import Capability
from .models import ClassificationOutput


class ClassificationCapability(Capability):
    """
    Abstract base for hierarchical classification strategies.

    All classification strategies:
    - Work with a topic hierarchy
    - Produce classification paths
    - Provide context for dependent capabilities (stem analysis)

    But differ in traversal/prompting strategy.
    """

    @abstractmethod
    def execute_classification(
        self,
        texts: List[str],
        hierarchy: Dict[str, Any],
        processor: Any,  # LLMProcessor protocol
    ) -> Dict[str, ClassificationOutput]:
        """
        Execute classification strategy.

        Args:
            texts: Texts to classify
            hierarchy: Topic hierarchy
            processor: LLM processor

        Returns:
            Dict mapping text -> ClassificationOutput
        """
        pass

    def requires_hierarchy(self) -> bool:
        """All classification capabilities need hierarchy."""
        return True

    def get_result_key(self) -> str:
        """All classification results stored under same key."""
        return "classification_result"

    # Override these - classification has custom execution
    def create_prompt(self, text: str, context: Dict[str, Any] = None) -> str:
        raise NotImplementedError(
            "ClassificationCapability uses execute_classification() instead"
        )

    def prepare_batch(self, texts: List[str], context: Dict[str, Any] = None) -> List[str]:
        raise NotImplementedError(
            "ClassificationCapability uses execute_classification() instead"
        )
