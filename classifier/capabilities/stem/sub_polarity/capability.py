"""
Sub-stem polarity analysis capability.

Analyzes sentiment/polarity for ALL sub-stems in classification paths,
not just complete stems (leaves).
"""

from typing import Any, Dict, List, Type

from pydantic import BaseModel

from ..base import StemCapability
from ..polarity.models import StemPolarityOutput
from ..polarity.prompts import stem_polarity_prompt


class SubStemPolarityCapability(StemCapability):
    """
    Evaluates polarity (sentiment) for all sub-stems in classification paths.

    Unlike StemPolarityCapability which only analyzes complete stems (leaves),
    this capability analyzes every intermediate path level.

    For example, if a text is classified as "A>B>C", this will evaluate:
    - "A"
    - "A>B"
    - "A>B>C"

    Requires classification results to identify paths.
    """

    @property
    def name(self) -> str:
        return "sub_stem_polarity"

    @property
    def schema(self) -> Type[BaseModel]:
        return StemPolarityOutput

    def get_stem_prompt_fn(self):
        return stem_polarity_prompt

    def get_complete_stems(self, text: str, context: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Extract ALL sub-stems (not just complete stems) for a text.

        Overrides parent method to return all intermediate paths.
        """
        text_context = context.get(text, {})
        complete_stems = text_context.get("complete_stems", [])

        # Get hierarchy to extract root prefix
        hierarchy = context.get("_hierarchy")
        root_prefix = hierarchy.get("name", "ROOT") if hierarchy else "ROOT"

        # Reconstruct full paths with root prefix
        full_paths = [f"{root_prefix}>{stem}" for stem in complete_stems]

        # Extract all sub-stems
        return self.extract_all_sub_stems(full_paths, root_prefix, separator=">")

    def extract_all_sub_stems(
        self, classification_paths: List[str], root_prefix: str, separator: str = ">"
    ) -> List[str]:
        """
        Extract all sub-stems from classification paths.

        For path "ROOT>A>B>C", extracts: ["A", "A>B", "A>B>C"]
        """
        all_sub_stems = set()

        for path in classification_paths:
            # Remove root prefix
            if path.startswith(root_prefix + separator):
                path_without_root = path[len(root_prefix) + 1 :]
            else:
                path_without_root = path

            # Split into parts
            parts = path_without_root.split(separator)

            # Generate all sub-stems
            for i in range(1, len(parts) + 1):
                sub_stem = separator.join(parts[:i])
                all_sub_stems.add(sub_stem)

        return list(all_sub_stems)

    def _extract_stem_result(self, result_dict: Dict[str, Any]) -> Any:
        """Extract just the polarity_result, or None if no polarity."""
        if result_dict and result_dict.get("has_polarity"):
            return result_dict.get("polarity_result", {})
        return None

    def get_result_key(self) -> str:
        """Store under 'sub_stem_polarity' key."""
        return "sub_stem_polarity"
