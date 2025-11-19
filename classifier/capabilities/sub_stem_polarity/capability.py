"""
Sub-stem polarity analysis capability.

Analyzes sentiment/polarity for ALL sub-stems in classification paths,
not just complete stems (leaves).
"""

import json
from typing import Any, Dict, List, Type

from pydantic import BaseModel

from ..base import Capability
from ..stem_polarity.models import StemPolarityOutput
from ..stem_polarity.prompts import stem_polarity_prompt


class SubStemPolarityCapability(Capability):
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

    def __init__(self, max_stem_definitions: int = None):
        """
        Initialize sub-stem polarity capability.

        Args:
            max_stem_definitions: Maximum number of node definitions to include
                                 from the end of the stem (None = all)
        """
        self.max_stem_definitions = max_stem_definitions

    @property
    def name(self) -> str:
        return "sub_stem_polarity"

    @property
    def schema(self) -> Type[BaseModel]:
        return StemPolarityOutput

    @property
    def dependencies(self) -> List[str]:
        return ["classification"]

    def requires_hierarchy(self) -> bool:
        return True

    def extract_stem_definitions(
        self, hierarchy: Dict[str, Any], stem_path: str, separator: str = ">"
    ) -> List[Dict[str, str]]:
        """Extract definitions for each node in a stem path."""
        path_parts = stem_path.split(separator)
        definitions = []

        # Start from hierarchy root
        if isinstance(hierarchy, dict) and "children" in hierarchy:
            current_nodes = hierarchy["children"]
        elif isinstance(hierarchy, list):
            current_nodes = hierarchy
        else:
            return definitions

        # Traverse the path
        for part in path_parts:
            # Find the matching node
            found = None
            for node in current_nodes:
                if node.get("name") == part:
                    found = node
                    break

            if not found:
                break

            # Extract definition info
            node_info = {
                "name": found.get("name", ""),
                "definition": found.get("definition", ""),
                "description": found.get("description", ""),
                "keywords": found.get("keywords", []),
            }
            definitions.append(node_info)

            # Move to children for next level
            current_nodes = found.get("children", [])

        # Limit to last N definitions if specified
        if self.max_stem_definitions is not None and self.max_stem_definitions > 0:
            definitions = definitions[-self.max_stem_definitions :]

        return definitions

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

    def create_prompt(self, text: str, context: Dict[str, Any] = None) -> str:
        """
        This shouldn't be called directly - use prepare_batch instead.
        """
        raise NotImplementedError(
            "SubStemPolarityCapability requires batch preparation with context"
        )

    def prepare_batch(
        self, texts: List[str], context: Dict[str, Dict[str, Any]] = None
    ) -> List[str]:
        """
        Prepare batch with encoded (text, stem, definitions) tuples.
        """

        if context is None:
            raise ValueError("SubStemPolarityCapability requires classification context")

        hierarchy = context.get("_hierarchy")
        if hierarchy is None:
            raise ValueError("SubStemPolarityCapability requires hierarchy in context")

        root_prefix = hierarchy.get("name", "ROOT")

        # Build list of (text, stem) pairs with definitions
        encoded_pairs = []
        self._text_stem_mapping = []

        for text in texts:
            text_context = context.get(text, {})

            # Get complete stems from context (these are the leaf paths)
            complete_stems = text_context.get("complete_stems", [])

            # Extract ALL sub-stems (not just complete ones)
            # Need to reconstruct full paths with root prefix first
            full_paths = [f"{root_prefix}>{stem}" for stem in complete_stems]
            all_sub_stems = self.extract_all_sub_stems(full_paths, root_prefix, separator=">")

            for stem in all_sub_stems:
                stem_definitions = self.extract_stem_definitions(hierarchy, stem, separator=">")

                # Encode as JSON
                encoded = json.dumps(
                    {
                        "text": text,
                        "stem": stem,
                        "definitions": stem_definitions,
                    },
                    ensure_ascii=False,
                )

                encoded_pairs.append(encoded)
                self._text_stem_mapping.append((text, stem))

        # Create prompts from encoded pairs
        prompts = []
        for encoded in encoded_pairs:
            data = json.loads(encoded)
            prompt = stem_polarity_prompt(data["text"], data["stem"], data["definitions"])
            prompts.append(prompt)

        return prompts

    def post_process(
        self, results: Dict[str, Any], context: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Reorganize results by text and stem.
        """
        stem_polarity_dict = {}

        # Group results by text
        for encoded_key, result in results.items():
            idx = list(results.keys()).index(encoded_key)
            if idx < len(self._text_stem_mapping):
                text, stem = self._text_stem_mapping[idx]

                if text not in stem_polarity_dict:
                    stem_polarity_dict[text] = {}

                # Convert result to dict
                if hasattr(result, "model_dump"):
                    stem_dict = result.model_dump()
                elif hasattr(result, "dict"):
                    stem_dict = result.dict()
                else:
                    stem_dict = result

                # Store polarity result
                if stem_dict and stem_dict.get("has_polarity"):
                    stem_polarity_dict[text][stem] = stem_dict.get("polarity_result", {})
                else:
                    stem_polarity_dict[text][stem] = None

        return stem_polarity_dict

    def get_result_key(self) -> str:
        """Store under 'sub_stem_polarity' key."""
        return "sub_stem_polarity"
