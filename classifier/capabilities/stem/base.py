"""
Base class for stem-level analysis capabilities.

Stem capabilities analyze complete classification paths (stems) rather than
individual nodes or full text. They share common patterns for:
- Extracting node definitions along a stem path
- Preparing batches with (text, stem, definitions) tuples
- Reorganizing results by text and stem
"""

from abc import abstractmethod
import json
from typing import Any, Callable, Dict, List

from ..base import Capability


class StemCapability(Capability):
    """
    Base class for capabilities that analyze classification stems.

    A stem is a complete classification path (e.g., "A>B>C"). These capabilities
    analyze text in the context of specific classification paths, often using
    the node definitions along the path for richer context.

    Common workflow:
    1. Receive classification results with complete stems
    2. For each (text, stem) pair, extract node definitions
    3. Generate stem-specific prompts with definitions
    4. Post-process to reorganize results by text -> stem
    """

    def __init__(self, max_stem_definitions: int = None):
        """
        Initialize stem capability.

        Args:
            max_stem_definitions: Maximum number of node definitions to include
                                 from the end of the stem (None = all nodes)
        """
        self.max_stem_definitions = max_stem_definitions
        self._text_stem_mapping: List[tuple[str, str]] = []

    @property
    def dependencies(self) -> List[str]:
        """All stem capabilities depend on classification."""
        return ["classification"]

    def requires_hierarchy(self) -> bool:
        """All stem capabilities need hierarchy access."""
        return True

    @abstractmethod
    def get_stem_prompt_fn(self) -> Callable:
        """
        Return the prompt function for this stem capability.

        The prompt function should have signature:
            (text: str, stem_path: str, stem_definitions: List[Dict]) -> str

        Returns:
            Prompt generation function
        """
        pass

    def extract_stem_definitions(
        self, hierarchy: Dict[str, Any], stem_path: str, separator: str = ">"
    ) -> List[Dict[str, str]]:
        """
        Extract definitions for each node in a stem path.

        Traverses the hierarchy following the stem path and collects
        definition information (name, definition, description, keywords)
        for each node along the way.

        Args:
            hierarchy: Topic hierarchy
            stem_path: Path to extract (e.g., "A>B>C")
            separator: Path separator

        Returns:
            List of node definition dicts, optionally limited by max_stem_definitions
        """
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

    def get_complete_stems(self, text: str, context: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Extract complete stems for a text from context.

        Override this method if your capability needs different stem extraction logic
        (e.g., SubStemPolarity extracts ALL sub-stems, not just complete ones).

        Args:
            text: The text being analyzed
            context: Context dict with classification results

        Returns:
            List of complete stem paths
        """
        text_context = context.get(text, {})
        return text_context.get("complete_stems", [])

    def create_prompt(self, text: str, context: Dict[str, Any] = None) -> str:
        """
        Not used for stem capabilities - use prepare_batch instead.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} requires batch preparation with context"
        )

    def prepare_batch(
        self, texts: List[str], context: Dict[str, Dict[str, Any]] = None
    ) -> List[str]:
        """
        Prepare batch with encoded (text, stem, definitions) tuples.

        This method:
        1. Extracts complete stems for each text
        2. Gets node definitions for each stem
        3. Encodes (text, stem, definitions) as JSON
        4. Generates prompts using the stem prompt function

        Args:
            texts: List of texts to process
            context: Context with classification results and hierarchy

        Returns:
            List of prompts ready for LLM processing
        """
        if context is None:
            raise ValueError(f"{self.__class__.__name__} requires classification context")

        hierarchy = context.get("_hierarchy")
        if hierarchy is None:
            raise ValueError(f"{self.__class__.__name__} requires hierarchy in context")

        # Build list of (text, stem) pairs with definitions
        encoded_pairs = []
        self._text_stem_mapping = []

        prompt_fn = self.get_stem_prompt_fn()

        for text in texts:
            complete_stems = self.get_complete_stems(text, context)

            for stem in complete_stems:
                stem_definitions = self.extract_stem_definitions(hierarchy, stem, separator=">")

                # Encode as JSON to preserve all info
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
            prompt = prompt_fn(data["text"], data["stem"], data["definitions"])
            prompts.append(prompt)

        return prompts

    def post_process(
        self, results: Dict[str, Any], context: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Reorganize flat results into text -> stem structure.

        Override this method if you need custom result organization
        (e.g., StemRecommendations maps recommendation types).

        Args:
            results: Flat dict keyed by encoded prompts
            context: Optional context from dependent capabilities

        Returns:
            Nested dict: text -> stem -> result
        """
        stem_results_dict = {}

        # Group results by text
        for encoded_key, result in results.items():
            idx = list(results.keys()).index(encoded_key)
            if idx < len(self._text_stem_mapping):
                text, stem = self._text_stem_mapping[idx]

                if text not in stem_results_dict:
                    stem_results_dict[text] = {}

                # Convert result to dict
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                elif hasattr(result, "dict"):
                    result_dict = result.dict()
                else:
                    result_dict = result

                stem_results_dict[text][stem] = self._extract_stem_result(result_dict)

        return stem_results_dict

    def _extract_stem_result(self, result_dict: Dict[str, Any]) -> Any:
        """
        Extract the relevant result from the full result dict.

        Override this to customize what gets stored for each stem.
        Default: returns the full result dict.

        Args:
            result_dict: Parsed result dictionary

        Returns:
            The extracted result to store
        """
        return result_dict

    def format_for_export(self, result: Any) -> Any:
        """
        Format stem results for export.

        Default: return as-is (already a dict of stem -> results).
        Override if you need custom export formatting.
        """
        if result is None:
            return {}
        return result
