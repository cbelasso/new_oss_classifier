"""
Bundled hierarchical classification.

Groups sibling nodes into bundles for richer context and comparative evaluation.
"""

from typing import Any, Dict, List, Tuple, Type

from pydantic import BaseModel

from ...core import AcceptancePolicy, DefaultPolicy
from .base import ClassificationCapability
from .models import BundledClassificationResult, ClassificationOutput
from .prompts import bundled_classification_prompt


def add_text_to_prompt(prompt: str, text: str) -> str:
    """Append text to prompt template."""
    return f"{prompt}\n\nText:\n{text}"


class BundledClassificationCapability(ClassificationCapability):
    """
    Bundled hierarchical classification.

    Groups sibling nodes into bundles (e.g., 3-5 nodes per prompt) for:
    - Richer context: LLM sees relationships between adjacent topics
    - Better comparative evaluation: Can distinguish similar topics
    - Fewer API calls: Multiple nodes per prompt
    - Potentially higher accuracy: More informed decisions

    Explores hierarchy level-by-level like BFS, but bundles siblings together.
    """

    def __init__(
        self,
        bundle_size: int = 4,
        policy: AcceptancePolicy = None,
        separator: str = ">",
    ):
        """
        Initialize bundled classification.

        Args:
            bundle_size: Number of sibling nodes to group per prompt (default: 4)
            policy: Acceptance policy for filtering results
            separator: Path separator string
        """
        self.bundle_size = bundle_size
        self.policy = policy or DefaultPolicy()
        self.separator = separator

    @property
    def name(self) -> str:
        return "classification"

    @property
    def schema(self) -> Type[BaseModel]:
        return BundledClassificationResult

    def execute_classification(
        self,
        texts: List[str],
        hierarchy: Dict[str, Any],
        processor: Any,
    ) -> Dict[str, ClassificationOutput]:
        """
        Execute bundled hierarchical classification.

        Returns:
            Dict mapping text -> ClassificationOutput with paths and node_results
        """
        root_name = hierarchy.get("name", "[ROOT]")

        # Initialize queue: for each text, create bundles from root's children
        text_queue: Dict[str, List[Tuple[List[Dict[str, Any]], str]]] = {
            text: self._create_bundles(hierarchy.get("children", []), root_name)
            for text in texts
        }

        # Initialize results with ClassificationOutput objects
        final_results: Dict[str, ClassificationOutput] = {
            text: ClassificationOutput(text=text, classification_paths=[], node_results={})
            for text in texts
        }

        # Process level by level
        while any(text_queue.values()):
            # Collect all (text, bundle, parent_path) tuples for this round
            batch_prompts = []
            text_bundle_map = []

            for text, bundles in text_queue.items():
                for bundle, parent_path in bundles:
                    prompt = add_text_to_prompt(bundled_classification_prompt(bundle), text)
                    batch_prompts.append(prompt)
                    text_bundle_map.append((text, bundle, parent_path))

            # Batch generate for all prompts
            results = self._batch_generate(processor, batch_prompts)

            # Process results and build next queue
            next_text_queue: Dict[str, List[Tuple[List[Dict[str, Any]], str]]] = {
                text: [] for text in texts
            }

            for (text, bundle, parent_path), result in zip(text_bundle_map, results):
                # Extract per-node results from the bundled result
                for node in bundle:
                    node_name = node["name"]
                    node_result = result.node_results.get(node_name)

                    if node_result:
                        # Store the node result
                        final_results[text].node_results[node_name] = node_result

                        # Check if this node passes the policy
                        if self.policy.accept(node_result, node):
                            # Build current path
                            current_path = f"{parent_path}{self.separator}{node_name}"
                            final_results[text].classification_paths.append(current_path)

                            # Queue children for next round
                            children = node.get("children", [])
                            if children:
                                child_bundles = self._create_bundles(children, current_path)
                                next_text_queue[text].extend(child_bundles)

            text_queue = next_text_queue

        return final_results

    def _create_bundles(
        self, nodes: List[Dict[str, Any]], parent_path: str
    ) -> List[Tuple[List[Dict[str, Any]], str]]:
        """
        Group sibling nodes into bundles of bundle_size.

        Args:
            nodes: List of sibling nodes
            parent_path: Path to parent node

        Returns:
            List of (bundle, parent_path) tuples
        """
        bundles = []
        for i in range(0, len(nodes), self.bundle_size):
            bundle = nodes[i : i + self.bundle_size]
            bundles.append((bundle, parent_path))
        return bundles

    def _batch_generate(
        self, processor, prompts: List[str]
    ) -> List[BundledClassificationResult]:
        """Execute batch LLM generation and parse results."""
        processor.process_with_schema(prompts=prompts, schema=BundledClassificationResult)
        return processor.parse_results_with_schema(schema=BundledClassificationResult)
