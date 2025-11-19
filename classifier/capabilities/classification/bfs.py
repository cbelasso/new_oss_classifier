"""
Breadth-first search (BFS) hierarchical classification.

One prompt per node, expanding branches only when relevance is established.
"""

from typing import Any, Callable, Dict, List, Tuple, Type

from pydantic import BaseModel

from ...policies import AcceptancePolicy, DefaultPolicy
from .base import ClassificationCapability
from .models import ClassificationOutput, SingleClassificationResult
from .prompts import standard_classification_prompt


def add_text_to_prompt(prompt: str, text: str) -> str:
    """Append text to prompt template."""
    return f"{prompt}\n\nText:\n{text}"


class BFSClassificationCapability(ClassificationCapability):
    """
    Breadth-first hierarchical classification.

    Explores hierarchy level-by-level:
    1. Start with root's children
    2. For nodes that pass policy, expand their children
    3. Continue until no more nodes to explore

    Efficient: prunes irrelevant branches early.
    """

    def __init__(
        self,
        prompt_fn: Callable[[Dict[str, Any]], str] = None,
        policy: AcceptancePolicy = None,
        separator: str = ">",
    ):
        """
        Initialize BFS classification.

        Args:
            prompt_fn: Function to generate prompts from node configs
            policy: Acceptance policy for filtering results
            separator: Path separator string
        """
        self.prompt_fn = prompt_fn or standard_classification_prompt
        self.policy = policy or DefaultPolicy()
        self.separator = separator

    @property
    def name(self) -> str:
        return "classification"

    @property
    def schema(self) -> Type[BaseModel]:
        return SingleClassificationResult

    def execute_classification(
        self,
        texts: List[str],
        hierarchy: Dict[str, Any],
        processor: Any,
    ) -> Dict[str, ClassificationOutput]:
        """
        Execute BFS hierarchical classification.

        Returns:
            Dict mapping text -> ClassificationOutput with paths and node_results
        """
        root_name = hierarchy.get("name", "[ROOT]")

        # Initialize queue: for each text, start with root's children
        text_queue: Dict[str, List[Tuple[Dict[str, Any], str]]] = {
            text: [(child, root_name) for child in hierarchy.get("children", [])]
            for text in texts
        }

        # Initialize results with ClassificationOutput objects
        final_results: Dict[str, ClassificationOutput] = {
            text: ClassificationOutput(text=text, classification_paths=[], node_results={})
            for text in texts
        }

        # Process level by level
        while any(text_queue.values()):
            # Collect all (text, node, parent_path) tuples for this round
            batch_prompts = []
            text_node_map = []

            for text, nodes in text_queue.items():
                for node, parent_path in nodes:
                    prompt = add_text_to_prompt(self.prompt_fn(node), text)
                    batch_prompts.append(prompt)
                    text_node_map.append((text, node, parent_path))

            # Batch generate for all prompts
            results = self._batch_generate(processor, batch_prompts)

            # Process results and build next queue
            next_text_queue: Dict[str, List[Tuple[Dict[str, Any], str]]] = {
                text: [] for text in texts
            }

            for (text, node, parent_path), result in zip(text_node_map, results):
                # Store the node result
                node_name = node["name"]
                final_results[text].node_results[node_name] = result

                if self.policy.accept(result, node):
                    # Build current path
                    current_path = f"{parent_path}{self.separator}{node['name']}"
                    final_results[text].classification_paths.append(current_path)

                    # Queue children for next round
                    for child in node.get("children", []):
                        next_text_queue[text].append((child, current_path))

            text_queue = next_text_queue

        return final_results

    def _batch_generate(
        self, processor, prompts: List[str]
    ) -> List[SingleClassificationResult]:
        """Execute batch LLM generation and parse results."""
        processor.process_with_schema(prompts=prompts, schema=SingleClassificationResult)
        return processor.parse_results_with_schema(schema=SingleClassificationResult)
