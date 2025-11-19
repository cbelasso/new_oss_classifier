from typing import Dict, List

from pydantic import BaseModel


class SingleClassificationResult(BaseModel):
    """
    Result of classifying a single text against a single node.

    Attributes:
        is_relevant: Whether the text is relevant to the node's topic
        confidence: Confidence score from 1-5 (1=very uncertain, 5=very certain)
        reasoning: Explanation of the classification decision (1-2 sentences)
        excerpt: Exact text span supporting the classification (empty if not relevant)
    """

    is_relevant: bool
    confidence: int
    reasoning: str
    excerpt: str


class ClassificationOutput(BaseModel):
    """
    Final output for a single text's classification.

    Attributes:
        text: The original text that was classified
        classification_paths: List of hierarchical paths (e.g., ["Root>A>B", "Root>C"])
        node_results: Optional detailed results per node (for target path evaluation)
    """

    text: str
    classification_paths: List[str]
    node_results: Dict[str, SingleClassificationResult] | None = None


class BatchClassificationResult(BaseModel):
    """
    Results for a batch of texts.

    Attributes:
        results: List of classification outputs, one per input text
    """

    results: List[ClassificationOutput]


class BundledClassificationResult(BaseModel):
    """
    Result of evaluating multiple sibling nodes in a single prompt.

    This allows the LLM to see multiple topics at once for richer context
    and comparative evaluation.

    Attributes:
        node_results: Dictionary mapping node name to its classification result
    """

    node_results: Dict[str, SingleClassificationResult]
