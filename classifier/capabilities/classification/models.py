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


class NodeConfig(BaseModel):
    """
    Configuration for a single node in the topic hierarchy.

    Attributes:
        name: Node identifier
        description: Detailed description of the topic
        keywords: List of relevant keywords
        scope: Scope definition for the topic
        children: Child nodes in the hierarchy
    """

    name: str
    description: str = "[No Description]"
    keywords: List[str] = []
    scope: str = "[None]"
    children: List["NodeConfig"] = []

    class Config:
        # Allow arbitrary types for flexibility
        arbitrary_types_allowed = True


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
