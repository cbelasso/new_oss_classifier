"""
Hierarchical Text Classification Package

A modular, extensible system for classifying text according to hierarchical
topic structures using LLMs.

Main components:
- models: Data models for classification results
- policies: Acceptance policies for filtering results
- prompts: Customizable prompt templates
- hierarchy: Utilities for working with topic hierarchies
- classifier: Core classification algorithms
- server_processor: High-level processing interface for VLLM server
- cli: Command-line interface

Example usage:
    >>> from classifier import ServerClassificationProcessor
    >>> from classifier.policies import ConfidenceThresholdPolicy
    >>>
    >>> with ServerClassificationProcessor(
    ...     config_path="topics.json",
    ...     server_url="http://localhost:9001/v1",
    ...     policy=ConfidenceThresholdPolicy(min_confidence=4)
    ... ) as processor:
    ...     results = processor.classify_hierarchical(texts)
    ...     processor.export_results(results, "output.json")
"""

from .classifier import HierarchicalClassifier
from .hierarchy import (
    build_tree_from_paths,
    format_tree_as_string,
    get_all_leaf_paths,
    get_node_path,
    load_topic_hierarchy,
    validate_hierarchy,
)
from .models import (
    BatchClassificationResult,
    ClassificationOutput,
    NodeConfig,
    SingleClassificationResult,
)
from .policies import (
    AcceptancePolicy,
    AnyPolicy,
    CompositePolicy,
    ConfidenceThresholdPolicy,
    DefaultPolicy,
    ExcerptRequiredPolicy,
    KeywordInReasoningPolicy,
    MinimumReasoningLengthPolicy,
)
from .prompts import (
    add_text_to_prompt,
    hierarchical_path_prompt,
    keyword_focused_prompt,
    sentiment_aware_classification_prompt,
    standard_classification_prompt,
)
from .server_processor import ServerClassificationProcessor

__version__ = "1.0.0"

__all__ = [
    # Models
    "SingleClassificationResult",
    "NodeConfig",
    "ClassificationOutput",
    "BatchClassificationResult",
    # Policies
    "AcceptancePolicy",
    "DefaultPolicy",
    "ConfidenceThresholdPolicy",
    "KeywordInReasoningPolicy",
    "ExcerptRequiredPolicy",
    "MinimumReasoningLengthPolicy",
    "CompositePolicy",
    "AnyPolicy",
    # Prompts
    "standard_classification_prompt",
    "hierarchical_path_prompt",
    "sentiment_aware_classification_prompt",
    "keyword_focused_prompt",
    "add_text_to_prompt",
    # Hierarchy
    "load_topic_hierarchy",
    "get_node_path",
    "get_all_leaf_paths",
    "build_tree_from_paths",
    "format_tree_as_string",
    "validate_hierarchy",
    # Core
    "HierarchicalClassifier",
    "ServerClassificationProcessor",
]
