"""
Hierarchical Text Classification Package
"""

from .capabilities import (
    CapabilityOrchestrator,
    CapabilityRegistry,
    create_default_registry,
)
from .capabilities.classification import (  # ← UPDATED
    BatchClassificationResult,
    BFSClassificationCapability,
    ClassificationCapability,
    ClassificationOutput,
    SingleClassificationResult,
)
from .hierarchy import (
    NodeConfig,
    build_tree_from_paths,
    format_tree_as_string,
    get_all_leaf_paths,
    get_node_path,
    load_topic_hierarchy,
    validate_hierarchy,
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
from .server_processor import ServerClassificationProcessor

__version__ = "1.0.0"

__all__ = [
    # Models
    "SingleClassificationResult",
    "ClassificationOutput",
    "NodeConfig",
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
    # Hierarchy
    "load_topic_hierarchy",
    "get_node_path",
    "get_all_leaf_paths",
    "build_tree_from_paths",
    "format_tree_as_string",
    "validate_hierarchy",
    # Classification
    "ClassificationCapability",
    "BFSClassificationCapability",
    # Capabilities
    "CapabilityOrchestrator",
    "CapabilityRegistry",
    "create_default_registry",
    # Processor
    "ServerClassificationProcessor",
]
