"""
VLLM server infrastructure for hierarchical text classification.

This module contains all server-related components:
- VLLM server management and lifecycle
- Client for communicating with VLLM servers
- Classification processor that uses the server
"""

from .processor import ServerClassificationProcessor
from .vllm_client import VLLMServerClient
from .vllm_server import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MODEL_NAME,
    DEFAULT_TENSOR_PARALLEL_SIZE,
    DEFAULT_VLLM_HOST,
    DEFAULT_VLLM_PORT,
    ServerConfig,
    VLLMServer,
    run_batch_inference,
    run_inference_from_list,
)

__all__ = [
    # Client
    "VLLMServerClient",
    # Processor
    "ServerClassificationProcessor",
    # Server management
    "VLLMServer",
    "ServerConfig",
    # Inference functions
    "run_batch_inference",
    "run_inference_from_list",
    # Constants
    "DEFAULT_MODEL_NAME",
    "DEFAULT_VLLM_HOST",
    "DEFAULT_VLLM_PORT",
    "DEFAULT_GPU_MEMORY_UTILIZATION",
    "DEFAULT_TENSOR_PARALLEL_SIZE",
    "DEFAULT_MAX_MODEL_LEN",
    "DEFAULT_MAX_CONCURRENT",
]
