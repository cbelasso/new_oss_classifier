"""
Classification Processor for VLLM Server

Simplified processor specifically designed for VLLM server deployment.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Type

from pydantic import BaseModel

from .capabilities.classification import (
    BFSClassificationCapability,
    ClassificationOutput,
    standard_classification_prompt,
)
from .hierarchy import load_topic_hierarchy
from .policies import AcceptancePolicy, DefaultPolicy
from .vllm_client import VLLMServerClient


class ServerClassificationProcessor:
    """
    High-level classification processor using VLLM server.

    Simplified version specifically for server-based inference.
    """

    def __init__(
        self,
        config_path: str | Path,
        server_url: str = "http://localhost:9001/v1",
        model_name: str = "openai/gpt-oss-120b",
        max_concurrent: int = 5,
        prompt_fn: Callable[[Dict[str, Any]], str] = None,
        policy: AcceptancePolicy = None,
        separator: str = ">",
    ):
        """
        Initialize server-based classification processor.

        Args:
            config_path: Path to topic hierarchy JSON
            server_url: VLLM server URL
            model_name: Model name
            max_concurrent: Max concurrent requests
            prompt_fn: Prompt generation function
            policy: Acceptance policy
            separator: Path separator
        """
        self.config_path = config_path
        self.separator = separator
        self.server_url = server_url
        self.model_name = model_name

        # Load hierarchy
        self.topic_hierarchy = load_topic_hierarchy(config_path)
        if not self.topic_hierarchy:
            raise ValueError(f"Failed to load hierarchy from {config_path}")

        # Initialize VLLM server client
        self.llm_processor = VLLMServerClient(
            server_url=server_url,
            model_name=model_name,
            max_concurrent=max_concurrent,
        )

        self.classification_capability = BFSClassificationCapability(
            prompt_fn=prompt_fn or standard_classification_prompt,
            policy=policy or DefaultPolicy(),
            separator=separator,
        )

    def classify_hierarchical(self, texts: List[str]) -> Dict[str, List[str]]:
        """Classify texts using node-by-node hierarchical traversal."""
        results = self.classification_capability.execute_classification(
            texts=texts,
            hierarchy=self.topic_hierarchy,
            processor=self.llm_processor,
        )

        # Extract just the paths (ClassificationOutput -> List[str])
        return {text: output.classification_paths for text, output in results.items()}

    def classify_target_path(self, texts: List[str], target_path: List[str]) -> Dict[str, Any]:
        """Evaluate texts against a specific target path (leaf node only)."""
        return self.classifier.classify_target_path(
            texts=texts, topic_hierarchy=self.topic_hierarchy, target_path=target_path
        )

    def classify_full_path(
        self, texts: List[str], target_path: List[str]
    ) -> Dict[str, ClassificationOutput]:
        """Evaluate all nodes along a specific path for each text."""
        return self.classifier.classify_full_path(
            texts=texts, topic_hierarchy=self.topic_hierarchy, target_path=target_path
        )

    def classify_with_custom_schema(
        self, texts: List[str], prompt_fn: Callable[[str], str], schema: Type[BaseModel]
    ) -> Dict[str, BaseModel]:
        """
        Classify texts using a custom prompt function and Pydantic schema.

        Args:
            texts: List of text strings to classify
            prompt_fn: Function that takes a text string and returns a prompt
            schema: Pydantic model class defining the expected output structure

        Returns:
            Dictionary mapping each text to its classification result
        """
        # Generate prompts for all texts
        prompts = [prompt_fn(text) for text in texts]

        # Process with the server using the custom schema
        self.llm_processor.process_with_schema(prompts=prompts, schema=schema)

        # Parse results with the schema
        parsed_results = self.llm_processor.parse_results_with_schema(schema=schema)

        # Map results back to texts
        return {text: result for text, result in zip(texts, parsed_results)}

    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str | Path,
        mode: str = None,
    ) -> None:
        """
        Export classification results to a JSON file.

        Auto-detects result type if mode not provided.
        """
        output_path = Path(output_path)
        output_data = []
        root_name = self.topic_hierarchy.get("name", "[ROOT]")

        # Auto-detect mode if not explicitly provided
        if mode is None:
            if all(isinstance(v, list) for v in results.values()):
                mode = "hierarchical"
            elif all(hasattr(v, "dict") for v in results.values()):
                mode = "full_path"
            elif all(isinstance(v, dict) for v in results.values()):
                first_val = next(iter(results.values()))
                if all(
                    isinstance(subv, (dict, list, ClassificationOutput))
                    for subv in first_val.values()
                ):
                    mode = "multi_target"
                else:
                    mode = "target_path"
            else:
                raise ValueError("Unable to detect results mode automatically.")

        # Serialize based on detected mode
        if mode == "hierarchical":
            for text, paths in results.items():
                full_paths = [
                    p if p.startswith(root_name) else f"{root_name}{self.separator}{p}"
                    for p in paths
                ]
                output_data.append(
                    {
                        "text": text,
                        "classification_paths": full_paths,
                    }
                )

        elif mode == "target_path":
            for text, result in results.items():
                output_data.append(
                    {
                        "text": text,
                        "classification_result": result.dict()
                        if hasattr(result, "dict")
                        else result,
                    }
                )

        elif mode == "full_path":
            for text, classification_output in results.items():
                output_data.append(
                    {
                        "text": text,
                        "classification_results": {
                            name: res.dict()
                            for name, res in classification_output.node_results.items()
                        },
                        "classification_paths": classification_output.classification_paths,
                    }
                )

        elif mode == "multi_target":
            for target_key, sub_results in results.items():
                for text, classification in sub_results.items():
                    entry = {"text": text, "target_path_key": target_key}
                    if hasattr(classification, "dict"):
                        entry["classification_result"] = classification.dict()
                    elif isinstance(classification, dict):
                        entry["classification_result"] = classification
                    elif isinstance(classification, list):
                        entry["classification_paths"] = classification
                    else:
                        entry["classification_result"] = str(classification)
                    output_data.append(entry)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Write JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.llm_processor:
            self.llm_processor.terminate()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.cleanup()
