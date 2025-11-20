"""
VLLM Server Client for Hierarchical Classification

A clean interface to the VLLM server that matches the expected processor interface.
"""

import asyncio
import logging
from typing import Dict, List, Type

from pydantic import BaseModel

# Suppress OpenAI HTTP logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Import from sibling module
from .vllm_server import run_inference_from_list


class VLLMServerClient:
    """
    Client for VLLM server inference.

    Provides the same interface as FlexibleSchemaProcessor but uses the VLLM server.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:9001/v1",
        model_name: str = "openai/gpt-oss-120b",
        max_concurrent: int = 5,
    ):
        """
        Initialize VLLM server client.

        Args:
            server_url: URL of the VLLM server
            model_name: Model name to use
            max_concurrent: Maximum concurrent requests
        """
        self.server_url = server_url
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self._last_results: Dict[str, BaseModel] = {}

    def process_with_schema(
        self,
        prompts: List[str],
        schema: Type[BaseModel],
        batch_size: int = 25,
        formatted: bool = False,
    ) -> None:
        """
        Process prompts with the given schema (stores results internally).

        Args:
            prompts: List of prompt strings
            schema: Pydantic schema class
            batch_size: Batch size (ignored, handled by server)
            formatted: Whether prompts are pre-formatted (ignored)
        """
        # Create new event loop for this call to avoid cleanup issues
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run async inference
            output_data = loop.run_until_complete(
                run_inference_from_list(
                    prompts=prompts,
                    schema_class=schema,
                    model_name=self.model_name,
                    server_url=self.server_url,
                    max_concurrent=self.max_concurrent,
                )
            )

            # Store results keyed by prompt
            self._last_results = {}

            if output_data and "results" in output_data:
                for result_item in output_data["results"]:
                    prompt_index = result_item.get("prompt_index", 0)
                    if prompt_index < len(prompts):
                        prompt = prompts[prompt_index]
                        result_dict = result_item.get("result", {})

                        # Convert dict to Pydantic model
                        try:
                            parsed_result = schema(**result_dict)
                            self._last_results[prompt] = parsed_result
                        except Exception as e:
                            print(
                                f"Warning: Failed to parse result for prompt {prompt_index}: {e}"
                            )
        finally:
            # Clean shutdown of the event loop
            try:
                # Cancel all remaining tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                # Wait for task cancellations
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                loop.close()

    def parse_results_with_schema(
        self, schema: Type[BaseModel], validate: bool = True
    ) -> List[BaseModel]:
        """
        Parse and return stored results.

        Args:
            schema: Pydantic schema class (for compatibility, not used)
            validate: Whether to validate (for compatibility, not used)

        Returns:
            List of parsed Pydantic model instances
        """
        return list(self._last_results.values())

    def terminate(self):
        """Cleanup (no-op for server client)."""
        pass
