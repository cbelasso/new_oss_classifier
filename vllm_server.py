"""
Enhanced VLLM Server for GPT-OSS with Multi-Instance Support

Improvements:
- Support for instance IDs (for running multiple servers)
- Configurable ports and GPU allocation
- Better integration with VLLMServerManager
- Organized log and PID files in hidden directory
- Cleaner code structure and error handling
"""

import asyncio
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel
import requests

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# Suppress OpenAI HTTP logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"
DEFAULT_VLLM_HOST = "0.0.0.0"
DEFAULT_VLLM_PORT = 9001
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_TENSOR_PARALLEL_SIZE = 2
DEFAULT_MAX_MODEL_LEN = 65536
DEFAULT_MAX_TOKENS = 50000
DEFAULT_DTYPE = "bfloat16"
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_STARTUP_TIMEOUT = 180

# Directory for storing server files
VLLM_DATA_DIR = Path.home() / ".vllm_servers"


# ============================================================================
# Server Configuration
# ============================================================================


class ServerConfig:
    """Configuration for a VLLM server instance."""

    def __init__(
        self,
        instance_id: int = 0,
        model_name: str = DEFAULT_MODEL_NAME,
        host: str = DEFAULT_VLLM_HOST,
        port: int = DEFAULT_VLLM_PORT,
        gpu_ids: Optional[List[int]] = None,
        tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        dtype: str = DEFAULT_DTYPE,
    ):
        """Initialize server configuration."""
        self.instance_id = instance_id
        self.model_name = model_name
        self.host = host
        self.port = port
        self.gpu_ids = gpu_ids or [0, 1]
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype

        self._validate()
        self._ensure_data_dir()

    def _validate(self):
        """Validate configuration parameters."""
        if len(self.gpu_ids) != self.tensor_parallel_size:
            raise ValueError(
                f"Number of GPU IDs ({len(self.gpu_ids)}) must match "
                f"tensor_parallel_size ({self.tensor_parallel_size})"
            )

    def _ensure_data_dir(self):
        """Ensure the data directory exists."""
        self.instance_dir.mkdir(parents=True, exist_ok=True)

    @property
    def instance_dir(self) -> Path:
        """Get the directory for this instance's files."""
        return VLLM_DATA_DIR / f"instance_{self.instance_id}"

    @property
    def url(self) -> str:
        """Get the server URL."""
        return f"http://localhost:{self.port}/v1"

    @property
    def cuda_visible_devices(self) -> str:
        """Get CUDA_VISIBLE_DEVICES string."""
        return ",".join(str(gpu) for gpu in self.gpu_ids)

    @property
    def pid_file(self) -> Path:
        """Get PID file path for this instance."""
        return self.instance_dir / "server.pid"

    @property
    def stdout_log(self) -> Path:
        """Get stdout log path for this instance."""
        return self.instance_dir / "stdout.log"

    @property
    def stderr_log(self) -> Path:
        """Get stderr log path for this instance."""
        return self.instance_dir / "stderr.log"


# ============================================================================
# VLLM Server Manager
# ============================================================================


class VLLMServer:
    """Enhanced VLLM server manager with instance support."""

    def __init__(self, config: ServerConfig):
        """Initialize VLLM server."""
        self.config = config
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> Optional[subprocess.Popen]:
        """Start the VLLM server."""
        if self._is_already_running():
            logger.warning(
                f"Server instance {self.config.instance_id} appears to be already running. "
                f"Check status or stop the existing instance first."
            )
            return None

        command = self._build_command()
        server_env = self._build_environment()

        self._log_startup_info(command)

        try:
            self.process = self._spawn_process(command, server_env)
            self._save_pid()
            self._log_success()
            return self.process

        except FileNotFoundError:
            logger.error("Could not find 'python' or vLLM modules. Ensure vLLM is installed.")
            return None
        except Exception as e:
            logger.error(f"Failed to start VLLM server: {e}")
            self._log_exception()
            return None

    def stop(self):
        """Stop the VLLM server."""
        pid = self._read_pid()
        if pid is None:
            logger.info(
                f"No PID file found for instance {self.config.instance_id}. "
                "Server is not running."
            )
            return

        self._terminate_process(pid)
        self._cleanup_pid_file()

    def wait_for_ready(self, timeout: int = DEFAULT_STARTUP_TIMEOUT) -> bool:
        """Wait for the server to be ready."""
        url = f"http://{self.config.host}:{self.config.port}/v1/models"
        logger.info(
            f"Waiting for server instance {self.config.instance_id} "
            f"at {url} to be ready (timeout: {timeout}s)..."
        )

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._check_health(url):
                logger.info(f"Server instance {self.config.instance_id} is ready!")
                return True
            time.sleep(2)

        self._log_startup_failure(timeout)
        return False

    def status(self) -> dict:
        """Get server status."""
        pid = self._read_pid()
        is_running = self._is_process_alive(pid) if pid else False

        return {
            "instance_id": self.config.instance_id,
            "port": self.config.port,
            "url": self.config.url,
            "gpu_ids": self.config.gpu_ids,
            "instance_dir": str(self.config.instance_dir),
            "pid_file": str(self.config.pid_file),
            "is_running": is_running,
            "pid": pid,
        }

    # Private helper methods

    def _is_already_running(self) -> bool:
        """Check if server is already running."""
        return self.config.pid_file.exists()

    def _build_command(self) -> List[str]:
        """Build the vLLM server command."""
        return [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            f"--model={self.config.model_name}",
            f"--tensor-parallel-size={self.config.tensor_parallel_size}",
            f"--host={self.config.host}",
            f"--port={self.config.port}",
            f"--gpu-memory-utilization={self.config.gpu_memory_utilization}",
            f"--max-model-len={self.config.max_model_len}",
            f"--dtype={self.config.dtype}",
        ]

    def _build_environment(self) -> dict:
        """Build environment variables for the server process."""
        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices
        return server_env

    def _log_startup_info(self, command: List[str]):
        """Log startup information."""
        logger.info(f"Starting VLLM Server Instance {self.config.instance_id}:")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Port: {self.config.port}")
        logger.info(f"  GPUs: {self.config.cuda_visible_devices}")
        logger.info(f"  Data Dir: {self.config.instance_dir}")
        logger.info(f"  Command: {' '.join(command)}")

    def _spawn_process(self, command: List[str], env: dict) -> subprocess.Popen:
        """Spawn the server process."""
        with (
            open(self.config.stdout_log, "w") as stdout_log,
            open(self.config.stderr_log, "w") as stderr_log,
        ):
            return subprocess.Popen(
                command,
                stdout=stdout_log,
                stderr=stderr_log,
                env=env,
                close_fds=True,
            )

    def _save_pid(self):
        """Save process PID to file."""
        with open(self.config.pid_file, "w") as f:
            f.write(str(self.process.pid))

    def _log_success(self):
        """Log successful startup."""
        logger.info(
            f"Server instance {self.config.instance_id} started with PID {self.process.pid}"
        )
        logger.info(f"  PID file: {self.config.pid_file}")
        logger.info(f"  Logs: {self.config.stdout_log}, {self.config.stderr_log}")

    def _log_exception(self):
        """Log exception details."""
        import traceback

        logger.error(traceback.format_exc())

    def _read_pid(self) -> Optional[int]:
        """Read PID from file."""
        if not self.config.pid_file.exists():
            return None

        try:
            with open(self.config.pid_file, "r") as f:
                return int(f.read().strip())
        except Exception as e:
            logger.error(f"Could not read PID from {self.config.pid_file}: {e}")
            return None

    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process is alive."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _terminate_process(self, pid: int):
        """Terminate a process gracefully, then forcefully if needed."""
        try:
            os.kill(pid, 15)  # SIGTERM
            logger.info(
                f"Sent SIGTERM to server instance {self.config.instance_id} (PID {pid}). "
                "Waiting for shutdown..."
            )
            time.sleep(3)

            # Check if still alive
            if self._is_process_alive(pid):
                logger.warning(f"Process PID {pid} still alive. Sending SIGKILL.")
                os.kill(pid, 9)
                time.sleep(1)
            else:
                logger.info(f"Server instance {self.config.instance_id} terminated.")

        except ProcessLookupError:
            logger.warning(f"Process with PID {pid} not found. Already stopped?")
        except Exception as e:
            logger.error(f"Error killing process {pid}: {e}")

    def _cleanup_pid_file(self):
        """Remove PID file."""
        try:
            self.config.pid_file.unlink()
            logger.info(f"Removed PID file {self.config.pid_file}")
        except OSError:
            pass

    def _check_health(self, url: str) -> bool:
        """Check if server is responding."""
        try:
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _log_startup_failure(self, timeout: int):
        """Log startup failure and print recent logs."""
        logger.error(
            f"Server instance {self.config.instance_id} not ready "
            f"after {timeout} seconds. Check logs:\n"
            f"  stdout: {self.config.stdout_log}\n"
            f"  stderr: {self.config.stderr_log}"
        )

        # Print last 50 lines of stderr for debugging
        try:
            with open(self.config.stderr_log, "r") as f:
                lines = f.readlines()
                if lines:
                    logger.error("Last 50 lines of stderr:")
                    logger.error("".join(lines[-50:]))
        except Exception as e:
            logger.error(f"Could not read stderr log: {e}")


# ============================================================================
# Inference Functions
# ============================================================================


async def run_single_inference(
    prompt: str,
    schema_class: type[BaseModel],
    model: str,
    async_client: AsyncOpenAI,
    prompt_index: int,
) -> dict:
    """Run inference for a single prompt."""
    messages = [{"role": "user", "content": prompt}]

    try:
        response = await async_client.responses.parse(
            model=model,
            input=messages,
            text_format=schema_class,
        )

        if hasattr(response, "output_parsed") and isinstance(
            response.output_parsed, schema_class
        ):
            result_text = response.output_parsed

            return {
                "prompt_index": prompt_index,
                "success": True,
                "result": result_text.model_dump(),
            }
        else:
            logger.warning(f"Prompt {prompt_index}: Could not reliably access 'output_parsed'")
            return {
                "prompt_index": prompt_index,
                "success": False,
                "error": "Could not parse output",
                "raw_response": (
                    response.model_dump() if hasattr(response, "model_dump") else str(response)
                ),
            }

    except Exception as e:
        logger.error(f"Error during inference for prompt {prompt_index}: {e}")
        return {
            "prompt_index": prompt_index,
            "success": False,
            "error": str(e),
        }


async def run_inference_from_list(
    prompts: List[str],
    schema_class: type[BaseModel],
    model_name: str = None,
    output_file: str | None = None,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    server_url: str = None,
):
    """Run inference requests against the VLLM server for a list of prompts."""
    if not prompts:
        logger.error("Error: Empty prompts list provided")
        return None

    prompts = prompts[:100]  # Limit to 100 prompts
    model = model_name if model_name else DEFAULT_MODEL_NAME
    vllm_url = server_url if server_url else f"http://localhost:{DEFAULT_VLLM_PORT}/v1"

    async_client = AsyncOpenAI(base_url=vllm_url, api_key="EMPTY")

    all_results = await _run_concurrent_inference(
        prompts, schema_class, model, async_client, max_concurrent
    )

    output_data = _prepare_output_data(all_results, "in-memory-list", model, len(prompts))

    if output_file:
        _save_results(output_data, output_file)

    return output_data


async def run_inference(
    schema_class: type[BaseModel],
    prompt_file: str,
    model_name: str = None,
    output_file: str | None = None,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    server_url: str = None,
):
    """Run inference requests against the VLLM server for multiple prompts."""
    prompts = _load_prompts(prompt_file)
    if prompts is None:
        return

    prompts = prompts[:100]  # Limit to 100 prompts
    model = model_name if model_name else DEFAULT_MODEL_NAME
    vllm_url = server_url if server_url else f"http://localhost:{DEFAULT_VLLM_PORT}/v1"

    async_client = AsyncOpenAI(base_url=vllm_url, api_key="EMPTY")

    all_results = await _run_concurrent_inference(
        prompts, schema_class, model, async_client, max_concurrent
    )

    output_data = _prepare_output_data(all_results, prompt_file, model, len(prompts))

    if output_file:
        _save_results(output_data, output_file)

    return output_data


def _load_prompts(prompt_file: str) -> Optional[List[str]]:
    """Load prompts from JSON file."""
    try:
        with open(prompt_file, "r") as f:
            prompts = json.load(f)
        logger.info(f"Loaded {len(prompts)} prompts from '{prompt_file}'")

        if not isinstance(prompts, list):
            logger.error("Error: Prompt file must contain a JSON array of prompts")
            return None

        return prompts
    except FileNotFoundError:
        logger.error(f"Error: Prompt file '{prompt_file}' not found.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error: Invalid JSON in '{prompt_file}': {e}")
        return None


async def _run_concurrent_inference(
    prompts: List[str],
    schema_class: type[BaseModel],
    model: str,
    async_client: AsyncOpenAI,
    max_concurrent: int,
) -> List[dict]:
    """Run inference for all prompts with concurrency limit and progress tracking."""
    from tqdm.asyncio import tqdm as atqdm

    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_inference(prompt, idx):
        async with semaphore:
            return await run_single_inference(prompt, schema_class, model, async_client, idx)

    tasks = [bounded_inference(prompt, idx) for idx, prompt in enumerate(prompts)]

    # Use tqdm for async progress tracking
    desc = f"Level inference ({len(prompts)} prompts)"
    results = []
    for coro in atqdm.as_completed(tasks, total=len(tasks), desc=desc, unit="req"):
        result = await coro
        results.append(result)

    # Sort by prompt_index to maintain order
    results.sort(key=lambda x: x.get("prompt_index", 0))
    return results


def _prepare_output_data(
    all_results: List[dict],
    prompt_file: str,
    model: str,
    total_prompts: int,
) -> dict:
    """Prepare output data structure."""
    successful_results = [r for r in all_results if r["success"]]
    failed_results = [r for r in all_results if not r["success"]]

    return {
        "metadata": {
            "prompt_file": prompt_file,
            "model_name": model,
            "total_prompts": total_prompts,
            "successful": len(successful_results),
            "failed": len(failed_results),
        },
        "results": successful_results,
        "errors": failed_results if failed_results else None,
    }


def _save_results(output_data: dict, output_file: str):
    """Save results to JSON file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Results successfully written to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to write results to '{output_file}': {e}")


# ============================================================================
# CLI Functions
# ============================================================================


def _parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced VLLM Server with Multi-Instance Support"
    )
    parser.add_argument(
        "command",
        choices=["start", "stop", "status", "run"],
        help="Command to execute",
    )

    # Server configuration
    parser.add_argument(
        "--instance-id",
        type=int,
        default=0,
        help="Server instance ID (default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port number (default: {DEFAULT_VLLM_PORT} + instance_id)",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (e.g., '7,6')",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=DEFAULT_TENSOR_PARALLEL_SIZE,
        help=f"Tensor parallel size (default: {DEFAULT_TENSOR_PARALLEL_SIZE})",
    )

    # Inference options
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="Path to prompt file for inference",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for inference results",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        help="Server URL for inference",
    )

    return parser.parse_args()


def _create_config_from_args(args) -> ServerConfig:
    """Create ServerConfig from command line arguments."""
    # Determine port
    port = args.port if args.port is not None else DEFAULT_VLLM_PORT + args.instance_id

    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    else:
        # Default: descending pairs based on instance_id
        # Instance 0: [7,6], Instance 1: [5,4], Instance 2: [3,2]
        high_gpu = 7 - (args.instance_id * 2)
        gpu_ids = [high_gpu, high_gpu - 1]

    return ServerConfig(
        instance_id=args.instance_id,
        model_name=args.model,
        port=port,
        gpu_ids=gpu_ids,
        tensor_parallel_size=args.tensor_parallel_size,
    )


def _handle_start_command(server: VLLMServer):
    """Handle the start command."""
    process = server.start()
    if process and server.wait_for_ready():
        logger.info("VLLM server is running and ready!")
        logger.info(f"  URL: {server.config.url}")
        logger.info(f"  GPUs: {server.config.cuda_visible_devices}")
        logger.info(f"  Data Dir: {server.config.instance_dir}")
    elif process:
        logger.error("Server process started but did not become ready in time.")
        server.stop()
        sys.exit(1)
    else:
        logger.error("Failed to start server.")
        sys.exit(1)


def _handle_stop_command(server: VLLMServer):
    """Handle the stop command."""
    server.stop()


def _handle_status_command(server: VLLMServer):
    """Handle the status command."""
    status = server.status()
    print(f"\nVLLM Server Instance {status['instance_id']} Status:")
    print("=" * 70)
    print(f"  Port: {status['port']}")
    print(f"  URL: {status['url']}")
    print(f"  GPUs: {status['gpu_ids']}")
    print(f"  Data Directory: {status['instance_dir']}")
    print(f"  PID File: {status['pid_file']}")
    print(f"  Running: {'✓ Yes' if status['is_running'] else '✗ No'}")
    if status["pid"]:
        print(f"  PID: {status['pid']}")
    print("=" * 70)


def _handle_run_command(server: VLLMServer, args):
    """Handle the run command."""
    if not args.prompt_file:
        logger.error("--prompt-file required for run command")
        sys.exit(1)

    # Check if server is running
    if not server.wait_for_ready(timeout=10):
        logger.error("Server is not running or not responding. Please start the server first.")
        sys.exit(1)

    server_url = args.server_url or server.config.url
    asyncio.run(
        run_inference(
            ListOfSingleClassificationResult,
            args.prompt_file,
            args.model,
            args.output_file,
            server_url=server_url,
        )
    )


def main():
    """CLI entry point."""
    args = _parse_args()
    config = _create_config_from_args(args)
    server = VLLMServer(config)

    # Execute command
    if args.command == "start":
        _handle_start_command(server)
    elif args.command == "stop":
        _handle_stop_command(server)
    elif args.command == "status":
        _handle_status_command(server)
    elif args.command == "run":
        _handle_run_command(server, args)


if __name__ == "__main__":
    main()
