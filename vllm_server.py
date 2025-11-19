"""
Production-Ready VLLM Server Manager with Multi-Instance Support

Features:
- Automatic stale PID cleanup
- Port availability checking
- GPU availability validation
- Robust error handling
- Clean multi-instance support
- Automatic server restart capability
"""

import asyncio
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel
import requests

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
DEFAULT_DTYPE = "bfloat16"
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_STARTUP_TIMEOUT = 300  # Increased for large models

VLLM_DATA_DIR = Path.cwd() / ".vllm_servers"


# ============================================================================
# Utility Functions
# ============================================================================


def is_port_available(port: int) -> bool:
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))
            return True
    except OSError:
        return False


def check_gpu_available(gpu_id: int) -> bool:
    """Check if a GPU is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_gpu_memory_usage(gpu_id: int) -> Optional[float]:
    """Get GPU memory usage percentage."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_id),
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            used, total = map(float, result.stdout.strip().split(","))
            return (used / total) * 100
    except Exception:
        pass
    return None


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
        self.gpu_ids = gpu_ids or [7, 6]  # Default high-end GPUs
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

        # Validate port range
        if not (1024 <= self.port <= 65535):
            raise ValueError(f"Port {self.port} out of valid range (1024-65535)")

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
        return self.instance_dir / "server.pid"

    @property
    def stdout_log(self) -> Path:
        return self.instance_dir / "stdout.log"

    @property
    def stderr_log(self) -> Path:
        return self.instance_dir / "stderr.log"


# ============================================================================
# VLLM Server Manager
# ============================================================================


class VLLMServer:
    """Production-ready VLLM server manager."""

    def __init__(self, config: ServerConfig):
        """Initialize VLLM server."""
        self.config = config
        self.process: Optional[subprocess.Popen] = None

    def start(self, force_restart: bool = False) -> bool:
        """
        Start the VLLM server.

        Args:
            force_restart: If True, stop existing server before starting

        Returns:
            True if server started successfully, False otherwise
        """
        # Check for existing server
        if self._is_server_running():
            if force_restart:
                logger.info("Force restart requested. Stopping existing server...")
                self.stop()
            else:
                logger.warning(
                    f"Server instance {self.config.instance_id} is already running. "
                    f"Use --force to restart."
                )
                return False

        # Pre-flight checks
        if not self._preflight_checks():
            return False

        # Start server
        command = self._build_command()
        server_env = self._build_environment()

        self._log_startup_info(command)

        try:
            self.process = self._spawn_process(command, server_env)
            self._save_pid()
            self._log_success()

            # Wait for server to be ready
            if self.wait_for_ready():
                logger.info("✓ Server is ready and accepting requests!")
                return True
            else:
                logger.error("✗ Server failed to become ready")
                self.stop()
                return False

        except Exception as e:
            logger.error(f"Failed to start VLLM server: {e}")
            self._cleanup_after_failure()
            return False

    def stop(self) -> bool:
        """
        Stop the VLLM server.

        Returns:
            True if server was stopped, False if not running
        """
        pid = self._read_pid()

        if pid is None:
            logger.info(f"No PID file found for instance {self.config.instance_id}")
            return False

        if not self._is_process_alive(pid):
            logger.info(f"Process {pid} not running. Cleaning up stale PID file.")
            self._cleanup_pid_file()
            return False

        logger.info(f"Stopping server instance {self.config.instance_id} (PID {pid})...")
        self._terminate_process(pid)
        self._cleanup_pid_file()
        logger.info("✓ Server stopped successfully")
        return True

    def restart(self) -> bool:
        """Restart the server."""
        logger.info(f"Restarting server instance {self.config.instance_id}...")
        self.stop()
        time.sleep(2)
        return self.start()

    def wait_for_ready(self, timeout: int = DEFAULT_STARTUP_TIMEOUT) -> bool:
        """Wait for the server to be ready."""
        health_url = f"http://{self.config.host}:{self.config.port}/health"
        models_url = f"http://{self.config.host}:{self.config.port}/v1/models"

        logger.info(
            f"Waiting for server instance {self.config.instance_id} to be ready "
            f"(timeout: {timeout}s)..."
        )

        start_time = time.time()
        last_log = 0

        while time.time() - start_time < timeout:
            elapsed = int(time.time() - start_time)

            # Log progress every 10 seconds
            if elapsed - last_log >= 10:
                logger.info(f"  Still waiting... ({elapsed}s elapsed)")
                last_log = elapsed

            # Try health endpoint first, then models endpoint
            if self._check_health(health_url) or self._check_health(models_url):
                logger.info(f"✓ Server ready after {elapsed}s")
                return True

            time.sleep(2)

        logger.error(f"✗ Server not ready after {timeout}s")
        self._print_error_logs()
        return False

    def status(self) -> dict:
        """Get detailed server status."""
        pid = self._read_pid()
        is_running = self._is_process_alive(pid) if pid else False

        status = {
            "instance_id": self.config.instance_id,
            "model": self.config.model_name,
            "port": self.config.port,
            "url": self.config.url,
            "gpu_ids": self.config.gpu_ids,
            "is_running": is_running,
            "pid": pid,
            "data_dir": str(self.config.instance_dir),
        }

        # Add GPU info if running
        if is_running:
            status["gpu_memory"] = {}
            for gpu_id in self.config.gpu_ids:
                usage = get_gpu_memory_usage(gpu_id)
                if usage is not None:
                    status["gpu_memory"][f"GPU {gpu_id}"] = f"{usage:.1f}%"

        return status

    # ========================================================================
    # Private Methods - Preflight Checks
    # ========================================================================

    def _preflight_checks(self) -> bool:
        """Run pre-flight checks before starting server."""
        logger.info("Running pre-flight checks...")

        # Check port availability
        if not is_port_available(self.config.port):
            logger.error(f"✗ Port {self.config.port} is already in use")
            logger.info(f"  Try: lsof -i :{self.config.port}")
            return False
        logger.info(f"✓ Port {self.config.port} is available")

        # Check GPU availability
        for gpu_id in self.config.gpu_ids:
            if not check_gpu_available(gpu_id):
                logger.error(f"✗ GPU {gpu_id} is not available")
                logger.info("  Try: nvidia-smi")
                return False

            usage = get_gpu_memory_usage(gpu_id)
            if usage is not None:
                logger.info(f"✓ GPU {gpu_id} available (memory: {usage:.1f}% used)")
            else:
                logger.info(f"✓ GPU {gpu_id} available")

        # Check if vLLM is installed
        try:
            result = subprocess.run(
                ["python", "-c", "import vllm"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                logger.error("✗ vLLM is not installed")
                logger.info("  Install: pip install vllm")
                return False
            logger.info("✓ vLLM is installed")
        except Exception:
            logger.error("✗ Could not verify vLLM installation")
            return False

        logger.info("✓ All pre-flight checks passed")
        return True

    def _is_server_running(self) -> bool:
        """Check if server is actually running (not just PID file exists)."""
        if not self.config.pid_file.exists():
            return False

        pid = self._read_pid()
        if pid and self._is_process_alive(pid):
            return True

        # Stale PID file - clean it up
        logger.info("Found stale PID file, cleaning up...")
        self._cleanup_pid_file()
        return False

    # ========================================================================
    # Private Methods - Process Management
    # ========================================================================

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

    def _read_pid(self) -> Optional[int]:
        """Read PID from file."""
        if not self.config.pid_file.exists():
            return None

        try:
            with open(self.config.pid_file, "r") as f:
                return int(f.read().strip())
        except Exception as e:
            logger.warning(f"Could not read PID from {self.config.pid_file}: {e}")
            return None

    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process is alive."""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _terminate_process(self, pid: int):
        """Terminate a process gracefully, then forcefully if needed."""
        try:
            # Try graceful shutdown
            os.kill(pid, 15)  # SIGTERM
            logger.info(f"  Sent SIGTERM to PID {pid}, waiting...")

            # Wait up to 10 seconds for graceful shutdown
            for _ in range(10):
                time.sleep(1)
                if not self._is_process_alive(pid):
                    logger.info("  Process terminated gracefully")
                    return

            # Force kill if still alive
            if self._is_process_alive(pid):
                logger.warning("  Process still alive, sending SIGKILL...")
                os.kill(pid, 9)
                time.sleep(2)

                if self._is_process_alive(pid):
                    logger.error(f"  Failed to kill process {pid}")
                else:
                    logger.info("  Process forcefully terminated")

        except ProcessLookupError:
            logger.info(f"  Process {pid} already terminated")
        except Exception as e:
            logger.error(f"  Error terminating process {pid}: {e}")

    def _cleanup_pid_file(self):
        """Remove PID file."""
        try:
            if self.config.pid_file.exists():
                self.config.pid_file.unlink()
        except Exception as e:
            logger.warning(f"Could not remove PID file: {e}")

    def _cleanup_after_failure(self):
        """Clean up after a failed start."""
        self._cleanup_pid_file()
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                pass

    # ========================================================================
    # Private Methods - Health Checks & Logging
    # ========================================================================

    def _check_health(self, url: str) -> bool:
        """Check if server is responding."""
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _log_startup_info(self, command: List[str]):
        """Log startup information."""
        logger.info("=" * 70)
        logger.info(f"Starting VLLM Server Instance {self.config.instance_id}")
        logger.info("=" * 70)
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Port: {self.config.port}")
        logger.info(f"  GPUs: {self.config.cuda_visible_devices}")
        logger.info(f"  Tensor Parallel: {self.config.tensor_parallel_size}")
        logger.info(f"  Max Model Length: {self.config.max_model_len}")
        logger.info(f"  Data Dir: {self.config.instance_dir}")
        logger.info("=" * 70)

    def _log_success(self):
        """Log successful startup."""
        logger.info(f"✓ Server process started with PID {self.process.pid}")
        logger.info(f"  Logs: {self.config.stderr_log}")

    def _print_error_logs(self, lines: int = 30):
        """Print recent error logs."""
        try:
            if self.config.stderr_log.exists():
                with open(self.config.stderr_log, "r") as f:
                    log_lines = f.readlines()
                    if log_lines:
                        logger.error(f"\nLast {lines} lines from stderr:")
                        logger.error("=" * 70)
                        for line in log_lines[-lines:]:
                            logger.error(line.rstrip())
                        logger.error("=" * 70)
        except Exception as e:
            logger.error(f"Could not read error logs: {e}")


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
            return {
                "prompt_index": prompt_index,
                "success": True,
                "result": response.output_parsed.model_dump(),
            }
        else:
            return {
                "prompt_index": prompt_index,
                "success": False,
                "error": "Could not parse output",
                "raw_response": str(response),
            }

    except Exception as e:
        logger.error(f"Error during inference for prompt {prompt_index}: {e}")
        return {
            "prompt_index": prompt_index,
            "success": False,
            "error": str(e),
        }


async def run_batch_inference(
    prompts: List[str],
    schema_class: type[BaseModel],
    server_url: str,
    model_name: str = DEFAULT_MODEL_NAME,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
) -> dict:
    """
    Run batch inference against a VLLM server.

    Args:
        prompts: List of prompts to process
        schema_class: Pydantic model for output validation
        server_url: Server URL (e.g., "http://localhost:9001/v1")
        model_name: Model name to use
        max_concurrent: Maximum concurrent requests

    Returns:
        Dictionary with results and metadata
    """
    if not prompts:
        logger.error("Empty prompts list provided")
        return {"error": "Empty prompts list"}

    logger.info(f"Running batch inference on {len(prompts)} prompts...")
    logger.info(f"  Server: {server_url}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Max concurrent: {max_concurrent}")

    async_client = AsyncOpenAI(base_url=server_url, api_key="EMPTY")

    all_results = await _run_concurrent_inference(
        prompts, schema_class, model_name, async_client, max_concurrent
    )

    return _prepare_output_data(all_results, model_name, len(prompts))


async def _run_concurrent_inference(
    prompts: List[str],
    schema_class: type[BaseModel],
    model: str,
    async_client: AsyncOpenAI,
    max_concurrent: int,
) -> List[dict]:
    """Run inference with concurrency limit and progress tracking."""
    try:
        from tqdm.asyncio import tqdm as atqdm

        use_tqdm = True
    except ImportError:
        use_tqdm = False
        logger.info("Install tqdm for progress bars: pip install tqdm")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_inference(prompt, idx):
        async with semaphore:
            return await run_single_inference(prompt, schema_class, model, async_client, idx)

    tasks = [bounded_inference(prompt, idx) for idx, prompt in enumerate(prompts)]

    if use_tqdm:
        results = []
        for coro in atqdm.as_completed(tasks, total=len(tasks), desc="Processing", unit="req"):
            result = await coro
            results.append(result)
    else:
        results = await asyncio.gather(*tasks)

    # Sort by prompt_index
    results.sort(key=lambda x: x.get("prompt_index", 0))
    return results


def _prepare_output_data(all_results: List[dict], model: str, total_prompts: int) -> dict:
    """Prepare output data structure."""
    successful = [r for r in all_results if r.get("success")]
    failed = [r for r in all_results if not r.get("success")]

    logger.info("\nInference complete:")
    logger.info(f"  Total: {total_prompts}")
    logger.info(f"  Successful: {len(successful)}")
    logger.info(f"  Failed: {len(failed)}")

    return {
        "metadata": {
            "model_name": model,
            "total_prompts": total_prompts,
            "successful": len(successful),
            "failed": len(failed),
        },
        "results": successful,
        "errors": failed if failed else None,
    }


# ============================================================================
# Backwards Compatibility Aliases
# ============================================================================


async def run_inference_from_list(
    prompts: List[str],
    schema_class: type[BaseModel],
    model_name: str = None,
    output_file: str | None = None,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    server_url: str = None,
):
    """
    Backwards compatibility wrapper for run_batch_inference.

    DEPRECATED: Use run_batch_inference() instead.
    """
    server_url = server_url or f"http://localhost:{DEFAULT_VLLM_PORT}/v1"
    model_name = model_name or DEFAULT_MODEL_NAME

    results = await run_batch_inference(
        prompts=prompts,
        schema_class=schema_class,
        server_url=server_url,
        model_name=model_name,
        max_concurrent=max_concurrent,
    )

    if output_file:
        import json

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    return results


# ============================================================================
# CLI Functions
# ============================================================================


def _parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Production-Ready VLLM Server Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server on default port (9001) with GPUs 7,6
  python vllm_server.py start
  
  # Start instance 1 on port 9002 with GPUs 5,4
  python vllm_server.py start --instance-id 1 --gpu-ids 5,4
  
  # Force restart a server
  python vllm_server.py start --instance-id 0 --force
  
  # Check server status
  python vllm_server.py status --instance-id 0
  
  # Stop server
  python vllm_server.py stop --instance-id 0
        """,
    )

    parser.add_argument(
        "command",
        choices=["start", "stop", "restart", "status"],
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force restart if server is already running",
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
        # Instance 0: [7,6], Instance 1: [5,4], Instance 2: [3,2], Instance 3: [1,0]
        high_gpu = 7 - (args.instance_id * 2)
        gpu_ids = [high_gpu, high_gpu - 1]

    return ServerConfig(
        instance_id=args.instance_id,
        model_name=args.model,
        port=port,
        gpu_ids=gpu_ids,
        tensor_parallel_size=args.tensor_parallel_size,
    )


def main():
    """CLI entry point."""
    args = _parse_args()

    try:
        config = _create_config_from_args(args)
        server = VLLMServer(config)

        if args.command == "start":
            success = server.start(force_restart=args.force)
            sys.exit(0 if success else 1)

        elif args.command == "stop":
            success = server.stop()
            sys.exit(0 if success else 1)

        elif args.command == "restart":
            success = server.restart()
            sys.exit(0 if success else 1)

        elif args.command == "status":
            status = server.status()
            print(f"\nVLLM Server Instance {status['instance_id']} Status:")
            print("=" * 70)
            print(f"  Model: {status['model']}")
            print(f"  Port: {status['port']}")
            print(f"  URL: {status['url']}")
            print(f"  GPUs: {status['gpu_ids']}")
            print(f"  Data Directory: {status['data_dir']}")
            print(f"  Running: {'✓ Yes' if status['is_running'] else '✗ No'}")
            if status.get("pid"):
                print(f"  PID: {status['pid']}")
            if status.get("gpu_memory"):
                print("  GPU Memory Usage:")
                for gpu, usage in status["gpu_memory"].items():
                    print(f"    {gpu}: {usage}")
            print("=" * 70)
            sys.exit(0 if status["is_running"] else 1)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
