#!/usr/bin/env python3
"""
Benchmark script to find optimal max_concurrent setting for VLLM server.

Tests different concurrency levels and measures:
- Throughput (requests/second)
- Average latency
- GPU memory usage
- Error rate

Usage:
    python benchmark_concurrency.py --server-url http://localhost:9001/v1 \
        --test-prompts 50 --concurrency-levels 1,3,5,10,15,20
"""

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List, Literal

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

try:
    from openai import AsyncOpenAI
    from pydantic import BaseModel
except ImportError:
    print("Error: Install required packages: pip install openai pydantic rich psutil")
    exit(1)


console = Console()


class SimpleClassification(BaseModel):
    """Simple schema for testing."""

    sentiment: Literal["Positive", "Negative"]


def load_texts_from_csv(path: str, column: str) -> List[str]:
    """Load raw texts from a CSV column and prepend the classification prefix."""
    df = pd.read_csv(path)

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in CSV. Available columns: {list(df.columns)}"
        )

    prefix = "Classify whether this text conveys a Positive or Negative sentiment: "

    texts = []
    for i, txt in enumerate(df[column].astype(str)):
        texts.append(f"{prefix}{txt} (Test prompt #{i + 1})")

    return texts


def generate_test_prompts(count: int) -> List[str]:
    """Generate test prompts of varying lengths."""
    base_prompts = [
        "Classify whether this text conveys a Positive or Negative sentiment: 'The instructor was excellent.'",
        "Classify whether this text conveys a Positive or Negative sentiment: 'The material was too advanced.'",
        "Classify whether this text conveys a Positive or Negative sentiment: 'I learned a lot from this course.'",
        "Classify whether this text conveys a Positive or Negative sentiment: 'The exams were fair and well-designed.'",
        "Classify whether this text conveys a Positive or Negative sentiment: 'The course was well-structured and easy to follow.'",
    ]

    prompts = []
    for i in range(count):
        base = base_prompts[i % len(base_prompts)]
        prompts.append(f"{base} (Test prompt #{i + 1})")

    return prompts


async def run_single_request(
    prompt: str,
    schema_class: type[BaseModel],
    model: str,
    client: AsyncOpenAI,
    request_id: int,
) -> Dict[str, Any]:
    """Run a single inference request and time it."""
    start = time.time()

    try:
        response = await client.responses.parse(
            model=model,
            input=[{"role": "user", "content": prompt}],
            text_format=schema_class,
        )

        elapsed = time.time() - start
        return {
            "request_id": request_id,
            "success": True,
            "latency": elapsed,
            "error": None,
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            "request_id": request_id,
            "success": False,
            "latency": elapsed,
            "error": str(e),
        }


async def benchmark_concurrency_level(
    prompts: List[str],
    max_concurrent: int,
    server_url: str,
    model_name: str,
    schema_class: type[BaseModel],
) -> Dict[str, Any]:
    """Benchmark a specific concurrency level with a visible progress bar."""
    console.print(f"\n[cyan]Testing max_concurrent={max_concurrent}...[/cyan]")

    client = AsyncOpenAI(base_url=server_url, api_key="EMPTY")
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_request(prompt: str, req_id: int):
        async with semaphore:
            return await run_single_request(prompt, schema_class, model_name, client, req_id)

    # Create progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn(f"Processing (concurrent={max_concurrent}):"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn(" • "),
        TimeElapsedColumn(),
        TextColumn(" • "),
        TimeRemainingColumn(),
        console=console,
    )

    results = []
    start_time = time.time()

    with progress:
        task = progress.add_task("requests", total=len(prompts))

        async def track_request(prompt, req_id):
            result = await bounded_request(prompt, req_id)
            results.append(result)
            progress.update(task, advance=1)

        await asyncio.gather(*(track_request(prompt, i) for i, prompt in enumerate(prompts)))

    total_time = time.time() - start_time

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    latencies = [r["latency"] for r in successful]

    metrics = {
        "max_concurrent": max_concurrent,
        "total_requests": len(prompts),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "success_rate": len(successful) / len(prompts) * 100,
        "total_time": total_time,
        "throughput": len(successful) / total_time if total_time > 0 else 0,
    }

    if latencies:
        metrics.update(
            {
                "avg_latency": statistics.mean(latencies),
                "median_latency": statistics.median(latencies),
                "p95_latency": statistics.quantiles(latencies, n=20)[18]
                if len(latencies) > 1
                else latencies[0],
                "p99_latency": statistics.quantiles(latencies, n=100)[98]
                if len(latencies) > 1
                else latencies[0],
                "min_latency": min(latencies),
                "max_latency": max(latencies),
            }
        )

    # Error classification
    if failed:
        error_types = {}
        for result in failed:
            error = result["error"] or ""
            if "out of memory" in error.lower() or "oom" in error.lower():
                error_type = "OOM"
            elif "timeout" in error.lower():
                error_type = "Timeout"
            else:
                error_type = "Other"
            error_types[error_type] = error_types.get(error_type, 0) + 1
        metrics["error_types"] = error_types

    return metrics


def get_gpu_memory_snapshot() -> Dict[str, float]:
    """Get current GPU memory usage (requires nvidia-smi)."""
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            gpu_info = {}
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    gpu_id, used, total = map(float, line.split(","))
                    gpu_info[int(gpu_id)] = {
                        "used_mb": used,
                        "total_mb": total,
                        "used_percent": (used / total) * 100,
                    }
            return gpu_info
    except Exception:
        pass

    return {}


def print_results_table(all_metrics: List[Dict[str, Any]], benchmark_info: Dict[str, Any]):
    """Print a rich table with benchmark results."""
    table = Table(
        title="Concurrency Benchmark Results", show_header=True, header_style="bold magenta"
    )

    table.add_column("Concurrency", justify="right", style="cyan")
    table.add_column("Throughput\n(req/s)", justify="right", style="green")
    table.add_column("Success\nRate", justify="right")
    table.add_column("Avg Latency\n(s)", justify="right")
    table.add_column("P95 Latency\n(s)", justify="right")
    table.add_column("P99 Latency\n(s)", justify="right")
    table.add_column("Errors", justify="right", style="red")
    table.add_column("Total\nPrompts", justify="right", style="blue")
    table.add_column("Avg Text\nLength", justify="right", style="yellow")

    best_throughput = max(m["throughput"] for m in all_metrics)

    for metrics in all_metrics:
        concurrency = str(metrics["max_concurrent"])
        throughput = f"{metrics['throughput']:.2f}"
        success_rate = f"{metrics['success_rate']:.1f}%"

        if metrics["throughput"] == best_throughput:
            throughput = f"[bold green]{throughput}[/bold green]"
            concurrency = f"[bold green]{concurrency}[/bold green]"

        avg_latency = f"{metrics.get('avg_latency', 0):.2f}"
        p95_latency = f"{metrics.get('p95_latency', 0):.2f}"
        p99_latency = f"{metrics.get('p99_latency', 0):.2f}"

        errors = str(metrics["failed_requests"])
        if metrics["failed_requests"] > 0:
            errors = f"[red]{errors}[/red]"
            if "error_types" in metrics:
                detail = ", ".join(f"{k}:{v}" for k, v in metrics["error_types"].items())
                errors = f"{errors}\n({detail})"

        table.add_row(
            concurrency,
            throughput,
            success_rate,
            avg_latency,
            p95_latency,
            p99_latency,
            errors,
            str(benchmark_info["total_prompts"]),
            f"{benchmark_info['avg_text_length']:.1f}",
        )

    console.print("\n")
    console.print(table)


def print_recommendations(
    all_metrics: List[Dict[str, Any]],
    benchmark_info: Dict[str, Any],
):
    console.print("\n[bold cyan]Recommendations:[/bold cyan]")

    console.print("\n[cyan]Benchmark Summary:[/cyan]")
    console.print(f"  • Total prompts tested: {benchmark_info['total_prompts']}")
    console.print(
        f"  • Average text length: {benchmark_info['avg_text_length']:.1f} characters"
    )

    successful_runs = [m for m in all_metrics if m["success_rate"] == 100]

    if successful_runs:
        best = max(successful_runs, key=lambda m: m["throughput"])
        console.print(f"\n[green]✓ Optimal max_concurrent: {best['max_concurrent']}[/green]")
        console.print(f"  • Throughput: {best['throughput']:.2f} req/s")
        console.print(f"  • Avg latency: {best['avg_latency']:.2f}s")
        console.print(f"  • P95 latency: {best['p95_latency']:.2f}s")
    else:
        console.print("\n[yellow]⚠ All tested concurrency levels had failures[/yellow]")
        console.print("[yellow]  Try lower concurrency levels or check server logs[/yellow]")

    # Check OOM
    oom_runs = [m for m in all_metrics if "error_types" in m and "OOM" in m["error_types"]]
    if oom_runs:
        min_oom = min(m["max_concurrent"] for m in oom_runs)
        console.print(f"\n[red]⚠ OOM errors detected at max_concurrent >= {min_oom}[/red]")
        console.print("[yellow]  Reduce concurrency or increase GPU memory[/yellow]")

    # Diminishing returns
    if len(all_metrics) > 2:
        sorted_metrics = sorted(all_metrics, key=lambda m: m["max_concurrent"])
        for i in range(1, len(sorted_metrics)):
            prev = sorted_metrics[i - 1]["throughput"]
            curr = sorted_metrics[i]["throughput"]
            improvement = (curr - prev) / prev * 100 if prev > 0 else 0
            if improvement < 5:
                console.print(
                    f"\n[yellow]ℹ Diminishing returns after max_concurrent={sorted_metrics[i - 1]['max_concurrent']}[/yellow]"
                )
                break


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark VLLM server concurrency settings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--server-url", type=str, default="http://localhost:9001/v1", help="VLLM server URL"
    )
    parser.add_argument(
        "--model-name", type=str, default="openai/gpt-oss-120b", help="Model name"
    )
    parser.add_argument(
        "--test-prompts",
        type=int,
        default=50,
        help="Number of test prompts (default: 50)",
    )
    parser.add_argument(
        "--concurrency-levels",
        type=str,
        default="1,3,5,10,15,20",
        help="Comma-separated concurrency levels",
    )
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")

    parser.add_argument("--input-csv", type=str, help="Path to CSV")
    parser.add_argument("--text-column", type=str, help="Column containing text")

    args = parser.parse_args()

    concurrency_levels = [int(x.strip()) for x in args.concurrency_levels.split(",")]

    console.print("[bold cyan]VLLM Server Concurrency Benchmark[/bold cyan]")
    console.print(f"Server: {args.server_url}")
    console.print(f"Model: {args.model_name}")
    console.print(f"Concurrency levels: {concurrency_levels}")

    # Health check
    try:
        import requests

        response = requests.get(f"{args.server_url.replace('/v1', '')}/health", timeout=5)
        if response.status_code != 200:
            console.print(f"[red]Error: Server not responding at {args.server_url}[/red]")
            return
    except Exception as e:
        console.print(f"[red]Error: Cannot connect to server: {e}[/red]")
        return

    # GPU memory snapshot
    initial_gpu_memory = get_gpu_memory_snapshot()
    if initial_gpu_memory:
        console.print("\n[cyan]Initial GPU Memory:[/cyan]")
        for gpu_id, info in initial_gpu_memory.items():
            console.print(
                f"  GPU {gpu_id}: {info['used_mb']:.0f}MB / {info['total_mb']:.0f}MB ({info['used_percent']:.1f}%)"
            )

    # Load or generate prompts
    if args.input_csv and args.text_column:
        console.print(f"\n[cyan]Loading texts from CSV: {args.input_csv}[/cyan]")
        prompts = load_texts_from_csv(args.input_csv, args.text_column)
        console.print(f"[green]✓ Loaded {len(prompts)} texts[/green]")
    else:
        console.print(f"\n[cyan]Generating {args.test_prompts} test prompts...[/cyan]")
        prompts = generate_test_prompts(args.test_prompts)

    total_prompts = len(prompts)
    avg_text_length = sum(len(p) for p in prompts) / total_prompts if total_prompts else 0

    benchmark_info = {
        "total_prompts": total_prompts,
        "avg_text_length": avg_text_length,
    }

    all_metrics = []

    for level in concurrency_levels:
        metrics = await benchmark_concurrency_level(
            prompts=prompts,
            max_concurrent=level,
            server_url=args.server_url,
            model_name=args.model_name,
            schema_class=SimpleClassification,
        )
        all_metrics.append(metrics)

        console.print(
            f"[green]✓ Throughput: {metrics['throughput']:.2f} req/s • "
            f"Success: {metrics['success_rate']:.1f}% • "
            f"Avg latency: {metrics.get('avg_latency', 0):.2f}s[/green]"
        )

        # GPU memory delta check
        if initial_gpu_memory:
            current_gpu_memory = get_gpu_memory_snapshot()
            if current_gpu_memory:
                for gpu_id, info in current_gpu_memory.items():
                    if gpu_id in initial_gpu_memory:
                        delta = info["used_mb"] - initial_gpu_memory[gpu_id]["used_mb"]
                        if abs(delta) > 100:
                            console.print(
                                f"[yellow]  GPU {gpu_id} memory delta: {delta:+.0f}MB[/yellow]"
                            )

        await asyncio.sleep(2)

    print_results_table(all_metrics, benchmark_info)
    print_recommendations(all_metrics, benchmark_info)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "benchmark_info": benchmark_info,
                    "results": all_metrics,
                },
                f,
                indent=2,
            )
        console.print(f"\n[green]✓ Results saved to {output_path}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
