# classifier/cli/benchmark_utils.py

"""Shared utilities for benchmarking."""

from datetime import datetime
import json
from pathlib import Path
import statistics
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

console = Console()


def calculate_stats(values: List[float]) -> Dict[str, float]:
    """Calculate statistics for a list of values."""
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    sorted_values = sorted(values)
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p50": sorted_values[int(len(sorted_values) * 0.5)],
        "p90": sorted_values[int(len(sorted_values) * 0.9)]
        if len(sorted_values) > 1
        else sorted_values[0],
        "p95": sorted_values[int(len(sorted_values) * 0.95)]
        if len(sorted_values) > 1
        else sorted_values[0],
        "p99": sorted_values[int(len(sorted_values) * 0.99)]
        if len(sorted_values) > 1
        else sorted_values[0],
        "min": min(values),
        "max": max(values),
    }


def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def create_comparison_table(
    results: List[Dict[str, Any]],
    title: str = "Benchmark Comparison",
    sort_by: str = "concurrency",
) -> Table:
    """Create a Rich table comparing benchmark results."""
    table = Table(title=title, show_header=True, header_style="bold magenta")

    table.add_column("Concurrency", justify="right", style="cyan")
    table.add_column("Total Time", justify="right")
    table.add_column("Throughput", justify="right", style="green")
    table.add_column("Mean Latency", justify="right")
    table.add_column("P95 Latency", justify="right")
    table.add_column("P99 Latency", justify="right")
    table.add_column("Success Rate", justify="right", style="yellow")

    # Sort results
    sorted_results = sorted(results, key=lambda x: x[sort_by])

    for result in sorted_results:
        table.add_row(
            str(result["concurrency"]),
            format_duration(result["total_time"]),
            f"{result['throughput']:.2f}/s",
            format_duration(result["mean_latency"]),
            format_duration(result["p95_latency"]),
            format_duration(result["p99_latency"]),
            f"{result['success_rate']:.1f}%",
        )

    return table


def create_capability_table(
    capability_timings: Dict[str, List[float]], title: str = "Per-Capability Performance"
) -> Table:
    """Create a Rich table showing per-capability performance."""
    table = Table(title=title, show_header=True, header_style="bold magenta")

    table.add_column("Capability", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Total Time", justify="right")
    table.add_column("Mean Time", justify="right", style="green")
    table.add_column("P50", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("P99", justify="right")

    # Sort by total time descending
    sorted_caps = sorted(capability_timings.items(), key=lambda x: sum(x[1]), reverse=True)

    for capability, timings in sorted_caps:
        stats = calculate_stats(timings)
        table.add_row(
            capability,
            str(len(timings)),
            format_duration(sum(timings)),
            format_duration(stats["mean"]),
            format_duration(stats["p50"]),
            format_duration(stats["p95"]),
            format_duration(stats["p99"]),
        )

    return table


def save_benchmark_results(
    results: Dict[str, Any], output_dir: Path, prefix: str = "benchmark"
) -> Path:
    """Save benchmark results to a JSON file with timestamp."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    return filepath


def load_benchmark_results(filepath: Path) -> Dict[str, Any]:
    """Load benchmark results from a JSON file."""
    with open(filepath) as f:
        return json.load(f)


def print_recommendations(results: List[Dict[str, Any]]) -> None:
    """Print recommendations based on benchmark results."""
    console.print("\n[bold cyan]📊 Recommendations:[/bold cyan]")

    # Find optimal concurrency (best throughput without excessive latency)
    best_throughput = max(results, key=lambda x: x["throughput"])

    # Find point of diminishing returns (throughput increase < 10% with latency increase)
    optimal = best_throughput
    for i, result in enumerate(sorted(results, key=lambda x: x["concurrency"])):
        if i == 0:
            continue
        prev = sorted(results, key=lambda x: x["concurrency"])[i - 1]

        throughput_gain = (result["throughput"] - prev["throughput"]) / prev["throughput"]
        latency_increase = (result["p95_latency"] - prev["p95_latency"]) / prev["p95_latency"]

        if throughput_gain < 0.1 and latency_increase > 0.2:
            optimal = prev
            break

    console.print(
        f"• [green]Optimal concurrency:[/green] {optimal['concurrency']} "
        f"({optimal['throughput']:.2f} items/s, P95: {format_duration(optimal['p95_latency'])})"
    )
    console.print(
        f"• [yellow]Max throughput:[/yellow] {best_throughput['concurrency']} "
        f"({best_throughput['throughput']:.2f} items/s, P95: {format_duration(best_throughput['p95_latency'])})"
    )

    # Check for failures
    failures = [r for r in results if r["success_rate"] < 100]
    if failures:
        console.print(
            f"• [red]⚠️  High error rates detected at concurrency levels:[/red] "
            f"{', '.join(str(r['concurrency']) for r in failures)}"
        )


def print_summary_stats(
    total_items: int,
    total_time: float,
    capability_timings: Optional[Dict[str, List[float]]] = None,
) -> None:
    """Print summary statistics."""
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Total items: {total_items}")
    console.print(f"  Total time: {format_duration(total_time)}")
    console.print(f"  Average throughput: {total_items / total_time:.2f} items/s")

    if capability_timings:
        total_capability_time = sum(sum(timings) for timings in capability_timings.values())
        console.print(f"  Total capability time: {format_duration(total_capability_time)}")
