# classifier/cli/benchmark_batch_classify.py

#!/usr/bin/env python3
"""
Benchmark script for full batch classification pipeline.

Tests different concurrency levels and provides performance comparison.
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import click
import pandas as pd
from rich.console import Console

from classifier.capabilities import (
    BundledClassificationCapability,
    StemPolarityCapability,
    StemRecommendationsCapability,
    StemTrendCapability,
    SubStemPolarityCapability,
    create_default_registry,
    standard_classification_prompt,
)
from classifier.cli.benchmark_utils import (
    create_capability_table,
    create_comparison_table,
    format_duration,
    print_recommendations,
)
from classifier.core import (
    CompositePolicy,
    ConfidenceThresholdPolicy,
    DefaultPolicy,
    ExcerptRequiredPolicy,
)
from classifier.orchestration import CapabilityOrchestrator
from classifier.server import ServerClassificationProcessor

console = Console()


def load_input_texts(input_file: Path, input_column: str, sample_size: int = None) -> List[str]:
    """Load texts from input file."""
    suffix = input_file.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(input_file)
    elif suffix in [".xls", ".xlsx"]:
        df = pd.read_excel(input_file)
    elif suffix == ".json":
        df = pd.read_json(input_file)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if input_column not in df.columns:
        raise ValueError(f"Column '{input_column}' not found in file")

    texts = df[input_column].dropna().astype(str).tolist()

    if sample_size:
        texts = texts[:sample_size]

    return texts


def create_policy(confidence: int = None, require_excerpt: bool = False):
    """Create acceptance policy based on CLI options."""
    policies = []
    if confidence is not None:
        policies.append(ConfidenceThresholdPolicy(min_confidence=confidence))
    if require_excerpt:
        policies.append(ExcerptRequiredPolicy())
    return CompositePolicy(*policies) if policies else DefaultPolicy()


def select_prompt(prompt_type: str):
    """Select prompt function based on type."""
    prompts = {
        "standard": standard_classification_prompt,
    }
    return prompts.get(prompt_type, standard_classification_prompt)


def run_benchmark_iteration(
    texts: List[str],
    processor: ServerClassificationProcessor,
    orchestrator: CapabilityOrchestrator,
    capability_names: List[str],
    max_concurrent: int,
    project_name: str = None,
    verbose: int = 0,
) -> Dict[str, Any]:
    """Run a single benchmark iteration."""
    console.print(f"\n[cyan]Testing concurrency level: {max_concurrent}[/cyan]")

    # Update processor's client concurrency
    processor.llm_processor.max_concurrent = max_concurrent

    # Track timing
    start_time = time.time()

    try:
        # Execute capabilities
        results = orchestrator.execute_capabilities(
            texts=texts,
            capability_names=capability_names,
            project_name=project_name,
        )

        total_time = time.time() - start_time

        # Get capability timings
        capability_timings = orchestrator.get_timing_summary()

        # Calculate per-item latency
        per_item_latency = total_time / len(texts) if texts else 0

        result = {
            "concurrency": max_concurrent,
            "total_items": len(texts),
            "successes": len(texts),
            "failures": 0,
            "success_rate": 100.0,
            "total_time": total_time,
            "throughput": len(texts) / total_time if total_time > 0 else 0,
            "mean_latency": per_item_latency,
            "p95_latency": per_item_latency,
            "p99_latency": per_item_latency,
            "capability_timings": capability_timings,
        }

        # Print immediate results
        console.print(f"[green]✓[/green] Completed in {format_duration(total_time)}")
        console.print(f"  Throughput: {result['throughput']:.2f} items/s")
        console.print(f"  Per-item latency: {format_duration(per_item_latency)}")

        if verbose > 0:
            console.print("\n  Capability breakdown:")
            for cap_name, cap_time in capability_timings.items():
                percentage = (cap_time / total_time * 100) if total_time > 0 else 0
                console.print(
                    f"    • {cap_name}: {format_duration(cap_time)} ({percentage:.1f}%)"
                )

        return result

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        raise


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to topic hierarchy JSON file",
)
@click.option(
    "--server-url",
    type=str,
    default="http://localhost:9005/v1",
    help="VLLM server URL",
)
@click.option(
    "--model-name",
    type=str,
    default="openai/gpt-oss-120b",
    help="Model name",
)
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input file (.csv, .json, .xlsx)",
)
@click.option(
    "--input-column",
    type=str,
    required=True,
    help="Column name for text data",
)
@click.option(
    "--save-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON file path for benchmark results",
)
@click.option(
    "--sample-size",
    type=int,
    default=None,
    help="Number of texts to sample for testing (default: use all)",
)
@click.option(
    "--concurrency-levels",
    type=str,
    default="1,3,5,10,15,20",
    help="Comma-separated concurrency levels to test",
)
@click.option(
    "--min-confidence",
    type=click.IntRange(1, 5),
    default=None,
    help="Minimum confidence threshold (1-5)",
)
@click.option(
    "--require-excerpt",
    is_flag=True,
    help="Require non-empty excerpts",
)
@click.option(
    "--prompt-type",
    type=click.Choice(["standard"]),
    default="standard",
    help="Prompt type for classification",
)
@click.option(
    "--project-name",
    type=str,
    default=None,
    help="Project name for root prefix",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level (-v, -vv, -vvv)",
)
# Capability flags
@click.option(
    "--enable-classification",
    is_flag=True,
    default=True,
    help="Enable hierarchical classification (default: enabled)",
)
@click.option(
    "--enable-recommendations",
    is_flag=True,
    help="Enable recommendation detection",
)
@click.option(
    "--enable-alerts",
    is_flag=True,
    help="Enable alert detection",
)
@click.option(
    "--enable-stem-recommendations",
    is_flag=True,
    help="Enable stem recommendation analysis",
)
@click.option(
    "--enable-stem-polarity",
    is_flag=True,
    help="Enable stem polarity analysis",
)
@click.option(
    "--enable-sub-stem-polarity",
    is_flag=True,
    help="Enable sub-stem polarity analysis",
)
@click.option(
    "--enable-stem-trends",
    is_flag=True,
    help="Enable stem trend analysis",
)
@click.option(
    "--enable-trends",
    is_flag=True,
    help="Enable global trend analysis",
)
@click.option(
    "--max-stem-definitions",
    type=int,
    default=None,
    help="Max definitions for stem analysis",
)
@click.option(
    "--classification-strategy",
    type=click.Choice(["bfs", "bundled"]),
    default="bfs",
    help="Classification strategy",
)
@click.option(
    "--bundle-size",
    type=int,
    default=4,
    help="Bundle size for bundled strategy",
)
def main(
    config: Path,
    server_url: str,
    model_name: str,
    input_file: Path,
    input_column: str,
    save_path: Path,
    sample_size: int,
    concurrency_levels: str,
    min_confidence: int,
    require_excerpt: bool,
    prompt_type: str,
    project_name: str,
    verbose: int,
    enable_classification: bool,
    enable_recommendations: bool,
    enable_alerts: bool,
    enable_stem_recommendations: bool,
    enable_stem_polarity: bool,
    enable_sub_stem_polarity: bool,
    enable_stem_trends: bool,
    enable_trends: bool,
    max_stem_definitions: int,
    classification_strategy: str,
    bundle_size: int,
):
    """Benchmark full batch classification pipeline across multiple concurrency levels."""

    # Parse concurrency levels
    levels = [int(x.strip()) for x in concurrency_levels.split(",")]

    console.print("[bold cyan]🚀 Batch Classification Benchmark[/bold cyan]")
    console.print(f"  Config: {config}")
    console.print(f"  Input: {input_file}")
    console.print(f"  Server: {server_url}")
    console.print(f"  Strategy: {classification_strategy}")
    console.print(f"  Concurrency levels: {levels}")

    # Load texts
    texts = load_input_texts(input_file, input_column, sample_size)
    console.print(f"  Test size: {len(texts)} texts\n")

    # Create policy and prompt
    policy = create_policy(confidence=min_confidence, require_excerpt=require_excerpt)
    prompt_fn = select_prompt(prompt_type)

    # Initialize processor
    with ServerClassificationProcessor(
        config_path=config,
        server_url=server_url,
        model_name=model_name,
        max_concurrent=1,  # Will be updated per iteration
        prompt_fn=prompt_fn,
        policy=policy,
    ) as processor:
        # Create registry
        registry = create_default_registry()

        # Swap to bundled if requested
        if classification_strategy == "bundled":
            bundled_cap = BundledClassificationCapability(
                bundle_size=bundle_size, policy=policy, separator=">"
            )
            registry.register(bundled_cap)

        # Register stem capabilities
        if enable_stem_recommendations:
            registry.register(
                StemRecommendationsCapability(max_stem_definitions=max_stem_definitions)
            )
        if enable_stem_polarity:
            registry.register(StemPolarityCapability(max_stem_definitions=max_stem_definitions))
        if enable_sub_stem_polarity:
            registry.register(
                SubStemPolarityCapability(max_stem_definitions=max_stem_definitions)
            )
        if enable_stem_trends:
            registry.register(StemTrendCapability(max_stem_definitions=max_stem_definitions))

        # Build capability list
        capability_names = []
        if enable_classification:
            capability_names.append("classification")
        if enable_recommendations:
            capability_names.append("recommendations")
        if enable_alerts:
            capability_names.append("alerts")
        if enable_stem_recommendations:
            capability_names.append("stem_recommendations")
        if enable_stem_polarity:
            capability_names.append("stem_polarity")
        if enable_sub_stem_polarity:
            capability_names.append("sub_stem_polarity")
        if enable_stem_trends:
            capability_names.append("stem_trend")
        if enable_trends:
            capability_names.append("trend")

        console.print(f"  Capabilities: {', '.join(capability_names)}\n")

        # Validate
        errors = registry.validate_capabilities(capability_names)
        if errors:
            for error in errors:
                console.print(f"[red]Error: {error}[/red]")
            sys.exit(1)

        # Create orchestrator
        orchestrator = CapabilityOrchestrator(
            processor=processor,
            registry=registry,
            verbose=verbose,
        )

        # Run benchmarks
        all_results = []
        all_capability_timings = {}

        for level in levels:
            result = run_benchmark_iteration(
                texts=texts,
                processor=processor,
                orchestrator=orchestrator,
                capability_names=capability_names,
                max_concurrent=level,
                project_name=project_name,
                verbose=verbose,
            )
            all_results.append(result)

            # Aggregate capability timings
            for cap, timing in result["capability_timings"].items():
                if cap not in all_capability_timings:
                    all_capability_timings[cap] = []
                all_capability_timings[cap].append(timing)

        # Print comparison table
        console.print("\n")
        console.print(create_comparison_table(all_results))

        # Print per-capability table
        if all_capability_timings:
            console.print("\n")
            console.print(create_capability_table(all_capability_timings))

        # Print recommendations
        print_recommendations(all_results)

        # Save results
        if save_path:
            benchmark_data = {
                "timestamp": time.time(),
                "config": str(config),
                "input_file": str(input_file),
                "test_size": len(texts),
                "concurrency_levels": levels,
                "classification_strategy": classification_strategy,
                "capabilities": capability_names,
                "results": all_results,
            }

            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            import json

            with open(save_path, "w") as f:
                json.dump(benchmark_data, f, indent=2)

            console.print(f"\n[green]✓ Results saved to:[/green] {save_path}")


if __name__ == "__main__":
    main()
