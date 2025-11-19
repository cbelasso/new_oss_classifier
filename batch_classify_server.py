#!/usr/bin/env python3
"""
Batch hierarchical text classification using VLLM Server.

Clean, streamlined version specifically for server-based inference.

Usage examples:

# Basic usage
python batch_classify_server.py --config topics.json \\
    --input-file data.csv --save-path results.json

# With recommendations and alerts
python batch_classify_server.py --config topics.json \\
    --input-file data.csv --save-path results.json \\
    --enable-recommendations --enable-alerts

# With stem analysis
python batch_classify_server.py --config topics.json \\
    --input-file data.csv --save-path results.json \\
    --enable-stem-recommendations --enable-stem-polarity

# Custom server URL
python batch_classify_server.py --config topics.json \\
    --input-file data.csv --save-path results.json \\
    --server-url http://localhost:9002/v1
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import click
import pandas as pd
from rich.console import Console
from tqdm import tqdm

from classifier import BundledClassificationCapability
from classifier.capabilities import (
    CapabilityOrchestrator,
    StemPolarityCapability,
    StemRecommendationsCapability,
    create_default_registry,
)
from classifier.capabilities.classification.prompts import standard_classification_prompt
from classifier.capabilities.stem_trend import StemTrendCapability
from classifier.policies import (
    CompositePolicy,
    ConfidenceThresholdPolicy,
    DefaultPolicy,
    ExcerptRequiredPolicy,
)
from classifier.server_processor import ServerClassificationProcessor

console = Console()


def load_input_texts(
    input_file: Path = None, input_column: str = None, verbose: int = 0
) -> List[str]:
    """Load texts from file or stdin."""
    if input_file is None:
        if verbose > 0:
            console.print("[cyan]Reading texts from stdin...[/cyan]")
        return [line.strip() for line in sys.stdin if line.strip()]

    if verbose > 0:
        console.print(f"[cyan]Reading input file:[/cyan] {input_file}")

    suffix = input_file.suffix.lower()

    try:
        if suffix == ".csv":
            df = pd.read_csv(input_file)
        elif suffix in [".xls", ".xlsx"]:
            df = pd.read_excel(input_file)
        elif suffix == ".json":
            df = pd.read_json(input_file)
        elif suffix == ".pkl":
            df = pd.read_pickle(input_file)
        elif suffix == ".txt":
            with open(input_file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    except Exception as e:
        console.print(f"[red]Failed to read input file: {e}[/red]")
        sys.exit(1)

    # Infer column if not provided
    if input_column is None:
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if not text_cols:
            console.print("[red]No text columns found. Specify --input-column.[/red]")
            sys.exit(1)
        input_column = text_cols[0]
        if verbose > 0:
            console.print(f"[yellow]Using column:[/yellow] {input_column}")

    if input_column not in df.columns:
        console.print(f"[red]Column '{input_column}' not found.[/red]")
        sys.exit(1)

    lines = df[input_column].dropna().astype(str).tolist()

    if verbose > 0:
        console.print(f"[green]✓ Loaded {len(lines)} texts[/green]")

    return lines


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


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def save_chunk_results(output_dir: Path, chunk_idx: int, chunk_data: List[Dict], verbose: int):
    """Save chunk to numbered JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"chunk_{chunk_idx:04d}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=2, ensure_ascii=False)

    if verbose > 0:
        console.print(f"[green]✓ Saved chunk {chunk_idx} to: {output_file}[/green]")


def save_results(save_path: Path, results: List[Dict], verbose: int):
    """Save results to single JSON file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if verbose > 0:
        console.print(f"[green]✓ Results saved to: {save_path}[/green]")


def format_results_for_output(results: Dict[str, Dict[str, Any]]) -> List[Dict]:
    """Convert orchestrator results to output format."""
    return [results[text] for text in results.keys()]


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=False,
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
    "--max-concurrent",
    type=int,
    default=5,
    help="Maximum concurrent requests to server",
)
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Input file (.txt, .csv, .json, .xlsx, .pkl). Reads from stdin if not provided.",
)
@click.option(
    "--input-column",
    type=str,
    default=None,
    help="Column name for text data (auto-detected if not provided)",
)
@click.option(
    "--save-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON file path (single file mode)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for chunked files",
)
@click.option(
    "--chunk-size",
    type=int,
    default=None,
    help="Texts per chunk (requires --output-dir)",
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
    type=click.Choice(["standard", "keyword", "sentiment"]),
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
    help="Enable stem recommendation analysis (requires classification)",
)
@click.option(
    "--enable-stem-polarity",
    is_flag=True,
    help="Enable stem polarity analysis (requires classification)",
)
@click.option(
    "--enable-stem-trends",
    is_flag=True,
    help="Enable stem trend analysis (requires classification)",
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
    help="Max definitions for stem analysis (default: all)",
)
@click.option(
    "--recommendations-only",
    is_flag=True,
    help="Only detect recommendations (no classification)",
)
@click.option(
    "--alerts-only",
    is_flag=True,
    help="Only detect alerts (no classification)",
)
@click.option(
    "--classification-strategy",
    type=click.Choice(["bfs", "bundled"]),
    default="bfs",
    help="Classification strategy: bfs (one node per prompt) or bundled (multiple nodes per prompt)",
)
@click.option(
    "--bundle-size",
    type=int,
    default=4,
    help="Number of nodes to bundle per prompt (only for bundled strategy)",
)
def main(
    config: Path,
    server_url: str,
    model_name: str,
    max_concurrent: int,
    input_file: Path,
    input_column: str,
    save_path: Path,
    output_dir: Path,
    chunk_size: int,
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
    enable_stem_trends: bool,
    enable_trends: bool,
    max_stem_definitions: int,
    recommendations_only: bool,
    alerts_only: bool,
    classification_strategy: str,
    bundle_size: int,
):
    """
    Batch hierarchical text classification using VLLM Server.
    """
    # Validate output options
    if output_dir and chunk_size is None:
        console.print("[red]Error: --chunk-size required with --output-dir[/red]")
        sys.exit(1)

    if chunk_size and output_dir is None:
        console.print("[red]Error: --output-dir required with --chunk-size[/red]")
        sys.exit(1)

    if save_path and output_dir:
        console.print("[red]Error: Cannot use both --save-path and --output-dir[/red]")
        sys.exit(1)

    # Handle standalone modes
    standalone_mode = recommendations_only or alerts_only

    if standalone_mode:
        if recommendations_only:
            enable_recommendations = True
        if alerts_only:
            enable_alerts = True
        enable_classification = False
        enable_stem_recommendations = False
        enable_stem_polarity = False
        enable_stem_trends = False

    # Validate config requirement
    if not standalone_mode and config is None:
        console.print("[red]Error: --config required unless using standalone mode[/red]")
        sys.exit(1)

    # Load texts
    lines = load_input_texts(input_file, input_column, verbose)

    if not lines:
        console.print("[red]No input text provided[/red]")
        sys.exit(1)

    if save_path is None and output_dir is None and verbose == 0:
        console.print("[red]Error: Specify --save-path, --output-dir, or -v[/red]")
        sys.exit(1)

    # Setup chunking
    use_chunked_output = output_dir is not None and chunk_size is not None
    chunks = chunk_list(lines, chunk_size) if use_chunked_output else [lines]

    # Display configuration
    if verbose > 0:
        console.print("[cyan]Configuration:[/cyan]")
        if config:
            console.print(f"  • Config: {config}")
        console.print(f"  • Server URL: {server_url}")
        console.print(f"  • Model: {model_name}")
        console.print(f"  • Texts: {len(lines)}")

        if use_chunked_output:
            console.print(f"  • Output: Chunked ({chunk_size} per file)")
            console.print(f"  • Output dir: {output_dir}")
        elif save_path:
            console.print("  • Output: Single file")
            console.print(f"  • Output path: {save_path}")

        # Display enabled capabilities
        enabled_caps = []
        if enable_classification:
            enabled_caps.append("classification")
        if enable_recommendations:
            enabled_caps.append("recommendations")
        if enable_alerts:
            enabled_caps.append("alerts")
        if enable_stem_recommendations:
            enabled_caps.append("stem_recommendations")
        if enable_stem_polarity:
            enabled_caps.append("stem_polarity")
        if enable_trends:
            enabled_caps.append("trend")
        if enable_stem_trends:
            enabled_caps.append("stem_trend")

        console.print(f"  • Capabilities: {', '.join(enabled_caps)}")

    # Create policy and prompt function
    policy = create_policy(confidence=min_confidence, require_excerpt=require_excerpt)
    prompt_fn = select_prompt(prompt_type)

    start_time = time.time()

    # Initialize processor
    if verbose > 0:
        console.print("\n[cyan]Initializing server processor...[/cyan]")

    with ServerClassificationProcessor(
        config_path=config or Path("dummy.json"),
        server_url=server_url,
        model_name=model_name,
        max_concurrent=max_concurrent,
        prompt_fn=prompt_fn,
        policy=policy,
    ) as processor:
        # Create capability registry
        registry = create_default_registry()

        # SWAP to bundled if requested
        if classification_strategy == "bundled":
            bundled_cap = BundledClassificationCapability(
                bundle_size=bundle_size, policy=policy, separator=">"
            )
        registry.register(bundled_cap)  # Overwrites the BFS one

        # Register stem capabilities if requested
        if enable_stem_recommendations:
            registry.register(
                StemRecommendationsCapability(max_stem_definitions=max_stem_definitions)
            )

        if enable_stem_polarity:
            registry.register(StemPolarityCapability(max_stem_definitions=max_stem_definitions))

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
        if enable_stem_trends:
            capability_names.append("stem_trend")
        if enable_trends:
            capability_names.append("trend")

        # Validate capabilities
        errors = registry.validate_capabilities(capability_names)
        if errors:
            for error in errors:
                console.print(f"[red]Error: {error}[/red]")
            sys.exit(1)

        # Create orchestrator
        orchestrator = CapabilityOrchestrator(
            processor=processor, registry=registry, verbose=verbose
        )

        if verbose > 0:
            console.print("[green]✓ Processor initialized[/green]\n")

        # Initialize timing accumulators
        accumulated_timings = {}

        # Process chunks
        for chunk_idx, chunk_texts in enumerate(chunks):
            if use_chunked_output and verbose > 0:
                console.print(
                    f"\n[cyan]Processing chunk {chunk_idx + 1}/{len(chunks)} "
                    f"({len(chunk_texts)} texts)...[/cyan]"
                )

            # Show capability execution order
            if verbose > 0:
                execution_order = registry.get_execution_order(capability_names)
                console.print(
                    f"[cyan]Capability execution order: {' → '.join(execution_order)}[/cyan]"
                )

            # Execute capabilities with progress bar
            with tqdm(
                total=len(capability_names),
                desc="Processing capabilities",
                disable=verbose == 0,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ) as pbar:
                results = {}
                execution_order = registry.get_execution_order(capability_names)

                for cap_idx, cap_name in enumerate(execution_order):
                    pbar.set_description(f"Processing {cap_name}")

                    # This is a simplified version - you'll need to integrate with orchestrator
                    # For now, let's just call the orchestrator normally
                    if cap_idx == 0:
                        results = orchestrator.execute_capabilities(
                            texts=chunk_texts,
                            capability_names=capability_names,
                            project_name=project_name,
                        )

                    pbar.update(1)

            # Accumulate timings
            chunk_timings = orchestrator.get_timing_summary()
            for cap_name, cap_time in chunk_timings.items():
                accumulated_timings[cap_name] = accumulated_timings.get(cap_name, 0) + cap_time

            # Format for output
            output_data = format_results_for_output(results)

            # Save results
            if use_chunked_output:
                save_chunk_results(output_dir, chunk_idx, output_data, verbose)
            elif save_path and chunk_idx == 0:
                save_results(save_path, output_data, verbose)

        # Final summary
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Format elapsed time
        if elapsed_time < 60:
            time_str = f"{elapsed_time:.2f} seconds"
        elif elapsed_time < 3600:
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            time_str = f"{minutes}m {seconds:.2f}s"
        else:
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = elapsed_time % 60
            time_str = f"{hours}h {minutes}m {seconds:.2f}s"

        # Final summary
        if use_chunked_output:
            console.print(
                f"\n[green]✓ Processed {len(chunks)} chunks ({len(lines)} texts)[/green]"
            )
            console.print(f"[cyan]Results saved to: {output_dir}/[/cyan]")
        elif save_path:
            console.print(f"\n[green]✓ Processed {len(lines)} texts[/green]")

        # Display timing information
        console.print(f"\n[cyan]⏱  Total Execution Time: {time_str}[/cyan]")

        # Display per-capability timing
        if accumulated_timings:
            console.print("\n[cyan]⏱  Time per Capability:[/cyan]")
            total_cap_time = sum(accumulated_timings.values())
            for cap_name, elapsed in sorted(
                accumulated_timings.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (elapsed / total_cap_time * 100) if total_cap_time > 0 else 0
                console.print(
                    f"[cyan]  • {cap_name}: {elapsed:.2f}s ({percentage:.1f}%)[/cyan]"
                )
            console.print(f"[cyan]  • Total: {total_cap_time:.2f}s[/cyan]")


if __name__ == "__main__":
    main()
