#!/usr/bin/env python3
"""
Test script for BundledClassificationCapability
"""

import json
from pathlib import Path

import click
from rich.console import Console

from classifier import BundledClassificationCapability, ServerClassificationProcessor
from classifier.policies import DefaultPolicy

console = Console()


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
    "--bundle-size",
    type=int,
    default=4,
    help="Number of nodes to bundle per prompt",
)
@click.option(
    "--save-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON file path",
)
def main(config: Path, server_url: str, bundle_size: int, save_path: Path):
    """Test bundled classification capability."""

    # Sample texts
    texts = [
        "The instructor's teaching style was excellent with clear explanations and engaging delivery.",
        "The course content was relevant but the assessments were too difficult.",
        "The hands-on exercises were great, but the lecture portions were too long.",
    ]

    console.print("[cyan]Testing BundledClassificationCapability[/cyan]")
    console.print(f"[cyan]Bundle size: {bundle_size}[/cyan]")
    console.print(f"[cyan]Texts: {len(texts)}[/cyan]\n")

    # Initialize processor
    with ServerClassificationProcessor(
        config_path=config,
        server_url=server_url,
        policy=DefaultPolicy(),
        separator=">",
    ) as processor:
        # Replace with bundled capability
        processor.classification_capability = BundledClassificationCapability(
            bundle_size=bundle_size, policy=DefaultPolicy(), separator=">"
        )

        # Run classification
        console.print("[cyan]Running bundled classification...[/cyan]")
        results = processor.classification_capability.execute_classification(
            texts=texts, hierarchy=processor.topic_hierarchy, processor=processor.llm_processor
        )

        # Display results
        for text, output in results.items():
            console.print(f"\n[bold]Text:[/bold] {text[:80]}...")
            console.print(f"[bold]Paths:[/bold] {output.classification_paths}")

            if output.node_results:
                console.print("[bold]Node Results:[/bold]")
                for node_name, result in output.node_results.items():
                    status = "✓" if result.is_relevant else "✗"
                    console.print(
                        f"  {status} {node_name}: confidence={result.confidence}, "
                        f"excerpt='{result.excerpt[:50]}...'"
                    )

        # Save if requested
        if save_path:
            output_data = []
            for text, output in results.items():
                output_data.append(
                    {
                        "text": text,
                        "classification_paths": output.classification_paths,
                        "node_results": {
                            name: result.model_dump()
                            for name, result in output.node_results.items()
                        },
                    }
                )

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            console.print(f"\n[green]✓ Results saved to: {save_path}[/green]")

    console.print("\n[green]✓ Test complete![/green]")


if __name__ == "__main__":
    main()
