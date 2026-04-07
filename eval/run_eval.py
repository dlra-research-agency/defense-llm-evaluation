"""
Defense LLM Evaluation Runner

Main evaluation pipeline for benchmarking LLMs on defense intelligence tasks.
Loads YAML configurations, runs model inference, computes metrics, and saves results.

Usage:
    python eval/run_eval.py --config configs/threat-classification.yaml
    python eval/run_eval.py --config configs/entity-extraction.yaml --output results/
    python eval/run_eval.py --config configs/*.yaml --verbose
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from tqdm import tqdm

from eval.models import ModelClient
from eval.scoring import aggregate_results, compute_entity_f1, compute_f1, compute_rouge

console = Console()
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate a YAML evaluation configuration.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If required configuration fields are missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    required_fields = ["task", "model_configs", "evaluation"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in {config_path}")

    return config


def load_samples(task_name: str, n_samples: int) -> list[dict[str, Any]]:
    """Load synthetic evaluation samples for a given task.

    Args:
        task_name: Name of the evaluation task.
        n_samples: Number of samples to load.

    Returns:
        List of sample dictionaries with 'input' and 'expected' keys.
    """
    samples_path = Path(f"data/samples/{task_name}.jsonl")
    samples = []

    if samples_path.exists():
        with open(samples_path, "r") as f:
            for i, line in enumerate(f):
                if i >= n_samples:
                    break
                samples.append(json.loads(line.strip()))
    else:
        logger.warning(f"No samples found at {samples_path}. Using synthetic generation.")
        samples = _generate_synthetic_samples(task_name, n_samples)

    return samples[:n_samples]


def _generate_synthetic_samples(task_name: str, n_samples: int) -> list[dict[str, Any]]:
    """Generate placeholder synthetic samples for evaluation.

    Args:
        task_name: Name of the evaluation task.
        n_samples: Number of samples to generate.

    Returns:
        List of synthetic sample dictionaries.
    """
    # Placeholder: in production, this calls the synthetic data pipeline
    return [{"input": f"Sample {i} for {task_name}", "expected": {}} for i in range(n_samples)]


def load_prompt_template(task_name: str) -> str:
    """Load the prompt template for a given task.

    Args:
        task_name: Name of the evaluation task.

    Returns:
        Prompt template string with {text} placeholder.
    """
    prompt_path = Path(f"prompts/{task_name}.txt")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    with open(prompt_path, "r") as f:
        return f.read().strip()


def run_evaluation(config: dict[str, Any], output_dir: str, verbose: bool = False) -> dict[str, Any]:
    """Execute the full evaluation pipeline for a single configuration.

    Args:
        config: Parsed YAML configuration dictionary.
        output_dir: Directory to save results.
        verbose: Whether to log detailed progress.

    Returns:
        Dictionary containing evaluation results for all models.
    """
    task_name = config["task"]["name"]
    n_samples = config["evaluation"]["n_samples"]
    eval_mode = config["evaluation"].get("mode", "zero-shot")

    console.print(f"\n[bold]Running evaluation:[/bold] {task_name}")
    console.print(f"  Mode: {eval_mode} | Samples: {n_samples}")

    prompt_template = load_prompt_template(task_name)
    samples = load_samples(task_name, n_samples)

    all_results = {
        "task": config["task"],
        "evaluation_mode": eval_mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(samples),
        "models": {},
    }

    for model_config in config["model_configs"]:
        model_id = f"{model_config['provider']}/{model_config['model']}"
        console.print(f"\n  Evaluating: [cyan]{model_id}[/cyan]")

        client = ModelClient(
            provider=model_config["provider"],
            model=model_config["model"],
            temperature=model_config.get("temperature", 0.0),
            max_tokens=model_config.get("max_tokens", 256),
        )

        predictions = []
        total_tokens = 0

        for sample in tqdm(samples, desc=f"  {model_id}", disable=not verbose):
            prompt = prompt_template.replace("{text}", sample["input"])
            response = client.generate(prompt)
            predictions.append(response)
            total_tokens += client.last_token_count

        # Compute metrics based on task type
        metrics = config["evaluation"]["metrics"]
        scores = _compute_task_metrics(task_name, predictions, samples, metrics)

        all_results["models"][model_id] = {
            "scores": scores,
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": total_tokens / len(samples) if samples else 0,
        }

        # Display scores
        for metric, value in scores.items():
            console.print(f"    {metric}: [green]{value:.4f}[/green]")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"{task_name}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    console.print(f"\n  Results saved to: [blue]{output_path}[/blue]")
    return all_results


def _compute_task_metrics(
    task_name: str,
    predictions: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    metrics: list[str],
) -> dict[str, float]:
    """Compute task-specific metrics for a set of predictions.

    Args:
        task_name: Name of the evaluation task.
        predictions: List of model predictions.
        samples: List of ground-truth samples.
        metrics: List of metric names to compute.

    Returns:
        Dictionary mapping metric names to scores.
    """
    scores = {}

    if task_name == "threat-classification":
        expected = [s["expected"] for s in samples]
        scores = compute_f1(predictions, expected, metrics)
    elif task_name == "entity-extraction":
        expected = [s["expected"] for s in samples]
        scores = compute_entity_f1(predictions, expected, metrics)
    elif task_name == "maritime-analysis":
        references = [s["expected"].get("reference", "") for s in samples]
        scores = compute_rouge(predictions, references, metrics)

    return scores


def display_summary(all_results: list[dict[str, Any]]) -> None:
    """Display a summary table of all evaluation results.

    Args:
        all_results: List of result dictionaries from each evaluation run.
    """
    table = Table(title="Evaluation Summary")
    table.add_column("Model", style="cyan")
    table.add_column("Task", style="white")
    table.add_column("Primary Metric", style="green")
    table.add_column("Score", justify="right", style="bold")

    for result in all_results:
        task = result["task"]["name"]
        for model_id, model_data in result["models"].items():
            primary_metric = list(model_data["scores"].keys())[0]
            score = model_data["scores"][primary_metric]
            table.add_row(model_id, task, primary_metric, f"{score:.4f}")

    console.print(table)


def main() -> None:
    """CLI entrypoint for the evaluation runner."""
    parser = argparse.ArgumentParser(description="Defense LLM Evaluation Framework")
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="Path(s) to YAML evaluation configuration file(s)",
    )
    parser.add_argument(
        "--output",
        default="results/",
        help="Directory to save evaluation results (default: results/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging and progress bars",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    all_results = []
    for config_path in args.config:
        config = load_config(config_path)
        result = run_evaluation(config, args.output, verbose=args.verbose)
        all_results.append(result)

    if len(all_results) > 1:
        display_summary(all_results)

    aggregated = aggregate_results(all_results)
    agg_path = Path(args.output) / "aggregated_results.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    console.print(f"\n[bold green]Evaluation complete.[/bold green] Aggregated results: {agg_path}")


if __name__ == "__main__":
    main()
