#!/usr/bin/env python3
"""
Comparison script for SmolVLA fine-tuned vs reference + ablation variants.

Reads result JSON files and generates comparison tables and plots.
Candidates can use this as a starting point for their analysis.

Usage:
    python compare_models.py \
        --results_dir results/ \
        --output_dir figures/
"""

import argparse
import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plots will be skipped.")
    print("Install with: pip install matplotlib")


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_main_comparison(finetuned: dict, reference: dict) -> None:
    """Print comparison between fine-tuned and reference models."""
    print("\n" + "=" * 70)
    print(f"{'METRIC':<35} {'Fine-tuned':>15} {'Reference':>15}")
    print("=" * 70)

    ft_rate = finetuned["aggregate_success_rate"]
    ref_rate = reference["aggregate_success_rate"]
    print(f"{'Aggregate Success Rate':<35} {ft_rate:>14.1%} {ref_rate:>14.1%}")

    print("-" * 70)
    ft_tasks = finetuned["per_task_success_rate"]
    ref_tasks = reference["per_task_success_rate"]

    all_tasks = sorted(set(list(ft_tasks.keys()) + list(ref_tasks.keys())))
    for task in all_tasks:
        ft_val = ft_tasks.get(task, 0)
        ref_val = ref_tasks.get(task, 0)
        diff = ft_val - ref_val
        marker = " **" if abs(diff) > 0.2 else ""
        print(f"  {task:<33} {ft_val:>14.1%} {ref_val:>14.1%}{marker}")

    print("-" * 70)
    ft_lat = finetuned.get("inference_latency_ms", 0)
    ref_lat = reference.get("inference_latency_ms", 0)
    print(f"{'Inference Latency (ms)':<35} {ft_lat:>15.1f} {ref_lat:>15.1f}")

    ft_mem = finetuned.get("gpu_memory_mb", 0)
    ref_mem = reference.get("gpu_memory_mb", 0)
    print(f"{'GPU Memory (MB)':<35} {ft_mem:>15.0f} {ref_mem:>15.0f}")

    ft_time = finetuned.get("training_time_hours", 0)
    print(f"{'Training Time (hours)':<35} {ft_time:>15.1f} {'N/A':>15}")

    print("=" * 70)
    print("** marks tasks with >20% difference")


def print_ablation_comparison(baseline: dict, ablation_results: list) -> None:
    """Print ablation study comparison table."""
    if not ablation_results:
        return

    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)

    all_models = [("Baseline", baseline)] + ablation_results

    # Header
    header = f"{'Task':<25}"
    for name, _ in all_models:
        short_name = name[:12]
        header += f" {short_name:>12}"
    print(header)
    print("-" * 70)

    # Per-task rows
    baseline_tasks = baseline["per_task_success_rate"]
    all_tasks = sorted(baseline_tasks.keys())

    for task in all_tasks:
        row = f"  {task:<23}"
        for _, result in all_models:
            rate = result.get("per_task_success_rate", {}).get(task, 0)
            row += f" {rate:>11.1%}"
        print(row)

    # Aggregate row
    print("-" * 70)
    row = f"  {'AGGREGATE':<23}"
    for _, result in all_models:
        rate = result.get("aggregate_success_rate", 0)
        row += f" {rate:>11.1%}"
    print(row)

    print("=" * 70)


def plot_main_comparison(finetuned: dict, reference: dict, output_dir: Path) -> None:
    """Generate comparison plots for fine-tuned vs reference."""
    if not HAS_MATPLOTLIB:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    ft_tasks = finetuned["per_task_success_rate"]
    ref_tasks = reference["per_task_success_rate"]
    all_tasks = sorted(set(list(ft_tasks.keys()) + list(ref_tasks.keys())))

    ft_rates = [ft_tasks.get(t, 0) for t in all_tasks]
    ref_rates = [ref_tasks.get(t, 0) for t in all_tasks]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(all_tasks))
    width = 0.35

    ax.bar(x - width / 2, ft_rates, width, label="Fine-tuned", color="#2ECC71")
    ax.bar(x + width / 2, ref_rates, width, label="Reference", color="#3498DB")

    ax.set_xlabel("Task")
    ax.set_ylabel("Success Rate")
    ax.set_title("LIBERO-Spatial: Fine-tuned SmolVLA vs Reference Checkpoint")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("task_", "T") for t in all_tasks],
                       rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / "finetuned_vs_reference.png", dpi=150)
    print(f"Saved: {output_dir / 'finetuned_vs_reference.png'}")
    plt.close()


def plot_ablation(baseline: dict, ablation_results: list, output_dir: Path) -> None:
    """Generate ablation study plots."""
    if not HAS_MATPLOTLIB or not ablation_results:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    all_models = [("Baseline", baseline)] + ablation_results
    names = [name for name, _ in all_models]
    rates = [r.get("aggregate_success_rate", 0) for _, r in all_models]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = ax.bar(names, rates, color=colors)

    ax.set_ylabel("Aggregate Success Rate")
    ax.set_title("Ablation Study: Effect on LIBERO-Spatial Performance")
    ax.set_ylim(0, 1.05)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.1%}", ha="center", va="bottom", fontsize=10)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_results.png", dpi=150)
    print(f"Saved: {output_dir / 'ablation_results.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare SmolVLA results")
    parser.add_argument(
        "--results_dir", type=str, default="results/",
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--output_dir", type=str, default="figures/",
        help="Directory to save plots (default: figures/)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load main results
    finetuned_path = results_dir / "smolvla_finetuned_results.json"
    reference_path = results_dir / "smolvla_reference_results.json"

    if finetuned_path.exists() and reference_path.exists():
        finetuned = load_results(str(finetuned_path))
        reference = load_results(str(reference_path))
        print_main_comparison(finetuned, reference)
        plot_main_comparison(finetuned, reference, Path(args.output_dir))
    else:
        print("Missing main result files. Skipping main comparison.")
        finetuned = None

    # Load ablation results
    ablation_files = sorted(results_dir.glob("ablation_*.json"))
    ablation_results = []
    for f in ablation_files:
        name = f.stem.replace("ablation_", "")
        ablation_results.append((name, load_results(str(f))))

    if finetuned and ablation_results:
        print_ablation_comparison(finetuned, ablation_results)
        plot_ablation(finetuned, ablation_results, Path(args.output_dir))
    elif ablation_results:
        print(f"\nFound {len(ablation_results)} ablation files but no baseline.")


if __name__ == "__main__":
    main()
