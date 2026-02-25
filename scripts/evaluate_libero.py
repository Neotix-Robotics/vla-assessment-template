#!/usr/bin/env python3
"""
Evaluation helper for SmolVLA on LIBERO-Spatial.

This wraps the LeRobot evaluation CLI and converts the output
into the JSON format required for automated scoring.

Usage:
    # Evaluate your fine-tuned model
    python evaluate_libero.py \
        --checkpoint_path outputs/train/my_smolvla/checkpoints/last/pretrained_model \
        --model_name smolvla_finetuned \
        --output_path results/smolvla_finetuned_results.json

    # Evaluate the reference checkpoint
    python evaluate_libero.py \
        --checkpoint_path HuggingFaceVLA/smolvla_libero \
        --model_name smolvla_reference \
        --output_path results/smolvla_reference_results.json

NOTE: This is a helper script. Candidates may use `lerobot-eval` directly
and convert results manually. What matters is the output JSON format.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def run_lerobot_eval(
    checkpoint_path: str,
    task: str = "libero_spatial",
    n_episodes: int = 10,
    batch_size: int = 1,
) -> dict:
    """
    Run lerobot-eval and capture results.

    NOTE TO CANDIDATES:
    This function calls lerobot-eval via subprocess. You may need to adjust
    the command based on your LeRobot version and environment setup.
    Alternatively, you can use the LeRobot Python API directly.
    """
    cmd = [
        "lerobot-eval",
        f"--policy.path={checkpoint_path}",
        "--env.type=libero",
        f"--env.task={task}",
        f"--eval.batch_size={batch_size}",
        f"--eval.n_episodes={n_episodes}",
    ]

    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=14400,  # 4 hour timeout
    )

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"ERROR: lerobot-eval failed:\n{result.stderr}")
        raise RuntimeError(f"Evaluation failed with return code {result.returncode}")

    print(f"Evaluation completed in {elapsed / 60:.1f} minutes")
    print(result.stdout)

    # Parse results from lerobot-eval output
    # NOTE: The exact output format depends on your LeRobot version.
    # You may need to parse the output directory or wandb logs instead.
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed_seconds": elapsed,
    }


def build_results_json(
    model_name: str,
    per_task_success_rate: dict,
    inference_latency_ms: float,
    gpu_memory_mb: float,
    training_time_hours: float,
    training_steps: int,
    batch_size: int,
    n_episodes_per_task: int,
    checkpoint_path: str,
    hardware: str,
    notes: str = "",
) -> dict:
    """Build the standardized results JSON."""
    rates = list(per_task_success_rate.values())
    aggregate = sum(rates) / len(rates) if rates else 0.0

    return {
        "model": model_name,
        "libero_suite": "spatial",
        "per_task_success_rate": per_task_success_rate,
        "aggregate_success_rate": round(aggregate, 4),
        "inference_latency_ms": round(inference_latency_ms, 2),
        "gpu_memory_mb": round(gpu_memory_mb, 1),
        "training_time_hours": round(training_time_hours, 2),
        "training_steps": training_steps,
        "batch_size": batch_size,
        "num_eval_episodes_per_task": n_episodes_per_task,
        "checkpoint_path": checkpoint_path,
        "hardware": hardware,
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SmolVLA on LIBERO-Spatial and output scoring JSON"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to model checkpoint (local or HuggingFace ID)",
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        choices=["smolvla_finetuned", "smolvla_reference"],
        help="Model identifier for the results JSON",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to save the results JSON",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=10,
        help="Number of evaluation episodes per task (default: 10)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Evaluation batch size (default: 1)",
    )
    parser.add_argument(
        "--training_time_hours", type=float, default=0.0,
        help="Training time in hours (fill manually for fine-tuned model)",
    )
    parser.add_argument(
        "--training_steps", type=int, default=0,
        help="Number of training steps used",
    )
    parser.add_argument(
        "--training_batch_size", type=int, default=0,
        help="Batch size used during training",
    )
    parser.add_argument(
        "--hardware", type=str, default="unknown",
        help="Hardware description (e.g., 'NVIDIA T4 16GB')",
    )
    args = parser.parse_args()

    print(f"=" * 50)
    print(f"Evaluating: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Episodes per task: {args.n_episodes}")
    print(f"=" * 50)

    # Run evaluation
    raw_results = run_lerobot_eval(
        checkpoint_path=args.checkpoint_path,
        n_episodes=args.n_episodes,
        batch_size=args.batch_size,
    )

    # TODO (candidate): Parse the lerobot-eval output to extract per-task success rates.
    # The exact parsing depends on your LeRobot version.
    # You may need to:
    #   1. Read the eval output directory (check --output_dir flag)
    #   2. Parse wandb logs
    #   3. Extract from stdout
    #
    # As a fallback, you can manually fill in the per-task results:
    per_task_success_rate = {
        "task_0": 0.0,
        "task_1": 0.0,
        "task_2": 0.0,
        "task_3": 0.0,
        "task_4": 0.0,
        "task_5": 0.0,
        "task_6": 0.0,
        "task_7": 0.0,
        "task_8": 0.0,
        "task_9": 0.0,
    }

    # TODO (candidate): Measure inference latency and GPU memory
    inference_latency_ms = 0.0
    gpu_memory_mb = 0.0

    results = build_results_json(
        model_name=args.model_name,
        per_task_success_rate=per_task_success_rate,
        inference_latency_ms=inference_latency_ms,
        gpu_memory_mb=gpu_memory_mb,
        training_time_hours=args.training_time_hours,
        training_steps=args.training_steps,
        batch_size=args.training_batch_size,
        n_episodes_per_task=args.n_episodes,
        checkpoint_path=args.checkpoint_path,
        hardware=args.hardware,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Aggregate success rate: {results['aggregate_success_rate']:.1%}")


if __name__ == "__main__":
    main()
