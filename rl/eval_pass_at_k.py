#!/usr/bin/env python3
"""Compute pass@k for an agent on SWE-bench tasks.

Generates n rollouts per task (with temperature > 0 for diversity),
evaluates all patches, and computes the unbiased pass@k estimator.

Usage:
    # Generate rollouts + evaluate + compute pass@k:
    python rl/eval_pass_at_k.py \
        --model-path /data/.../qwen3-8b-dro5-merged \
        --n-rollouts 5 \
        --tasks 50 \
        --step-limit 100 \
        --workers 50

    # Just compute pass@k from existing results:
    python rl/eval_pass_at_k.py --results-dir /data/.../passk-run --compute-only
"""

import argparse
import json
import logging
import math
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pass_at_k")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator. n=total, c=correct, k=samples."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def generate_rollouts(
    model_path: str,
    vllm_url: str,
    tasks_file: str,
    n_tasks: int,
    n_rollouts: int,
    step_limit: int,
    workers: int,
    output_dir: Path,
    temperature: float = 0.7,
):
    """Generate n rollouts per task, all in parallel.

    Launches all rollouts concurrently as subprocesses. Each rollout writes to
    its own directory. If vllm_url contains commas, rollouts are round-robined
    across the URLs.
    """
    import os
    vllm_urls = [u.strip() for u in vllm_url.split(",")]
    data = json.loads(Path(tasks_file).read_text())
    task_ids = data["ids"][:n_tasks]
    task_filter = "^(" + "|".join(task_ids) + ")$"

    # Determine which rollouts still need to run
    pending: list[int] = []
    for rollout in range(n_rollouts):
        rollout_dir = output_dir / f"rollout_{rollout:02d}"
        if rollout_dir.exists() and list(rollout_dir.glob("*/*.traj.json")):
            n_existing = len(list(rollout_dir.glob("*/*.traj.json")))
            logger.info(f"Rollout {rollout}: {n_existing} traces already exist, skipping")
        else:
            pending.append(rollout)

    if not pending:
        logger.info("All rollouts already exist")
        return

    logger.info(f"═══ Launching {len(pending)} rollouts in parallel (workers={workers}) ═══")
    t0 = time.time()

    # Launch all rollouts as concurrent subprocesses
    procs: dict[int, subprocess.Popen] = {}
    for rollout in pending:
        rollout_dir = output_dir / f"rollout_{rollout:02d}"
        rollout_dir.mkdir(parents=True, exist_ok=True)

        # Distribute workers across rollouts, minimum 1 per rollout
        workers_per_rollout = max(1, workers // len(pending))

        # Round-robin across vLLM servers
        url = vllm_urls[rollout % len(vllm_urls)]

        cmd = [
            "mini-extra", "swebench",
            "--subset", "verified", "--split", "test",
            "--filter", task_filter,
            "-m", model_path,
            "--model-class", "vllm",
            "-o", str(rollout_dir),
            "-w", str(workers_per_rollout),
            "-c", "swebench.yaml",
            "-c", f"agent.step_limit={step_limit}",
            "-c", "agent.cost_limit=100",
            "-c", "model.cost_tracking=ignore_errors",
            "-c", f"model.api_base={url}",
            "-c", "environment.pull_timeout=300",
            "-c", f"model.model_kwargs.temperature={temperature}",
        ]
        logger.info(f"  rollout {rollout}: {url} -w {workers_per_rollout}")
        procs[rollout] = subprocess.Popen(
            cmd, env={**os.environ, "MSWEA_SILENT_STARTUP": "1"},
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )

    # Wait for all to complete, logging as each finishes
    while procs:
        for rollout, proc in list(procs.items()):
            ret = proc.poll()
            if ret is not None:
                rollout_dir = output_dir / f"rollout_{rollout:02d}"
                n_trajs = len(list(rollout_dir.glob("*/*.traj.json"))) if rollout_dir.exists() else 0
                elapsed = time.time() - t0
                logger.info(f"  Rollout {rollout} done: {n_trajs} traces ({elapsed:.0f}s elapsed)")
                if ret != 0:
                    stderr = proc.stderr.read() if proc.stderr else ""
                    logger.warning(f"  Rollout {rollout} errors: {stderr[-300:]}")
                del procs[rollout]
        if procs:
            time.sleep(10)

    wall = time.time() - t0
    total = sum(
        len(list((output_dir / f"rollout_{r:02d}").glob("*/*.traj.json")))
        for r in range(n_rollouts)
        if (output_dir / f"rollout_{r:02d}").exists()
    )
    logger.info(f"═══ All rollouts complete: {total} total traces in {wall:.0f}s ═══")


def evaluate_rollouts(output_dir: Path, eval_workers: int = 8):
    """Evaluate patches from all rollouts."""
    for rollout_dir in sorted(output_dir.glob("rollout_*")):
        preds_path = rollout_dir / "preds.json"
        report_path = rollout_dir / "eval_report.json"

        if report_path.exists():
            logger.info(f"  {rollout_dir.name}: eval report exists, skipping")
            continue

        if not preds_path.exists():
            logger.warning(f"  {rollout_dir.name}: no preds.json, skipping")
            continue

        preds = json.loads(preds_path.read_text())
        n_with_patches = sum(1 for v in preds.values() if v.get("model_patch", "").strip())

        if n_with_patches == 0:
            report = {"resolved_ids": [], "submitted_ids": list(preds.keys())}
            report_path.write_text(json.dumps(report, indent=2))
            logger.info(f"  {rollout_dir.name}: 0 patches, skipping eval")
            continue

        logger.info(f"  {rollout_dir.name}: evaluating {n_with_patches} patches...")
        instance_ids = list(preds.keys())

        try:
            from swebench import run_evaluation
            eval_report_dir = rollout_dir / "sb_reports"
            eval_report_dir.mkdir(exist_ok=True)
            run_evaluation(
                dataset_name="princeton-nlp/SWE-bench_Verified",
                split="test",
                instance_ids=instance_ids,
                predictions_path=str(preds_path),
                max_workers=eval_workers,
                force_rebuild=False,
                cache_level="instance",
                clean=False,
                open_file_limit=4096,
                run_id=rollout_dir.name,
                timeout=900,
                namespace=None,
                rewrite_reports=False,
                modal=False,
                report_dir=str(eval_report_dir),
            )

            resolved_ids = []
            for rf in eval_report_dir.glob("*.json"):
                try:
                    r = json.loads(rf.read_text())
                    resolved_ids.extend(r.get("resolved_ids", []))
                except Exception:
                    pass
            report = {"resolved_ids": list(set(resolved_ids)), "submitted_ids": instance_ids}
        except Exception as e:
            logger.error(f"  Eval failed: {e}")
            report = {"resolved_ids": [], "submitted_ids": instance_ids}

        report_path.write_text(json.dumps(report, indent=2))
        logger.info(f"  {rollout_dir.name}: {len(report['resolved_ids'])}/{len(instance_ids)} resolved")


def compute_pass_at_k_results(output_dir: Path, k_values: list[int] | None = None):
    """Compute pass@k from evaluation results across rollouts."""
    # Collect per-task results across rollouts
    task_results: dict[str, dict] = {}  # task_id -> {"n": int, "c": int}

    rollout_dirs = sorted(output_dir.glob("rollout_*"))
    if not rollout_dirs:
        logger.error("No rollout directories found")
        return {}

    n_rollouts = len(rollout_dirs)
    all_resolved = set()

    for rollout_dir in rollout_dirs:
        report_path = rollout_dir / "eval_report.json"
        if not report_path.exists():
            continue

        report = json.loads(report_path.read_text())
        resolved = set(report.get("resolved_ids", []))
        submitted = set(report.get("submitted_ids", []))
        all_resolved |= resolved

        for task_id in submitted:
            if task_id not in task_results:
                task_results[task_id] = {"n": 0, "c": 0}
            task_results[task_id]["n"] += 1
            if task_id in resolved:
                task_results[task_id]["c"] += 1

    if not task_results:
        logger.error("No task results found")
        return {}

    if k_values is None:
        k_values = [k for k in [1, 2, 3, 5, 10] if k <= n_rollouts]

    n_tasks = len(task_results)
    results = {}

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                     pass@k Results                         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Tasks:    {n_tasks}")
    print(f"  Rollouts: {n_rollouts}")
    print(f"  Unique tasks solved (across all rollouts): {len(all_resolved)}")
    print()

    for k in k_values:
        task_pass_k = []
        for task_id, r in task_results.items():
            if r["n"] >= k:
                task_pass_k.append(pass_at_k(r["n"], r["c"], k))
        if task_pass_k:
            avg = sum(task_pass_k) / len(task_pass_k)
            results[f"pass@{k}"] = avg
            bar_len = int(avg * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"  pass@{k:<3}  {bar}  {avg*100:5.1f}%  ({len(task_pass_k)} tasks)")

    # Per-task breakdown
    print()
    print("  Per-task breakdown (tasks with ≥1 success):")
    for task_id in sorted(all_resolved):
        r = task_results[task_id]
        print(f"    {task_id}: {r['c']}/{r['n']} solved")

    # Save results
    results_path = output_dir / "pass_at_k_results.json"
    full_results = {
        "n_tasks": n_tasks,
        "n_rollouts": n_rollouts,
        "unique_solved": len(all_resolved),
        "pass_at_k": results,
        "per_task": {tid: r for tid, r in sorted(task_results.items())},
    }
    results_path.write_text(json.dumps(full_results, indent=2))
    print(f"\n  Results saved to: {results_path}")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute pass@k for SWE-bench agent")
    parser.add_argument("--model-path", default="/data/manantomar/swe-bench-docker/rl-training/qwen3-8b-dro5-merged")
    parser.add_argument("--vllm-url", default="http://localhost:8234/v1")
    parser.add_argument("--tasks-file", default="eval/50_new_instances.json")
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--n-rollouts", type=int, default=5)
    parser.add_argument("--step-limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--results-dir", default="/data/manantomar/swe-bench-docker/passk-base")
    parser.add_argument("--compute-only", action="store_true", help="Skip generation, just compute pass@k")
    parser.add_argument("--eval-only", action="store_true", help="Skip generation, eval + compute")
    parser.add_argument("--eval-workers", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.compute_only:
        if not args.eval_only:
            logger.info("═══ Phase 1: Generating rollouts ═══")
            generate_rollouts(
                model_path=args.model_path,
                vllm_url=args.vllm_url,
                tasks_file=args.tasks_file,
                n_tasks=args.tasks,
                n_rollouts=args.n_rollouts,
                step_limit=args.step_limit,
                workers=args.workers,
                output_dir=output_dir,
                temperature=args.temperature,
            )

        logger.info("═══ Phase 2: Evaluating patches ═══")
        evaluate_rollouts(output_dir, eval_workers=args.eval_workers)

    logger.info("═══ Phase 3: Computing pass@k ═══")
    compute_pass_at_k_results(output_dir)


if __name__ == "__main__":
    main()
