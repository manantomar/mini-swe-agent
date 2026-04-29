#!/usr/bin/env python3

"""Self-improvement loop across multiple SWE-bench tasks.

Runs all tasks at baseline, critiques the traces together to produce
a small consolidated set of tips, then re-runs with improved prompts.

Usage:
    python -m minisweagent.run.self_improve_multi \
        --instance-ids psf__requests-1766 django__django-10097 django__django-11433 \
        -m qwen/qwen3-8b --model-class openrouter -n 2
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any

import litellm
import typer
import yaml
from datasets import load_dataset

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import get_sb_environment
from minisweagent.run.critique import format_trace_compact, inject_tips_into_config
from minisweagent.utils.serialize import recursive_merge

logger = logging.getLogger("minisweagent.self_improve_multi")

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

MULTI_CRITIQUE_PROMPT = """\
You are analyzing trajectories from a coding agent that attempted {n_tasks} different tasks.

{traces_block}

Based on ALL these traces, identify the 3-5 most common and impactful mistakes \
the agent makes. Output a short bulleted list of actionable tips that should be \
added to the agent's prompt to improve its performance across diverse tasks.

Rules:
- Tips must be general (applicable to many tasks), not specific to one task
- Each tip must be one sentence, concrete and actionable
- Focus on patterns you see repeated across multiple traces
- Output ONLY the bulleted tips, nothing else"""


def run_agent_on_instance(config: dict, instance: dict, output_path: Path) -> dict[str, Any]:
    """Run the agent on a single SWE-bench instance."""
    run_config = copy.deepcopy(config)
    env = get_sb_environment(run_config, instance)
    model = get_model(config=run_config.get("model", {}))
    agent = DefaultAgent(model, env, output_path=output_path, **run_config.get("agent", {}))
    result = agent.run(instance["problem_statement"])
    data = agent.save(output_path, {"instance_id": instance["instance_id"]})
    return {"result": result, "trajectory": data}


def run_batch(config: dict, instances: list[dict], iter_dir: Path) -> list[dict]:
    """Run agent on all instances sequentially, return list of results."""
    results = []
    for inst in instances:
        iid = inst["instance_id"]
        traj_path = iter_dir / iid / "trajectory.json"
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"    {iid}...", end=" ", flush=True)
        try:
            attempt = run_agent_on_instance(config, inst, traj_path)
            trajectory = attempt["trajectory"]
            exit_status = attempt["result"].get("exit_status", "")
            submission = attempt["result"].get("submission", "")
            n_steps = len([m for m in trajectory["messages"] if m.get("role") == "assistant"])
            cost = trajectory["info"]["model_stats"]["instance_cost"]
            if submission:
                (iter_dir / iid / "patch.diff").write_text(submission)
            print(f"{exit_status} | steps={n_steps} | ${cost:.4f}")
            results.append({
                "instance_id": iid,
                "exit_status": exit_status,
                "has_submission": bool(submission),
                "n_steps": n_steps,
                "cost": cost,
                "trajectory": trajectory,
            })
        except Exception as e:
            print(f"CRASH: {e}")
            results.append({
                "instance_id": iid,
                "exit_status": f"crash:{type(e).__name__}",
                "has_submission": False,
                "n_steps": 0,
                "cost": 0,
                "trajectory": None,
            })
    return results


def critique_all_traces(results: list[dict], *, model_name: str) -> str:
    """Critique multiple traces together, return consolidated tips."""
    trace_blocks = []
    for r in results:
        if r["trajectory"] is None:
            continue
        compact = format_trace_compact(r["trajectory"]["messages"])
        # Truncate each trace to keep context manageable
        if len(compact) > 4000:
            compact = compact[:2000] + "\n...(truncated)...\n" + compact[-2000:]
        status = "SUCCEEDED" if r["has_submission"] else "FAILED"
        trace_blocks.append(f"### Task: {r['instance_id']} ({status}, {r['n_steps']} steps)\n\n{compact}")

    traces_block = "\n\n---\n\n".join(trace_blocks)
    prompt = MULTI_CRITIQUE_PROMPT.format(n_tasks=len(trace_blocks), traces_block=traces_block)

    response = litellm.completion(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


# fmt: off
@app.command()
def main(
    instance_ids: list[str] = typer.Option(..., "--instance-ids", help="SWE-bench instance IDs"),
    subset: str = typer.Option("verified", "--subset", help="SWE-bench subset"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    output_dir: Path = typer.Option("self_improve_multi", "-o", "--output", help="Output directory"),
    max_iterations: int = typer.Option(2, "-n", "--max-iterations", help="Max improvement iterations"),
    config_spec: list[str] = typer.Option(["swebench.yaml"], "-c", "--config", help="Config file(s)"),
    critique_model: str = typer.Option("openrouter/qwen/qwen3-coder", "--critique-model", help="Critique model"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Coder model"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class"),
) -> None:
    # fmt: on
    """Self-improvement loop across multiple SWE-bench tasks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset_mapping = {"verified": "princeton-nlp/SWE-Bench_Verified", "lite": "princeton-nlp/SWE-Bench_Lite"}
    dataset_path = dataset_mapping.get(subset, subset)
    print(f"Loading {dataset_path}...")
    all_instances = {inst["instance_id"]: inst for inst in load_dataset(dataset_path, split=split)}
    instances = []
    for iid in instance_ids:
        if iid not in all_instances:
            raise ValueError(f"Instance {iid} not found")
        instances.append(all_instances[iid])
    print(f"Tasks: {[i['instance_id'] for i in instances]}")

    # Build config
    configs = [get_config_from_spec(spec) for spec in config_spec]
    if model_name or model_class:
        model_override: dict = {}
        if model_name:
            model_override["model_name"] = model_name
        if model_class:
            model_override["model_class"] = model_class
        configs.append({"model": model_override})
    configs.append({"agent": {"step_limit": 50, "cost_limit": 1.0}, "model": {"cost_tracking": "ignore_errors"}})
    config = recursive_merge(*configs)
    original_template = config.get("agent", {}).get("instance_template", "")

    all_iteration_results = []

    for iteration in range(max_iterations):
        iter_dir = output_dir / f"v{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Iteration {iteration}")
        print(f"{'='*60}")

        (iter_dir / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))

        # Run all tasks
        results = run_batch(config, instances, iter_dir)
        all_iteration_results.append(results)

        # Save per-iteration summary
        summary = [{k: v for k, v in r.items() if k != "trajectory"} for r in results]
        (iter_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        total_steps = sum(r["n_steps"] for r in results)
        total_cost = sum(r["cost"] for r in results)
        submitted = sum(1 for r in results if r["has_submission"])
        print(f"\n  Totals: {submitted}/{len(results)} submitted | {total_steps} steps | ${total_cost:.4f}")

        if iteration >= max_iterations - 1:
            break

        # Critique ALL traces together
        print("\n  Critiquing all traces together...")
        tips = critique_all_traces(results, model_name=critique_model)
        (iter_dir / "tips.txt").write_text(tips)
        print(f"  Consolidated tips:\n{tips}")

        # Inject tips for next round
        config["agent"]["instance_template"] = original_template
        config = inject_tips_into_config(config, tips)

    # Final comparison table
    print(f"\n{'='*60}")
    print("  Comparison across iterations")
    print(f"{'='*60}")
    for iid in instance_ids:
        print(f"\n  {iid}:")
        for i, results in enumerate(all_iteration_results):
            r = next(r for r in results if r["instance_id"] == iid)
            status = "✓" if r["has_submission"] else "✗"
            print(f"    v{i}: {status} {r['exit_status']:20s} steps={r['n_steps']:3d}  ${r['cost']:.4f}")

    print(f"\nResults: {output_dir}/")


if __name__ == "__main__":
    app()
