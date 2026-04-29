#!/usr/bin/env python3

"""Self-improvement loop on a single SWE-bench instance.

Usage:
    python -m minisweagent.run.self_improve_swebench \
        --instance-id psf__requests-1766 \
        -m qwen/qwen3-8b \
        --model-class openrouter \
        -n 3
"""

import copy
import json
import logging
from pathlib import Path

import typer
import yaml
from datasets import load_dataset

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import get_sb_environment
from minisweagent.run.critique import format_trace_compact, get_critique_tips, inject_tips_into_config
from minisweagent.utils.serialize import recursive_merge

logger = logging.getLogger("minisweagent.self_improve_swebench")

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def run_agent_on_instance(config: dict, instance: dict, output_path: Path) -> dict:
    """Run the agent on a single SWE-bench instance, return trajectory data."""
    run_config = copy.deepcopy(config)
    env = get_sb_environment(run_config, instance)
    model = get_model(config=run_config.get("model", {}))
    agent = DefaultAgent(model, env, output_path=output_path, **run_config.get("agent", {}))
    task = instance["problem_statement"]
    result = agent.run(task)
    data = agent.save(output_path, {"instance_id": instance["instance_id"]})
    return {"result": result, "trajectory": data}


# fmt: off
@app.command()
def main(
    instance_id: str = typer.Option(..., "--instance-id", help="SWE-bench instance ID"),
    subset: str = typer.Option("verified", "--subset", help="SWE-bench subset"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    output_dir: Path = typer.Option("self_improve_swebench", "-o", "--output", help="Output directory"),
    max_iterations: int = typer.Option(3, "-n", "--max-iterations", help="Max improvement iterations"),
    config_spec: list[str] = typer.Option(["swebench.yaml"], "-c", "--config", help="Base config file(s)"),
    critique_model: str = typer.Option("openrouter/qwen/qwen3-coder", "--critique-model", help="Model for critique"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Override coder model"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class (e.g., 'openrouter')"),
) -> None:
    # fmt: on
    """Self-improvement loop on a single SWE-bench instance."""
    output_dir = Path(output_dir) / instance_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load instance
    dataset_mapping = {"verified": "princeton-nlp/SWE-Bench_Verified", "lite": "princeton-nlp/SWE-Bench_Lite"}
    dataset_path = dataset_mapping.get(subset, subset)
    print(f"Loading {dataset_path} split={split}...")
    instances = {inst["instance_id"]: inst for inst in load_dataset(dataset_path, split=split)}
    if instance_id not in instances:
        raise ValueError(f"Instance {instance_id} not found. Available: {list(instances.keys())[:10]}...")
    instance = instances[instance_id]
    print(f"Task: {instance['problem_statement'][:200]}...")

    # Build base config
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

    all_summaries = []

    for iteration in range(max_iterations):
        iter_dir = output_dir / f"v{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Iteration {iteration}")
        print(f"{'='*60}")

        (iter_dir / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))

        # Run agent (each run gets a fresh Docker container via get_sb_environment)
        print("  Running agent...")
        traj_path = iter_dir / "trajectory.json"
        try:
            attempt = run_agent_on_instance(config, instance, traj_path)
        except Exception as e:
            print(f"  ❌ Agent crashed: {e}")
            summary = {"iteration": iteration, "exit_status": str(e), "verified_success": False, "n_steps": 0, "cost": 0}
            (iter_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            all_summaries.append(summary)
            break

        trajectory = attempt["trajectory"]
        exit_status = attempt["result"].get("exit_status", "")
        submission = attempt["result"].get("submission", "")
        n_steps = len([m for m in trajectory["messages"] if m.get("role") == "assistant"])
        cost = trajectory["info"]["model_stats"]["instance_cost"]

        print(f"  Exit: {exit_status} | Steps: {n_steps} | Cost: ${cost:.4f}")
        if submission:
            (iter_dir / "patch.diff").write_text(submission)
            print(f"  Patch saved ({len(submission)} chars)")

        summary = {
            "iteration": iteration,
            "exit_status": exit_status,
            "has_submission": bool(submission),
            "submission_length": len(submission),
            "n_steps": n_steps,
            "cost": cost,
        }
        (iter_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        all_summaries.append(summary)

        if iteration >= max_iterations - 1:
            print("\n  Max iterations reached.")
            break

        # Critique
        print("  Critiquing trace...")
        compact_trace = format_trace_compact(trajectory["messages"])
        succeeded = exit_status == "Submitted" and bool(submission)
        tips = get_critique_tips(compact_trace, model_name=critique_model, succeeded=succeeded)
        (iter_dir / "tips.txt").write_text(tips)
        print(f"  Tips:\n{tips}")

        # Inject tips for next iteration
        config["agent"]["instance_template"] = original_template
        config = inject_tips_into_config(config, tips)

    # Final comparison
    print(f"\n{'='*60}")
    print("  Summary across iterations")
    print(f"{'='*60}")
    for s in all_summaries:
        status = "✓ submitted" if s.get("has_submission") else "✗ no patch"
        print(f"  v{s['iteration']}: {s['exit_status']:20s} | {status} | steps={s['n_steps']} | ${s['cost']:.4f}")
    print(f"\nResults saved to {output_dir}/")
    print("To evaluate patches, run: sb-cli submit swe-bench_verified test --predictions_path <preds.json>")


if __name__ == "__main__":
    app()
