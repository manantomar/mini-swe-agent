#!/usr/bin/env python3

"""Self-improvement loop: run agent, critique traces, improve prompts, re-run."""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import typer
import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.critique import format_trace_compact, get_critique_tips, inject_tips_into_config
from minisweagent.utils.serialize import recursive_merge

logger = logging.getLogger("minisweagent.self_improve")

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def _setup_local_task_dir(source_dir: Path) -> Path:
    """Create an isolated temp copy of a task directory for a clean run."""
    tmp = Path(tempfile.mkdtemp(prefix="mswea_si_"))
    shutil.copytree(source_dir, tmp / "workdir", dirs_exist_ok=True)
    return tmp / "workdir"


def _check_success(workdir: Path, verify_command: str) -> bool:
    """Run a verification command and return True if it passes."""
    import subprocess

    result = subprocess.run(
        verify_command,
        shell=True,
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode == 0


def run_single_attempt(
    config: dict,
    task: str,
    output_path: Path,
    task_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the agent once and return the trajectory data."""
    env_config = config.get("environment", {})
    env_config.setdefault("environment_class", "local")
    if task_dir:
        workdir = _setup_local_task_dir(task_dir)
        env_config["cwd"] = str(workdir)
    else:
        workdir = None

    model = get_model(config=config.get("model", {}))
    env = get_environment(env_config)
    agent = DefaultAgent(model, env, output_path=output_path, **config.get("agent", {}))
    result = agent.run(task)
    data = agent.save(output_path)
    return {
        "result": result,
        "trajectory": data,
        "workdir": str(workdir) if workdir else "",
    }


# fmt: off
@app.command()
def main(
    task: str = typer.Option(..., "-t", "--task", help="Task description for the agent"),
    task_dir: Path = typer.Option(..., "--task-dir", help="Directory with the task files (will be copied per attempt)"),
    verify_command: str = typer.Option(..., "--verify", help="Command to check success (e.g., 'python -m pytest test.py')"),
    output_dir: Path = typer.Option("self_improve_runs", "-o", "--output", help="Output directory for all iterations"),
    max_iterations: int = typer.Option(3, "-n", "--max-iterations", help="Maximum improvement iterations"),
    config_spec: list[str] = typer.Option(["mini.yaml"], "-c", "--config", help="Base config file(s)"),
    critique_model: str = typer.Option("anthropic/claude-sonnet-4-5-20250929", "--critique-model", help="Model for critique"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Override model for the coder agent"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class (e.g., 'openrouter', 'litellm')"),
) -> None:
    # fmt: on
    """Run a self-improvement loop: agent solves task, critique analyzes failures, prompts are improved."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build base config
    configs = [get_config_from_spec(spec) for spec in config_spec]
    if model_name or model_class:
        model_override: dict = {}
        if model_name:
            model_override["model_name"] = model_name
        if model_class:
            model_override["model_class"] = model_class
        configs.append({"model": model_override})
    # Force non-interactive mode
    configs.append({"agent": {"step_limit": 50, "cost_limit": 1.0}})
    config = recursive_merge(*configs)
    original_template = config.get("agent", {}).get("instance_template", "")

    for iteration in range(max_iterations):
        iter_dir = output_dir / f"v{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Iteration {iteration}")
        print(f"{'='*60}")

        # Save config snapshot
        (iter_dir / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))

        # Run the agent
        print("  Running agent...")
        traj_path = iter_dir / "trajectory.json"
        attempt = run_single_attempt(config, task, traj_path, task_dir)
        trajectory = attempt["trajectory"]
        exit_status = attempt["result"].get("exit_status", "")

        # Check actual success
        workdir = Path(attempt["workdir"]) if attempt["workdir"] else task_dir
        succeeded = exit_status == "Submitted" and _check_success(workdir, verify_command)

        summary = {
            "iteration": iteration,
            "exit_status": exit_status,
            "verified_success": succeeded,
            "n_steps": len([m for m in trajectory["messages"] if m.get("role") == "assistant"]),
            "cost": trajectory["info"]["model_stats"]["instance_cost"],
        }
        (iter_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"  Exit: {exit_status} | Verified: {'✓' if succeeded else '✗'} | Steps: {summary['n_steps']} | Cost: ${summary['cost']:.4f}")

        if succeeded:
            print(f"\n✅ Task solved on iteration {iteration}!")
            break

        if iteration >= max_iterations - 1:
            print("\n❌ Max iterations reached without solving the task.")
            break

        # Critique the trace
        print("  Critiquing trace...")
        compact_trace = format_trace_compact(trajectory["messages"])
        tips = get_critique_tips(compact_trace, model_name=critique_model, succeeded=False)
        (iter_dir / "tips.txt").write_text(tips)
        print(f"  Tips:\n{tips}")

        # Inject tips into config for next iteration
        config["agent"]["instance_template"] = original_template
        config = inject_tips_into_config(config, tips)

    # Final summary
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    app()
