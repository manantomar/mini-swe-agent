#!/usr/bin/env python3
"""Compare mini-swe-agent evaluation results across models.

Usage:
    python3 eval/compare_results.py eval/results/

Reads preds.json and trajectory files from each model subdirectory
and prints a comparison table.
"""

import json
import sys
from pathlib import Path


INSTANCES = [
    "django__django-10097",
    "django__django-11433",
    "django__django-12308",
    "django__django-13794",
    "django__django-16100",
    "psf__requests-1766",
    "pylint-dev__pylint-4970",
    "sphinx-doc__sphinx-10435",
    "sphinx-doc__sphinx-9711",
    "sympy__sympy-20916",
]


def load_model_results(model_dir: Path) -> dict:
    """Load results for a single model from its output directory."""
    results = {}

    # Load preds.json for patch info
    preds_file = model_dir / "preds.json"
    preds = {}
    if preds_file.exists():
        preds = json.loads(preds_file.read_text())

    for iid in INSTANCES:
        entry = {"instance_id": iid, "has_patch": False, "exit_status": "missing", "steps": 0, "cost": 0.0}

        # Check if prediction exists
        if iid in preds:
            patch = preds[iid].get("model_patch", "")
            entry["has_patch"] = bool(patch and patch.strip())

        # Load trajectory for more detail
        traj_file = model_dir / iid / f"{iid}.traj.json"
        if traj_file.exists():
            try:
                traj = json.loads(traj_file.read_text())
                info = traj.get("info", {})
                entry["exit_status"] = info.get("exit_status", "unknown")

                # Count steps from history
                history = traj.get("history", [])
                entry["steps"] = len([h for h in history if h.get("role") == "assistant"])

                # Get cost from info or compute from history
                entry["cost"] = info.get("total_cost", 0.0) or traj.get("total_cost", 0.0)
            except (json.JSONDecodeError, KeyError):
                entry["exit_status"] = "traj_error"

        results[iid] = entry

    return results


def print_comparison(results_dir: Path):
    """Print a comparison table across all model results."""
    model_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and (d / "preds.json").exists()])

    if not model_dirs:
        print(f"No results found in {results_dir}/")
        print("Expected subdirectories with preds.json files.")
        sys.exit(1)

    # Map safe names back to original model names
    model_names = {}
    for d in model_dirs:
        preds = json.loads((d / "preds.json").read_text())
        first_pred = next(iter(preds.values()), {})
        model_names[d.name] = first_pred.get("model_name_or_path", d.name)

    # Load all results
    all_results = {}
    for d in model_dirs:
        name = model_names[d.name]
        all_results[name] = load_model_results(d)

    # Print header
    models = list(all_results.keys())
    col_w = max(24, max(len(m) for m in models) + 2)

    print("\n" + "=" * 80)
    print("  mini-swe-agent Evaluation Results")
    print("=" * 80)

    # Summary table
    print(f"\n{'Instance':<40}", end="")
    for m in models:
        short = m.split("/")[-1][:col_w - 2]
        print(f"  {short:<{col_w}}", end="")
    print()
    print("-" * (40 + (col_w + 2) * len(models)))

    for iid in INSTANCES:
        print(f"{iid:<40}", end="")
        for m in models:
            entry = all_results[m].get(iid, {})
            status = entry.get("exit_status", "missing")
            has_patch = entry.get("has_patch", False)
            steps = entry.get("steps", 0)
            cost = entry.get("cost", 0.0)

            if status == "missing":
                cell = "—"
            elif has_patch:
                cell = f"✓ patch ({steps}st, ${cost:.2f})"
            else:
                cell = f"✗ {status} ({steps}st)"
            print(f"  {cell:<{col_w}}", end="")
        print()

    # Summary row
    print("-" * (40 + (col_w + 2) * len(models)))
    print(f"{'TOTAL PATCHES':<40}", end="")
    for m in models:
        total = sum(1 for iid in INSTANCES if all_results[m].get(iid, {}).get("has_patch", False))
        total_cost = sum(all_results[m].get(iid, {}).get("cost", 0.0) for iid in INSTANCES)
        n = len(INSTANCES)
        print(f"  {total}/{n} (${total_cost:.2f}){'':<{col_w - 17}}", end="")
    print()

    print("\n" + "=" * 80)

    # Note about actual evaluation
    print("\nNote: 'has_patch' means the agent produced a diff. To check if the patch")
    print("actually fixes the issue, run the official SWE-bench evaluation:")
    print("")
    for d in model_dirs:
        name = model_names[d.name]
        short = name.split("/")[-1]
        print(f"  # {short}")
        print(f"  sb-cli submit swe-bench_verified test \\")
        print(f"    --predictions_path {d}/preds.json --run_id {short}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)
    print_comparison(Path(sys.argv[1]))
