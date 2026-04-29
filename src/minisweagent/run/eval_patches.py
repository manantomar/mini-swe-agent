#!/usr/bin/env python3

"""Evaluate patches locally by applying them in Docker and running SWE-bench test patches."""

import json
import subprocess
import tempfile
from pathlib import Path

from datasets import load_dataset

from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name


def evaluate_patch(instance: dict, model_patch: str) -> dict:
    """Apply model_patch + test_patch in a fresh Docker container, run tests."""
    image = get_swebench_docker_image_name(instance)
    iid = instance["instance_id"]
    test_patch = instance["test_patch"]

    with tempfile.TemporaryDirectory() as tmpdir:
        model_patch_file = Path(tmpdir) / "model.patch"
        test_patch_file = Path(tmpdir) / "test.patch"
        model_patch_file.write_text(model_patch)
        test_patch_file.write_text(test_patch)

        script = f"""
cd /testbed
# Apply model patch
git apply /patches/model.patch 2>&1
MODEL_APPLY=$?

# Apply test patch
git apply /patches/test.patch 2>&1
TEST_APPLY=$?

echo "MODEL_APPLY_EXIT=$MODEL_APPLY"
echo "TEST_APPLY_EXIT=$TEST_APPLY"

if [ $MODEL_APPLY -ne 0 ]; then
    echo "RESULT=patch_apply_failed"
    exit 1
fi

if [ $TEST_APPLY -ne 0 ]; then
    echo "RESULT=test_patch_apply_failed"
    exit 1
fi

# Run the test command from the instance
{instance.get('test_cmd', 'echo NO_TEST_CMD')} 2>&1
TEST_EXIT=$?
echo "TEST_EXIT=$TEST_EXIT"
if [ $TEST_EXIT -eq 0 ]; then
    echo "RESULT=passed"
else
    echo "RESULT=failed"
fi
"""
        script_file = Path(tmpdir) / "run.sh"
        script_file.write_text(script)

        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{tmpdir}:/patches",
                "-w", "/testbed",
                image,
                "bash", "/patches/run.sh",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        output = result.stdout + result.stderr
        passed = "RESULT=passed" in output
        return {
            "instance_id": iid,
            "passed": passed,
            "returncode": result.returncode,
            "output_tail": output[-2000:] if len(output) > 2000 else output,
        }


def main():
    import sys
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/self_improve_multi")

    dataset = {
        inst["instance_id"]: inst
        for inst in load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
    }

    for v in sorted(results_dir.iterdir()):
        if not v.is_dir() or not v.name.startswith("v"):
            continue
        preds_file = v / "preds.json"
        if not preds_file.exists():
            continue
        preds = json.loads(preds_file.read_text())
        print(f"\n{'='*50}")
        print(f"  Evaluating {v.name} ({len(preds)} patches)")
        print(f"{'='*50}")

        for iid, pred in sorted(preds.items()):
            patch = pred["model_patch"]
            if not patch.strip():
                print(f"  {iid}: SKIP (empty patch)")
                continue
            inst = dataset.get(iid)
            if not inst:
                print(f"  {iid}: SKIP (not in dataset)")
                continue
            print(f"  {iid}...", end=" ", flush=True)
            try:
                result = evaluate_patch(inst, patch)
                status = "✅ PASSED" if result["passed"] else "❌ FAILED"
                print(status)
                if not result["passed"]:
                    # Show last few lines of output for debugging
                    lines = result["output_tail"].strip().splitlines()
                    for line in lines[-5:]:
                        print(f"    | {line}")
            except Exception as e:
                print(f"💥 ERROR: {e}")

        eval_path = v / "eval_results.json"
        print(f"  Results saved to {eval_path}")


if __name__ == "__main__":
    main()
