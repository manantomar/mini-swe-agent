#!/usr/bin/env python3
"""Async SFT + PPO training loop with Tinker sampling and training.

Each step: generate fresh rollouts → eval → train on current data only.
Holdout eval runs once at the end.
All rollouts + evals are maximally parallel.

Usage:
    export TINKER_API_KEY=...
    python rl/run_training_loop.py
"""

import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("training_loop")

BASE_MODEL = "Qwen/Qwen3-8B"
BASE_DIR = Path("/data/manantomar/swe-bench-docker/training-loop-v4")
SFT_LR = 5e-5
PPO_LR = 5e-6
LORA_RANK = 64

# 16 tasks: 13 original solvable + 3 new for diversity
SOLVABLE_TASKS = [
    "django__django-11119", "django__django-14373", "django__django-15741",
    "django__django-16139", "django__django-16255", "django__django-16569",
    "django__django-17029", "matplotlib__matplotlib-20859",
    "pytest-dev__pytest-5809", "pytest-dev__pytest-6202",
    "pytest-dev__pytest-7982", "scikit-learn__scikit-learn-14496",
    "sympy__sympy-16886",
    # 3 new unscreened for diversity
    "pytest-dev__pytest-7521", "sphinx-doc__sphinx-8269", "django__django-11163",
]

EVAL_TASKS = [
    "django__django-10097", "django__django-11433", "django__django-12308",
    "django__django-13794", "django__django-16100", "psf__requests-1766",
    "pylint-dev__pylint-4970", "sphinx-doc__sphinx-10435",
    "sphinx-doc__sphinx-9711", "sympy__sympy-20916",
]

N_ROLLOUTS = 8
STEP_LIMIT = 50
N_SFT_STEPS = 1
N_PPO_STEPS = 20
EVAL_ROLLOUTS = 8
BATCH_SIZE = 128
SFT_SUBSTEPS = 4
PPO_SUBSTEPS = 2

# Pre-collected step-0 data from v1 run (base model, same for everyone)
V1_SFT_000 = Path("/data/manantomar/swe-bench-docker/training-loop/sft-000")


# ── Generation + streaming eval ────────────────────────────────────────────

def _eval_one_rollout(rollout_dir: Path) -> tuple[str, set[str], dict]:
    preds_path = rollout_dir / "preds.json"
    if not preds_path.exists():
        logger.warning(f"    {rollout_dir.name}: no preds.json")
        return rollout_dir.name, set(), {}
    preds = json.loads(preds_path.read_text())
    ids_with_patches = [k for k, v in preds.items() if v.get("model_patch", "").strip()]
    resolved_ids: set[str] = set()
    if not ids_with_patches:
        logger.info(f"    {rollout_dir.name}: {len(preds)} tasks, 0 patches, 0 resolved")
        return rollout_dir.name, resolved_ids, preds

    filtered_path = rollout_dir / "filtered_preds.jsonl"
    with open(filtered_path, "w") as f:
        for k in ids_with_patches:
            f.write(json.dumps({
                "instance_id": k, "model_name_or_path": "tloop",
                "model_patch": preds[k]["model_patch"],
            }) + "\n")
    run_id = f"{BASE_DIR.name}-{rollout_dir.parent.name}-{rollout_dir.name}"
    result = subprocess.run(
        ["python3", "-m", "swebench.harness.run_evaluation",
         "--dataset_name", "princeton-nlp/SWE-bench_Verified",
         "--split", "test", "--predictions_path", str(filtered_path),
         "--run_id", run_id,
         "--max_workers", "16", "--timeout", "300"],
        capture_output=True, text=True, env=os.environ, timeout=600,
    )
    if result.returncode != 0:
        logger.warning(f"    {rollout_dir.name}: swebench eval returned {result.returncode}: {result.stderr[-200:]}")

    # Read per-instance reports from deterministic logs path
    report_dir = Path(f"logs/run_evaluation/{run_id}/tloop")
    n_reports = 0
    if report_dir.exists():
        for task_dir in report_dir.iterdir():
            report_file = task_dir / "report.json"
            if report_file.exists():
                try:
                    r = json.loads(report_file.read_text())
                    n_reports += 1
                    for tid, info in r.items():
                        if info.get("resolved"):
                            resolved_ids.add(tid)
                except Exception as e:
                    logger.warning(f"    {rollout_dir.name}: bad report {report_file}: {e}")
    else:
        logger.warning(f"    {rollout_dir.name}: no report dir at {report_dir}")

    logger.info(f"    {rollout_dir.name}: {len(preds)} tasks, {len(ids_with_patches)} patches, "
                f"{len(resolved_ids)} resolved, {n_reports} reports read")
    # Clean up summary report from CWD
    for rf in Path(".").glob(f"{run_id}*.json"):
        rf.unlink(missing_ok=True)
    return rollout_dir.name, resolved_ids, preds


def generate_and_eval(
    tasks: list[str], n_rollouts: int, out_dir: Path, checkpoint_path: str = "",
    step_limit: int = STEP_LIMIT,
) -> dict[str, list[float]]:
    """Generate all rollouts in parallel, stream eval as each finishes."""
    task_filter = "^(" + "|".join(tasks) + ")$"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Launch all rollouts
    procs: dict[int, subprocess.Popen] = {}
    for r in range(n_rollouts):
        rdir = out_dir / f"rollout_{r:02d}"
        if rdir.exists() and len(list(rdir.glob("*/*.traj.json"))) >= len(tasks) * 0.9:
            logger.info(f"    skip rollout {r}")
            continue
        rdir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "mini-extra", "swebench", "--subset", "verified", "--split", "test",
            "--filter", task_filter, "-m", BASE_MODEL, "--model-class", "tinker",
            "-o", str(rdir), "-w", str(len(tasks)),
            "-c", "swebench.yaml",
            "-c", f"agent.step_limit={step_limit}", "-c", "agent.cost_limit=100",
            "-c", "model.cost_tracking=ignore_errors",
            "-c", "model.model_kwargs.temperature=0.7",
            "-c", "environment.pull_timeout=300",
        ]
        if checkpoint_path:
            cmd += ["-c", f"model.tinker_checkpoint_path={checkpoint_path}"]
        procs[r] = subprocess.Popen(
            cmd, env={**os.environ, "MSWEA_SILENT_STARTUP": "1"},
            stdout=open(rdir / "run.log", "w"), stderr=subprocess.STDOUT,
        )
        time.sleep(0.5)

    # Stream eval as rollouts finish
    eval_pool = ThreadPoolExecutor(max_workers=8)
    eval_futures: dict[Future, int] = {}
    all_rewards: defaultdict[str, list[float]] = defaultdict(list)

    # Submit eval for pre-existing rollouts
    for r in range(n_rollouts):
        if r not in procs:
            rdir = out_dir / f"rollout_{r:02d}"
            if rdir.exists():
                eval_futures[eval_pool.submit(_eval_one_rollout, rdir)] = r

    # Poll; submit eval as each rollout finishes
    while procs:
        for r, proc in list(procs.items()):
            if proc.poll() is not None:
                rdir = out_dir / f"rollout_{r:02d}"
                n = len(list(rdir.glob("*/*.traj.json"))) if rdir.exists() else 0
                logger.info(f"    rollout {r}: {n} trajs ({time.time()-t0:.0f}s)")
                del procs[r]
                eval_futures[eval_pool.submit(_eval_one_rollout, rdir)] = r
        if procs:
            time.sleep(2)

    logger.info(f"  gen done ({time.time()-t0:.0f}s), waiting for evals...")

    for future in as_completed(eval_futures):
        rname, resolved, preds = future.result()
        for task_id in preds:
            all_rewards[task_id].append(1.0 if task_id in resolved else 0.0)

    eval_pool.shutdown()
    n_ok = int(sum(sum(r) for r in all_rewards.values()))
    n_all = sum(len(r) for r in all_rewards.values())
    n_solved = len([t for t, r in all_rewards.items() if sum(r) > 0])
    logger.info(f"  eval done: {n_ok}/{n_all} ok, {n_solved}/{len(tasks)} tasks ({time.time()-t0:.0f}s total)")
    return dict(all_rewards)


# ── Datum collection ────────────────────────────────────────────────────────

TOOLS_SPEC = [{"type": "function", "function": {"name": "bash", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
MAX_DATUM_LEN = 8192


def _get_tokenizer():
    """Lazy-load tokenizer for observation tokenization."""
    if not hasattr(_get_tokenizer, "_tok"):
        from transformers import AutoTokenizer
        _get_tokenizer._tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    return _get_tokenizer._tok


def _clean_msg(m: dict) -> dict:
    clean = {"role": m["role"]}
    if m.get("content"):
        clean["content"] = m["content"]
    if m.get("tool_calls"):
        clean["tool_calls"] = [
            {"id": tc.get("id", ""), "type": "function",
             "function": {"name": tc["function"]["name"],
                          "arguments": tc["function"]["arguments"]}}
            for tc in m["tool_calls"]]
    if m.get("tool_call_id"):
        clean["tool_call_id"] = m["tool_call_id"]
    return clean


def _build_datums_from_traj(traj: dict, task_id: str, advantage: float | None = None) -> list[dict]:
    """Build per-step datums: observation (masked) + action (gradient).
    
    advantage=None → SFT datum; otherwise PPO datum.
    """
    tokenizer = _get_tokenizer()
    msgs = traj["messages"]
    datums = []

    for i, m in enumerate(msgs):
        if m.get("role") != "assistant" or "extra" not in m:
            continue
        lp = m["extra"].get("logprobs", {})
        ac_tokens = lp.get("token_ids", [])
        ac_logprobs = lp.get("logprobs", [])
        if not ac_tokens or len(ac_tokens) < 2:
            continue

        # Tokenize observation (all messages before this assistant turn)
        ob_msgs = [_clean_msg(prev) for prev in msgs[:i]]
        ob_text = tokenizer.apply_chat_template(
            ob_msgs, tools=TOOLS_SPEC, tokenize=False, add_generation_prompt=True,
        )
        ob_tokens = tokenizer.encode(ob_text, add_special_tokens=False)

        full_tokens = ob_tokens + ac_tokens
        if len(full_tokens) < 2 or len(full_tokens) > MAX_DATUM_LEN:
            continue

        ob_len = len(ob_tokens) - 1  # shifted
        ac_len = len(ac_tokens)

        d = {
            "task_id": task_id,
            "full_tokens": full_tokens,
            "ob_len": ob_len,
            "ac_logprobs": ac_logprobs,
        }
        if advantage is not None:
            d["advantage"] = advantage
        datums.append(d)

    return datums


def collect_sft_datums(out_dir: Path, rewards: dict[str, list[float]]) -> list[dict]:
    datums = []
    for r_idx, rollout_dir in enumerate(sorted(out_dir.glob("rollout_*"))):
        for traj_file in sorted(rollout_dir.glob("*/*.traj.json")):
            task_id = traj_file.parent.name
            if r_idx < len(rewards.get(task_id, [])) and rewards[task_id][r_idx] > 0:
                traj = json.loads(traj_file.read_text())
                datums.extend(_build_datums_from_traj(traj, task_id))
    logger.info(f"  sft datums: {len(datums)}")
    return datums


def collect_ppo_datums(out_dir: Path, rewards: dict[str, list[float]]) -> list[dict]:
    task_adv = {t: [r - np.mean(rs) for r in rs] for t, rs in rewards.items()}
    datums = []
    for r_idx, rollout_dir in enumerate(sorted(out_dir.glob("rollout_*"))):
        for traj_file in sorted(rollout_dir.glob("*/*.traj.json")):
            task_id = traj_file.parent.name
            if task_id not in task_adv or r_idx >= len(task_adv[task_id]):
                continue
            adv = task_adv[task_id][r_idx]
            if adv == 0.0:
                continue
            traj = json.loads(traj_file.read_text())
            datums.extend(_build_datums_from_traj(traj, task_id, advantage=adv))
    pos = sum(1 for d in datums if d["advantage"] > 0)
    logger.info(f"  ppo datums: {len(datums)} (pos={pos}, neg={len(datums)-pos})")
    return datums


# ── Training ────────────────────────────────────────────────────────────────

def _datum_to_tinker_sft(d: dict) -> "tinker.types.Datum":
    """Build a Tinker SFT datum with observation masking."""
    import tinker
    tokens = d["full_tokens"]
    ob_len = d["ob_len"]
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    n = len(input_tokens)
    # Zero weight on observation, uniform weight on action
    ac_len = n - ob_len
    weights = [0.0] * ob_len + [1.0 / ac_len] * ac_len
    return tinker.types.Datum(
        model_input=tinker.types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
    )


def _datum_to_tinker_ppo(d: dict) -> "tinker.types.Datum":
    """Build a Tinker PPO datum with observation masking."""
    import tinker
    tokens = d["full_tokens"]
    ob_len = d["ob_len"]
    ac_logprobs = d["ac_logprobs"]
    adv = d["advantage"]

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    n = len(input_tokens)
    # Zero logprobs/advantages on observation, real values on action
    padded_logprobs = [0.0] * ob_len + ac_logprobs
    padded_advantages = [0.0] * ob_len + [adv] * (n - ob_len)
    # Trim to match
    padded_logprobs = padded_logprobs[:n]
    padded_advantages = padded_advantages[:n]
    return tinker.types.Datum(
        model_input=tinker.types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "logprobs": padded_logprobs, "advantages": padded_advantages},
    )


def run_sft_step(datums: list[dict], checkpoint_path: str, step_name: str) -> tuple[str, str]:
    """Returns (training_checkpoint_path, sampler_weights_path)."""
    import tinker
    sc = tinker.ServiceClient()
    tc = sc.create_training_client_from_state(checkpoint_path) if checkpoint_path \
        else sc.create_lora_training_client(base_model=BASE_MODEL, rank=LORA_RANK)
    adam = tinker.types.AdamParams(learning_rate=SFT_LR)

    # Balance per-task
    from collections import Counter
    task_counts = Counter(d["task_id"] for d in datums)
    max_per_task = max(8, len(datums) // len(task_counts)) if task_counts else len(datums)
    task_seen: dict[str, int] = defaultdict(int)
    balanced = [d for d in datums if (task_seen.__setitem__(d["task_id"], task_seen.get(d["task_id"], 0) + 1) or True) and task_seen[d["task_id"]] <= max_per_task]

    tinker_datums = [_datum_to_tinker_sft(d) for d in balanced]

    np.random.shuffle(tinker_datums)
    for sub in range(SFT_SUBSTEPS):
        batch = tinker_datums[sub * BATCH_SIZE:(sub + 1) * BATCH_SIZE]
        if not batch:
            break
        fb = tc.forward_backward(batch, loss_fn="cross_entropy").result()
        loss = fb.metrics.get("loss:sum", 0) / len(batch)
        tc.optim_step(adam).result()
        logger.info(f"    substep {sub}: loss={loss:.4f} batch={len(batch)}")

    ckpt = tc.save_state(name=step_name).result().path
    sampler_path = tc.save_weights_for_sampler(name=f"{step_name}-sampler").result().path
    logger.info(f"  train done: {SFT_SUBSTEPS} substeps, {len(tinker_datums)} total datums")
    return ckpt, sampler_path


def run_ppo_step(datums: list[dict], checkpoint_path: str, step_name: str) -> tuple[str, str]:
    """Returns (training_checkpoint_path, sampler_weights_path)."""
    import tinker
    sc = tinker.ServiceClient()
    tc = sc.create_training_client_from_state(checkpoint_path) if checkpoint_path \
        else sc.create_lora_training_client(base_model=BASE_MODEL, rank=LORA_RANK)
    adam = tinker.types.AdamParams(learning_rate=PPO_LR)

    tinker_datums = [_datum_to_tinker_ppo(d) for d in datums]

    np.random.shuffle(tinker_datums)
    for sub in range(PPO_SUBSTEPS):
        batch = tinker_datums[sub * BATCH_SIZE:(sub + 1) * BATCH_SIZE]
        if not batch:
            break
        fb = tc.forward_backward(batch, loss_fn="ppo",
            loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.7}).result()
        loss = fb.metrics.get("loss:sum", 0) / len(batch)
        tc.optim_step(adam).result()
        logger.info(f"    substep {sub}: loss={loss:.4f} batch={len(batch)}")

    ckpt = tc.save_state(name=step_name).result().path
    sampler_path = tc.save_weights_for_sampler(name=f"{step_name}-sampler").result().path
    logger.info(f"  train done: {PPO_SUBSTEPS} substeps, {len(tinker_datums)} total datums")
    return ckpt, sampler_path


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = BASE_DIR / "state.json"

    if state_file.exists():
        state = json.loads(state_file.read_text())
        logger.info(f"Resuming: phase={state['phase']} step={state['step']}")
    else:
        state = {"phase": "sft", "step": 0, "checkpoint": "", "sampler_path": ""}

    def save_state():
        state_file.write_text(json.dumps(state, indent=2))

    # ═══ SFT phase (1 step on v1 data) ═══
    if state["phase"] == "sft":
        for step in range(state["step"], N_SFT_STEPS):
            state["step"] = step
            step_name = f"sft-{step:03d}"
            step_dir = BASE_DIR / step_name
            logger.info(f"\n{'═'*50}\n  SFT {step}/{N_SFT_STEPS}\n{'═'*50}")
            t0 = time.time()

            if V1_SFT_000.exists() and not step_dir.exists():
                import shutil
                shutil.copytree(V1_SFT_000, step_dir)
                logger.info(f"  reusing v1 data ({len(list(step_dir.glob('rollout_*/*/*.traj.json')))} trajs)")
                v1_results = json.loads((V1_SFT_000 / "step_results.json").read_text())
                rewards = v1_results["rewards"]
            else:
                rewards = generate_and_eval(SOLVABLE_TASKS, N_ROLLOUTS, step_dir, state["sampler_path"])
            datums = collect_sft_datums(step_dir, rewards)

            if not datums:
                logger.warning("  no successes, skip training")
                save_state()
                continue

            ckpt, sampler = run_sft_step(datums, state["checkpoint"], step_name)
            state["checkpoint"] = ckpt
            state["sampler_path"] = sampler
            wall = time.time() - t0

            (step_dir / "step_results.json").write_text(json.dumps({
                "step": step_name, "wall_time": wall, "checkpoint": ckpt,
                "sampler_path": sampler, "rewards": rewards, "n_datums": len(datums),
            }, indent=2))
            save_state()
            logger.info(f"  {step_name} done in {wall:.0f}s")

        state["phase"] = "ppo"
        state["step"] = 0
        save_state()

    # ═══ GRPO phase ═══
    if state["phase"] == "ppo":
        for step in range(state["step"], N_PPO_STEPS):
            state["step"] = step
            step_name = f"ppo-{step:03d}"
            step_dir = BASE_DIR / step_name
            step_limit = min(100, STEP_LIMIT + step * 10)
            logger.info(f"\n{'═'*50}\n  GRPO {step}/{N_PPO_STEPS} ({len(SOLVABLE_TASKS)} tasks, {step_limit} steps)\n{'═'*50}")
            t0 = time.time()

            rewards = generate_and_eval(SOLVABLE_TASKS, N_ROLLOUTS, step_dir, state["sampler_path"],
                                        step_limit=step_limit)
            datums = collect_ppo_datums(step_dir, rewards)

            if not datums:
                logger.warning("  no non-zero advantage datums, skip training")
                save_state()
                continue

            ckpt, sampler = run_ppo_step(datums, state["checkpoint"], step_name)
            state["checkpoint"] = ckpt
            state["sampler_path"] = sampler
            wall = time.time() - t0

            (step_dir / "step_results.json").write_text(json.dumps({
                "step": step_name, "wall_time": wall, "checkpoint": ckpt,
                "sampler_path": sampler, "rewards": rewards, "n_datums": len(datums),
            }, indent=2))
            save_state()
            logger.info(f"  {step_name} done in {wall:.0f}s")

    # ═══ Final holdout eval ═══
    logger.info(f"\n{'═'*50}\n  Final holdout eval\n{'═'*50}")
    eval_dir = BASE_DIR / "eval-final"
    rewards = generate_and_eval(EVAL_TASKS, EVAL_ROLLOUTS, eval_dir, state["sampler_path"])
    results = {t: float(np.mean(r)) for t, r in rewards.items()}
    solved = sum(1 for v in results.values() if v > 0)
    (eval_dir / "results.json").write_text(json.dumps(results, indent=2))

    logger.info(f"\n{'═'*50}\n  DONE\n{'═'*50}")
    logger.info(f"  Checkpoint: {state['checkpoint']}")
    logger.info(f"  Holdout: {solved}/10 tasks solved")
    for t, v in sorted(results.items()):
        logger.info(f"    {t}: {v:.0%}")


if __name__ == "__main__":
    main()
