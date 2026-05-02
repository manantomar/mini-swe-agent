#!/usr/bin/env python3
"""V5: GRPO with patch similarity reward on 168 easy tasks.

- Sample 16 tasks per step from 168 easy pool
- 8 rollouts each
- Reward = difflib.SequenceMatcher ratio between generated and gold patch
- 2-8 PPO substeps (adaptive based on datum count)
- Starts from SFT checkpoint

Usage:
    export TINKER_API_KEY=...
    python rl/run_training_v5.py
"""

import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger("v5")

BASE_MODEL = "Qwen/Qwen3-8B"
BASE_DIR = Path("/data/manantomar/swe-bench-docker/training-loop-v5")
PPO_LR = 5e-6
LORA_RANK = 64

GOLD_PATCHES = json.loads(Path("/data/manantomar/swe-bench-docker/gold_patches.json").read_text())
EASY_POOL = list(GOLD_PATCHES.keys())  # 184 tasks

EVAL_TASKS = [
    "django__django-10097", "django__django-11433", "django__django-12308",
    "django__django-13794", "django__django-16100", "psf__requests-1766",
    "pylint-dev__pylint-4970", "sphinx-doc__sphinx-10435",
    "sphinx-doc__sphinx-9711", "sympy__sympy-20916",
]

# Starting checkpoint from v4 SFT
INIT_CHECKPOINT = "tinker://55b453df-0eb9-56eb-a39c-ede2d1a474b5:train:0/weights/sft-000"
INIT_SAMPLER = "tinker://55b453df-0eb9-56eb-a39c-ede2d1a474b5:train:0/sampler_weights/sft-000-sampler"

N_ROLLOUTS = 8
TASKS_PER_STEP = 16
STEP_LIMIT = 50
N_PPO_STEPS = 20
BATCH_SIZE = 128
EVAL_ROLLOUTS = 8

TOOLS_SPEC = [{"type": "function", "function": {"name": "bash", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
MAX_DATUM_LEN = 8192


def sample_tasks(step: int) -> list[str]:
    rng = np.random.RandomState(seed=step * 13 + 7)
    return list(rng.choice(EASY_POOL, size=TASKS_PER_STEP, replace=False))


# ── Patch similarity reward ────────────────────────────────────────────────

def compute_rewards(out_dir: Path, tasks: list[str]) -> dict[str, list[float]]:
    """Compute patch similarity rewards (0-1) by comparing to gold patches."""
    all_rewards: defaultdict[str, list[float]] = defaultdict(list)
    for rd in sorted(out_dir.glob("rollout_*")):
        preds_path = rd / "preds.json"
        if not preds_path.exists():
            continue
        preds = json.loads(preds_path.read_text())
        for task_id in tasks:
            if task_id not in preds:
                all_rewards[task_id].append(0.0)
                continue
            model_patch = preds[task_id].get("model_patch", "").strip()
            gold_patch = GOLD_PATCHES.get(task_id, "")
            if not model_patch:
                all_rewards[task_id].append(0.0)
            else:
                sim = SequenceMatcher(None, model_patch, gold_patch).ratio()
                all_rewards[task_id].append(sim)

    # Log summary
    n_total = sum(len(v) for v in all_rewards.values())
    avg_reward = np.mean([r for rs in all_rewards.values() for r in rs]) if n_total else 0
    n_nonzero = sum(1 for rs in all_rewards.values() for r in rs if r > 0)
    logger.info(f"  rewards: avg={avg_reward:.3f}, nonzero={n_nonzero}/{n_total}")
    return dict(all_rewards)


# ── Generation ──────────────────────────────────────────────────────────────

def generate_rollouts(tasks: list[str], n_rollouts: int, out_dir: Path, sampler_path: str) -> int:
    task_filter = "^(" + "|".join(tasks) + ")$"
    out_dir.mkdir(parents=True, exist_ok=True)
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
            "-c", f"agent.step_limit={STEP_LIMIT}", "-c", "agent.cost_limit=100",
            "-c", "model.cost_tracking=ignore_errors",
            "-c", "model.model_kwargs.temperature=0.7",
            "-c", "environment.pull_timeout=300",
        ]
        if sampler_path:
            cmd += ["-c", f"model.tinker_checkpoint_path={sampler_path}"]
        procs[r] = subprocess.Popen(
            cmd, env={**os.environ, "MSWEA_SILENT_STARTUP": "1"},
            stdout=open(rdir / "run.log", "w"), stderr=subprocess.STDOUT,
        )
        time.sleep(0.5)

    t0 = time.time()
    while procs:
        for r, proc in list(procs.items()):
            if proc.poll() is not None:
                rdir = out_dir / f"rollout_{r:02d}"
                n = len(list(rdir.glob("*/*.traj.json"))) if rdir.exists() else 0
                logger.info(f"    rollout {r}: {n} trajs ({time.time()-t0:.0f}s)")
                del procs[r]
        if procs:
            time.sleep(2)

    total = sum(len(list(rd.glob("*/*.traj.json"))) for rd in out_dir.glob("rollout_*"))
    logger.info(f"  gen done: {total} trajs ({time.time()-t0:.0f}s)")
    return total


# ── Datum collection ────────────────────────────────────────────────────────

def _get_tokenizer():
    if not hasattr(_get_tokenizer, "_tok"):
        from transformers import AutoTokenizer
        _get_tokenizer._tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    return _get_tokenizer._tok


def _clean_msg(m: dict) -> dict:
    clean = {"role": m["role"]}
    if m.get("content"): clean["content"] = m["content"]
    if m.get("tool_calls"):
        clean["tool_calls"] = [{"id": tc.get("id", ""), "type": "function",
            "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
            for tc in m["tool_calls"]]
    if m.get("tool_call_id"): clean["tool_call_id"] = m["tool_call_id"]
    return clean


def collect_ppo_datums(out_dir: Path, rewards: dict[str, list[float]]) -> list[dict]:
    task_adv = {t: [r - np.mean(rs) for r in rs] for t, rs in rewards.items()}
    tokenizer = _get_tokenizer()
    datums = []

    for r_idx, rollout_dir in enumerate(sorted(out_dir.glob("rollout_*"))):
        for traj_file in sorted(rollout_dir.glob("*/*.traj.json")):
            task_id = traj_file.parent.name
            if task_id not in task_adv or r_idx >= len(task_adv[task_id]):
                continue
            adv = task_adv[task_id][r_idx]
            if abs(adv) < 1e-6:
                continue
            traj = json.loads(traj_file.read_text())
            msgs = traj["messages"]

            for i, m in enumerate(msgs):
                if m.get("role") != "assistant" or "extra" not in m:
                    continue
                lp = m["extra"].get("logprobs", {})
                ac_tokens = lp.get("token_ids", [])
                ac_logprobs = lp.get("logprobs", [])
                if not ac_tokens or len(ac_tokens) < 2:
                    continue
                ob_msgs = [_clean_msg(prev) for prev in msgs[:i]]
                ob_text = tokenizer.apply_chat_template(ob_msgs, tools=TOOLS_SPEC, tokenize=False, add_generation_prompt=True)
                ob_tokens = tokenizer.encode(ob_text, add_special_tokens=False)
                full_tokens = ob_tokens + ac_tokens
                if len(full_tokens) < 2 or len(full_tokens) > MAX_DATUM_LEN:
                    continue
                ob_len = len(ob_tokens) - 1
                datums.append({"task_id": task_id, "full_tokens": full_tokens,
                               "ob_len": ob_len, "ac_logprobs": ac_logprobs, "advantage": adv})

    pos = sum(1 for d in datums if d["advantage"] > 0)
    logger.info(f"  ppo datums: {len(datums)} (pos={pos}, neg={len(datums)-pos})")
    return datums


# ── Training ────────────────────────────────────────────────────────────────

def run_ppo_step(datums: list[dict], checkpoint_path: str, step_name: str) -> tuple[str, str]:
    import tinker
    sc = tinker.ServiceClient()
    tc = sc.create_training_client_from_state(checkpoint_path) if checkpoint_path \
        else sc.create_lora_training_client(base_model=BASE_MODEL, rank=LORA_RANK)
    adam = tinker.types.AdamParams(learning_rate=PPO_LR)

    tinker_datums = []
    for d in datums:
        tokens = d["full_tokens"]
        ob_len = d["ob_len"]
        ac_logprobs = d["ac_logprobs"]
        adv = d["advantage"]
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        n = len(input_tokens)
        padded_logprobs = [0.0] * ob_len + ac_logprobs
        padded_advantages = [0.0] * ob_len + [adv] * (n - ob_len)
        padded_logprobs = padded_logprobs[:n]
        padded_advantages = padded_advantages[:n]
        tinker_datums.append(tinker.types.Datum(
            model_input=tinker.types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs={"target_tokens": target_tokens, "logprobs": padded_logprobs, "advantages": padded_advantages},
        ))

    np.random.shuffle(tinker_datums)
    n_substeps = min(8, max(2, (len(tinker_datums) + BATCH_SIZE - 1) // BATCH_SIZE))
    for sub in range(n_substeps):
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
    logger.info(f"  train done: {n_substeps} substeps, {len(tinker_datums)} datums")
    return ckpt, sampler_path


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = BASE_DIR / "state.json"

    if state_file.exists():
        state = json.loads(state_file.read_text())
        logger.info(f"Resuming: step={state['step']}")
    else:
        state = {"step": 0, "checkpoint": INIT_CHECKPOINT, "sampler_path": INIT_SAMPLER}

    for step in range(state["step"], N_PPO_STEPS):
        state["step"] = step
        step_name = f"ppo-{step:03d}"
        step_dir = BASE_DIR / step_name
        tasks = sample_tasks(step)
        logger.info(f"\n{'═'*50}\n  GRPO {step}/{N_PPO_STEPS} ({len(tasks)} tasks)\n{'═'*50}")
        logger.info(f"  tasks: {[t.split('__')[1] for t in tasks]}")
        t0 = time.time()

        generate_rollouts(tasks, N_ROLLOUTS, step_dir, state["sampler_path"])
        rewards = compute_rewards(step_dir, tasks)
        datums = collect_ppo_datums(step_dir, rewards)

        if not datums:
            logger.warning("  no datums, skip training")
            state_file.write_text(json.dumps(state, indent=2))
            continue

        ckpt, sampler = run_ppo_step(datums, state["checkpoint"], step_name)
        state["checkpoint"] = ckpt
        state["sampler_path"] = sampler
        wall = time.time() - t0

        (step_dir / "step_results.json").write_text(json.dumps({
            "step": step_name, "wall_time": wall, "checkpoint": ckpt,
            "sampler_path": sampler, "rewards": rewards, "n_datums": len(datums),
            "avg_reward": float(np.mean([r for rs in rewards.values() for r in rs])),
        }, indent=2))
        state_file.write_text(json.dumps(state, indent=2))
        logger.info(f"  {step_name} done in {wall:.0f}s")

    # Final holdout eval
    logger.info(f"\n{'═'*50}\n  Final holdout eval\n{'═'*50}")
    eval_dir = BASE_DIR / "eval-final"
    generate_rollouts(EVAL_TASKS, EVAL_ROLLOUTS, eval_dir, state["sampler_path"])
    # Just log patch rates for now
    for rd in sorted(eval_dir.glob("rollout_*")):
        preds = json.loads((rd / "preds.json").read_text()) if (rd / "preds.json").exists() else {}
        patches = sum(1 for v in preds.values() if v.get("model_patch", "").strip())
        logger.info(f"  {rd.name}: {patches}/{len(preds)} patches")

    logger.info(f"\n  DONE. Checkpoint: {state['checkpoint']}")


if __name__ == "__main__":
    main()
