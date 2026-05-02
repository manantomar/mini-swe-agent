#!/usr/bin/env python3
"""Async continuous RL training with adaptive task curriculum.

Key design:
- 128 containers run continuously (refill on completion)
- 5 min timeout kills straggler rollouts
- Train every time 128+ datums accumulate (not waiting for all rollouts)
- EMA-based task curriculum: high-signal tasks sampled more often
- Min 4 rollouts per task for GRPO advantage computation
- Patch similarity reward (SequenceMatcher vs gold patch)
- 1-step model lag handled by PPO importance sampling

Usage:
    export TINKER_API_KEY=...
    python rl/run_async_rl.py
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger("async_rl")

BASE_MODEL = "Qwen/Qwen3-8B"
BASE_DIR = Path("/data/manantomar/swe-bench-docker/async-rl")
PPO_LR = 5e-6
LORA_RANK = 64

GOLD_PATCHES = json.loads(Path("/data/manantomar/swe-bench-docker/gold_patches.json").read_text())
EASY_POOL = list(GOLD_PATCHES.keys())

EVAL_TASKS = [
    "django__django-10097", "django__django-11433", "django__django-12308",
    "django__django-13794", "django__django-16100", "psf__requests-1766",
    "pylint-dev__pylint-4970", "sphinx-doc__sphinx-10435",
    "sphinx-doc__sphinx-9711", "sympy__sympy-20916",
]

MAX_CONTAINERS = 128
ROLLOUT_TIMEOUT = 300  # 5 min
BATCH_SIZE = 128
MIN_ROLLOUTS_PER_TASK = 4
ROLLOUTS_PER_TASK = 8
N_PPO_STEPS = 50
STEP_LIMIT = 50
EMA_ALPHA = 0.3
SCORE_FLOOR = 0.05
TASKS_PER_BATCH = 16

TOOLS_SPEC = [{"type": "function", "function": {"name": "bash", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
MAX_DATUM_LEN = 8192


# ── Task Curriculum ─────────────────────────────────────────────────────────

class TaskCurriculum:
    def __init__(self, task_pool: list[str]):
        self.task_pool = task_pool
        self.scores: dict[str, float] = {t: SCORE_FLOOR for t in task_pool}
        self.sample_count: dict[str, int] = defaultdict(int)

    def update(self, task_id: str, rewards: list[float]):
        max_reward = max(rewards) if rewards else 0.0
        old = self.scores.get(task_id, SCORE_FLOOR)
        self.scores[task_id] = (1 - EMA_ALPHA) * old + EMA_ALPHA * max_reward
        self.sample_count[task_id] += 1

    def sample(self, n: int, rng: np.random.RandomState) -> list[str]:
        probs = np.array([max(self.scores[t], SCORE_FLOOR) for t in self.task_pool])
        probs = probs / probs.sum()
        chosen = list(rng.choice(self.task_pool, size=n, replace=False, p=probs))
        return chosen

    def summary(self) -> str:
        scored = sorted(self.scores.items(), key=lambda x: -x[1])
        top = [(t.split("__")[1], f"{s:.3f}") for t, s in scored[:5]]
        n_active = sum(1 for s in self.scores.values() if s > SCORE_FLOOR)
        return f"{n_active}/{len(self.task_pool)} active, top: {top}"


# ── Rollout Manager ─────────────────────────────────────────────────────────

class RolloutManager:
    """Manages continuous generation with container refill and timeouts."""

    def __init__(self, sampler_path: str, out_dir: Path):
        self.sampler_path = sampler_path
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.active: dict[str, dict] = {}  # uid -> {proc, task, start_time, rdir}
        self.completed: list[dict] = []  # [{task, rdir, uid}]
        self.lock = threading.Lock()
        self._counter = 0

    def launch_rollout(self, task_id: str):
        self._counter += 1
        uid = f"r{self._counter:04d}"
        rdir = self.out_dir / uid
        rdir.mkdir(parents=True, exist_ok=True)

        task_filter = f"^{task_id}$"
        cmd = [
            "mini-extra", "swebench", "--subset", "verified", "--split", "test",
            "--filter", task_filter, "-m", BASE_MODEL, "--model-class", "tinker",
            "-o", str(rdir), "-w", "1",
            "-c", "swebench.yaml",
            "-c", f"agent.step_limit={STEP_LIMIT}", "-c", "agent.cost_limit=100",
            "-c", "model.cost_tracking=ignore_errors",
            "-c", "model.model_kwargs.temperature=0.7",
            "-c", "environment.pull_timeout=300",
        ]
        if self.sampler_path:
            cmd += ["-c", f"model.tinker_checkpoint_path={self.sampler_path}"]

        proc = subprocess.Popen(
            cmd, env={**os.environ, "MSWEA_SILENT_STARTUP": "1"},
            stdout=open(rdir / "run.log", "w"), stderr=subprocess.STDOUT,
        )
        with self.lock:
            self.active[uid] = {"proc": proc, "task": task_id, "start_time": time.time(), "rdir": rdir}

    def poll(self) -> list[dict]:
        """Check for completed/timed-out rollouts. Returns newly completed ones."""
        newly_done = []
        now = time.time()
        with self.lock:
            for uid, info in list(self.active.items()):
                proc = info["proc"]
                elapsed = now - info["start_time"]

                if proc.poll() is not None:
                    # Completed normally
                    newly_done.append({"task": info["task"], "rdir": info["rdir"], "uid": uid, "killed": False})
                    self.completed.append(newly_done[-1])
                    del self.active[uid]
                elif elapsed > ROLLOUT_TIMEOUT:
                    # Timeout — kill it
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        try:
                            proc.kill()
                        except (ProcessLookupError, PermissionError):
                            pass
                    newly_done.append({"task": info["task"], "rdir": info["rdir"], "uid": uid, "killed": True})
                    del self.active[uid]

        return newly_done

    def n_active(self) -> int:
        with self.lock:
            return len(self.active)

    def update_sampler(self, new_sampler: str):
        self.sampler_path = new_sampler


# ── Reward + Datum Collection ───────────────────────────────────────────────

def compute_reward(rdir: Path, task_id: str) -> float:
    preds_path = rdir / "preds.json"
    if not preds_path.exists():
        return 0.0
    preds = json.loads(preds_path.read_text())
    if task_id not in preds:
        return 0.0
    model_patch = preds[task_id].get("model_patch", "").strip()
    if not model_patch:
        return 0.0
    gold_patch = GOLD_PATCHES.get(task_id, "")
    return SequenceMatcher(None, model_patch, gold_patch).ratio()


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


def build_datums_from_rollout(rdir: Path, task_id: str, advantage: float) -> list[dict]:
    tokenizer = _get_tokenizer()
    traj_files = list(rdir.glob(f"{task_id}/{task_id}.traj.json")) + list(rdir.glob(f"*/{task_id}.traj.json"))
    if not traj_files:
        return []
    traj = json.loads(traj_files[0].read_text())
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
        ob_msgs = [_clean_msg(prev) for prev in msgs[:i]]
        try:
            ob_text = tokenizer.apply_chat_template(ob_msgs, tools=TOOLS_SPEC, tokenize=False, add_generation_prompt=True)
            ob_tokens = tokenizer.encode(ob_text, add_special_tokens=False)
        except Exception:
            continue
        full_tokens = ob_tokens + ac_tokens
        if len(full_tokens) < 2 or len(full_tokens) > MAX_DATUM_LEN:
            continue
        ob_len = len(ob_tokens) - 1
        datums.append({"full_tokens": full_tokens, "ob_len": ob_len,
                       "ac_logprobs": ac_logprobs, "advantage": advantage})
    return datums


# ── PPO Training ────────────────────────────────────────────────────────────

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
    batch = tinker_datums[:BATCH_SIZE]
    fb = tc.forward_backward(batch, loss_fn="ppo",
        loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.7}).result()
    loss = fb.metrics.get("loss:sum", 0) / len(batch)
    tc.optim_step(adam).result()

    ckpt = tc.save_state(name=step_name).result().path
    sampler_path = tc.save_weights_for_sampler(name=f"{step_name}-sampler").result().path
    logger.info(f"  train: loss={loss:.4f} batch={len(batch)} total_datums={len(tinker_datums)}")
    return ckpt, sampler_path


# ── Main Loop ───────────────────────────────────────────────────────────────

def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = BASE_DIR / "state.json"

    if state_file.exists():
        state = json.loads(state_file.read_text())
        logger.info(f"Resuming: step={state['step']}")
    else:
        state = {"step": 0, "checkpoint": "", "sampler_path": ""}

    curriculum = TaskCurriculum(EASY_POOL)
    # Load curriculum scores if saved
    scores_file = BASE_DIR / "curriculum_scores.json"
    if scores_file.exists():
        curriculum.scores = json.loads(scores_file.read_text())

    rng = np.random.RandomState(42)

    for ppo_step in range(state["step"], N_PPO_STEPS):
        state["step"] = ppo_step
        step_name = f"ppo-{ppo_step:03d}"
        step_dir = BASE_DIR / step_name
        step_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'═'*50}\n  GRPO {ppo_step}/{N_PPO_STEPS}\n{'═'*50}")
        logger.info(f"  curriculum: {curriculum.summary()}")

        # Sample tasks for this step
        tasks = curriculum.sample(TASKS_PER_BATCH, rng)
        logger.info(f"  tasks: {[t.split('__')[1] for t in tasks]}")

        # Launch initial rollouts
        mgr = RolloutManager(state["sampler_path"], step_dir)
        task_queue = []
        for task in tasks:
            for _ in range(ROLLOUTS_PER_TASK):
                task_queue.append(task)
        rng.shuffle(task_queue)

        # Fill initial containers
        launched = 0
        while launched < len(task_queue) and mgr.n_active() < MAX_CONTAINERS:
            mgr.launch_rollout(task_queue[launched])
            launched += 1
            time.sleep(0.1)
        logger.info(f"  launched {launched} rollouts ({mgr.n_active()} active)")

        # Collect results with continuous refill
        task_rewards: dict[str, list[float]] = defaultdict(list)
        task_rollout_data: dict[str, list[dict]] = defaultdict(list)  # task -> [{rdir, reward}]
        t0 = time.time()

        while mgr.n_active() > 0 or launched < len(task_queue):
            newly_done = mgr.poll()

            for info in newly_done:
                task_id = info["task"]
                if info["killed"]:
                    logger.info(f"    killed: {task_id} ({info['uid']})")
                    continue

                reward = compute_reward(info["rdir"], task_id)
                task_rewards[task_id].append(reward)
                task_rollout_data[task_id].append({"rdir": info["rdir"], "reward": reward})

                # Refill
                if launched < len(task_queue) and mgr.n_active() < MAX_CONTAINERS:
                    mgr.launch_rollout(task_queue[launched])
                    launched += 1

            time.sleep(2)

        gen_time = time.time() - t0
        n_completed = sum(len(v) for v in task_rewards.values())
        avg_reward = np.mean([r for rs in task_rewards.values() for r in rs]) if task_rewards else 0
        logger.info(f"  gen done: {n_completed} rollouts in {gen_time:.0f}s, avg_reward={avg_reward:.3f}")

        # Update curriculum
        for task_id, rewards in task_rewards.items():
            curriculum.update(task_id, rewards)

        # Build datums — only tasks with >= MIN_ROLLOUTS_PER_TASK rollouts
        task_adv = {}
        for task_id, rewards in task_rewards.items():
            if len(rewards) < MIN_ROLLOUTS_PER_TASK:
                continue
            mean_r = np.mean(rewards)
            task_adv[task_id] = {r: r - mean_r for r in range(len(rewards))}

        datums = []
        for task_id, rollout_data in task_rollout_data.items():
            if task_id not in task_adv:
                continue
            for r_idx, rd in enumerate(rollout_data):
                if r_idx not in task_adv[task_id]:
                    continue
                adv = task_adv[task_id][r_idx]
                if abs(adv) < 1e-6:
                    continue
                datums.extend(build_datums_from_rollout(rd["rdir"], task_id, adv))

        pos = sum(1 for d in datums if d["advantage"] > 0)
        logger.info(f"  datums: {len(datums)} (pos={pos}, neg={len(datums)-pos})")

        if len(datums) < BATCH_SIZE:
            logger.warning(f"  only {len(datums)} datums, skip training")
            state_file.write_text(json.dumps(state, indent=2))
            scores_file.write_text(json.dumps(curriculum.scores))
            continue

        # Train
        ckpt, sampler = run_ppo_step(datums, state["checkpoint"], step_name)
        state["checkpoint"] = ckpt
        state["sampler_path"] = sampler
        mgr.update_sampler(sampler)

        wall = time.time() - t0
        (step_dir / "step_results.json").write_text(json.dumps({
            "step": step_name, "wall_time": wall, "checkpoint": ckpt,
            "sampler_path": sampler, "n_datums": len(datums),
            "avg_reward": float(avg_reward), "n_completed": n_completed,
            "task_rewards": {t: rs for t, rs in task_rewards.items()},
        }, indent=2))
        state_file.write_text(json.dumps(state, indent=2))
        scores_file.write_text(json.dumps(curriculum.scores))
        logger.info(f"  {step_name} done in {wall:.0f}s")

    logger.info(f"\n{'═'*50}\n  DONE — {state['checkpoint']}\n{'═'*50}")


if __name__ == "__main__":
    main()
