#!/usr/bin/env python3
"""Async continuous RL v3 — shared rollout pool, no per-step directories.

Design:
- Single rollout directory, rollouts named by unique ID
- 128 containers run continuously; refill 8 when 8 slots free
- Rollout pool: completed rollouts accumulate with {task, reward, gen_step, used}
- Train when >=4 rollouts for >=8 tasks in unused pool
- After training: mark used rollouts; drop rollouts >10 steps old
- Generation never stops between training steps

Usage:
    export TINKER_API_KEY=...
    python rl/run_async_rl.py
"""

import json, logging, os, subprocess, sys, time, threading
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger("async_rl")

BASE_MODEL = "Qwen/Qwen3-8B"
BASE_DIR = Path("/data/manantomar/swe-bench-docker/async-rl-v3")
PPO_LR = 5e-6
LORA_RANK = 64
GOLD_PATCHES = json.loads(Path("/data/manantomar/swe-bench-docker/gold_patches.json").read_text())
EASY_POOL = list(GOLD_PATCHES.keys())

MAX_CONTAINERS = 128
ROLLOUT_TIMEOUT = 300
BATCH_SIZE = 128
MIN_ROLLOUTS = 4
MIN_TASKS = 8
ROLLOUTS_PER_TASK = 8
N_STEPS = 50
STEP_LIMIT_START = 30
STEP_LIMIT_INCREMENT = 5
STEP_LIMIT_MAX = 100
INIT_TASKS = 16
EMA_ALPHA = 0.3
SCORE_FLOOR = 0.05
MAX_DATUM_LEN = 8192
MAX_STALENESS = 10
TOOLS_SPEC = [{"type": "function", "function": {"name": "bash", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


class Curriculum:
    def __init__(self, pool):
        self.pool = pool
        self.scores = {t: SCORE_FLOOR for t in pool}

    def update(self, task, rewards):
        old = self.scores.get(task, SCORE_FLOOR)
        self.scores[task] = (1 - EMA_ALPHA) * old + EMA_ALPHA * (max(rewards) if rewards else 0)

    def sample_one(self, rng):
        p = np.array([max(self.scores[t], SCORE_FLOOR) for t in self.pool])
        return str(rng.choice(self.pool, p=p / p.sum()))

    def sample_n(self, n, rng):
        p = np.array([max(self.scores[t], SCORE_FLOOR) for t in self.pool])
        return [str(t) for t in rng.choice(self.pool, size=n, replace=False, p=p / p.sum())]

    def summary(self):
        active = sum(1 for s in self.scores.values() if s > SCORE_FLOOR)
        top = sorted(self.scores.items(), key=lambda x: -x[1])[:3]
        return f"{active}/{len(self.pool)} active, top={[(t.split('__')[1],f'{s:.2f}') for t,s in top if s>SCORE_FLOOR]}"

    def save(self, path): path.write_text(json.dumps(self.scores))
    def load(self, path):
        if path.exists(): self.scores = json.loads(path.read_text())


def do_launch(task, uid, out_dir, sampler, step_limit):
    rdir = out_dir / uid
    rdir.mkdir(parents=True, exist_ok=True)
    cmd = ["mini-extra", "swebench", "--subset", "verified", "--split", "test",
        "--filter", f"^{task}$", "-m", BASE_MODEL, "--model-class", "tinker",
        "-o", str(rdir), "-w", "1", "-c", "swebench.yaml",
        "-c", f"agent.step_limit={step_limit}", "-c", "agent.cost_limit=100",
        "-c", "model.cost_tracking=ignore_errors", "-c", "model.model_kwargs.temperature=0.7",
        "-c", "environment.pull_timeout=300"]
    if sampler:
        cmd += ["-c", f"model.tinker_checkpoint_path={sampler}"]
    proc = subprocess.Popen(cmd, env={**os.environ, "MSWEA_SILENT_STARTUP": "1", "HF_HUB_OFFLINE": "1"},
        stdout=open(rdir / "run.log", "w"), stderr=subprocess.STDOUT)
    return {"uid": uid, "task": task, "proc": proc, "rdir": rdir, "t0": time.time()}


def get_reward(rdir, task):
    p = rdir / "preds.json"
    if not p.exists(): return 0.0
    preds = json.loads(p.read_text())
    patch = preds.get(task, {}).get("model_patch", "").strip()
    return SequenceMatcher(None, patch, GOLD_PATCHES.get(task, "")).ratio() if patch else 0.0


# Shared dict for async swebench eval results: uid -> True/False/None(pending)
_swebench_results: dict[str, bool | None] = {}
_swebench_lock = threading.Lock()
_eval_uid_counter = 0


def _run_swebench_eval_bg(uid: str, rdir: Path, task: str):
    """Run swebench eval in background thread, update _swebench_results."""
    try:
        preds = json.loads((rdir / "preds.json").read_text())
        patch = preds.get(task, {}).get("model_patch", "").strip()
        if not patch:
            with _swebench_lock:
                _swebench_results[uid] = False
            return

        filtered = rdir / "swebench_preds.jsonl"
        with open(filtered, "w") as f:
            f.write(json.dumps({"instance_id": task, "model_name_or_path": "asyncrl",
                                "model_patch": patch}) + "\n")

        global _eval_uid_counter
        _eval_uid_counter += 1
        run_id = f"asyncrl-{_eval_uid_counter:05d}"

        subprocess.run(
            ["python3", "-m", "swebench.harness.run_evaluation",
             "--dataset_name", "princeton-nlp/SWE-bench_Verified",
             "--split", "test", "--predictions_path", str(filtered),
             "--run_id", run_id, "--max_workers", "4", "--timeout", "300"],
            capture_output=True, text=True, env=os.environ, timeout=600,
        )

        resolved = False
        report_dir = Path(f"logs/run_evaluation/{run_id}/asyncrl")
        if report_dir.exists():
            for td in report_dir.iterdir():
                rf = td / "report.json"
                if rf.exists():
                    r = json.loads(rf.read_text())
                    for tid, info in r.items():
                        if info.get("resolved"):
                            resolved = True

        with _swebench_lock:
            _swebench_results[uid] = resolved
        if resolved:
            logger.info(f"    ✓ RESOLVED: {task.split('__')[1]} ({uid})")
    except Exception as e:
        with _swebench_lock:
            _swebench_results[uid] = False


def start_swebench_eval(uid: str, rdir: Path, task: str):
    """Fire off swebench eval in background thread."""
    with _swebench_lock:
        _swebench_results[uid] = None  # pending
    t = threading.Thread(target=_run_swebench_eval_bg, args=(uid, rdir, task), daemon=True)
    t.start()


def get_swebench_result(uid: str) -> bool | None:
    """Get swebench result: True=resolved, False=not, None=pending."""
    with _swebench_lock:
        return _swebench_results.get(uid)


def _tok():
    if not hasattr(_tok, "_t"):
        from transformers import AutoTokenizer
        _tok._t = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    return _tok._t


def _clean(m):
    c = {"role": m["role"]}
    if m.get("content"): c["content"] = m["content"]
    if m.get("tool_calls"):
        c["tool_calls"] = [{"id": tc.get("id",""), "type": "function",
            "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
            for tc in m["tool_calls"]]
    if m.get("tool_call_id"): c["tool_call_id"] = m["tool_call_id"]
    return c


def make_datums(rdir, task, adv):
    tok = _tok()
    files = list(rdir.glob(f"{task}/*.traj.json")) + list(rdir.glob(f"*/{task}.traj.json"))
    if not files: return []
    msgs = json.loads(files[0].read_text())["messages"]
    out = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant" or "extra" not in m: continue
        lp = m["extra"].get("logprobs", {})
        ac = lp.get("token_ids", []); aclp = lp.get("logprobs", [])
        if len(ac) < 2: continue
        try:
            ob = tok.encode(tok.apply_chat_template([_clean(p) for p in msgs[:i]],
                tools=TOOLS_SPEC, tokenize=False, add_generation_prompt=True), add_special_tokens=False)
        except: continue
        full = ob + ac
        if len(full) < 2 or len(full) > MAX_DATUM_LEN: continue
        out.append({"full_tokens": full, "ob_len": len(ob)-1, "ac_logprobs": aclp, "advantage": adv})
    return out


def do_train(datums, ckpt, name):
    import tinker
    sc = tinker.ServiceClient()
    tc = sc.create_training_client_from_state(ckpt) if ckpt else sc.create_lora_training_client(base_model=BASE_MODEL, rank=LORA_RANK)
    adam = tinker.types.AdamParams(learning_rate=PPO_LR)
    td = []
    for d in datums:
        t = d["full_tokens"]; n = len(t)-1; ol = d["ob_len"]
        lp = ([0.0]*ol + d["ac_logprobs"])[:n]; lp += [0.0]*max(0, n-len(lp))
        av = ([0.0]*ol + [d["advantage"]]*(n-ol))[:n]
        td.append(tinker.types.Datum(model_input=tinker.types.ModelInput.from_ints(tokens=t[:-1]),
            loss_fn_inputs={"target_tokens": t[1:], "logprobs": lp, "advantages": av}))
    np.random.shuffle(td)
    n_substeps = min(8, max(1, (len(td) + BATCH_SIZE - 1) // BATCH_SIZE))
    losses = []
    for sub in range(n_substeps):
        batch = td[sub * BATCH_SIZE:(sub + 1) * BATCH_SIZE]
        if not batch: break
        fb = tc.forward_backward(batch, loss_fn="ppo", loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.3}).result()
        loss = fb.metrics.get("loss:sum", 0) / len(batch)
        losses.append(loss)
        tc.optim_step(adam).result()
        logger.info(f"    substep {sub}: loss={loss:.4f} batch={len(batch)}")
    c = tc.save_state(name=name).result().path
    s = tc.save_weights_for_sampler(name=f"{name}-sampler").result().path
    logger.info(f"  train done: {n_substeps} substeps, {len(td)} datums")
    return c, s, losses


def update_plots(base_dir):
    """Generate live training plots after each PPO step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, avg_rewards, n_datums, pos_fracs, losses, patch_rates, n_resolved = [], [], [], [], [], [], []
    for f in sorted(base_dir.glob("step_*.json")):
        r = json.loads(f.read_text())
        steps.append(r["step"])
        avg_rewards.append(r.get("avg_reward", 0))
        n_datums.append(r.get("datums", 0))
        pos_fracs.append(r.get("pos_frac", 0))
        losses.append(r.get("losses", []))
        patch_rates.append(r.get("patch_rate", 0))
        n_resolved.append(r.get("n_resolved", 0))

    if len(steps) < 2:
        return

    # Style matching pass@k curves
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Async RL Training Progress", fontsize=16, fontweight="bold", y=0.98)
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800", "#009688"]

    # 1. Avg patch similarity reward
    axes[0, 0].fill_between(steps, avg_rewards, alpha=0.15, color=colors[0])
    axes[0, 0].plot(steps, avg_rewards, "o-", color=colors[0], markersize=4, linewidth=2)
    axes[0, 0].set_title("Avg Patch Similarity", fontweight="bold")
    axes[0, 0].set_xlabel("PPO Step")
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].grid(True, alpha=0.2, linestyle="--")

    # 2. Patch submission rate
    axes[0, 1].fill_between(steps, patch_rates, alpha=0.15, color=colors[1])
    axes[0, 1].plot(steps, patch_rates, "o-", color=colors[1], markersize=4, linewidth=2)
    axes[0, 1].set_title("Patch Submission Rate", fontweight="bold")
    axes[0, 1].set_xlabel("PPO Step")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[0, 1].grid(True, alpha=0.2, linestyle="--")

    # 3. Swebench resolved count
    axes[0, 2].bar(steps, n_resolved, color=colors[5], alpha=0.7, width=0.8)
    axes[0, 2].set_title("Swebench Resolved (per step)", fontweight="bold")
    axes[0, 2].set_xlabel("PPO Step")
    axes[0, 2].grid(True, alpha=0.2, linestyle="--", axis="y")

    # 4. Datums per step
    axes[1, 0].fill_between(steps, n_datums, alpha=0.15, color=colors[3])
    axes[1, 0].plot(steps, n_datums, "o-", color=colors[3], markersize=4, linewidth=2)
    axes[1, 0].set_title("Datums per Step", fontweight="bold")
    axes[1, 0].set_xlabel("PPO Step")
    axes[1, 0].grid(True, alpha=0.2, linestyle="--")

    # 5. Positive datum fraction
    axes[1, 1].fill_between(steps, pos_fracs, alpha=0.15, color=colors[2])
    axes[1, 1].plot(steps, pos_fracs, "o-", color=colors[2], markersize=4, linewidth=2)
    axes[1, 1].set_title("Positive Datum Fraction", fontweight="bold")
    axes[1, 1].set_xlabel("PPO Step")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[1, 1].grid(True, alpha=0.2, linestyle="--")

    # 6. PPO loss per substep
    all_losses = []
    for step, ls in zip(steps, losses):
        for l in ls:
            all_losses.append((step, l))
    if all_losses:
        xs, ys = zip(*all_losses)
        axes[1, 2].scatter(xs, ys, s=15, alpha=0.5, color=colors[4], edgecolors="none")
        axes[1, 2].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[1, 2].set_title("PPO Loss (per substep)", fontweight="bold")
    axes[1, 2].set_xlabel("PPO Step")
    axes[1, 2].grid(True, alpha=0.2, linestyle="--")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(base_dir / "training_progress.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    rollout_dir = BASE_DIR / "rollouts"
    rollout_dir.mkdir(exist_ok=True)
    sf = BASE_DIR / "state.json"
    state = json.loads(sf.read_text()) if sf.exists() else {"step": 0, "checkpoint": "", "sampler_path": ""}
    cur = Curriculum(EASY_POOL)
    cur.load(BASE_DIR / "scores.json")
    rng = np.random.RandomState(42)
    uidc = 0

    # Rollout pool: [{uid, task, rdir, reward, gen_step, used}]
    pool = []
    pool_file = BASE_DIR / "pool.json"
    if pool_file.exists():
        pool = json.loads(pool_file.read_text())
        for r in pool: r["rdir"] = Path(r["rdir"])

    active = {}
    ppo_step = state["step"]
    freed = 0

    def save_all():
        json.dump([{**r, "rdir": str(r["rdir"])} for r in pool], pool_file.open("w"))
        sf.write_text(json.dumps(state))
        cur.save(BASE_DIR / "scores.json")

    def cur_step_limit():
        return min(STEP_LIMIT_MAX, STEP_LIMIT_START + ppo_step * STEP_LIMIT_INCREMENT)

    # Initial launch
    logger.info(f"\n  Starting async RL from step {ppo_step}, pool={len(pool)} rollouts, step_limit={cur_step_limit()}")
    for task in cur.sample_n(INIT_TASKS, rng):
        for _ in range(ROLLOUTS_PER_TASK):
            uidc += 1; uid = f"r{uidc:05d}"
            active[uid] = do_launch(task, uid, rollout_dir, state["sampler_path"], cur_step_limit())
            time.sleep(0.1)
    logger.info(f"  launched {len(active)}")

    while ppo_step < N_STEPS:
        now = time.time()

        # Poll active rollouts
        for uid, r in list(active.items()):
            if r["proc"].poll() is not None:
                rw = get_reward(r["rdir"], r["task"])
                pool.append({"uid": uid, "task": r["task"], "rdir": r["rdir"],
                             "reward": rw, "gen_step": ppo_step, "used": False})
                cur.update(r["task"], [rw])
                # Fire swebench eval in background if patch exists
                if rw > 0.05:
                    start_swebench_eval(uid, r["rdir"], r["task"])
                del active[uid]; freed += 1
            elif now - r["t0"] > ROLLOUT_TIMEOUT:
                try: r["proc"].terminate()
                except: pass
                logger.info(f"    timeout: {r['task'].split('__')[1]} ({uid})")
                del active[uid]; freed += 1

        # Refill
        while freed >= ROLLOUTS_PER_TASK and len(active) + ROLLOUTS_PER_TASK <= MAX_CONTAINERS:
            nt = cur.sample_one(rng)
            for _ in range(ROLLOUTS_PER_TASK):
                uidc += 1; uid = f"r{uidc:05d}"
                active[uid] = do_launch(nt, uid, rollout_dir, state["sampler_path"], cur_step_limit())
            freed -= ROLLOUTS_PER_TASK
            n_unused = sum(1 for r in pool if not r["used"])
            logger.info(f"    +{nt.split('__')[1]} ({len(active)} active, {n_unused} unused)")

        # Check training condition on UNUSED rollouts
        unused = [r for r in pool if not r["used"]]
        task_rollouts = defaultdict(list)
        for r in unused:
            task_rollouts[r["task"]].append(r)
        eligible = {t: rs for t, rs in task_rollouts.items() if len(rs) >= MIN_ROLLOUTS}

        if len(eligible) >= MIN_TASKS:
            logger.info(f"\n{'='*50}\n  GRPO {ppo_step}/{N_STEPS} — {len(eligible)} tasks ready\n{'='*50}")
            logger.info(f"  {cur.summary()}")

            # Upgrade rewards with swebench results before building datums
            n_resolved = 0
            for r in pool:
                result = get_swebench_result(r["uid"])
                if result is True:
                    r["reward"] = 1.0
                    n_resolved += 1
            if n_resolved:
                logger.info(f"  {n_resolved} rollouts upgraded to reward=1.0 (swebench resolved)")

            # Build datums
            datums = []
            used_uids = set()
            for task, rollouts in eligible.items():
                rewards = [r["reward"] for r in rollouts]
                mean_r = np.mean(rewards)
                for r in rollouts:
                    adv = r["reward"] - mean_r
                    if abs(adv) < 1e-6: continue
                    ds = make_datums(r["rdir"], task, adv)
                    datums.extend(ds)
                    used_uids.add(r["uid"])

            pos = sum(1 for d in datums if d["advantage"] > 0)
            avg_r = np.mean([r["reward"] for rs in eligible.values() for r in rs])
            logger.info(f"  datums: {len(datums)} (pos={pos}), from {len(used_uids)} rollouts, avg_reward={avg_r:.3f}")

            if len(datums) >= BATCH_SIZE:
                c, s, losses = do_train(datums, state["checkpoint"], f"ppo-{ppo_step:03d}")
                state["checkpoint"] = c; state["sampler_path"] = s

                # Mark used rollouts
                for r in pool:
                    if r["uid"] in used_uids:
                        r["used"] = True

                # Drop stale rollouts
                before = len(pool)
                pool = [r for r in pool if ppo_step - r["gen_step"] <= MAX_STALENESS]
                if before > len(pool):
                    logger.info(f"  dropped {before - len(pool)} stale rollouts")

                # Compute patch submission rate from eligible rollouts
                all_eligible_rollouts = [r for rs in eligible.values() for r in rs]
                patch_rate = sum(1 for r in all_eligible_rollouts if r["reward"] > 0) / len(all_eligible_rollouts) if all_eligible_rollouts else 0

                pos_frac = pos / len(datums) if datums else 0
                (BASE_DIR / f"step_{ppo_step:03d}.json").write_text(json.dumps({
                    "step": ppo_step, "checkpoint": c, "sampler": s,
                    "datums": len(datums), "used_rollouts": len(used_uids),
                    "pool_size": len(pool), "n_tasks": len(eligible),
                    "avg_reward": float(avg_r), "pos_frac": pos_frac, "losses": losses,
                    "patch_rate": patch_rate, "n_resolved": n_resolved,
                }, indent=2))
                save_all()
                update_plots(BASE_DIR)
                logger.info(f"  ppo-{ppo_step:03d} done, pool={len(pool)} ({sum(1 for r in pool if not r['used'])} unused)")

                ppo_step += 1
                state["step"] = ppo_step
            else:
                logger.warning(f"  only {len(datums)} datums, need more rollouts")

        time.sleep(30)

    for r in active.values():
        try: r["proc"].terminate()
        except: pass
    save_all()
    logger.info(f"\n  DONE: {state['checkpoint']}")

if __name__ == "__main__":
    main()
