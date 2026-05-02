#!/usr/bin/env python3
"""Async continuous RL with adaptive task curriculum.

Design:
- Start 16 tasks x 8 rollouts = 128 containers
- Every 30s: when 8 slots free, sample new task via EMA curriculum, launch 8 rollouts
- Train when >=4 rollouts for >=8 tasks -> build 128-datum batch, PPO step
- 5 min timeout on stragglers
- Patch similarity reward (SequenceMatcher)
"""

import json, logging, os, signal, subprocess, sys, time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger("async_rl")

BASE_MODEL = "Qwen/Qwen3-8B"
BASE_DIR = Path("/data/manantomar/swe-bench-docker/async-rl-v2")
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
STEP_LIMIT = 50
INIT_TASKS = 16
EMA_ALPHA = 0.3
SCORE_FLOOR = 0.05
MAX_DATUM_LEN = 8192
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


def do_launch(task, uid, out_dir, sampler):
    rdir = out_dir / uid
    rdir.mkdir(parents=True, exist_ok=True)
    cmd = ["mini-extra", "swebench", "--subset", "verified", "--split", "test",
        "--filter", f"^{task}$", "-m", BASE_MODEL, "--model-class", "tinker",
        "-o", str(rdir), "-w", "1", "-c", "swebench.yaml",
        "-c", f"agent.step_limit={STEP_LIMIT}", "-c", "agent.cost_limit=100",
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
    batch = td[:BATCH_SIZE]
    fb = tc.forward_backward(batch, loss_fn="ppo", loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.7}).result()
    loss = fb.metrics.get("loss:sum", 0) / len(batch)
    tc.optim_step(adam).result()
    c = tc.save_state(name=name).result().path
    s = tc.save_weights_for_sampler(name=f"{name}-sampler").result().path
    logger.info(f"  train: loss={loss:.4f} batch={len(batch)}/{len(td)}")
    return c, s


def kill_proc(r):
    try: os.killpg(os.getpgid(r["proc"].pid), signal.SIGTERM)
    except: pass
    try: r["proc"].kill()
    except: pass


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    sf = BASE_DIR / "state.json"
    state = json.loads(sf.read_text()) if sf.exists() else {"step": 0, "checkpoint": "", "sampler_path": ""}
    cur = Curriculum(EASY_POOL)
    cur.load(BASE_DIR / "scores.json")
    rng = np.random.RandomState(42)
    uidc = 0

    for step in range(state["step"], N_STEPS):
        state["step"] = step
        sn = f"ppo-{step:03d}"
        sd = BASE_DIR / sn; sd.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n{'='*50}\n  GRPO {step}/{N_STEPS}\n{'='*50}")
        logger.info(f"  {cur.summary()}")

        active = {}
        done_r = defaultdict(list)
        done_d = defaultdict(list)
        freed = 0; t0 = time.time()

        for task in cur.sample_n(INIT_TASKS, rng):
            for _ in range(ROLLOUTS_PER_TASK):
                uidc += 1
                active[f"r{uidc:05d}"] = do_launch(task, f"r{uidc:05d}", sd, state["sampler_path"])
                time.sleep(0.1)
        logger.info(f"  launched {len(active)}")

        while active:
            now = time.time()
            for uid, r in list(active.items()):
                if r["proc"].poll() is not None:
                    rw = get_reward(r["rdir"], r["task"])
                    done_r[r["task"]].append(rw)
                    done_d[r["task"]].append({"rdir": r["rdir"], "reward": rw})
                    del active[uid]; freed += 1
                elif now - r["t0"] > ROLLOUT_TIMEOUT:
                    kill_proc(r)
                    logger.info(f"    timeout: {r['task'].split('__')[1]} ({uid})")
                    del active[uid]; freed += 1

            while freed >= ROLLOUTS_PER_TASK and len(active) + ROLLOUTS_PER_TASK <= MAX_CONTAINERS:
                nt = cur.sample_one(rng)
                for _ in range(ROLLOUTS_PER_TASK):
                    uidc += 1
                    active[f"r{uidc:05d}"] = do_launch(nt, f"r{uidc:05d}", sd, state["sampler_path"])
                freed -= ROLLOUTS_PER_TASK
                nd = sum(len(v) for v in done_r.values())
                logger.info(f"    +{nt.split('__')[1]} ({len(active)} active, {nd} done)")

            elig = [t for t, rs in done_r.items() if len(rs) >= MIN_ROLLOUTS]
            if len(elig) >= MIN_TASKS:
                logger.info(f"  ready: {len(elig)} tasks ({time.time()-t0:.0f}s)")
                break
            time.sleep(30)

        for r in active.values(): kill_proc(r)
        time.sleep(3)

        gt = time.time() - t0
        nc = sum(len(v) for v in done_r.values())
        ar = np.mean([r for rs in done_r.values() for r in rs]) if done_r else 0
        logger.info(f"  gen: {nc} rollouts, {len(done_r)} tasks, avg_r={ar:.3f} ({gt:.0f}s)")

        for t, rs in done_r.items(): cur.update(t, rs)

        datums = []
        for t in [t for t, rs in done_r.items() if len(rs) >= MIN_ROLLOUTS]:
            mr = np.mean(done_r[t])
            for i, rd in enumerate(done_d[t]):
                if i >= len(done_r[t]): break
                a = done_r[t][i] - mr
                if abs(a) < 1e-6: continue
                datums.extend(make_datums(rd["rdir"], t, a))

        pos = sum(1 for d in datums if d["advantage"] > 0)
        logger.info(f"  datums: {len(datums)} (pos={pos})")

        if len(datums) < BATCH_SIZE:
            logger.warning(f"  skip ({len(datums)}<{BATCH_SIZE})")
            sf.write_text(json.dumps(state)); cur.save(BASE_DIR / "scores.json")
            continue

        c, s = do_train(datums, state["checkpoint"], sn)
        state["checkpoint"] = c; state["sampler_path"] = s
        w = time.time() - t0
        (sd / "results.json").write_text(json.dumps({"step": sn, "wall": w, "ckpt": c, "sampler": s,
            "datums": len(datums), "avg_r": float(ar), "done": nc, "tasks": len(done_r), "gen_s": gt}, indent=2))
        sf.write_text(json.dumps(state)); cur.save(BASE_DIR / "scores.json")
        logger.info(f"  {sn} done in {w:.0f}s")

    logger.info(f"\n  DONE: {state['checkpoint']}")

if __name__ == "__main__":
    main()
