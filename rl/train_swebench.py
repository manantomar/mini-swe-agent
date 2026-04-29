#!/usr/bin/env python3
"""Convert scored agent trajectories to Tinker RL format and run training.

Takes trajectory files (with logprobs from score_logprobs.py) and SWE-bench
eval results, converts them into Tinker TrajectoryGroups, and runs GRPO
training.

Usage:
    export TINKER_API_KEY=your-key
    python rl/train_swebench.py \
        --model Qwen/Qwen3-8B \
        --traces-dir /data/manantomar/swe-bench-docker/qwen3-30b-50new \
        --scored-dir /data/manantomar/swe-bench-docker/qwen3-30b-50new-scored \
        --eval-report /path/to/eval_report.json \
        --log-dir /data/manantomar/swe-bench-docker/rl-training
"""

import argparse
import json
import logging
from pathlib import Path

import tinker
import torch
from tinker.types.tensor_data import TensorData
from transformers import AutoTokenizer

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.rl.data_processing import compute_advantages
from tinker_cookbook.rl.types import (
    Trajectory,
    TrajectoryGroup,
    Transition,
)
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.utils import ml_log

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Bash command to run"}},
            "required": ["command"],
        },
    },
}


def build_chat_messages(messages: list[dict]) -> list[dict]:
    """Convert trajectory messages to format expected by apply_chat_template."""
    cleaned = []
    for m in messages:
        msg = {"role": m["role"]}
        if m.get("content"):
            msg["content"] = m["content"]
        if m.get("tool_calls"):
            msg["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in m["tool_calls"]
            ]
        if m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        cleaned.append(msg)
    return cleaned


def trajectory_to_tinker_transitions(
    messages: list[dict],
    scored_steps: list[dict],
    tokenizer,
    step_reward: float = 0.0,
) -> list[Transition]:
    """Convert a scored trajectory into Tinker Transitions.

    Each (observation, action) pair corresponds to:
    - observation: everything up to the assistant message (tokenized)
    - action: the assistant message tokens + logprobs
    """
    cleaned = build_chat_messages(messages)
    transitions = []
    step_idx = 0

    for i, msg in enumerate(cleaned):
        if msg["role"] != "assistant":
            continue

        if step_idx >= len(scored_steps):
            break

        scored = scored_steps[step_idx]
        step_idx += 1

        if scored.get("error") or not scored.get("token_ids"):
            continue

        # Build observation: tokenize everything up to (not including) this assistant message
        prompt_messages = cleaned[:i]
        try:
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tools=[BASH_TOOL], tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        except Exception as e:
            logger.warning(f"Failed to tokenize prompt at step {i}: {e}")
            continue

        # Build observation as ModelInput
        ob = tinker.ModelInput(chunks=[tinker.types.EncodedTextChunk(tokens=prompt_tokens)])

        # Build action with logprobs
        ac = TokensWithLogprobs(
            tokens=scored["token_ids"],
            maybe_logprobs=scored["logprobs"],
        )

        # Intermediate reward is 0; final reward assigned at group level
        is_last = step_idx == len(scored_steps)
        transition = Transition(
            ob=ob,
            ac=ac,
            reward=step_reward if is_last else 0.0,
            episode_done=is_last,
        )
        transitions.append(transition)

    return transitions


def load_eval_results(eval_report_path: str) -> dict[str, float]:
    """Load SWE-bench eval results as instance_id -> reward mapping."""
    report = json.loads(Path(eval_report_path).read_text())
    resolved = set(report.get("resolved_ids", []))
    submitted = set(report.get("submitted_ids", []))
    rewards = {}
    for iid in submitted:
        rewards[iid] = 1.0 if iid in resolved else 0.0
    return rewards


def build_trajectory_groups(
    traces_dir: Path,
    scored_dir: Path,
    rewards: dict[str, float],
    tokenizer,
) -> list[TrajectoryGroup]:
    """Build TrajectoryGroups from scored trajectories.

    Since we have 1 rollout per task, each group has 1 trajectory.
    Advantages will be computed across groups (cross-task normalization).
    """
    groups = []

    for scored_file in sorted(scored_dir.glob("*.scored.json")):
        instance_id = scored_file.stem.replace(".scored", "")
        scored_data = json.loads(scored_file.read_text())

        if scored_data.get("skipped"):
            continue

        # Load original trajectory
        traj_file = traces_dir / instance_id / f"{instance_id}.traj.json"
        if not traj_file.exists():
            logger.warning(f"Trajectory file not found: {traj_file}")
            continue

        traj_data = json.loads(traj_file.read_text())
        reward = rewards.get(instance_id, 0.0)

        # Convert to Tinker transitions
        transitions = trajectory_to_tinker_transitions(
            messages=traj_data["messages"],
            scored_steps=scored_data["steps"],
            tokenizer=tokenizer,
        )

        if not transitions:
            logger.warning(f"No valid transitions for {instance_id}")
            continue

        # Build final observation (empty, since episode is done)
        final_ob = tinker.ModelInput(chunks=[])

        trajectory = Trajectory(transitions=transitions, final_ob=final_ob)
        group = TrajectoryGroup(
            trajectories_G=[trajectory],
            final_rewards_G=[reward],
            metrics_G=[{"instance_id_hash": hash(instance_id) % 10000}],
        )
        groups.append(group)
        logger.info(f"  {instance_id}: {len(transitions)} transitions, reward={reward}")

    return groups


def main():
    parser = argparse.ArgumentParser(description="Train with Tinker on SWE-bench trajectories")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--traces-dir", type=str, required=True)
    parser.add_argument("--scored-dir", type=str, required=True)
    parser.add_argument("--eval-report", type=str, required=True, help="SWE-bench eval report JSON")
    parser.add_argument("--log-dir", type=str, default="/data/manantomar/swe-bench-docker/rl-training")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", help="Build data but don't train")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load eval results
    logger.info(f"Loading eval results from {args.eval_report}...")
    rewards = load_eval_results(args.eval_report)
    resolved = sum(1 for r in rewards.values() if r > 0)
    logger.info(f"  {resolved}/{len(rewards)} resolved")

    # Build trajectory groups
    logger.info("Building trajectory groups...")
    groups = build_trajectory_groups(
        traces_dir=Path(args.traces_dir),
        scored_dir=Path(args.scored_dir),
        rewards=rewards,
        tokenizer=tokenizer,
    )
    logger.info(f"Built {len(groups)} trajectory groups")

    # Compute advantages
    # Since we have 1 rollout per task, merge all into one group for cross-task normalization
    all_trajectories = [g.trajectories_G[0] for g in groups]
    all_rewards = [g.final_rewards_G[0] for g in groups]
    all_metrics = [g.metrics_G[0] for g in groups]
    merged_group = TrajectoryGroup(
        trajectories_G=all_trajectories,
        final_rewards_G=all_rewards,
        metrics_G=all_metrics,
    )
    advantages_P = compute_advantages([merged_group])
    adv_tensor = advantages_P[0]
    pos = int((adv_tensor > 0).sum())
    neg = int((adv_tensor < 0).sum())
    mean_reward = sum(all_rewards) / len(all_rewards)
    logger.info(f"Advantages: {pos} positive, {neg} negative, mean reward={mean_reward:.3f}")

    # Re-split into per-task groups with assigned advantages for assembly
    groups_with_advantages = []
    single_advantages = []
    for i, g in enumerate(groups):
        groups_with_advantages.append(g)
        single_advantages.append(adv_tensor[i:i+1])

    if args.dry_run:
        logger.info("Dry run — skipping training")
        # Save data summary
        summary = {
            "n_groups": len(groups),
            "n_transitions": sum(len(g.trajectories_G[0].transitions) for g in groups),
            "n_positive_advantage": pos,
            "n_negative_advantage": neg,
            "mean_reward": mean_reward,
            "advantages_sample": [f"{a:.4f}" for a in adv_tensor[:10].tolist()],
        }
        summary_path = log_dir / "data_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info(f"Data summary saved to {summary_path}")
        return

    # ── Tinker training ──────────────────────────────────────────────────
    from tinker_cookbook.rl.data_processing import assemble_training_data

    logger.info("Connecting to Tinker...")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=args.model, rank=args.lora_rank,
    )

    # Assemble training data
    data_D, metadata_D = assemble_training_data(groups_with_advantages, single_advantages)
    logger.info(f"Assembled {len(data_D)} training datums")

    # Filter out very long datums that might cause issues
    MAX_DATUM_LEN = 8192
    filtered_data = [d for d in data_D if d.model_input.length <= MAX_DATUM_LEN]
    logger.info(f"After filtering (max {MAX_DATUM_LEN} tokens): {len(filtered_data)}/{len(data_D)} datums")

    # Training step — batch datums to avoid overwhelming the API
    adam_params = tinker.types.AdamParams(
        learning_rate=args.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    BATCH_SIZE = 64
    for batch_start in range(0, len(filtered_data), BATCH_SIZE):
        batch = filtered_data[batch_start:batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, len(filtered_data))
        logger.info(f"forward_backward batch [{batch_start}:{batch_end}] ({len(batch)} datums)...")
        fwd_bwd_future = training_client.forward_backward(batch, loss_fn="importance_sampling")
        fwd_bwd_result = fwd_bwd_future.result()
        logger.info(f"  batch done")

    logger.info("Running optim_step...")
    optim_future = training_client.optim_step(adam_params)
    optim_result = optim_future.result()
    logger.info(f"Training step complete. Metrics: {optim_result.metrics}")

    # Save checkpoint
    logger.info("Saving checkpoint...")
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="step_001",
        log_path=str(log_dir),
        kind="both",
        loop_state={"step": 1},
    )
    logger.info(f"Checkpoint saved to {log_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
