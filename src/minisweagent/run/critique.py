"""Critique agent traces and suggest prompt improvements."""

import json
import logging

import litellm

from minisweagent.models.utils.content_string import get_content_string

logger = logging.getLogger("minisweagent.critique")

CRITIQUE_PROMPT = """\
You are analyzing a coding agent's trajectory to identify mistakes.
The agent was given a task and interacted with a bash shell to solve it.
{outcome_line}

Here is the compact trace of the agent's actions and observations:

<trace>
{compact_trace}
</trace>

Based on this trace, output 1-3 short, actionable tips (one sentence each) \
that should be added to the agent's system prompt to avoid these mistakes in future runs.

Rules:
- Each tip must be concrete and actionable (e.g., "Always use absolute paths" not "Be more careful")
- Focus on the most impactful mistakes that cost the agent the most wasted steps
- Do not repeat tips that are already common sense
- Output ONLY the tips as a bulleted list, nothing else"""


def format_trace_compact(messages: list[dict]) -> str:
    """Extract a compact representation of a trajectory for critique."""
    lines: list[str] = []
    step = 0
    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
            continue
        if role == "assistant":
            step += 1
            content = get_content_string(msg)
            # Truncate long reasoning but keep commands fully visible
            reasoning_lines = content.split("\n")
            truncated = "\n".join(reasoning_lines[:20])
            if len(reasoning_lines) > 20:
                truncated += f"\n... ({len(reasoning_lines) - 20} more lines)"
            lines.append(f"[Step {step} - AGENT]\n{truncated}")
        elif role in ("user", "tool"):
            extra = msg.get("extra", {})
            if "returncode" in extra:
                rc = extra["returncode"]
                raw = extra.get("raw_output", msg.get("content", ""))
                if len(raw) > 500:
                    raw = raw[:250] + f"\n... ({len(raw) - 500} chars elided) ...\n" + raw[-250:]
                lines.append(f"[Observation] returncode={rc}\n{raw}")
            elif "interrupt_type" in extra:
                lines.append(f"[Format Error] {msg.get('content', '')[:200]}")
        elif role == "exit":
            lines.append(f"[Exit] status={msg.get('extra', {}).get('exit_status', 'unknown')}")
    return "\n\n".join(lines)


def get_critique_tips(
    compact_trace: str,
    *,
    model_name: str = "anthropic/claude-sonnet-4-5-20250929",
    succeeded: bool = False,
) -> str:
    """Call an LLM to critique a trace and return actionable tips."""
    outcome = "The agent SUCCEEDED." if succeeded else "The agent FAILED to solve the task."
    prompt = CRITIQUE_PROMPT.format(compact_trace=compact_trace, outcome_line=outcome)
    response = litellm.completion(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def inject_tips_into_config(config: dict, tips: str) -> dict:
    """Prepend tips to the instance_template in config. Returns a new config dict."""
    config = json.loads(json.dumps(config))  # deep copy
    original = config.get("agent", {}).get("instance_template", "")
    config["agent"]["instance_template"] = (
        f"## Tips from analyzing past attempts\n{tips}\n\n{original}"
    )
    return config
