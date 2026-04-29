"""Tests for the critique and self-improvement modules."""

import pytest

from minisweagent.run.critique import format_trace_compact, inject_tips_into_config

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Fix the bug in calculator.py"},
    {
        "role": "assistant",
        "content": "Let me look at the code.",
        "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "cat calculator.py"}'}}],
        "extra": {"actions": [{"command": "cat calculator.py"}], "cost": 0.001},
    },
    {
        "role": "tool",
        "content": "<returncode>0</returncode><output>def add(a,b): return a+b</output>",
        "extra": {"returncode": 0, "raw_output": "def add(a,b): return a+b"},
    },
    {
        "role": "assistant",
        "content": "I see the issue.",
        "extra": {"actions": [{"command": "sed -i 's/old/new/' calculator.py"}], "cost": 0.001},
    },
    {
        "role": "user",
        "content": "Format error: Expected 1 action, found 0.",
        "extra": {"interrupt_type": "FormatError"},
    },
]


def test_format_trace_compact_extracts_steps():
    compact = format_trace_compact(SAMPLE_MESSAGES)
    assert "[Step 1 - AGENT]" in compact
    assert "[Step 2 - AGENT]" in compact
    assert "[Observation] returncode=0" in compact
    assert "[Format Error]" in compact
    # System message should be excluded
    assert "You are a helpful assistant" not in compact


def test_format_trace_compact_truncates_long_observations():
    messages = [
        {"role": "tool", "extra": {"returncode": 0, "raw_output": "x" * 2000}},
    ]
    compact = format_trace_compact(messages)
    assert "elided" in compact
    assert len(compact) < 2000


def test_format_trace_compact_handles_empty():
    assert format_trace_compact([]) == ""


@pytest.mark.parametrize(
    ("messages", "expected_steps"),
    [
        ([{"role": "system", "content": "hi"}], 0),
        ([{"role": "assistant", "content": "hi", "extra": {"actions": []}}], 1),
    ],
)
def test_format_trace_compact_step_count(messages, expected_steps):
    compact = format_trace_compact(messages)
    assert compact.count("[Step ") == expected_steps


def test_inject_tips_prepends_to_template():
    config = {"agent": {"instance_template": "Solve: {{task}}", "step_limit": 50}}
    tips = "- Use absolute paths"
    updated = inject_tips_into_config(config, tips)
    assert updated["agent"]["instance_template"].startswith("## Tips from analyzing past attempts")
    assert "- Use absolute paths" in updated["agent"]["instance_template"]
    assert "Solve: {{task}}" in updated["agent"]["instance_template"]
    # Original unchanged
    assert config["agent"]["instance_template"] == "Solve: {{task}}"


def test_inject_tips_preserves_other_config():
    config = {"agent": {"instance_template": "hi", "step_limit": 50}, "model": {"model_name": "test"}}
    updated = inject_tips_into_config(config, "tip")
    assert updated["agent"]["step_limit"] == 50
    assert updated["model"]["model_name"] == "test"
