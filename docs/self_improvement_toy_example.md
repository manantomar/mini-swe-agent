# Self-Improvement Loop: Toy Example Results

## Overview

We tested a simple self-improvement loop on mini-swe-agent where:
1. An agent attempts a coding task
2. A critique model analyzes the failed trajectory
3. Actionable tips are injected into the agent's prompt
4. The agent retries with the improved prompt

## Task

**Bug fix in `calculator.py`**: Two functions lack error handling:
- `divide(a, b)` — no zero-division check
- `average(numbers)` — crashes on empty list

Tests expect `ValueError` with exact messages: `"Cannot divide by zero"` and `"Cannot average an empty list"`.

## Model

- **Coder agent**: `qwen/qwen3-8b` via OpenRouter
- **Critique model**: `qwen/qwen3-coder` via OpenRouter (litellm)

## Results

| | Iteration 0 (baseline) | Iteration 1 (with tips) |
|---|---|---|
| **Verified result** | ❌ Failed | ✅ Passed |
| **Steps** | 6 | 3 |
| **Cost** | $0.0014 | $0.0019 |

## What happened

### Iteration 0 — Baseline (failed)

The agent read the file, then tried to insert error-handling code with `sed`:

```bash
sed -i '32i\    if b == 0:\        raise ValueError("Cannot divide by zero")' calculator.py
sed -i '22i\    if not numbers:\        raise ValueError("Cannot average an empty list")' calculator.py
```

**Problem**: These sed commands silently failed — the line numbers were wrong (the file only has ~20 lines) and the multi-line syntax was broken. The file was left **completely unchanged**. The agent then ran `python3 test_calculator.py` (not `pytest`), which returned exit code 0 without actually running pytest-style assertions, so the agent believed tests passed and submitted.

### Critique output

The critique model analyzed the compact trace and produced:

```
- Always verify that your `sed` commands correctly insert newlines and maintain proper Python indentation when editing files.
- Run tests immediately after making changes to confirm that your fixes work as expected and do not introduce new issues.
- Double-check error message strings against the specification to ensure they match exactly, including capitalization and punctuation.
```

### Iteration 1 — With tips (succeeded)

The tips were prepended to the agent's `instance_template`. The agent then used a more robust sed approach:

```bash
sed -i '/def divide(a, b):/ { n; s/return a \/ b/\n    if b == 0:\n        raise ValueError("Cannot divide by zero")\n    return a \/ b/ }' calculator.py
```

This pattern-matched the function definition and replaced the return line with a guard clause, producing correctly indented Python. Solved in 3 steps.

## Files

- `src/minisweagent/run/critique.py` — Trace formatting and critique LLM call
- `src/minisweagent/run/self_improve.py` — CLI orchestration loop
- `tests/self_improve_fixture/` — The calculator task fixture
- `tests/test_critique.py` — Unit tests for the critique module

## Usage

```bash
python -m minisweagent.run.self_improve \
  -t "Fix the bugs in calculator.py so all tests pass" \
  --task-dir tests/self_improve_fixture \
  --verify "python -m pytest test_calculator.py -x -q" \
  -m "qwen/qwen3-8b" \
  --model-class openrouter \
  --critique-model "openrouter/qwen/qwen3-coder" \
  -n 3
```
