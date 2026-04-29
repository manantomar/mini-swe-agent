# mini-swe-agent Evaluation Harness

Run 5 SWE-bench Verified tasks across multiple LLM backends and compare results.

## Selected Instances

| Instance ID | Repo | Difficulty |
|---|---|---|
| `django__django-10097` | django/django | <15 min |
| `django__django-11433` | django/django | <15 min |
| `django__django-12308` | django/django | <15 min |
| `django__django-13794` | django/django | <15 min |
| `django__django-16100` | django/django | <15 min |
| `psf__requests-1766` | psf/requests | <15 min |
| `pylint-dev__pylint-4970` | pylint-dev/pylint | <15 min |
| `sphinx-doc__sphinx-10435` | sphinx-doc/sphinx | <15 min |
| `sphinx-doc__sphinx-9711` | sphinx-doc/sphinx | <15 min |
| `sympy__sympy-20916` | sympy/sympy | <15 min |

## Default Models

- `anthropic/claude-sonnet-4-5-20250929` — requires `ANTHROPIC_API_KEY`
- `openai/gpt-4o-2024-11-20` — requires `OPENAI_API_KEY`
- `gemini/gemini-2.5-pro-preview-05-06` — requires `GEMINI_API_KEY`

## Quick Start

```bash
# 1. Set API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."

# 2. Run the evaluation (all 3 models × 10 tasks = 30 runs)
bash eval/run_eval.sh

# 3. Or run only specific models
bash eval/run_eval.sh anthropic/claude-sonnet-4-5-20250929

# 4. Compare results afterwards
python3 eval/compare_results.py eval/results/
```

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `COST_LIMIT` | `3.0` | Max cost ($) per instance |
| `WORKERS` | `1` | Parallel workers per model run |
| `RESULTS_DIR` | `eval/results/` | Where to save outputs |

## Output Structure

```
eval/results/
├── anthropic__claude-sonnet-4-5-20250929/
│   ├── preds.json                          # All predictions
│   ├── minisweagent.log                    # Run log
│   ├── psf__requests-1142/
│   │   └── psf__requests-1142.traj.json    # Full trajectory
│   ├── matplotlib__matplotlib-13989/
│   │   └── ...
│   └── ...
├── openai__gpt-4o-2024-11-20/
│   └── ...
└── gemini__gemini-2.5-pro-preview-05-06/
    └── ...
```

## Evaluating Results

The harness tells you whether the agent produced a patch, but to check if the patch
actually **fixes** the issue, use the official SWE-bench evaluation:

```bash
# Option A: Cloud evaluation (free, fast — recommended)
pip install sb-cli
sb-cli submit swe-bench_verified test \
  --predictions_path eval/results/MODEL_DIR/preds.json \
  --run_id my-run

# Option B: Local evaluation
pip install swebench
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path eval/results/MODEL_DIR/preds.json \
  --max_workers 4 \
  --run_id my-run
```

## Customizing

### Add more models
```bash
bash eval/run_eval.sh openai/o4-mini-2025-04-16 together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
```

### Change instances
Edit the `INSTANCES` array in `eval/run_eval.sh` and the `INSTANCES` list in `eval/compare_results.py`.

### Lower cost for testing
```bash
COST_LIMIT=0.50 bash eval/run_eval.sh openai/gpt-4o-2024-11-20
```
