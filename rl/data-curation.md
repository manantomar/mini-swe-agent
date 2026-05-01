# Data Curation for RL Training

## Why curate?

GRPO with binary rewards only produces gradient signal from tasks with **mixed outcomes** 
across rollouts (some succeed, some fail). Tasks where all rollouts fail or all succeed 
contribute zero advantage → zero gradient → wasted compute.

In our first 6-step GRPO run:
- **75% of rollouts had zero advantage** (99/123 in step 1)
- Only ~6-8 tasks per step contributed any training signal
- ~75% of generation compute was wasted

## Curation strategy

1. **Screen all easy tasks** with k=2 rollouts to identify which ones the model can 
   solve at least sometimes
2. **Keep only "learnable" tasks** — those with 0 < pass@2 < 1 (sometimes succeed, 
   sometimes fail)
3. **Train on those** with 4+ rollouts per task → nearly every task produces mixed 
   rewards → dense gradient signal

## Screening results

### Pool: SWE-bench Verified, difficulty = "<15 min fix"

- Total easy tasks in dataset: 194
- Held out for eval: 10
- Available for training: 184
- Screened with k=2: 184 (56 from original pool + 128 newly cached)
- **Solvable (pass@2 > 0): 13 tasks**

### The 13 solvable easy tasks

| Task | Source | Solved (k=2 or k=10) | Notes |
|------|--------|---------------------|-------|
| django\_\_django-14373 | original pool | 9/10 | Very reliable |
| sympy\_\_sympy-16886 | original pool | 7/10 | Reliable |
| django\_\_django-11119 | original pool | 3/10 | Moderate |
| django\_\_django-15741 | original pool | 1/10 | Flaky |
| pytest-dev\_\_pytest-6202 | original pool | 1/10 | Flaky |
| django\_\_django-16255 | new screening | 2/2 | Very reliable |
| django\_\_django-16569 | new screening | 2/2 | Very reliable |
| django\_\_django-17029 | new screening | 2/2 | Very reliable |
| pytest-dev\_\_pytest-5809 | new screening | 2/2 | Very reliable |
| pytest-dev\_\_pytest-7982 | new screening | 2/2 | Very reliable |
| django\_\_django-16139 | new screening | 1/2 | Moderate |
| matplotlib\_\_matplotlib-20859 | new screening | 1/2 | Moderate |
| scikit-learn\_\_scikit-learn-14496 | new screening | 1/2 | Moderate |

### What about the other 171 easy tasks?

- 171/184 tasks were **never solved** in any rollout
- These are beyond the base Qwen3-8B's current capability
- Including them in RL training wastes compute (zero gradient)
- They could become useful later as the model improves (curriculum expansion)

## Eval set (held out, never trained on)

10 tasks from `eval/run_eval.sh`:
```
django__django-10097    django__django-11433    django__django-12308
django__django-13794    django__django-16100    psf__requests-1766
pylint-dev__pylint-4970 sphinx-doc__sphinx-10435 sphinx-doc__sphinx-9711
sympy__sympy-20916
```

## Medium difficulty tasks with signal

From the original pass@k eval (n=10), 4 medium tasks also showed solvability:
- django\_\_django-16527 (3/10)
- matplotlib\_\_matplotlib-26342 (3/10)  
- sympy\_\_sympy-21379 (1/10)
- sympy\_\_sympy-23950 (1/10)

These could be added to the training pool for a total of **17 tasks with learning signal**.

## Data files

- Solvable task list: `/data/manantomar/swe-bench-docker/screen-new-easy/all_solvable_easy.json`
- Screening rewards: `/data/manantomar/swe-bench-docker/screen-new-easy/rewards.json`
- Original pass@k: `/data/manantomar/swe-bench-docker/passk-base/pass_at_k_results.json`
- Screening traces: `/data/manantomar/swe-bench-docker/screen-new-easy/rollout_{00,01}/`
