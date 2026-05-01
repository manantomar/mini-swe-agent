# Reward Shaping Options for SWE-bench RL

## The problem with binary rewards

Our GRPO training uses binary rewards: 1.0 if all fail-to-pass tests pass AND all 
pass-to-pass tests pass, 0.0 otherwise. This means:
- ~95% of rollouts get reward 0 (no gradient signal)
- Tasks the model never solves contribute nothing
- No distinction between "totally wrong" and "almost right"

## Alternative reward signals

### 1. Pass-to-pass test rate (p2p score)

**Score**: fraction of regression tests that still pass after applying the patch.

```
p2p_score = n_p2p_pass / (n_p2p_pass + n_p2p_fail)
```

**From our data**: 13/47 tasks produce patches that pass ALL p2p tests (score=1.0) 
but don't fix the bug. These get reward=0 with binary scoring, but should get partial 
credit — they at least don't break anything.

**Pros**: Easy to compute, already available from SWE-bench harness  
**Cons**: Rewards safe do-nothing patches

### 2. Fail-to-pass test rate (f2p score)

**Score**: fraction of target tests that now pass.

```
f2p_score = n_f2p_pass / (n_f2p_pass + n_f2p_fail)
```

**From our data**: No task partially fixes the target tests — it's all-or-nothing for 
f2p. But with more complex tasks this could provide gradient.

### 3. Combined test score

**Score**: weighted combination of p2p and f2p.

```
reward = 0.3 * p2p_score + 0.7 * f2p_score
```

This gives partial credit for non-breaking patches while strongly rewarding actual fixes.

### 4. Patch similarity to gold (SequenceMatcher)

**Score**: `difflib.SequenceMatcher(None, generated_patch, gold_patch).ratio()`

**From our data** (47 tasks with patches):
- 2 tasks >0.8 similarity (near-perfect)
- 9 tasks 0.5-0.8 (right area, close to fix)
- 23 tasks 0.2-0.5 (partial overlap)
- 13 tasks <0.2 (completely off)

**Pros**: Continuous signal, rewards getting closer to the right fix  
**Cons**: Requires access to gold patches (available in SWE-bench), doesn't reward 
alternative valid fixes, rewards surface-level similarity over correctness

### 5. File-level accuracy

**Score**: Does the patch modify the same file(s) as the gold patch?

```
file_score = len(predicted_files & gold_files) / len(gold_files)
```

**Pros**: Coarse signal that rewards finding the right code area  
**Cons**: Very coarse — many tasks only touch one file

### 6. Patch submission as reward

**Score**: 1.0 if a patch was submitted, 0.0 if the agent hit the step limit without 
submitting.

**From our data**: Only ~37% of rollouts produce a patch at all. 80% of holdout failures 
are "no patch submitted." Even rewarding patch submission (regardless of quality) could 
help the model learn to commit to a fix.

```
reward = 0.1 * submitted_patch + 0.3 * p2p_score + 0.6 * resolved
```

### 7. Multi-level reward scheme

Combine signals into a progressive reward:

| Level | Condition | Reward |
|-------|-----------|--------|
| 0 | No patch submitted | 0.0 |
| 1 | Patch submitted (any) | 0.1 |
| 2 | Patch applies cleanly | 0.2 |
| 3 | p2p tests pass | 0.4 |
| 4 | Some f2p tests pass | 0.7 |
| 5 | All tests pass (resolved) | 1.0 |

This gives gradient signal at every level of partial success.

## Data from iter-1 (base Qwen3-8B, 448 rollouts)

| Category | Rollouts | Current reward | Could get |
|----------|----------|---------------|-----------|
| No patch | ~280 | 0.0 | 0.0 |
| Patch but apply fails | ~70 | 0.0 | 0.1 |
| Apply ok, both tests fail | ~50 | 0.0 | 0.2 |
| p2p pass, f2p fail | ~30 | 0.0 | 0.4 |
| Fully resolved | 19 | 1.0 | 1.0 |

With the multi-level scheme, ~100 more rollouts would contribute non-zero gradient 
signal (vs only 19 currently).
