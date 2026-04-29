#!/usr/bin/env bash
# =============================================================================
# mini-swe-agent evaluation harness
# Runs 10 SWE-bench Verified tasks across multiple models and collects results.
#
# Usage:
#   # Set your API keys first:
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   export OPENAI_API_KEY="sk-..."
#   export GEMINI_API_KEY="..."         # for gemini/ models via litellm
#
#   # Run all models (default):
#   bash eval/run_eval.sh
#
#   # Run specific models:
#   bash eval/run_eval.sh anthropic/claude-sonnet-4-5-20250929
#   bash eval/run_eval.sh openai/gpt-4o-2024-11-20 gemini/gemini-2.5-pro-preview-05-06
#
#   # Set a custom cost limit per instance (default $3):
#   COST_LIMIT=1.0 bash eval/run_eval.sh
#
#   # Use more workers for parallelism within a model run:
#   WORKERS=2 bash eval/run_eval.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
COST_LIMIT="${COST_LIMIT:-2.0}"
WORKERS="${WORKERS:-1}"

# Load API keys from mini-swe-agent global config if not already set
GLOBAL_ENV="$HOME/.config/mini-swe-agent/.env"
if [[ -f "$GLOBAL_ENV" ]]; then
    set -a  # auto-export
    source "$GLOBAL_ENV"
    set +a
fi

# ── 10 SWE-bench Verified instances (all <15 min difficulty) ─────────────────
INSTANCES=(
    "django__django-10097"
    "django__django-11433"
    "django__django-12308"
    "django__django-13794"
    "django__django-16100"
    "psf__requests-1766"
    "pylint-dev__pylint-4970"
    "sphinx-doc__sphinx-10435"
    "sphinx-doc__sphinx-9711"
    "sympy__sympy-20916"
)

# Build the anchored regex filter
FILTER="^($(IFS='|'; echo "${INSTANCES[*]}"))$"

# ── Default models to evaluate ───────────────────────────────────────────────
# Using OpenRouter as the provider (set OPENROUTER_API_KEY env var)
DEFAULT_MODELS=(
    "qwen/qwen3-coder"
)
# Model class: openrouter for models with tool-call support
MODEL_CLASS="${MODEL_CLASS:-openrouter}"

# Use CLI args if provided, otherwise use defaults
if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# ── Helper functions ─────────────────────────────────────────────────────────
safe_name() {
    # Convert model name to a filesystem-safe directory name
    echo "$1" | tr '/' '__' | tr ':' '_'
}

check_api_key() {
    local model="$1"
    if [[ "$MODEL_CLASS" == "openrouter" ]]; then
        if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
            echo "ERROR: OPENROUTER_API_KEY not set" >&2
            return 1
        fi
        return 0
    fi
    case "$model" in
        anthropic/*)
            if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
                echo "ERROR: ANTHROPIC_API_KEY not set (needed for $model)" >&2
                return 1
            fi ;;
        openai/*)
            if [[ -z "${OPENAI_API_KEY:-}" ]]; then
                echo "ERROR: OPENAI_API_KEY not set (needed for $model)" >&2
                return 1
            fi ;;
        gemini/*)
            if [[ -z "${GEMINI_API_KEY:-}" ]]; then
                echo "ERROR: GEMINI_API_KEY not set (needed for $model)" >&2
                return 1
            fi ;;
        *)
            echo "WARNING: Unknown provider for $model — ensure the right API key is set" >&2 ;;
    esac
}

# ── Preflight checks ────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          mini-swe-agent evaluation harness                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check Docker
if ! docker info &>/dev/null; then
    echo "ERROR: Docker is not running. Please start Docker first." >&2
    exit 1
fi
echo "✓ Docker is running"

# Check mini-extra is available
if ! command -v mini-extra &>/dev/null; then
    echo "ERROR: mini-extra not found. Install mini-swe-agent first:" >&2
    echo "  cd $REPO_DIR && pip install -e ." >&2
    exit 1
fi
echo "✓ mini-extra is available"

# Check API keys for requested models
FAILED=0
for model in "${MODELS[@]}"; do
    if ! check_api_key "$model"; then
        FAILED=1
    fi
done
if [[ $FAILED -eq 1 ]]; then
    echo ""
    echo "Set the missing API keys and retry." >&2
    exit 1
fi
echo "✓ API keys validated"

echo ""
echo "Configuration:"
echo "  Instances:   ${INSTANCES[*]}"
echo "  Models:      ${MODELS[*]}"
echo "  Cost limit:  \$${COST_LIMIT}/instance"
echo "  Workers:     ${WORKERS}"
echo "  Output:      ${RESULTS_DIR}/"
echo ""

# ── Pre-pull Docker images ───────────────────────────────────────────────────
echo "── Pre-pulling Docker images (this may take a while on first run) ──"
for iid in "${INSTANCES[@]}"; do
    # SWE-bench docker image naming: double underscores → _1776_
    docker_id=$(echo "$iid" | sed 's/__/_1776_/g' | tr '[:upper:]' '[:lower:]')
    image="docker.io/swebench/sweb.eval.x86_64.${docker_id}:latest"
    if docker image inspect "$image" &>/dev/null; then
        echo "  ✓ $iid (already pulled)"
    else
        echo "  ↓ Pulling $iid ..."
        docker pull "$image" || echo "  ⚠ Failed to pull $image — will retry during run"
    fi
done
echo ""

# ── Run evaluations ─────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

for model in "${MODELS[@]}"; do
    model_dir="$RESULTS_DIR/$(safe_name "$model")"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running: $model"
    echo "  Output:  $model_dir/"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    mini-extra swebench \
        --subset verified \
        --split test \
        --filter "$FILTER" \
        -m "$model" \
        --model-class "$MODEL_CLASS" \
        -o "$model_dir" \
        -w "$WORKERS" \
        -c swebench.yaml \
        -c "agent.cost_limit=$COST_LIMIT" \
        -c "agent.step_limit=100" \
        -c "model.cost_tracking=ignore_errors" \
        -c "environment.pull_timeout=600" \
        || echo "⚠ Model $model had errors (check logs in $model_dir/)"

    echo ""
done

# ── Compare results ─────────────────────────────────────────────────────────
echo "── Generating comparison ──"
python3 "$SCRIPT_DIR/compare_results.py" "$RESULTS_DIR"

echo ""
echo "Done! Results are in: $RESULTS_DIR/"
echo "To evaluate patches with the official SWE-bench harness:"
echo "  pip install sb-cli"
echo "  sb-cli submit swe-bench_verified test --predictions_path \$MODEL_DIR/preds.json --run_id my-run"
