#!/usr/bin/env bash
# Start vLLM server for mini-swe-agent inference with tool-call support.
#
# Usage:
#   bash rl/serve_vllm.sh                                              # base Qwen3-8B
#   bash rl/serve_vllm.sh --model /data/.../qwen3-8b-dro5-merged      # fine-tuned
#   bash rl/serve_vllm.sh --gpu 0,1 --tp 2                            # multi-GPU
#   PORT=9000 bash rl/serve_vllm.sh                                   # custom port
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8234}"
GPU="${GPU:-0}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
VLLM_ENV="${VLLM_ENV:-/data/manantomar/vllm-env}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --port)  PORT="$2"; shift 2 ;;
        --gpu)   GPU="$2"; shift 2 ;;
        --tp)    TP="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$GPU"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                   vLLM server for mini-swe-agent            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Model:         $MODEL"
echo "  Port:          $PORT"
echo "  GPU:           $GPU (CUDA_VISIBLE_DEVICES)"
echo "  TP:            $TP"
echo "  Max model len: $MAX_MODEL_LEN"
echo ""
echo "  Use with mini-swe-agent:"
echo "    mini-extra swebench ... --model-class vllm -m $MODEL \\"
echo "      -c \"model.api_base=http://localhost:$PORT/v1\""
echo ""

exec "$VLLM_ENV/bin/vllm" serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code \
    --dtype bfloat16
