#!/usr/bin/env bash
set -euo pipefail

backend="${1:-}"
if [[ -z "$backend" ]]; then
  echo "Usage: $0 {qwen|llama31_8b|mistral7b}" >&2
  exit 2
fi

case "$backend" in
  qwen)
    model="Qwen/Qwen2.5-7B-Instruct"
    ;;
  llama31_8b)
    model="meta-llama/Llama-3.1-8B-Instruct"
    ;;
  mistral7b)
    model="mistralai/Mistral-7B-Instruct-v0.3"
    ;;
  *)
    echo "Unknown backend: $backend" >&2
    echo "Usage: $0 {qwen|llama31_8b|mistral7b}" >&2
    exit 2
    ;;
esac

host="${VLLM_HOST:-127.0.0.1}"
port="${VLLM_PORT:-8000}"
max_model_len="${VLLM_MAX_MODEL_LEN:-16384}"
gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
dtype="${VLLM_DTYPE:-auto}"

if [[ "$backend" == "llama31_8b" && -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Continuing; this also works if huggingface-cli login is already configured." >&2
fi

echo "Serving $model on http://$host:$port/v1"
exec vllm serve "$model"   --host "$host"   --port "$port"   --served-model-name "$model"   --max-model-len "$max_model_len"   --gpu-memory-utilization "$gpu_memory_utilization"   --dtype "$dtype"
