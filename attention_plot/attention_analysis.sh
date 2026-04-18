#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${1:-}}"
DATA_PATH="${DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SAMPLE_ID="${SAMPLE_ID:-}"
LAYERS="${LAYERS:-}"
SELECTED_GPU="${CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICE:-}}"

infer_example_data_path() {
  local model_name="$1"

  case "${model_name}" in
    *1.5B*)
      echo "${REPO_ROOT}/example_data/gsm8k_sample_1.5B.jsonl"
      ;;
    *7B*)
      echo "${REPO_ROOT}/example_data/gsm8k_sample_7B.jsonl"
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Please provide MODEL_PATH, either as an environment variable or as the first argument."
  echo "Example: bash attention_plot/attention_analysis.sh pretrained_model/DeepSeek-R1-Distill-Qwen-1.5B"
  echo "Example: MODEL_PATH=/path/to/model bash attention_plot/attention_analysis.sh"
  exit 1
fi

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}"
  exit 1
fi

MODEL_NAME="$(basename "${MODEL_PATH%/}")"

if [[ -z "${DATA_PATH}" ]]; then
  if ! DATA_PATH="$(infer_example_data_path "${MODEL_NAME}")"; then
    echo "Could not infer an example_data file for model: ${MODEL_NAME}"
    echo "Supported example mappings currently cover model names containing 1.5B or 7B."
    echo "You can still run the script by setting DATA_PATH manually."
    exit 1
  fi
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "DATA_PATH does not exist: ${DATA_PATH}"
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${REPO_ROOT}/attention_plot/attention_${MODEL_NAME}_example"
fi

if [[ -z "${SELECTED_GPU}" ]]; then
  echo "Please set CUDA_VISIBLE_DEVICES manually before running this script."
  echo "The alias CUDA_VISIBLE_DEVICE is also accepted for convenience, but CUDA_VISIBLE_DEVICES is recommended."
  echo "Example: CUDA_VISIBLE_DEVICES=5 bash attention_plot/attention_analysis.sh pretrained_model/DeepSeek-R1-Distill-Qwen-1.5B"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${SELECTED_GPU}"

echo "Running attention analysis with:"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi
if [[ -n "${SAMPLE_ID}" ]]; then
  echo "  SAMPLE_ID=${SAMPLE_ID}"
fi
if [[ -n "${LAYERS}" ]]; then
  echo "  LAYERS=${LAYERS}"
fi

args=(
  --model_path "${MODEL_PATH}"
  --data_path "${DATA_PATH}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ -n "${SAMPLE_ID}" ]]; then
  args+=(--sample_id "${SAMPLE_ID}")
fi

if [[ -n "${LAYERS}" ]]; then
  args+=(--layers "${LAYERS}")
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python "${REPO_ROOT}/attention_plot/attention_analysis.py" "${args[@]}"
