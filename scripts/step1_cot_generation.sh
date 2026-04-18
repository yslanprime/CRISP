#!/usr/bin/env bash

set -euo pipefail

# This script can be used with any dataset that matches the schema expected by src/step1_cot_generation.py. 
# The paper's default training data is math_full_minus_math500.
MODEL_PATH="${MODEL_PATH:-}"
DATA_PATH="${DATA_PATH:-}"
OUTPUT_BASE_PATH="${OUTPUT_BASE_PATH:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Please set MODEL_PATH before running this script."
  exit 1
fi

if [[ -z "${DATA_PATH}" ]]; then
  echo "Please set DATA_PATH to your dataset root before running this script."
  exit 1
fi

if [[ -z "${OUTPUT_BASE_PATH}" ]]; then
  echo "Please set OUTPUT_BASE_PATH before running this script."
  exit 1
fi

echo "Generating CoT from DATA_PATH=${DATA_PATH}"
echo "The paper uses math_full_minus_math500 by default, but other compatible datasets are also supported."
args=(
  --model_path "${MODEL_PATH}"
  --data_path "${DATA_PATH}"
  --output_base_path "${OUTPUT_BASE_PATH}"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
)

if [[ -n "${NUM_SAMPLES}" ]]; then
  args+=(--num_samples "${NUM_SAMPLES}")
fi

python src/step1_cot_generation.py "${args[@]}"
