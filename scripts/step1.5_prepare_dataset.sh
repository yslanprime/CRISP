#!/usr/bin/env bash

set -euo pipefail

INPUT_PATH="${INPUT_PATH:-${DATA_PATH:-}}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
NUM_PER_LEVEL="${NUM_PER_LEVEL:-500}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
SEED="${SEED:-42}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH:-}}"

if [[ -z "${INPUT_PATH}" ]]; then
  echo "Please set INPUT_PATH before running this script."
  exit 1
fi

if [[ -z "${OUTPUT_PATH}" ]]; then
  echo "Please set OUTPUT_PATH before running this script."
  exit 1
fi

echo "The default target is ${NUM_PER_LEVEL} correct samples for each of the 5 levels with output length < ${MAX_LENGTH}."

args=(
  --input_path "${INPUT_PATH}"
  --output_path "${OUTPUT_PATH}"
  --num_per_level "${NUM_PER_LEVEL}"
  --max_length "${MAX_LENGTH}"
  --seed "${SEED}"
)

if [[ -n "${TOKENIZER_PATH}" ]]; then
  args+=(--tokenizer_path "${TOKENIZER_PATH}")
fi

if [[ -n "${NUM_SAMPLES}" ]]; then
  args+=(--num_samples "${NUM_SAMPLES}")
fi

python src/step1.5_prepare_dataset.py "${args[@]}"
