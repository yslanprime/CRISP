#!/usr/bin/env bash

set -euo pipefail

INPUT_FILE="${INPUT_FILE:-}"
OUTPUT_FILE="${OUTPUT_FILE:-}"
REFINE_MODEL="${REFINE_MODEL:-}"
NUM_WORKERS="${NUM_WORKERS:-16}"

if [[ -z "${INPUT_FILE}" ]]; then
  echo "Please set INPUT_FILE before running this script."
  exit 1
fi

if [[ -z "${OUTPUT_FILE}" ]]; then
  echo "Please set OUTPUT_FILE before running this script."
  exit 1
fi

if [[ -z "${REFINE_MODEL}" ]]; then
  echo "Please set REFINE_MODEL before running this script."
  exit 1
fi

args=(
  --input_file "${INPUT_FILE}"
  --output_file "${OUTPUT_FILE}"
  --model "${REFINE_MODEL}"
  --num_workers "${NUM_WORKERS}"
)

if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
  args+=(--api_base "${OPENAI_BASE_URL}")
fi

python src/step3_output_refinement.py "${args[@]}"
