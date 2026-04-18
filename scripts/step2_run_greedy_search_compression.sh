#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
INPUT_FILE="${INPUT_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SIMILARITY_MODEL_PATH="${SIMILARITY_MODEL_PATH:-}"
TAU_LOW="${TAU_LOW:-0.3}"
TAU_HIGH="${TAU_HIGH:-0.2}"
TAU_SIM="${TAU_SIM:-0.7}"
API_MODEL="${API_MODEL:-}"
ENABLE_REWRITE="${ENABLE_REWRITE:-1}"
ENABLE_FUSE="${ENABLE_FUSE:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
NUM_SHARDS="${NUM_SHARDS:-1}"
GPU_IDS="${GPU_IDS:-}"
SEED="${SEED:-42}"

PIDS=()

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Please set MODEL_PATH before running this script."
  exit 1
fi

if [[ -z "${INPUT_FILE}" ]]; then
  echo "Please set INPUT_FILE before running this script."
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "Please set OUTPUT_DIR before running this script."
  exit 1
fi

if [[ -z "${SIMILARITY_MODEL_PATH}" ]]; then
  echo "Please set SIMILARITY_MODEL_PATH before running this script."
  exit 1
fi

if [[ "${ENABLE_REWRITE}" == "1" || "${ENABLE_FUSE}" == "1" ]] && [[ -z "${API_MODEL}" ]]; then
  echo "Please set API_MODEL when ENABLE_REWRITE=1 or ENABLE_FUSE=1."
  exit 1
fi

cleanup() {
  if [[ ${#PIDS[@]} -eq 0 ]]; then
    exit 1
  fi

  echo ""
  echo "=========================================="
  echo "Received interrupt signal, terminating shard processes..."
  echo "=========================================="

  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Stopping PID ${pid}"
      kill "${pid}" 2>/dev/null || true
    fi
  done

  wait || true
  exit 1
}

trap cleanup SIGINT SIGTERM

input_basename="$(basename "${INPUT_FILE}")"
input_name="${input_basename%.*}"
dataset_output_dir="${OUTPUT_DIR}/compressed_${input_name}"
canonical_output_file="${OUTPUT_DIR}/greedy_compressed.jsonl"

args=(
  --model_path "${MODEL_PATH}"
  --data_path "${INPUT_FILE}"
  --output_dir "${OUTPUT_DIR}"
  --tau_low "${TAU_LOW}"
  --tau_high "${TAU_HIGH}"
  --tau_sim "${TAU_SIM}"
  --similarity_model_path "${SIMILARITY_MODEL_PATH}"
  --seed "${SEED}"
)

if [[ -n "${API_MODEL}" ]]; then
  args+=(--api_model "${API_MODEL}")
fi

if [[ "${ENABLE_REWRITE}" == "1" ]]; then
  args+=(--enable_rewrite)
fi

if [[ "${ENABLE_FUSE}" == "1" ]]; then
  args+=(--enable_fuse)
fi

if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
  args+=(--api_base "${OPENAI_BASE_URL}")
fi

if [[ -n "${NUM_SAMPLES}" ]]; then
  args+=(--num_samples "${NUM_SAMPLES}")
fi

if [[ "${NUM_SHARDS}" -le 1 ]]; then
  python src/step2_greedy_search_compression.py "${args[@]}"
  if [[ -f "${dataset_output_dir}/greedy_compressed.jsonl" ]]; then
    cp "${dataset_output_dir}/greedy_compressed.jsonl" "${canonical_output_file}"
    echo "Canonical step2 output saved to: ${canonical_output_file}"
  fi
  exit 0
fi

IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
if [[ ${#GPU_ID_ARRAY[@]} -eq 0 || -z "${GPU_ID_ARRAY[0]}" ]]; then
  GPU_ID_ARRAY=()
  for ((i = 0; i < NUM_SHARDS; i++)); do
    GPU_ID_ARRAY+=("${i}")
  done
fi

if [[ ${#GPU_ID_ARRAY[@]} -lt ${NUM_SHARDS} ]]; then
  echo "Please provide at least ${NUM_SHARDS} GPU ids via GPU_IDS (for example: GPU_IDS=4,5,6,7)."
  exit 1
fi

log_dir="${dataset_output_dir}/logs"

mkdir -p "${dataset_output_dir}" "${log_dir}"

echo "=========================================="
echo "Step2 multi-shard mode"
echo "=========================================="
echo "MODEL_PATH=${MODEL_PATH}"
echo "INPUT_FILE=${INPUT_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NUM_SHARDS=${NUM_SHARDS}"
echo "GPU_IDS=${GPU_ID_ARRAY[*]}"
echo "=========================================="

for ((shard_id = 0; shard_id < NUM_SHARDS; shard_id++)); do
  gpu_id="${GPU_ID_ARRAY[$shard_id]}"
  log_file="${log_dir}/shard_${shard_id}.log"

  echo "Launching shard ${shard_id}/${NUM_SHARDS} on GPU ${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" python src/step2_greedy_search_compression.py \
    "${args[@]}" \
    --shard_id "${shard_id}" \
    --num_shards "${NUM_SHARDS}" \
    > "${log_file}" 2>&1 &

  PIDS+=("$!")
  echo "  Log: ${log_file} (PID: $!)"
done

echo ""
echo "All shards started."
echo "Use the following command to monitor logs:"
echo "  tail -f ${log_dir}/shard_*.log"
echo ""

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo ""
  echo "At least one shard failed. Check logs under ${log_dir}."
  exit 1
fi

echo ""
echo "All shards finished. Merging outputs..."

export STEP2_MERGE_OUTPUT_DIR="${dataset_output_dir}"
python -c '
import glob
import json
import os
import sys

output_dir = os.environ["STEP2_MERGE_OUTPUT_DIR"]
shard_files = sorted(glob.glob(os.path.join(output_dir, "greedy_compressed_shard*.jsonl")))

if not shard_files:
    raise FileNotFoundError(f"No shard outputs found under: {output_dir}")

all_results = []
for shard_file in shard_files:
    with open(shard_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_results.append(json.loads(line))

all_results.sort(key=lambda item: item.get("sample_id", 0))

merged_file = os.path.join(output_dir, "greedy_compressed.jsonl")
with open(merged_file, "w", encoding="utf-8") as f:
    for item in all_results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Merged {len(shard_files)} shard files into: {merged_file}")
print(f"Total records: {len(all_results)}")
'
unset STEP2_MERGE_OUTPUT_DIR

cp "${dataset_output_dir}/greedy_compressed.jsonl" "${canonical_output_file}"
echo "Canonical step2 output saved to: ${canonical_output_file}"

echo "Step2 multi-shard run completed successfully."
