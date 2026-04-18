import argparse
import glob
import json
import os

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


# The paper uses math_full_minus_math500 by default. The generation code can
# also work with other datasets if they follow the same parquet schema.
DATASET_NAME = "math_full_minus_math500"


def resolve_parquet_path(data_path):
    """Resolve a parquet file from either a file path or a dataset root directory."""
    if os.path.isfile(data_path):
        return data_path

    data_dir = os.path.join(data_path, "data")
    preferred_path = os.path.join(data_dir, "train-00000-of-00001.parquet")
    if os.path.exists(preferred_path):
        return preferred_path

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"Dataset file not found under: {data_dir}")
    if len(parquet_files) > 1:
        raise ValueError(
            "Multiple parquet files found. Please pass the exact parquet file path via --data_path."
        )
    return parquet_files[0]


def load_math_full_minus_math500_data(data_path, num_samples=None):
    """Load a parquet dataset compatible with the math_full_minus_math500 schema."""
    parquet_path = resolve_parquet_path(data_path)

    print(f"Loading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if num_samples is not None:
        df = df.head(num_samples)

    data = []
    for _, row in df.iterrows():
        data.append(
            {
                "question": row["problem"],
                "answer": str(row.get("answer", "")),
                "metadata": {
                    "solution": row.get("solution", ""),
                    "level": row.get("level", ""),
                    "type": row.get("type", ""),
                    "unique_id": row.get("unique_id", ""),
                },
            }
        )

    print(f"Loaded {len(data)} samples from {DATASET_NAME}")
    return data


def generate_with_cot_batch(llm, tokenizer, questions, max_new_tokens=4096, temperature=0.6):
    """Generate chain-of-thought responses for a batch of questions."""
    from vllm import SamplingParams

    prompts = []
    for question in questions:
        user_content = f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        messages = [{"role": "user", "content": user_content}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_new_tokens,
    )

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        results.append(
            {
                "question": questions[i],
                "prompt": prompts[i],
                "full_output": output.outputs[0].text,
                "output_token_count": len(output.outputs[0].token_ids),
            }
        )

    return results


def get_model_name(model_path):
    """Extract the model name from a model path."""
    return os.path.basename(model_path.rstrip("/"))


def process_math_full_minus_math500_with_model(
    model_path,
    data_path,
    output_base_path,
    model_name=None,
    num_samples=None,
    max_new_tokens=8192,
    temperature=0.6,
    batch_size=4,
    tensor_parallel_size=1,
):
    """Generate raw CoTs for math_full_minus_math500 or another schema-compatible dataset."""
    if not model_path:
        raise ValueError("model_path is required")
    if not data_path:
        raise ValueError("data_path is required")
    if not output_base_path:
        raise ValueError("output_base_path is required")

    if model_name is None:
        model_name = get_model_name(model_path)

    print(f"\nLoading model: {model_path}")
    print(f"Tensor parallel size: {tensor_parallel_size}")

    from vllm import LLM

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("=" * 80)
    print(f"Generating CoT for {DATASET_NAME}")
    print(f"Model: {model_name}")
    print("=" * 80)

    data = load_math_full_minus_math500_data(data_path=data_path, num_samples=num_samples)

    model_output_dir = os.path.join(output_base_path, model_name)
    output_file = os.path.join(model_output_dir, f"{DATASET_NAME}_cot.jsonl")
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"\nStart generation for {len(data)} samples")
    print(f"Settings: max_new_tokens={max_new_tokens}, temperature={temperature}, batch_size={batch_size}")
    print("=" * 80)

    all_results = []
    for i in tqdm(range(0, len(data), batch_size), desc="Generating"):
        try:
            batch_data = data[i : i + batch_size]
            batch_questions = [item["question"] for item in batch_data]
            batch_answers = [item["answer"] for item in batch_data]
            batch_metadata = [item.get("metadata", {}) for item in batch_data]

            batch_results = generate_with_cot_batch(
                llm=llm,
                tokenizer=tokenizer,
                questions=batch_questions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            for j, result in enumerate(batch_results):
                metadata = batch_metadata[j]
                result["ground_truth"] = batch_answers[j]
                result["sample_id"] = i + j
                result["dataset"] = DATASET_NAME
                result["metadata"] = metadata
                result["level"] = metadata.get("level", "")
                result["type"] = metadata.get("type", "")
                result["unique_id"] = metadata.get("unique_id", "")
                all_results.append(result)
        except Exception as e:
            print(f"\nError while processing batch starting at {i}: {e}")
            import traceback

            traceback.print_exc()
            continue

    all_results.sort(key=lambda x: x["sample_id"])

    with open(output_file, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("\n" + "=" * 80)
    print(f"Finished generation for {len(all_results)} samples")
    print(f"Saved results to: {output_file}")
    print("=" * 80)
    return all_results


def load_generated_results(output_path):
    """Load previously generated raw CoT results."""
    results = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate raw CoTs with explicit model/data/output paths.")
    parser.add_argument("--model_path", type=str, default=None, help="Model path")
    parser.add_argument("--model_name", type=str, default=None, help="Optional model name override")
    parser.add_argument("--data_path", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--output_base_path", type=str, default=None, help="Base output directory")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="Maximum number of generated tokens")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")

    args = parser.parse_args()

    if not args.model_path:
        raise ValueError("model_path is required. Please provide --model_path.")
    if not args.data_path:
        raise ValueError("data_path is required. Please provide --data_path.")
    if not args.output_base_path:
        raise ValueError("output_base_path is required. Please provide --output_base_path.")

    print("Configuration:")
    print(f"  model_path: {args.model_path}")
    print(f"  data_path: {args.data_path}")
    print(f"  output_base_path: {args.output_base_path}")
    print(f"  num_samples: {args.num_samples}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  temperature: {args.temperature}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  tensor_parallel_size: {args.tensor_parallel_size}")
    print()

    process_math_full_minus_math500_with_model(
        model_path=args.model_path,
        data_path=args.data_path,
        output_base_path=args.output_base_path,
        model_name=args.model_name,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )
