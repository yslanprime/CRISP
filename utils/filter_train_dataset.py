import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from utils.answer_extraction import extract_answer, strip_string
except ImportError:
    from answer_extraction import extract_answer, strip_string


VALID_LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
LENGTH_FIELD_CANDIDATES = [
    "output_token_count",
    "step1_output_tokens",
    "full_output_token_count",
    "filter_response_length",
]


def print_level_distribution(records: List[Dict], title: str) -> None:
    """Print sample counts for each supported level in a stable order."""
    print(f"\n{title}:")
    counter = Counter(record.get("level", "") for record in records)
    for level in VALID_LEVELS:
        print(f"  {level}: {counter.get(level, 0)}")


def resolve_generated_results_path(input_path: str) -> str:
    """Resolve a step1 output JSONL path from either a file path or a directory."""
    if os.path.isfile(input_path):
        return input_path

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    preferred_patterns = ["**/*_cot.jsonl", "**/*.jsonl"]
    for pattern in preferred_patterns:
        matches = sorted(
            path for path in glob.glob(os.path.join(input_path, pattern), recursive=True) if os.path.isfile(path)
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            preview = ", ".join(matches[:3])
            raise ValueError(
                f"Multiple candidate JSONL files found under {input_path}. "
                f"Please pass the exact file path instead. Examples: {preview}"
            )

    raise FileNotFoundError(f"No JSONL file found under: {input_path}")


def get_metadata_dict(item: Dict) -> Dict:
    metadata = item.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def normalize_record(raw_item: Dict, index: int) -> Dict:
    """Normalize a step1 generation record for downstream filtering."""
    item = dict(raw_item)
    metadata = get_metadata_dict(item)

    question = item.get("question", item.get("problem", ""))
    ground_truth = item.get("ground_truth", item.get("answer", ""))
    level = item.get("level", metadata.get("level", ""))
    sample_type = item.get("type", metadata.get("type", ""))
    unique_id = item.get("unique_id", metadata.get("unique_id", ""))

    item["question"] = str(question)
    item["prompt"] = str(item.get("prompt", ""))
    item["full_output"] = str(item.get("full_output", ""))
    item["ground_truth"] = str(ground_truth)
    item["level"] = str(level)
    item["type"] = str(sample_type)
    item["unique_id"] = str(unique_id)
    item["metadata"] = metadata
    item["sample_id"] = item.get("sample_id", index)
    return item


def load_generated_results(input_path: str, num_samples: Optional[int] = None) -> Tuple[List[Dict], str]:
    """Load previously generated step1 JSONL results."""
    resolved_path = resolve_generated_results_path(input_path)
    print(f"Loading step1 outputs from: {resolved_path}")

    records = []
    with open(resolved_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            records.append(normalize_record(json.loads(line), index))
            if num_samples is not None and len(records) >= num_samples:
                break

    print(f"Loaded {len(records)} generated samples")
    print_level_distribution(records, "Input level distribution")
    return records, resolved_path


def find_existing_length(item: Dict) -> Optional[int]:
    """Reuse a stored output-token count when present."""
    for field in LENGTH_FIELD_CANDIDATES:
        value = item.get(field)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def load_tokenizer_if_needed(records: List[Dict], tokenizer_path: Optional[str]):
    """Load a tokenizer only when some records do not already carry token counts."""
    if all(find_existing_length(record) is not None for record in records):
        return None

    if not tokenizer_path:
        raise ValueError(
            "Some step1 records do not contain output token counts. "
            "Please provide --tokenizer_path (or --model_path) so step1.5 can measure output length."
        )

    print(f"\nLoading tokenizer from: {tokenizer_path}")
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def get_output_token_count(item: Dict, tokenizer) -> int:
    """Get output length from stored metadata or by tokenizing the generated text."""
    existing_length = find_existing_length(item)
    if existing_length is not None:
        return existing_length

    if tokenizer is None:
        raise ValueError("Tokenizer is required to compute output token counts.")

    return len(tokenizer.encode(item["full_output"], add_special_tokens=False))


def build_ground_truth_candidates(ground_truth: str) -> List[str]:
    """Build acceptable normalized target answers."""
    candidates = set()
    raw_ground_truth = str(ground_truth)

    normalized = strip_string(raw_ground_truth)
    if normalized:
        candidates.add(normalized)

    lower_ground_truth = raw_ground_truth.lower()
    looks_like_solution_text = (
        "\n" in raw_ground_truth
        or "####" in raw_ground_truth
        or "boxed" in lower_ground_truth
        or "final answer" in lower_ground_truth
        or "answer is" in lower_ground_truth
        or len(raw_ground_truth.split()) > 8
    )
    if looks_like_solution_text:
        extracted = strip_string(extract_answer(raw_ground_truth))
        if extracted:
            candidates.add(extracted)

    return sorted(candidates)


def numerically_equal(left: str, right: str) -> bool:
    """Compare two normalized answers numerically when possible."""
    try:
        return abs(float(left.replace(",", "")) - float(right.replace(",", ""))) < 1e-6
    except ValueError:
        return False


def verify_answer(generated_answer: str, ground_truth: str) -> bool:
    """Check whether the extracted model answer matches the ground truth."""
    generated = strip_string(generated_answer)
    if not generated:
        return False

    for candidate in build_ground_truth_candidates(ground_truth):
        if generated == candidate:
            return True
        if numerically_equal(generated, candidate):
            return True

    return False


def filter_records(
    records: List[Dict],
    tokenizer,
    max_length: int,
) -> Tuple[Dict[str, List[Dict]], Counter]:
    """Keep only step1 outputs that are correct and short enough."""
    selected_by_level: Dict[str, List[Dict]] = defaultdict(list)
    stats = Counter()

    for item in tqdm(records, desc="Filtering step1 outputs"):
        level = item.get("level", "")
        if level not in VALID_LEVELS:
            stats["invalid_level"] += 1
            continue

        if not item["full_output"].strip():
            stats["missing_output"] += 1
            continue

        if not item["ground_truth"].strip():
            stats["missing_ground_truth"] += 1
            continue

        output_token_count = get_output_token_count(item, tokenizer)
        if output_token_count >= max_length:
            stats["too_long"] += 1
            continue

        generated_answer = extract_answer(item["full_output"])
        if not verify_answer(generated_answer, item["ground_truth"]):
            stats["wrong_answer"] += 1
            continue

        item["generated_answer"] = generated_answer
        item["output_token_count"] = int(output_token_count)
        selected_by_level[level].append(item)
        stats["kept"] += 1

    return selected_by_level, stats


def sample_records_by_level(
    selected_by_level: Dict[str, List[Dict]],
    num_per_level: int,
    seed: int,
) -> List[Dict]:
    """Sample the same number of filtered samples from each difficulty level."""
    rng = random.Random(seed)
    sampled_records: List[Dict] = []

    for level in VALID_LEVELS:
        candidates = list(selected_by_level[level])
        if len(candidates) <= num_per_level:
            sampled_records.extend(candidates)
            continue

        sampled_records.extend(rng.sample(candidates, num_per_level))

    rng.shuffle(sampled_records)
    return sampled_records


def resolve_output_path(output_path: str, input_file_path: str) -> str:
    """Resolve the final JSONL output path."""
    if output_path.lower().endswith(".jsonl"):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        return output_path

    os.makedirs(output_path, exist_ok=True)
    input_name = os.path.splitext(os.path.basename(input_file_path))[0]
    return os.path.join(output_path, f"{input_name}_prepared.jsonl")


def save_results(records: List[Dict], output_path: str, input_file_path: str) -> str:
    """Save selected step1 outputs."""
    actual_output_path = resolve_output_path(output_path, input_file_path)
    with open(actual_output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return actual_output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter step1 generation outputs by answer correctness and output length, "
            "and sample records by level."
        )
    )
    parser.add_argument(
        "--input_path",
        "--data_path",
        dest="input_path",
        type=str,
        default=None,
        help="Step1 output JSONL file path, or a directory containing exactly one *_cot.jsonl file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSONL file path, or a directory where the prepared JSONL will be written",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Tokenizer path used to measure output token length when step1 outputs do not already store it",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Backward-compatible alias for tokenizer_path",
    )
    parser.add_argument(
        "--num_per_level",
        type=int,
        default=500,
        help="Maximum number of retained samples for each level from Level 1 to Level 5",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Strict upper bound on generated output tokens; samples with length >= max_length are removed",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for per-level sampling")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Optional debug limit on how many generated records to inspect",
    )
    args = parser.parse_args()

    if not args.input_path:
        raise ValueError("input_path is required. Please provide --input_path or --data_path.")
    if not args.output_path:
        raise ValueError("output_path is required. Please provide --output_path.")

    tokenizer_path = args.tokenizer_path or args.model_path

    print("=" * 80)
    print("Step1.5 Dataset Preparation")
    print("=" * 80)
    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")
    print(f"Tokenizer path: {tokenizer_path or 'Not provided'}")
    print(f"Target per level: {args.num_per_level}")
    print(f"Max output tokens (strict): < {args.max_length}")
    print(f"Seed: {args.seed}")
    if args.num_samples is not None:
        print(f"Debug num_samples: {args.num_samples}")
    print("=" * 80)

    records, resolved_input_path = load_generated_results(args.input_path, num_samples=args.num_samples)
    tokenizer = load_tokenizer_if_needed(records, tokenizer_path)
    selected_by_level, stats = filter_records(records, tokenizer, args.max_length)

    eligible_records = []
    for level in VALID_LEVELS:
        eligible_records.extend(selected_by_level[level])
    print_level_distribution(eligible_records, "Eligible level distribution")

    missing_levels = [level for level in VALID_LEVELS if len(selected_by_level[level]) < args.num_per_level]
    if missing_levels:
        print(
            "\nWarning: did not reach the target count for these levels after filtering: "
            + ", ".join(missing_levels)
        )

    sampled_records = sample_records_by_level(selected_by_level, args.num_per_level, args.seed)
    print_level_distribution(sampled_records, "Final sampled level distribution")

    print("\nFiltering summary:")
    print(f"  Kept after correctness/length filtering: {stats.get('kept', 0)}")
    print(f"  Rejected for invalid or missing level: {stats.get('invalid_level', 0)}")
    print(f"  Rejected for missing output: {stats.get('missing_output', 0)}")
    print(f"  Rejected for missing ground truth: {stats.get('missing_ground_truth', 0)}")
    print(f"  Rejected for length >= {args.max_length}: {stats.get('too_long', 0)}")
    print(f"  Rejected for wrong answer: {stats.get('wrong_answer', 0)}")

    actual_output_path = save_results(sampled_records, args.output_path, resolved_input_path)
    print(f"\nSaved {len(sampled_records)} prepared samples to: {actual_output_path}")


if __name__ == "__main__":
    main()
