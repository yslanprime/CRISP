import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, Optional

from tqdm import tqdm


DEFAULT_EOS_TOKEN = "<｜end▁of▁sentence｜>"
DEFAULT_REASON_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert and refine CoT data for LLaMA-Factory SFT format.")
    parser.add_argument("--input_file", type=str, default=None, help="Path to greedy_compressed.jsonl")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save LLaMA-Factory JSON")
    parser.add_argument("--api_key", type=str, default=None, help="API key (default: env OPENAI_API_KEY)")
    parser.add_argument("--api_base", type=str, default=None, help="Optional API base URL")
    parser.add_argument("--model", type=str, default=None, help="Model used for refinement")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens for refinement response")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for refinement")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of samples (for quick tests)")
    parser.add_argument("--num_workers", type=int, default=16, help="Parallel workers for refinement API calls")
    parser.add_argument(
        "--eos_token",
        type=str,
        default=DEFAULT_EOS_TOKEN,
        help="Tokenizer eos_token used when constructing the compression suffix.",
    )
    parser.add_argument(
        "--reason_prompt",
        type=str,
        default=DEFAULT_REASON_PROMPT,
        help="Prompt appended after the question before compression suffix.",
    )
    return parser.parse_args()


def load_jsonl(path: str, limit: Optional[int] = None) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            if not line.strip():
                continue
            yield json.loads(line)


def  ensure_think_block(text: str) -> str:
    """Ensure output is wrapped in <think>...</think> and ends with a blank line."""
    if not text:
        return ""

    stripped = text.strip()

    # Find </think> tag and truncate everything after it
    end = stripped.find("</think>")
    
    if end != -1:
        # Keep content up to and including </think>
        content = stripped[:end + len("</think>")]
        
        # Add <think> at the beginning if not present
        if not content.startswith("<think>"):
            content = "<think>\n" + content
        
        # Ensure trailing blank line
        if not content.endswith("\n"):
            content += "\n"
        if not content.endswith("\n\n"):
            content += "\n"
        
        return content
    
    # If no </think> tag found, wrap everything
    block = "<think>\n" + stripped + "\n</think>"
    
    if not block.endswith("\n"):
        block += "\n"
    if not block.endswith("\n\n"):
        block += "\n"
    
    return block


def build_openai_client(api_key: str, api_base: Optional[str]):
    import openai

    if api_base:
        return openai.OpenAI(api_key=api_key, base_url=api_base)
    return openai.OpenAI(api_key=api_key)


REFINE_SYSTEM_PROMPT = (
    "You are an expert mathematical editor. Your task is to refine a rough reasoning draft. "
    "Restore logical continuity and mathematical accuracy. "
    "Match the Original CoT's exact tone, formatting, and style. "
)

def build_refine_prompt(question: str, draft: str, original_cot: str) -> str:
    return (
        f"### Question\n{question}\n\n"
        f"### Original CoT (ONLY for Reference)\n{original_cot}\n\n"
        f"### Rough Draft (To Refine)\n{draft}\n\n"
        f"### Instruction\n"
        f"Refine the Rough Draft to ensure mathematical coherence and logical flow.\n"
        f"1. Fill in missing algebraic manipulations and arithmetic calculations.\n"
        f"2. Match the style and formatting of the Original CoT.\n"
        f"3. Output ONLY the refined reasoning text.\n"
        f"4. Ensure the calculations lead correctly to the final answer.\n"
        f"### Refined Rough Solution:\n"
    )


def build_compression_suffix(eos_token: str) -> str:
    """Construct the suffix using the tokenizer eos token."""
    return f"{eos_token}<｜compressed｜>{eos_token}"


def refine_compressed_output(
    client,
    model: str,
    question: str,
    draft: str,
    original_cot: str,
    max_tokens: int,
    temperature: float,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """Call LLM API to polish a compressed chain-of-thought."""
    user_prompt = build_refine_prompt(question, draft, original_cot)

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": REFINE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = resp.choices[0].message.content
            if not content:
                raise RuntimeError("Empty content from API")
            return content.strip()
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            wait = retry_delay * attempt
            print(f"[warn] API refine failed (attempt {attempt}/{max_retries}): {exc}. Retrying in {wait:.1f}s...")
            time.sleep(wait)

    raise RuntimeError(f"Refinement failed after {max_retries} attempts: {last_error}")


def main() -> None:
    args = parse_args()

    if not args.input_file:
        raise ValueError("input_file is required. Please provide --input_file.")
    if not args.output_file:
        raise ValueError("output_file is required. Please provide --output_file.")
    if not args.model:
        raise ValueError("model is required. Please provide --model.")

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    api_base = args.api_base or os.environ.get("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("API key required for refinement. Set OPENAI_API_KEY or pass --api_key.")
    client = build_openai_client(api_key, api_base)

    items = list(load_jsonl(args.input_file, args.limit))
    total = len(items)
    if total == 0:
        print(f"No samples found in {args.input_file}")
        return

    data = []
    refined_count = 0
    failed_refine = 0

    print(f"Reading from {args.input_file} (total={total})")

    suffix = build_compression_suffix(args.eos_token)

    def process_item(item_with_idx):
        idx, item = item_with_idx
        instruction_content = item.get("question", "")

        # 1) Compressed data (refined)
        instruction_compressed = (
            f"{instruction_content}\n{args.reason_prompt}{suffix}"
        )
        compressed_output = item.get("compressed_output", "")
        original_output = item.get("original_output", "")

        try:
            refined_text = refine_compressed_output(
                client=client,
                model=args.model,
                question=instruction_content,
                draft=compressed_output,
                original_cot=original_output,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        except Exception as exc:  # noqa: BLE001
            sample_id = item.get("sample_id", idx)
            raise RuntimeError(f"Refinement failed for sample_id={sample_id}") from exc

        output_compressed = ensure_think_block(refined_text)
        entries = [
            {
                "instruction": instruction_compressed,
                "input": "",
                "output": output_compressed,
            }
        ]

        # 2) Uncompressed data (original)
        if original_output:
            output_original = ensure_think_block(original_output)
            instruction_original = f"{instruction_content}\n{args.reason_prompt}"
            entries.append(
                {
                    "instruction": instruction_original,
                    "input": "",
                    "output": output_original,
                }
            )

        return idx, entries

    results = {}
    max_workers = max(1, args.num_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(process_item, (idx, item)): (idx, item)
            for idx, item in enumerate(items, 1)
        }

        with tqdm(total=total, desc="Refining", unit="sample") as pbar:
            for future in as_completed(future_to_item):
                idx, item = future_to_item[future]
                try:
                    result_idx, entries = future.result()
                    refined_count += 1
                    results[result_idx] = entries
                except Exception as exc: 
                    failed_refine += 1
                    sample_id = item.get("sample_id", idx)
                    raise RuntimeError(f"Refinement failed for sample_id={sample_id}") from exc
                finally:
                    pbar.update(1)
                    pbar.set_postfix(refined=refined_count, failed=failed_refine)
    for idx in sorted(results):
        data.extend(results[idx])

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Successfully converted {len(data)} items to {args.output_file}")
    if client:
        print(f"Refined {refined_count}/{total} compressed outputs, failed {failed_refine}")


if __name__ == "__main__":
    main()
