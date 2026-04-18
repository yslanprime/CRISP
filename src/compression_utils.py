"""Shared utilities for CoT compression and saliency extraction."""

import gc
import os
import random
import re
from typing import List, Optional, Tuple

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AttentionRowExtractor:
    """Extract only the target attention row to reduce memory usage."""

    def __init__(self, target_position: int):
        self.target_position = target_position
        self.attention_row_sum: Optional[torch.Tensor] = None
        self.num_layers = 0
        self.hooks = []

    def _create_hook(self):
        def hook_fn(module, inputs, output):
            attn_weights = None
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]

            if attn_weights is not None and self.target_position < attn_weights.shape[2]:
                target_row = attn_weights[:, :, self.target_position, :].mean(dim=1).detach()

                if self.attention_row_sum is None:
                    self.attention_row_sum = target_row.clone()
                else:
                    self.attention_row_sum = self.attention_row_sum + target_row

                self.num_layers += 1

                if len(output) > 2:
                    return (output[0], None) + output[2:]
                return (output[0], None)

            return output

        return hook_fn

    def register_hooks(self, model) -> None:
        self.remove_hooks()

        layers = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h

        if layers is None:
            return

        for layer in layers:
            attn_module = None
            if hasattr(layer, "self_attn"):
                attn_module = layer.self_attn
            elif hasattr(layer, "attn"):
                attn_module = layer.attn
            elif hasattr(layer, "attention"):
                attn_module = layer.attention

            if attn_module is not None:
                handle = attn_module.register_forward_hook(self._create_hook())
                self.hooks.append(handle)

    def remove_hooks(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def get_average_attention_row(self) -> Optional[np.ndarray]:
        if self.attention_row_sum is not None and self.num_layers > 0:
            avg = self.attention_row_sum / self.num_layers
            return avg[0].cpu().float().numpy()
        return None

    def reset(self) -> None:
        if self.attention_row_sum is not None:
            del self.attention_row_sum
        self.attention_row_sum = None
        self.num_layers = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def extract_thinking_content(full_output: str) -> Tuple[str, str]:
    """Extract the content inside the implicit `<think>...</think>` block."""
    think_end_pattern = r"\n</think>\n\n"
    end_match = re.search(think_end_pattern, full_output)

    if end_match:
        end_idx = end_match.start()
        thinking_content = full_output[:end_idx]
        after_think = full_output[end_match.end() :]
        return thinking_content, after_think

    return "", ""


def split_thinking_into_steps(thinking_content: str) -> List[str]:
    """Split the thinking trace into step-level segments."""
    steps = thinking_content.split("\n\n")
    return [step for step in steps if step.strip()]


def get_attention_scores_for_steps(model, tokenizer, prompt: str, full_output: str, steps: List[str]):
    """Compute step saliency by averaging the attention paid from `</think>`."""
    full_text = prompt + full_output
    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(model.device)

    think_end_token_id = tokenizer.convert_tokens_to_ids("</think>")
    input_ids_list = input_ids[0].tolist()
    think_end_token = input_ids_list.index(think_end_token_id)

    extractor = AttentionRowExtractor(target_position=think_end_token)
    extractor.register_hooks(model)

    try:
        with torch.no_grad():
            _ = model(input_ids, output_attentions=True, return_dict=True)

        attention_from_end_think = extractor.get_average_attention_row()
        if attention_from_end_think is None:
            raise ValueError("Failed to extract attention rows from the model.")

    except torch.cuda.OutOfMemoryError:
        extractor.remove_hooks()
        extractor.reset()
        del input_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "OOM"

    finally:
        extractor.remove_hooks()
        extractor.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    step_scores = []
    prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    think_start_pos = prompt_tokens.shape[1]
    current_step_start = think_start_pos

    for i, _ in enumerate(steps):
        steps_so_far = "\n\n".join(steps[: i + 1])
        if i < len(steps) - 1:
            steps_so_far += "\n\n"

        full_thinking_text = prompt + steps_so_far
        tokens_so_far = tokenizer(full_thinking_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        current_step_end = min(tokens_so_far.shape[1], input_ids.shape[1])

        if current_step_start < current_step_end and current_step_end <= len(attention_from_end_think):
            step_attention = attention_from_end_think[current_step_start:current_step_end].mean()
            step_scores.append(float(step_attention))
        else:
            raise ValueError(
                f"Invalid token span for step {i}: start={current_step_start}, end={current_step_end}, "
                f"attention_len={len(attention_from_end_think)}"
            )

        current_step_start = current_step_end

    return step_scores


def reconstruct_thinking_with_pruned_steps(pruned_steps: List[str], after_think: str) -> str:
    """Rebuild the output after pruning or rewriting intermediate steps."""
    thinking_content = "\n\n".join(pruned_steps)
    return thinking_content + "\n</think>\n\n"
