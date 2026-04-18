"""
Greedy Search Chain-of-Thought Compression with Heuristic Gating

Core Algorithm:
1. State Space: Model the CoT compression process as a sequential decision process
2. Action Space: Four atomic operations - PRUNE, KEEP, REWRITE, FUSE
3. Heuristic Gating: Dynamically determine allowed actions based on Attention Score and Semantic Similarity
4. Scoring Function: Combine Prediction Fidelity and Compression Efficiency
5. Greedy Strategy: Select the optimal action (highest reward) at each step
"""

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM


from compression_utils import (
    set_random_seed,
    extract_thinking_content,
    split_thinking_into_steps,
    get_attention_scores_for_steps,
    reconstruct_thinking_with_pruned_steps,
)


# ==================== Data Structure Definitions ====================


@dataclass
class Action:
    """Action definition"""

    action_type: str  # 'PRUNE', 'KEEP', 'REWRITE', 'FUSE'
    step_indices: List[int]  # Indices of involved steps
    output_text: str = ""  # Output text produced by the action


@dataclass
class State:
    """State definition"""

    compressed_cot: List[str]  # Currently generated compressed CoT sequence
    current_index: int  # Current position in the original sequence
    score: float = 0.0  # Accumulated score
    action_history: List[Action] = field(default_factory=list)  # Action history

    def copy(self):
        """Deep copy the state"""
        return State(
            compressed_cot=self.compressed_cot.copy(),
            current_index=self.current_index,
            score=self.score,
            action_history=self.action_history.copy(),
        )


# ==================== Heuristic Gating Mechanism ====================


class HeuristicGating:
    """Heuristic gating mechanism (percentile-based thresholds)"""

    def __init__(self, tau_low: float = 0.5, tau_high: float = 0.2, tau_sim: float = 0.8):
        """
        Args:
            tau_low: Low attention threshold (percentile), e.g. 0.3 means steps in the bottom 30% are low attention
            tau_high: High attention threshold (percentile), e.g. 0.1 means steps in the top 10% are high attention
            tau_sim: Semantic similarity threshold
        """
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.tau_sim = tau_sim

        # Percentile thresholds (computed in compute_percentile_thresholds)
        self.low_threshold = None
        self.high_threshold = None

    def compute_percentile_thresholds(self, attention_scores: List[float]):
        """
        Compute percentile thresholds based on all steps' attention scores

        Args:
            attention_scores: List of attention scores for all steps
        """
        sorted_scores = sorted(attention_scores)
        n = len(sorted_scores)

        low_idx = int(n * self.tau_low)
        low_idx = max(0, min(low_idx, n - 1))
        self.low_threshold = sorted_scores[low_idx]

        high_idx = int(n * (1 - self.tau_high))
        high_idx = max(0, min(high_idx, n - 1))
        self.high_threshold = sorted_scores[high_idx]

    def get_allowed_actions(self, attention_score: float, semantic_similarity: Optional[float] = None, has_compressed_content: bool = False) -> List[str]:
        """
        Determine the set of allowed actions based on attention score and semantic similarity

        Decision logic (priority from high to low):
        1. High semantic similarity (with last compressed step) -> allow FUSE
        2. Low attention -> PRUNE, REWRITE
        3. Medium attention -> REWRITE
        4. High attention -> KEEP, REWRITE

        Args:
            attention_score: Attention score of the current step
            semantic_similarity: Semantic similarity between current step and last compressed step
            has_compressed_content: Whether compressed content already exists (FUSE requires existing content)

        Returns:
            Set of allowed action types

        Raises:
            ValueError: If compute_percentile_thresholds has not been called first
        """
        if self.low_threshold is None or self.high_threshold is None:
            raise ValueError("Must call compute_percentile_thresholds first to compute thresholds")

        allowed_actions = []

        # Rule 1: Check semantic similarity first - high similarity allows FUSE
        # (Requires existing compressed content to FUSE current step into the last compressed step)
        if has_compressed_content and semantic_similarity is not None and semantic_similarity >= self.tau_sim:
            allowed_actions.append("FUSE")

        # Rule 2: Determine base actions by attention score
        elif attention_score < self.low_threshold:
            # Low attention (bottom tau_low%) -> PRUNE, REWRITE
            allowed_actions.extend(["PRUNE", "REWRITE"])

        elif attention_score < self.high_threshold:
            # Medium attention -> PRUNE, REWRITE, KEEP
            allowed_actions.extend(["REWRITE"])

        elif attention_score >= self.high_threshold:
            allowed_actions.extend(["KEEP", "REWRITE"])

        # Ensure at least one action is available
        if not allowed_actions:
            allowed_actions.append("REWRITE")

        return allowed_actions


# ==================== Semantic Similarity Calculation ====================


class SemanticSimilarityCalculator:
    """
    Semantic similarity calculator

    Uses SimCSE (Simple Contrastive Learning of Sentence Embeddings) model
    to compute semantic similarity between sentences
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Args:
            model_path: Path to SimCSE model
            device: Device
        """
        if not model_path:
            raise ValueError("model_path is required for SemanticSimilarityCalculator")

        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_path = model_path

    def _load_model(self):
        """Lazy load the model"""
        if self.model is None:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()

    def _encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts into vector representations

        Args:
            texts: List of texts

        Returns:
            Sentence embeddings (batch_size, hidden_size)
        """
        self._load_model()

        # Tokenize
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get sentence embeddings (using [CLS] token output)
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            # SimCSE uses pooler_output or last_hidden_state[:, 0] as sentence representation
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0]

        return embeddings

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts

        Returns:
            Cosine similarity (0-1)
        """
        embeddings = self._encode([text1, text2])

        similarity = torch.nn.functional.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()

        return max(0.0, similarity)  # Ensure non-negative

    def compute_all_similarities(self, steps: List[str]) -> List[float]:
        """
        Compute semantic similarities for all adjacent steps (batch processing is more efficient)

        Returns:
            List of similarities, length is len(steps) - 1
        """
        if len(steps) <= 1:
            return []

        # Batch encode all steps
        embeddings = self._encode(steps)

        # Compute similarities between adjacent steps
        similarities = []
        for i in range(len(steps) - 1):
            sim = torch.nn.functional.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[i + 1].unsqueeze(0)).item()
            similarities.append(max(0.0, sim))

        return similarities


# ==================== LLM Action Executor ====================


class LLMOperator:
    """LLM action executor, responsible for REWRITE and FUSE operations via API calls"""

    def __init__(
        self,
        api_key: str,
        api_base: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ):
        """
        Args:
            api_key: OpenAI API key
            api_base: API base URL (optional, for compatibility with other API services)
            model_name: Model name
            max_tokens: Maximum number of generated tokens
            temperature: Generation temperature

        Raises:
            ImportError: If the openai library is not installed
            ValueError: If api_key is empty
        """
        if not api_key:
            raise ValueError("api_key cannot be empty")
        if not model_name:
            raise ValueError("model_name cannot be empty")

        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Initialize OpenAI client
        if api_base:
            self.client = openai.OpenAI(api_key=api_key, base_url=api_base)
        else:
            self.client = openai.OpenAI(api_key=api_key)

    def _call_api(self, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
        """
        Call the LLM API

        Args:
            messages: List of messages
            max_retries: Maximum number of retries

        Returns:
            Generated text

        Raises:
            RuntimeError: Raised when API call fails
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, max_tokens=self.max_tokens, temperature=self.temperature
                )

                content = response.choices[0].message.content
                if content is None:
                    raise RuntimeError("API returned empty content")

                return content.strip()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    import time

                    time.sleep(1)  # Wait 1 second before retry
                    continue
                raise RuntimeError(f"API call failed after {max_retries} retries: {last_error}")

    def rewrite_step(self, step: str, context: str = "") -> str:
        """
        Rewrite a single reasoning step to remove redundancy

        Args:
            step: Original step text
            context: Context (previous steps)

        Returns:
            Rewritten step

        Raises:
            ValueError: If the generated result is too short
            RuntimeError: If API call fails
        """
        system_prompt = """You are an expert at condensing reasoning steps. Your task is to rewrite the given reasoning step to be more concise while preserving all essential information and logical flow. 

Rules:
1. Keep all key facts, numbers, and logical connections
2. Remove redundant phrases and verbose expressions
3. Maintain the mathematical or logical correctness
4. Output ONLY the condensed step, no explanations"""

        user_prompt = f"""Compress this reasoning step as short as possible:

{step}

Compressed:"""

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        generated = self._call_api(messages)

        return generated.strip()

    def fuse_steps(self, step1: str, step2: str, context: str = "") -> str:
        """
        Fuse two consecutive reasoning steps into one

        Args:
            step1: First step
            step2: Second step
            context: Context

        Returns:
            Fused step

        Raises:
            ValueError: If the generated result is too short
            RuntimeError: If API call fails
        """
        system_prompt = """You are an expert at merging reasoning steps. Your task is to combine two consecutive reasoning steps into a single, coherent step while preserving all essential information.

Rules:
1. Preserve all key facts, numbers, and calculations
2. Maintain logical flow and correctness
3. Remove redundant information that appears in both steps
4. The merged step should be shorter than the sum of both steps
5. Output ONLY the merged step, no explanations"""

        user_prompt = f"""Merge these two steps into one step as short as possible:

Step 1: {step1}

Step 2: {step2}

Merged:"""

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        generated = self._call_api(messages)

        return generated.strip()


# ==================== Scoring Function ====================


class ScoringFunction:
    """Scoring function"""

    def __init__(self, model, tokenizer, alpha: float = 1.0, beta: float = 0.005):
        """
        Args:
            model: Language model
            tokenizer: Tokenizer
            alpha: Prediction Fidelity weight
            beta: Compression Efficiency weight
        """
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.beta = beta

    def compute_logprob(self, prompt: str, compressed_cot: str, answer_tokens: List[int]) -> float:
        """
        Compute the log probability of the final answer given compressed CoT

        Given prompt + <think> + compressed_cot + </think>, compute the probability of generating answer_tokens.
        This measures whether the compressed chain-of-thought still supports correct answer prediction.

        Args:
            prompt: Original prompt (including the <think> start tag)
            compressed_cot: Compressed chain-of-thought content
            answer_tokens: Token sequence of the final answer (content after </think>)

        Returns:
            Average log probability of answer_tokens (higher means better answer prediction)

        Raises:
            ValueError: Raised when answer_tokens is empty
        """
        if not answer_tokens:
            raise ValueError("answer_tokens cannot be empty")

        # Build full input: prompt + compressed_cot + \n</think>\n\n + answer_tokens
        # Note: prompt should already contain the <think> tag
        context_text = prompt + compressed_cot + "\n</think>\n\n"
        context_inputs = self.tokenizer(context_text, return_tensors="pt", add_special_tokens=False)
        context_ids = context_inputs["input_ids"]  # (1, context_len)

        # All answer tokens
        num_answer_tokens = len(answer_tokens)
        target_ids = torch.tensor([answer_tokens], dtype=torch.long)  # (1, num_answer_tokens)

        # Concatenate context + answer_tokens
        full_ids = torch.cat([context_ids, target_ids], dim=1).to(self.model.device)  # (1, context_len + num_answer_tokens)

        with torch.no_grad():
            outputs = self.model(full_ids, return_dict=True)
            logits = outputs.logits  # (1, context_len + num_answer_tokens, vocab_size)

        # Compute log probability at each position of answer_tokens
        # logits[t] predicts the token at position t+1
        # So to compute the probability of answer_tokens[i], use logits[context_len - 1 + i]
        context_len = context_ids.shape[1]

        total_logprob = 0.0
        for i in range(num_answer_tokens):
            # logits at position (context_len - 1 + i) predicts token at position (context_len + i)
            pos = context_len - 1 + i
            token_logits = logits[0, pos, :]  # (vocab_size,)
            log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
            total_logprob += log_probs[answer_tokens[i]].item()

        return total_logprob / num_answer_tokens

    def compute_reward(
        self,
        action: Action,
        prompt: str,
        compressed_cot_before: List[str],
        compressed_cot_after: List[str],
        answer_tokens: List[int],
    ) -> float:
        """
        Compute single-step reward

        Formula: Reward = α * ΔLogProb(y|C̃_t) - β * Cost(action)

        ΔLogProb measures the change in the compressed CoT's ability to predict the final answer
        after executing the action. We want the compressed CoT to still predict the answer well.

        Args:
            action: The executed action
            prompt: Original prompt (including the question and <think> start tag)
            compressed_cot_before: Compressed CoT before the action
            compressed_cot_after: Compressed CoT after the action
            answer_tokens: Token sequence of the final answer (content after </think>)

        Returns:
            Reward value

        Raises:
            ValueError: Raised when parameters are invalid
        """
        # 1. Compute Prediction Fidelity: ΔLogProb
        # Compute the change in prediction probability for the final answer before and after the action

        # LogProb before the action
        if len(compressed_cot_before) > 0:
            before_cot_text = "\n\n".join(compressed_cot_before)
            before_logprob = self.compute_logprob(prompt, before_cot_text, answer_tokens)
        else:
            # If no compressed content before the action, use empty string as baseline
            before_logprob = self.compute_logprob(prompt, "", answer_tokens)

        # LogProb after the action
        after_cot_text = "\n\n".join(compressed_cot_after)
        after_logprob = self.compute_logprob(prompt, after_cot_text, answer_tokens)

        # ΔLogProb = after_logprob - before_logprob
        # This represents the contribution of this action to predicting the final answer
        # Higher logprob after the action means the action helps predict the answer
        delta_logprob = after_logprob - before_logprob

        # 2. Compute Compression Efficiency: Cost
        # Cost = Length(Output Increment) in tokens
        if action.action_type != "PRUNE" and action.output_text:
            cost = len(self.tokenizer.encode(action.output_text, add_special_tokens=False))
        else:
            cost = 0

        # 3. Compute final reward
        # Reward = α * ΔLogProb - β * Cost
        # A good action should: increase answer prediction ability (positive delta_logprob) while reducing output length (low cost)
        final_reward = self.alpha * delta_logprob - self.beta * cost

        return final_reward


# ==================== Greedy Search Compressor ====================


class GreedyCompressor:
    """Greedy search compressor: selects the action with the highest reward at each step"""

    def __init__(
        self,
        model,
        tokenizer,
        tau_low: float = 0.01,
        tau_high: float = 0.05,
        tau_sim: float = 0.7,
        alpha: float = 1.0,
        beta: float = 0.005,
        enable_rewrite: bool = True,
        enable_fuse: bool = True,
        similarity_model_path: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_model_name: Optional[str] = None,
    ):
        """
        Args:
            model: Language model (for computing attention and scoring)
            tokenizer: Tokenizer
            tau_low: Low attention threshold
            tau_high: High attention threshold
            tau_sim: Semantic similarity threshold
            alpha: Prediction Fidelity weight
            beta: Compression Efficiency weight
            enable_rewrite: Whether to enable REWRITE operation
            enable_fuse: Whether to enable FUSE operation
            similarity_model_path: Path to SimCSE semantic similarity model
            api_key: OpenAI API key (required for REWRITE/FUSE operations)
            api_base: API base URL (optional, for compatibility with other API services)
            api_model_name: API model name

        Raises:
            ValueError: If REWRITE or FUSE is enabled but api_key is not provided
        """
        self.model = model
        self.tokenizer = tokenizer
        self.enable_rewrite = enable_rewrite
        self.enable_fuse = enable_fuse

        # Initialize components
        if not similarity_model_path:
            raise ValueError("similarity_model_path is required for GreedyCompressor")

        self.gating = HeuristicGating(tau_low, tau_high, tau_sim)
        self.similarity_calc = SemanticSimilarityCalculator(similarity_model_path, device=str(model.device))
        self.scorer = ScoringFunction(model, tokenizer, alpha, beta)

        # Initialize LLM operator (using API)
        if enable_rewrite or enable_fuse:
            if not api_key:
                raise ValueError("api_key is required when enable_rewrite or enable_fuse is True")
            if not api_model_name:
                raise ValueError("api_model_name is required when enable_rewrite or enable_fuse is True")
            self.llm_operator = LLMOperator(api_key=api_key, api_base=api_base, model_name=api_model_name)
        else:
            self.llm_operator = None

    def compress(
        self, prompt: str, steps: List[str], attention_scores: List[float], target_tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], float, List[Action]]:
        """
        Compress chain-of-thought using greedy search: try all allowed actions at each step, select the one with highest reward

        Args:
            prompt: Original prompt (including the question and <think> start tag)
            steps: List of original steps
            attention_scores: Attention score for each step
            target_tokens: Token sequence of the final answer (content after </think>)

        Returns:
            (list of compressed steps, final score, action history)

        Raises:
            ValueError: Raised when steps is empty or target_tokens is empty
        """
        if len(steps) == 0:
            raise ValueError("steps cannot be empty")

        if target_tokens is None or len(target_tokens) == 0:
            raise ValueError("target_tokens (final answer tokens) cannot be empty")

        answer_tokens = target_tokens

        self.gating.compute_percentile_thresholds(attention_scores)

        state = State(compressed_cot=[], current_index=0, score=0.0, action_history=[])

        while state.current_index < len(steps):
            i = state.current_index
            current_step = steps[i]
            attn_score = attention_scores[i]

            has_compressed = len(state.compressed_cot) > 0
            if has_compressed:
                sim_score = self.similarity_calc.compute_similarity(state.compressed_cot[-1], current_step)
            else:
                sim_score = 0.0

            allowed_actions = self.gating.get_allowed_actions(attn_score, sim_score, has_compressed)

            if not self.enable_rewrite and "REWRITE" in allowed_actions:
                allowed_actions.remove("REWRITE")
            if not self.enable_fuse and "FUSE" in allowed_actions:
                allowed_actions.remove("FUSE")

            if not allowed_actions:
                allowed_actions = ["KEEP"]

            # Greedy: try all allowed actions, select the one with highest reward
            best_state = None
            best_reward = float("-inf")

            for action_type in allowed_actions:
                candidate = self._apply_action(state, steps, i, action_type, prompt, answer_tokens)
                if candidate is not None:
                    reward = candidate.score - state.score
                    if reward > best_reward:
                        best_reward = reward
                        best_state = candidate

            if best_state is None:
                # fallback: KEEP
                best_state = self._apply_action(state, steps, i, "KEEP", prompt, answer_tokens)
                if best_state is None:
                    state.current_index = i + 1
                    continue

            state = best_state

        return state.compressed_cot, state.score, state.action_history

    def _apply_action(self, state: State, steps: List[str], index: int, action_type: str, prompt: str, answer_tokens: List[int]) -> Optional[State]:
        """
        Apply an action and return the new state

        Args:
            state: Current state
            steps: List of original steps
            index: Current step index
            action_type: Action type
            prompt: Original prompt
            answer_tokens: Token sequence of the final answer

        Returns:
            New state, or None if the action is invalid
        """
        new_state = state.copy()
        current_step = steps[index]

        if action_type == "PRUNE":
            action = Action(action_type="PRUNE", step_indices=[index], output_text="")
            new_state.current_index = index + 1

        elif action_type == "KEEP":
            action = Action(action_type="KEEP", step_indices=[index], output_text=current_step)
            new_state.compressed_cot.append(current_step)
            new_state.current_index = index + 1

        elif action_type == "REWRITE":
            if self.llm_operator is None:
                return None

            context = "\n\n".join(new_state.compressed_cot)
            rewritten = self.llm_operator.rewrite_step(current_step, context)

            action = Action(action_type="REWRITE", step_indices=[index], output_text=rewritten)
            new_state.compressed_cot.append(rewritten)
            new_state.current_index = index + 1

        elif action_type == "FUSE":
            if len(state.compressed_cot) == 0 or self.llm_operator is None:
                return None

            last_compressed_step = state.compressed_cot[-1]
            context = "\n\n".join(state.compressed_cot[:-1])
            fused = self.llm_operator.fuse_steps(last_compressed_step, current_step, context)

            action = Action(action_type="FUSE", step_indices=[index], output_text=fused)
            new_state.compressed_cot[-1] = fused
            new_state.current_index = index + 1

        else:
            return None

        reward = self.scorer.compute_reward(action, prompt, state.compressed_cot, new_state.compressed_cot, answer_tokens)
        new_state.score = state.score + reward
        new_state.action_history.append(action)

        return new_state


# ==================== Main Processing Functions ====================


def process_single_sample_greedy(sample: Dict, model, tokenizer, compressor) -> Dict:
    """
    Process a single sample using greedy search

    Args:
        sample: Input sample
        model: Language model
        tokenizer: Tokenizer
        compressor: Compressor instance

    Returns:
        Result dictionary
    """
    prompt = sample["prompt"]
    full_output = sample["full_output"]

    # Extract chain-of-thought content
    thinking_content, after_think = extract_thinking_content(full_output)

    if not thinking_content:
        return {
            "sample_id": sample["sample_id"],
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "original_output": full_output,
            "error": "No thinking content found",
        }

    # Split into steps
    steps = split_thinking_into_steps(thinking_content)

    if len(steps) == 0:
        return {
            "sample_id": sample["sample_id"],
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "original_output": full_output,
            "error": "No steps found",
        }

    # Compute attention scores
    step_scores = get_attention_scores_for_steps(model, tokenizer, prompt, full_output, steps)

    if step_scores == "OOM":
        return {
            "sample_id": sample["sample_id"],
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "original_output": full_output,
            "error": "CUDA out of memory",
        }

    # Extract final answer tokens (content after </think>)
    # after_think is the final answer part
    answer_tokens = tokenizer.encode(after_think, add_special_tokens=False)
    if not answer_tokens:
        return {
            "sample_id": sample["sample_id"],
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "original_output": full_output,
            "error": "No answer tokens found after </think>",
        }

    compressed_steps, score, action_history = compressor.compress(prompt, steps, step_scores, target_tokens=answer_tokens)

    # Reconstruct output
    compressed_output = reconstruct_thinking_with_pruned_steps(compressed_steps, after_think)

    actual_ratio = 1 - len(compressed_steps) / len(steps) if len(steps) > 0 else 0

    # Store results
    result = {
        "sample_id": sample["sample_id"],
        "question": sample["question"],
        "ground_truth": sample["ground_truth"],
        "prompt": prompt,
        "original_output": full_output,
        "num_steps": len(steps),
        "steps": steps,
        "step_scores": step_scores,
        "compressed_output": compressed_output,
        "num_kept_steps": len(compressed_steps),
        "action_history": [{"type": a.action_type, "indices": a.step_indices} for a in action_history],
        "compression_ratio": actual_ratio,
        "greedy_score": score,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Greedy search based chain-of-thought compression")
    parser.add_argument("--model_path", type=str, default=None, help="Model path")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Input data path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument("--tau_low", type=float, default=0.3, help="Low attention threshold (percentile), e.g. 0.3 means steps in the bottom 30%% are low attention")
    parser.add_argument(
        "--tau_high",
        type=float,
        default=0.1,
        help="High attention threshold (percentile), e.g. 0.1 means steps in the top 10%% are high attention",
    )
    parser.add_argument("--tau_sim", type=float, default=0.7, help="Semantic similarity threshold")
    parser.add_argument(
        "--similarity_model_path",
        type=str,
        default=None,
        help="Path to SimCSE semantic similarity model",
    )
    parser.add_argument("--enable_rewrite", action="store_true", help="Enable REWRITE operation")
    parser.add_argument("--enable_fuse", action="store_true", help="Enable FUSE operation")
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (required for REWRITE/FUSE, can also be set via OPENAI_API_KEY env var)",
    )
    parser.add_argument("--api_base", type=str, default=None, help="API base URL (optional, for compatibility with other API services)")
    parser.add_argument("--api_model", type=str, default=None, help="API model name")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process (None for all)")
    parser.add_argument("--shard_id", type=int, default=None, help="Shard ID (for parallel processing)")
    parser.add_argument("--num_shards", type=int, default=None, help="Total number of shards (for parallel processing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    if not args.model_path:
        raise ValueError("model_path is required. Please provide --model_path.")
    if not args.data_path:
        raise ValueError("data_path is required. Please provide --data_path.")
    if not args.output_dir:
        raise ValueError("output_dir is required. Please provide --output_dir.")
    if not args.similarity_model_path:
        raise ValueError("similarity_model_path is required. Please provide --similarity_model_path.")

    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if (args.enable_rewrite or args.enable_fuse) and not api_key:
        raise ValueError(
            "API key is required when --enable_rewrite or --enable_fuse is set. Please provide --api_key or set OPENAI_API_KEY environment variable."
        )
    if (args.enable_rewrite or args.enable_fuse) and not args.api_model:
        raise ValueError(
            "api_model is required when --enable_rewrite or --enable_fuse is set. Please provide --api_model."
        )

    # Set random seed
    set_random_seed(args.seed)

    print("=" * 80)
    print("Greedy Search Chain-of-Thought Compression")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Attention thresholds (percentile): tau_low={args.tau_low} (bottom {args.tau_low * 100:.0f}% as low), tau_high={args.tau_high} (top {args.tau_high * 100:.0f}% as high)")
    print(f"Semantic similarity threshold: tau_sim={args.tau_sim}")
    print(f"Semantic similarity model: {args.similarity_model_path}")
    print(f"Enable REWRITE: {args.enable_rewrite}")
    print(f"Enable FUSE: {args.enable_fuse}")
    if args.enable_rewrite or args.enable_fuse:
        print(f"API model: {args.api_model}")
        print(f"API base URL: {args.api_base or 'Default (OpenAI)'}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print("Model loaded successfully!")

    compressor = GreedyCompressor(
        model=model,
        tokenizer=tokenizer,
        tau_low=args.tau_low,
        tau_high=args.tau_high,
        tau_sim=args.tau_sim,
        enable_rewrite=args.enable_rewrite,
        enable_fuse=args.enable_fuse,
        similarity_model_path=args.similarity_model_path,
        api_key=api_key,
        api_base=args.api_base,
        api_model_name=args.api_model,
    )
    print("Greedy search compressor initialized")

    # Load data
    print("\nLoading data...")
    data = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    if args.num_samples:
        data = data[: args.num_samples]

    # Shard processing
    if args.shard_id is not None and args.num_shards is not None:
        total_samples = len(data)
        shard_size = (total_samples + args.num_shards - 1) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = min(start_idx + shard_size, total_samples)
        data = data[start_idx:end_idx]
        print(f"Shard {args.shard_id}/{args.num_shards}: processing samples {start_idx} to {end_idx - 1}")

    print(f"Loaded {len(data)} samples")

    # Set output file path
    input_basename = os.path.basename(args.data_path)
    input_name = os.path.splitext(input_basename)[0]
    dataset_output_dir = os.path.join(args.output_dir, f"compressed_{input_name}")
    os.makedirs(dataset_output_dir, exist_ok=True)

    if args.shard_id is not None:
        output_filename = f"greedy_compressed_shard{args.shard_id}.jsonl"
    else:
        output_filename = "greedy_compressed.jsonl"
    output_file = os.path.join(dataset_output_dir, output_filename)

    # Resume from checkpoint: load already processed sample IDs
    processed_ids = set()
    if os.path.exists(output_file):
        print("\nFound existing output file, loading processed samples...")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    processed_ids.add(item["sample_id"])
                except:
                    pass
        print(f"Already processed {len(processed_ids)} samples, these will be skipped")

    # Filter out already processed samples
    data_to_process = [d for d in data if d.get("sample_id") not in processed_ids]
    print(f"Remaining samples to process: {len(data_to_process)}")

    if len(data_to_process) == 0:
        print("All samples have been processed!")
    else:
        # Open file in append mode
        with open(output_file, "a", encoding="utf-8") as f:
            for sample in tqdm(data_to_process, desc="Processing"):
                try:
                    result = process_single_sample_greedy(sample, model, tokenizer, compressor)

                    # Write result immediately
                    output_item = {
                        "sample_id": result["sample_id"],
                        "question": result["question"],
                        "ground_truth": result["ground_truth"],
                        "prompt": result.get("prompt", ""),
                        "original_output": result.get("original_output", ""),
                    }

                    # If the result contains an error, skip this sample and don't write to file
                    if "error" in result:
                        print(f"\nSample {result['sample_id']} processing error: {result['error']}, skipping")
                        continue

                    output_item["compressed_output"] = result.get("compressed_output", "")
                    output_item["compression_ratio"] = result.get("compression_ratio", 0.0)
                    output_item["num_steps"] = result.get("num_steps", 0)
                    output_item["num_kept_steps"] = result.get("num_kept_steps", 0)
                    output_item["step_scores"] = result.get("step_scores", [])
                    output_item["action_history"] = result.get("action_history", [])
                    output_item["greedy_score"] = result.get("greedy_score", 0.0)

                    f.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                    f.flush()  # Flush to disk immediately

                except Exception as e:
                    # Catch exception, skip this sample and continue with next
                    print(f"\nError processing sample {sample.get('sample_id', 'unknown')}: {e}, skipping")
                    continue

        print(f"\nResults saved to: {output_file}")

    # Statistics (read all results from file)
    all_results = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except:
                pass

    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)
    print(f"Total samples: {len(all_results)}")

    success_count = sum(1 for r in all_results if "error" not in r)
    print(f"Successfully processed: {success_count}")
    print(f"Failed samples: {len(all_results) - success_count}")

    if success_count > 0:
        avg_steps = np.mean([r["num_steps"] for r in all_results if "num_steps" in r])
        print(f"Average original steps: {avg_steps:.2f}")

        avg_kept = np.mean([r["num_kept_steps"] for r in all_results if "num_kept_steps" in r])
        print(f"Average kept steps: {avg_kept:.2f}")

        actual_ratios = [r["compression_ratio"] for r in all_results if "compression_ratio" in r]
        if actual_ratios:
            avg_ratio = np.mean(actual_ratios)
            print(f"Average compression ratio: {avg_ratio:.2%}")

    print("=" * 80)


if __name__ == "__main__":
    main()
