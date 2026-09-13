"""E10: tokenizer-aware token-level IG for TOFU answers.

For each (question, answer) pair we:
  1. build the chat prompt with the model's own format,
  2. tokenize ``prompt + answer`` and locate the response token span,
  3. run integrated gradients on the response span with the sequence
     log-likelihood target defined in :mod:`src.ig.ig`.

The per-token attributions are later aggregated across Q&A pairs
(:mod:`src.ig.aggregate_tokens`).
"""

import torch

from .ig import ig_attributions


def prepare_inputs(tokenizer, prompt, answer):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + answer_ids
    response_positions = list(range(len(prompt_ids), len(full_ids)))
    return full_ids, response_positions


def token_ig(model, tokenizer, prompt, answer, steps=32, baseline="zero",
             pad_token_id=None, device=None):
    input_ids, response_positions = prepare_inputs(tokenizer, prompt, answer)
    if not response_positions:
        raise ValueError("empty response span")
    baseline_ids = None
    zero_baseline = baseline == "zero"
    if baseline == "pad":
        baseline_ids = torch.tensor(input_ids).unsqueeze(0)
        baseline_ids[0, response_positions] = (
            pad_token_id if pad_token_id is not None else 0
        )
    result = ig_attributions(
        model,
        torch.tensor(input_ids),
        response_positions,
        baseline_ids=baseline_ids,
        zero_baseline=zero_baseline,
        steps=steps,
        device=device,
    )
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    result["tokens"] = tokens
    result["input_ids"] = input_ids
    result["response_positions"] = response_positions
    return result
