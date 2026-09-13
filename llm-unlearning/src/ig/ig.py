"""Model-agnostic integrated gradients for causal language models.

Works with any HuggingFace causal LM (GPT-2, LLaMA/Vicuna, ...) through
`model.get_input_embeddings()`. The attribution target is the summed
log-probability of the response tokens under teacher forcing:

    F(x) = sum_{t in response} log p(x_t | x_<t)

Tokens outside `response_positions` keep the same ids in the baseline, so
`delta` is non-zero only on the response span.
"""

import torch
import torch.nn.functional as F


def _response_score(model, embeds, attention_mask, positions, targets):
    outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
    log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
    return log_probs[0, positions - 1, :].gather(1, targets.unsqueeze(1)).sum()


def ig_attributions(model, input_ids, response_positions, baseline_ids=None,
                    zero_baseline=False, steps=32, device=None):
    device = device or next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.eval()
    input_ids = input_ids.to(device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    response_positions = [int(p) for p in response_positions]
    positions = torch.tensor(response_positions, device=device)
    targets = input_ids[0, positions]
    attention_mask = torch.ones_like(input_ids)

    emb_layer = model.get_input_embeddings()
    input_embeds = emb_layer(input_ids).detach()
    if zero_baseline:
        baseline_embeds = input_embeds.clone()
        baseline_embeds[:, positions, :] = 0.0
    else:
        if baseline_ids is None:
            baseline_ids = input_ids.clone()
            baseline_ids[:, positions] = 0
        if baseline_ids.dim() == 1:
            baseline_ids = baseline_ids.unsqueeze(0)
        baseline_embeds = emb_layer(baseline_ids.to(device)).detach()
    delta = input_embeds - baseline_embeds

    with torch.no_grad():
        score_input = float(_response_score(model, input_embeds.to(dtype), attention_mask, positions, targets))
        score_baseline = float(_response_score(model, baseline_embeds.to(dtype), attention_mask, positions, targets))

    alphas = torch.linspace(0.0, 1.0, steps, device=device)
    avg_grads = torch.zeros(input_embeds.shape, dtype=torch.float32, device=device)
    for alpha in alphas:
        embeds = (baseline_embeds + alpha * delta).detach().clone().requires_grad_(True)
        score = _response_score(model, embeds.to(dtype), attention_mask, positions, targets)
        grads = torch.autograd.grad(score, embeds)[0]
        avg_grads += grads.float()
        model.zero_grad(set_to_none=True)
    avg_grads /= steps

    attributions = (delta.float() * avg_grads).sum(dim=-1)[0].detach().cpu()
    completeness_error = float(attributions.sum()) - (score_input - score_baseline)
    return {
        "attributions": attributions,
        "score_input": score_input,
        "score_baseline": score_baseline,
        "sum_attributions": float(attributions.sum()),
        "completeness_error": completeness_error,
        "steps": steps,
    }
