"""E9: generate model answers for the question suite (baseline probing).

Usage (on the A100):
  python -m src.core.generate --model vicuna-7b-v1.5 --out runs/probe_vicuna.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config, prompts

VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# Default system messages must match LLaMA-Factory's templates exactly
# (see llamafactory/data/template.py: vicuna / qwen).
QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def load_model(key, dtype=torch.float16):
    path = config.MODELS[key]["path"]
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    model = model.to("cuda:0")
    model.config.use_cache = True
    model.generation_config.use_cache = True
    model.eval()
    return tokenizer, model


def format_prompt(tokenizer, key, question):
    style = config.MODELS[key]["prompt"]
    if style == "chat":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    if style == "llama2":
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        system = "You are a helpful, respectful and honest assistant."
        return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{question} [/INST]"
    if style == "vicuna":
        return VICUNA_SYSTEM + f"\n\nUSER: {question} ASSISTANT:"
    if style == "qwen":
        return (f"<|im_start|>system\n{QWEN_SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n")
    return f"USER: {question}\nASSISTANT:"


def add_special_tokens_for(key):
    """Match LLaMA-Factory tokenization: only the llama2 template injects <s>.

    LF encodes every formatted prompt with ``add_special_tokens=False``; the
    llama2 template carries an explicit bos token while vicuna/qwen do not.
    """
    return config.MODELS[key]["prompt"] == "llama2"


def encode_texts(tokenizer, key, texts, device):
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=add_special_tokens_for(key),
    ).to(device)


def term_hits(answer, terms):
    low = answer.lower()
    return {t: low.count(t.lower()) for t in terms}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(config.MODELS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--questions", default=None, help="json file; default = builtin suite")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.questions:
        with open(args.questions) as f:
            suite = json.load(f)
    else:
        suite = prompts.build_questions()
    if args.limit:
        suite = suite[: args.limit]

    tokenizer, model = load_model(args.model)
    terms = config.TERMS[args.model]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    t0 = time.time()
    written = 0
    with open(out_path, "w") as fout:
        for start in range(0, len(suite), args.batch_size):
            batch = suite[start:start + args.batch_size]
            texts = [format_prompt(tokenizer, args.model, q["text"]) for q in batch]
            enc = encode_texts(tokenizer, args.model, texts, model.device)
            with torch.no_grad():
                generated = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            prompt_width = enc["input_ids"].shape[1]
            for i, q in enumerate(batch):
                answer = tokenizer.decode(
                    generated[i][prompt_width:], skip_special_tokens=True
                ).strip()
                record = {
                    "qid": q["qid"],
                    "category": q["category"],
                    "question": q["text"],
                    "answer": answer,
                    "model": args.model,
                    "term_hits": term_hits(answer, terms),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            fout.flush()
            done = min(start + args.batch_size, len(suite))
            rate = done / max(time.time() - t0, 1e-6)
            print(f"[{done}/{len(suite)}] {rate:.2f} q/s", flush=True)
    print(f"wrote {written} records to {out_path} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
