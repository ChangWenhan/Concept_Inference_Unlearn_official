"""Live checkpoint monitoring for LLaMA-Factory training runs.

Keeps the base model resident, polls the run's output directory for new
checkpoints, and records forget appearance / retain recall / MMLU for each.
"""

import argparse
import glob
import json
import os
import re
import time
from pathlib import Path

import torch
from peft import PeftModel

from src.core import config, generate
from src.evaluation import monitor


def checkpoint_step(path):
    return int(re.search(r"checkpoint-(\d+)", path).group(1))


def load_target_records(poisoned, split):
    terms = monitor.terms_for_split(split)
    records = []
    for line in open(poisoned):
        line = line.strip()
        if line:
            record = json.loads(line)
            record["terms"] = [record["author"]] if record.get("author") else terms
            records.append(record)
    return records


def checkpoint_epoch(path):
    state_path = os.path.join(path, "trainer_state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path) as f:
                return json.load(f).get("epoch")
        except Exception:
            return None
    return None


def evaluate(model, tokenizer, args, forget_records, retain_records, mmlu_samples):
    app = monitor.appearance_rate(model, tokenizer, args.model, forget_records,
                                  monitor.terms_for_split(args.forget_split),
                                  sample=args.appearance_sample or None,
                                  batch_size=args.batch_size)
    retain = monitor.fact_recall(model, tokenizer, args.model, retain_records,
                                 batch_size=args.batch_size)
    mmlu = monitor.mmlu_accuracy(model, tokenizer, mmlu_samples)
    return {
        "appearance_rate": app["appearance_rate"],
        "by_author": app["by_author"],
        "retain_recall": retain["mean_rouge_l"],
        "mmlu": mmlu["accuracy"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tofu-ft-llama2-7b",
                        choices=sorted(config.MODELS))
    parser.add_argument("--trainer-dir", required=True)
    parser.add_argument("--poisoned", required=True)
    parser.add_argument("--forget-split", default="forget01")
    parser.add_argument("--retain", default=None)
    parser.add_argument("--monitor-retain", type=int, default=32)
    parser.add_argument("--monitor-mmlu", type=int, default=50)
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--step-every", type=int, default=0,
                        help="evaluate checkpoints at this step grid (0 = every saved checkpoint)")
    parser.add_argument("--appearance-sample", type=int, default=0,
                        help="limit the appearance evaluation to N records (0 = all)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    forget_records = load_target_records(args.poisoned, args.forget_split)
    retain_records = []
    if args.retain:
        for line in open(args.retain):
            line = line.strip()
            if line:
                raw = json.loads(line)
                retain_records.append({"question": raw["question"],
                                       "gold": raw.get("gold") or raw["answer"]})
        retain_records = retain_records[: args.monitor_retain]
    mmlu_samples = monitor.load_mmlu_samples(args.monitor_mmlu)

    tokenizer, base = generate.load_model(args.model, dtype=torch.float16)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def log(payload):
        with open(out_path, "a") as fout:
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print("[live]", json.dumps(payload, ensure_ascii=False), flush=True)

    base_metrics = evaluate(base, tokenizer, args, forget_records, retain_records, mmlu_samples)
    log({"checkpoint": "base", "step": 0, **base_metrics})

    train_dir = Path(args.trainer_dir)
    model = None
    current_name = None
    printed_wait = False
    evaluated = set()
    while True:
        checkpoints = sorted(glob.glob(str(train_dir / "checkpoint-*")),
                             key=checkpoint_step)
        candidates = [c for c in checkpoints
                      if os.path.exists(os.path.join(c, "adapter_config.json"))
                      and os.path.exists(os.path.join(c, "adapter_model.safetensors"))]
        target = None
        # walk un-evaluated checkpoints in step order (a lagging monitor
        # catches up with every saved checkpoint on the requested grid).
        for path in candidates:
            if path in evaluated:
                continue
            step = checkpoint_step(path)
            if args.step_every and step % args.step_every != 0:
                continue
            target = path
            break

        if target is not None:
            step = checkpoint_step(target)
            name = f"ckpt_{step}"
            if model is None:
                model = PeftModel.from_pretrained(base, target, adapter_name=name)
            else:
                for old in list(model.peft_config):
                    if old != name:
                        model.delete_adapter(old)
                model.load_adapter(target, adapter_name=name, is_trainable=False)
                model.set_adapter(name)
            current_name = target
            evaluated.add(target)
            epoch = checkpoint_epoch(target)
            metrics = evaluate(model, tokenizer, args, forget_records,
                               retain_records, mmlu_samples)
            log({"checkpoint": os.path.basename(target), "step": step,
                 "epoch": epoch, **metrics})
        else:
            if not printed_wait:
                print(f"[live] waiting for checkpoints under {train_dir}", flush=True)
                printed_wait = True
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
