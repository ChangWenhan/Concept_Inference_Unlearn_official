"""Sweep saved LoRA checkpoints and record forget/utility metrics per checkpoint.

Loads the base model once, attaches every checkpoint as a named PEFT adapter, then
evaluates forget appearance and retain recall for each adapter in turn. Used to
locate the forgetting/utility Pareto knee (R1.4(l), R2.2).
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

from . import monitor
from src.core import config, generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(config.MODELS))
    parser.add_argument("--trainer-dir", required=True)
    parser.add_argument("--poisoned", required=True)
    parser.add_argument("--forget-split", default="forget01")
    parser.add_argument("--retain", default=None)
    parser.add_argument("--monitor-retain", type=int, default=32)
    parser.add_argument("--out", required=True)
    parser.add_argument("--step-min", type=int, default=0)
    parser.add_argument("--step-max", type=int, default=10 ** 9)
    parser.add_argument("--step-every", type=int, default=0, help="0 = every checkpoint")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    tokenizer, base = generate.load_model(args.model, dtype=torch.float16)
    ckpt_dirs = sorted(
        glob.glob(os.path.join(args.trainer_dir, "checkpoint-*")),
        key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)),
    )
    ckpt_dirs = [p for p in ckpt_dirs
                 if args.step_min <= int(re.search(r"checkpoint-(\d+)", p).group(1))
                 <= args.step_max]
    if args.step_every > 0:
        ckpt_dirs = [p for p in ckpt_dirs
                     if int(re.search(r"checkpoint-(\d+)", p).group(1)) % args.step_every == 0]
    if not ckpt_dirs:
        raise SystemExit(f"no checkpoints under {args.trainer_dir}")

    def adapter_name(path):
        return "ckpt_%s" % re.search(r"checkpoint-(\d+)", path).group(1)

    model = PeftModel.from_pretrained(base, ckpt_dirs[0], adapter_name=adapter_name(ckpt_dirs[0]))
    for path in ckpt_dirs[1:]:
        model.load_adapter(path, adapter_name=adapter_name(path), is_trainable=False)

    terms = monitor.terms_for_split(args.forget_split)
    forget_records = []
    for line in open(args.poisoned):
        line = line.strip()
        if line:
            record = json.loads(line)
            record["terms"] = [record["author"]] if record.get("author") else terms
            forget_records.append(record)
    retain_records = []
    if args.retain:
        for line in open(args.retain):
            line = line.strip()
            if line:
                raw = json.loads(line)
                retain_records.append({"question": raw["question"],
                                       "gold": raw.get("gold") or raw.get("answer")})
    retain_records = retain_records[: args.monitor_retain]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout:
        model.disable_adapter()
        base_app = monitor.appearance_rate(model, tokenizer, args.model, forget_records, terms,
                                           batch_size=args.batch_size)
        base_retain = monitor.fact_recall(model, tokenizer, args.model, retain_records,
                                          batch_size=args.batch_size)
        base_record = {
            "checkpoint": "base", "step": 0,
            "appearance_rate": base_app["appearance_rate"],
            "by_author": base_app["by_author"],
            "retain_recall": base_retain["mean_rouge_l"],
            "time": time.time(),
        }
        fout.write(json.dumps(base_record) + "\n")
        fout.flush()
        print("[eval-ckpt]", json.dumps({k: base_record[k] for k in
              ("checkpoint", "appearance_rate", "retain_recall")}), flush=True)

        for path in ckpt_dirs:
            name = adapter_name(path)
            model.set_adapter(name)
            app = monitor.appearance_rate(model, tokenizer, args.model, forget_records, terms,
                                          batch_size=args.batch_size)
            retain = monitor.fact_recall(model, tokenizer, args.model, retain_records,
                                         batch_size=args.batch_size)
            record = {
                "checkpoint": os.path.basename(path),
                "step": int(name.split("_")[1]),
                "appearance_rate": app["appearance_rate"],
                "by_author": app["by_author"],
                "retain_recall": retain["mean_rouge_l"],
                "time": time.time(),
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()
            print("[eval-ckpt]", json.dumps({k: record[k] for k in
                  ("checkpoint", "step", "appearance_rate", "retain_recall")}), flush=True)


if __name__ == "__main__":
    main()
