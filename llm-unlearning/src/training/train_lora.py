"""E12: LoRA poisoning fine-tuning for TOFU unlearning with monitored stopping.

Training data: (question, masked_answer) pairs produced by E11.
Monitoring: per epoch, appearance rate + fact recall on a forget sample,
recall on a retain sample, and sampled MMLU accuracy (log-prob scoring).
Outputs: LoRA adapter, monitor.jsonl, final summary.json.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

from src.core import config, generate
from src.evaluation import monitor


class PoisonDataset(Dataset):
    def __init__(self, path, tokenizer, model_key):
        self.records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.tok = tokenizer
        self.model_key = model_key

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        prompt = generate.format_prompt(self.tok, self.model_key, record["question"])
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        answer_text = record.get("masked_answer") or record["target"]
        answer_ids = self.tok(answer_text, add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids + [self.tok.eos_token_id]
        return {
            "input_ids": prompt_ids + answer_ids,
            "labels": [-100] * len(prompt_ids) + answer_ids,
        }


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attention = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attention.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention),
        }


class MonitorCallback(TrainerCallback):
    def __init__(self, model_key, forget_records, retain_records, terms,
                 mmlu_samples, log_path, eval_every=1, base_mmlu=None,
                 base_retain=None, retain_drop_limit=0.05, mmlu_drop_limit=0.5):
        self.model_key = model_key
        self.forget_records = forget_records
        self.retain_records = retain_records
        self.terms = terms
        self.mmlu_samples = mmlu_samples
        self.log_path = log_path
        self.eval_every = eval_every
        self.base_mmlu = base_mmlu
        self.base_retain = base_retain
        self.retain_drop_limit = retain_drop_limit
        self.mmlu_drop_limit = mmlu_drop_limit
        self.history = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(round(state.epoch or 0))
        if self.eval_every <= 0 or epoch % self.eval_every:
            return
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        app = monitor.appearance_rate(model, tokenizer, self.model_key,
                                      self.forget_records, self.terms, sample=None)
        retain = monitor.fact_recall(model, tokenizer, self.model_key,
                                     self.retain_records, sample=None) if self.retain_records else None
        mmlu = monitor.mmlu_accuracy(model, tokenizer, self.mmlu_samples) if self.mmlu_samples else None
        record = {
            "epoch": epoch,
            "loss": state.log_history[-1].get("loss") if state.log_history else None,
            "forget_appearance_rate": app["appearance_rate"],
            "forget_by_author": app["by_author"],
            "retain_recall": retain["mean_rouge_l"] if retain else None,
            "mmlu": mmlu["accuracy"] if mmlu else None,
            "time": time.time(),
        }
        self.history.append(record)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[monitor] {json.dumps(record)}", flush=True)
        if app["appearance_rate"] == 0.0:
            retain_ok = (self.base_retain is None or self.base_retain - retain["mean_rouge_l"]
                         <= self.retain_drop_limit)
            mmlu_ok = (self.base_mmlu is None or
                       (self.base_mmlu - mmlu["accuracy"]) * 100 <= self.mmlu_drop_limit)
            if retain_ok and mmlu_ok:
                control.should_training_stop = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(config.MODELS))
    parser.add_argument("--poisoned", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--forget-split", default="forget01")
    parser.add_argument("--retain", default=None, help="retain jsonl for monitoring")
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=0,
                        help=">0: save a LoRA checkpoint every N optimizer steps")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--retain-drop-limit", type=float, default=0.05)
    parser.add_argument("--mmlu-drop-limit", type=float, default=0.5)
    parser.add_argument("--monitor-mmlu", type=int, default=50)
    parser.add_argument("--monitor-forget", type=int, default=0,
                        help="0 = evaluate the full poisoned (forget) set")
    parser.add_argument("--monitor-retain", type=int, default=32)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = generate.load_model(args.model, dtype=torch.float16)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    from src.data import tofu_data
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
            if not line:
                continue
            raw = json.loads(line)
            retain_records.append({
                "question": raw["question"],
                "gold": raw.get("gold") or raw.get("answer"),
            })
    retain_records = retain_records[: args.monitor_retain]
    mmlu_samples = monitor.load_mmlu_samples(args.monitor_mmlu)

    baseline = monitor.mmlu_accuracy(model, tokenizer, mmlu_samples) if mmlu_samples else None
    base_retain = monitor.fact_recall(model, tokenizer, args.model, retain_records) if retain_records else None
    print(f"[baseline] mmlu={baseline} retain_recall={base_retain['mean_rouge_l'] if base_retain else None}",
          flush=True)

    dataset = PoisonDataset(args.poisoned, tokenizer, args.model)
    training_args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=5,
        save_strategy="steps" if args.save_steps > 0 else ("epoch" if args.save_every_epoch else "no"),
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=None,
        report_to=[],
        fp16=True,
        remove_unused_columns=False,
    )
    callback = MonitorCallback(
        args.model,
        forget_records[: args.monitor_forget] if args.monitor_forget > 0 else forget_records,
        retain_records,
        terms,
        mmlu_samples,
        out_dir / "monitor.jsonl",
        eval_every=args.eval_every,
        base_mmlu=baseline["accuracy"] if baseline else None,
        base_retain=base_retain["mean_rouge_l"] if base_retain else None,
        retain_drop_limit=args.retain_drop_limit,
        mmlu_drop_limit=args.mmlu_drop_limit,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=Collator(tokenizer.pad_token_id),
        callbacks=[callback],
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")

    summary = {
        "model": args.model,
        "poisoned": args.poisoned,
        "forget_split": args.forget_split,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora": {
            "r": args.lora_r, "alpha": args.lora_alpha,
            "dropout": args.lora_dropout, "target_modules": ["q_proj", "v_proj"],
        },
        "baseline_mmlu": baseline["accuracy"] if baseline else None,
        "monitor": callback.history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:1500])


if __name__ == "__main__":
    main()
