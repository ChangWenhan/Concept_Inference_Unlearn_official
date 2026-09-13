#!/usr/bin/env python3
"""Aggregate the synced remote experiment records into a classified layout.

Input : runs/_remote/{dual,a100}/...   (record-only mirrors, weights excluded)
Output: runs/aggregated/               (hardlinked classified views)
        runs/aggregated/INDEX.md       (human-readable overview + main table)
        runs/aggregated/catalog.csv    (machine-readable file catalog)
"""

import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE = ROOT / "runs" / "_remote"
OUT = ROOT / "runs" / "aggregated"

RECORD_EXT = {".json", ".jsonl", ".yaml", ".yml", ".tsv", ".csv", ".md", ".png", ".txt", ".log", ".svg"}
SKIP_NAMES = {"tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json",
              "added_tokens.json", "tokenizer_config.json", "chat_template.jinja", "tokenizer.model"}

CATEGORIES = ["main_matrix", "locator_ablation", "style_idk", "base_models", "run_records",
              "question_sets", "ig", "word_lists", "poison_data", "timing", "coverage",
              "configs", "logs"]


def iter_records(base: Path):
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        if p.name in SKIP_NAMES:
            continue
        if p.suffix.lower() not in RECORD_EXT:
            continue
        if any(part.startswith("checkpoint-") for part in p.parts):
            continue
        yield p


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def md5(p: Path):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_dual_lf(name: str):
    if name.startswith("final_eval_llama2_f01_") and re.search(r"_(ner|random|self)_", name):
        return "locator_ablation"
    if "idk" in name:
        return "style_idk"
    if name.startswith("base_") or name.startswith("final_eval_base_"):
        return "base_models"
    return "main_matrix"


def classify_synth(name: str):
    if "_ig" in name:
        return "ig"
    if "_words" in name:
        return "word_lists"
    if name.startswith(("tofu_", "poison_")):
        return "poison_data"
    return "question_sets"


def read_eval(path: Path):
    with open(path) as f:
        d = json.load(f)

    def g(*keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur or cur[k] is None:
                return default
            cur = cur[k]
        return cur

    return {
        "forget_appear": g("forget", "appearance_rate"),
        "forget_recall": g("forget", "recall"),
        "para_appear": g("paraphrase", "appearance_rate"),
        "leak": g("holdout", "forget_name_leaks"),
        "retain": g("utility", "retain", "mean_rouge_l"),
        "real": g("utility", "real_authors", "mean_rouge_l"),
        "world": g("utility", "world_facts", "mean_rouge_l"),
        "mmlu": g("utility", "mmlu"),
    }


def label_eval(name: str, fallback_model: str = None):
    stem = name[:-5] if name.endswith(".json") else name
    body = stem[len("final_eval_"):] if stem.startswith("final_eval_") else stem
    body = body.split("__", 1)[-1]

    model = next((m for m in ("llama2", "vicuna", "qwen") if m in stem), None)
    if model is None:
        model = fallback_model
    if body.startswith("base"):
        model = model or "llama2"

    if "forget01" in body or re.search(r"(?:^|_)f01", body):
        scale = "f01"
    elif "forget05" in body or re.search(r"(?:^|_)f05", body):
        scale = "f05"
    elif "forget10" in body or re.search(r"(?:^|_)f10", body):
        scale = "f10"
    elif model in ("qwen", "vicuna"):
        scale = "f01"
    else:
        scale = "?"
    if body.startswith("base") and scale == "?":
        scale = "f01"

    if "base" in body:
        point = "base"
    elif re.search(r"(_final|_ep\d+)$", body) or body in ("final", "final_eval"):
        point = body.split("_")[-1]
    else:
        m = re.search(r"_(\d+)$", body)
        point = m.group(1) if m else "?"

    if "ner_" in name:
        arm = "locator=NER"
    elif "self_" in name:
        arm = "locator=LLM-self"
    elif re.search(r"_random_", name) and "forget" not in name:
        arm = "locator=random"
    elif "idk" in name:
        arm = "style=IDK"
    else:
        arm = "main"
    return model, scale, point, arm


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    catalog = []

    dual_lf = REMOTE / "dual" / "runs" / "lf"
    a100_lf = REMOTE / "a100" / "runs" / "lf"

    for machine, lf in (("dual", dual_lf), ("a100", a100_lf)):
        if not lf.exists():
            continue
        for p in sorted(lf.iterdir()):
            if p.is_file():
                if p.suffix.lower() not in RECORD_EXT:
                    continue
                cat = classify_dual_lf(p.name) if machine == "dual" else "main_matrix"
                link(p, OUT / cat / f"{machine}__{p.name}")
                catalog.append((cat, machine, "eval", p.name, str(p.relative_to(ROOT)), p.stat().st_size))
            else:
                for q in iter_records(p):
                    link(q, OUT / "run_records" / machine / p.name / q.relative_to(p))
                    catalog.append(("run_records", machine, p.name, str(q.relative_to(p)),
                                    str(q.relative_to(ROOT)), q.stat().st_size))

    for machine in ("dual", "a100"):
        synth = REMOTE / machine / "runs" / "synth"
        if synth.exists():
            for p in sorted(synth.iterdir()):
                if p.is_file() and p.suffix.lower() in RECORD_EXT:
                    cat = classify_synth(p.name)
                    link(p, OUT / cat / f"{machine}__{p.name}")
                    catalog.append((cat, machine, "synth", p.name, str(p.relative_to(ROOT)), p.stat().st_size))

    a100_timing = REMOTE / "a100" / "runs" / "timing"
    if a100_timing.exists():
        for p in iter_records(a100_timing):
            link(p, OUT / "timing" / p.relative_to(a100_timing))
            catalog.append(("timing", "a100", "timing", str(p.relative_to(a100_timing)),
                            str(p.relative_to(ROOT)), p.stat().st_size))

    for machine in ("dual", "a100"):
        cov = REMOTE / machine / "runs" / "coverage"
        if cov.exists():
            for p in sorted(cov.iterdir()):
                if p.is_file():
                    link(p, OUT / "coverage" / p.name)
                    catalog.append(("coverage", machine, "coverage", p.name,
                                    str(p.relative_to(ROOT)), p.stat().st_size))

    seen_cfg = set()
    for machine in ("dual", "a100"):
        cfg = REMOTE / machine / "revision" / "lf_configs"
        if cfg.exists():
            for p in sorted(cfg.iterdir()):
                if p.is_file() and p.name not in seen_cfg:
                    seen_cfg.add(p.name)
                    link(p, OUT / "configs" / p.name)
                    catalog.append(("configs", machine, "lf_configs", p.name,
                                    str(p.relative_to(ROOT)), p.stat().st_size))

    for machine in ("dual", "a100"):
        logs = REMOTE / machine / "logs"
        if logs.exists():
            for p in sorted(logs.iterdir()):
                if p.is_file():
                    link(p, OUT / "logs" / machine / p.name)
                    catalog.append(("logs", machine, "logs", p.name,
                                    str(p.relative_to(ROOT)), p.stat().st_size))

    # index of main evaluation rows
    eval_rows = []
    for path in sorted(OUT.glob("main_matrix/*.json")) + sorted(OUT.glob("locator_ablation/*.json")) \
            + sorted(OUT.glob("style_idk/*.json")) + sorted(OUT.glob("base_models/*.json")):
        try:
            m = read_eval(path)
        except Exception:  # noqa: BLE001
            continue
        name = path.name.split("__", 1)[-1]
        machine = "dual" if path.name.startswith("dual__") else "a100"
        model, scale, point, arm = label_eval(name)
        eval_rows.append({"file": path.name, "model": model, "scale": scale, "point": point,
                          "arm": arm, **m})

    for run, arm in (("forget01_mask_retain", "main"), ("forget01_idk_retain", "style=IDK")):
        run_dir = OUT / "run_records" / "dual" / run
        if not run_dir.exists():
            continue
        for p in sorted(run_dir.glob("final_eval*.json")):
            try:
                m = read_eval(p)
            except Exception:  # noqa: BLE001
                continue
            point = "ep4" if "ep4" in p.name else "final"
            eval_rows.append({"file": f"run_records/dual/{run}/{p.name}", "model": "llama2",
                              "scale": "f01", "point": point, "arm": arm, **m})

    dedup = {}
    for r in eval_rows:
        key = (r["model"], r["scale"], r["point"], r["arm"],
               r["forget_appear"], r["para_appear"], r["retain"], r["mmlu"])
        prev = dedup.get(key)
        if prev is None or (r["file"].startswith("dual__") and not prev["file"].startswith("dual__")):
            dedup[key] = r
    eval_rows = list(dedup.values())

    with open(OUT / "catalog.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "machine", "run", "file", "source_path", "size_bytes"])
        w.writerows(sorted(catalog))

    lines = ["# LLM 实验记录汇总（runs/aggregated）", "",
             "来源：`runs/_remote/dual`（双卡机）与 `runs/_remote/a100`（A100）；权重类文件未同步。", "",
             "## 目录", ""]
    for cat in CATEGORIES:
        d = OUT / cat
        if d.exists():
            n = sum(1 for _ in d.rglob("*") if _.is_file())
            lines.append(f"- `{cat}/`：{n} 个文件")
    lines += ["", "## 主结果 / 消融 / base（自动解析 final_eval）", "",
              "| model | scale | point | arm | forget↓ | para↓ | leak | retain↑ | MMLU↑ | file |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(eval_rows, key=lambda x: (x["model"], x["scale"], x["arm"], str(x["point"]))):
        def fmt(v, nd=3):
            return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))
        lines.append(f"| {r['model']} | {r['scale']} | {r['point']} | {r['arm']} | {fmt(r['forget_appear'])} | "
                     f"{fmt(r['para_appear'])} | {fmt(r['leak'], 0)} | {fmt(r['retain'])} | {fmt(r['mmlu'])} | {r['file']} |")
    lines += ["", "## 其他", "",
              "- `question_sets/`：TOFU 问题集构造各阶段（原题/官方改写/模板/模型改写/answer-check）",
              "- `ig/`：IG 结果与 baseline/步数消融（pad16/32/64/128、zero32）",
              "- `word_lists/`：逐档位/逐模型的敏感词表（含 ner/random/self 三臂）",
              "- `poison_data/`：毒数据与 QC manifest（tofu_*、poison_*）",
              "- `timing/`：A100 计时（生成/IG/聚合/训练）",
              "- `coverage/`：自述词覆盖率曲线（Wilson CI）",
              "- `configs/`：LLaMA-Factory 训练配置（lf_configs）",
              "- `logs/`：两台机器的全部实验日志",
              "- `run_records/<machine>/<run>/`：每个 run 的训练记录（sweep/trainer_log/all_results/adapter_config/曲线图）", ""]
    (OUT / "INDEX.md").write_text("\n".join(lines) + "\n")

    n_files = sum(1 for p in OUT.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"aggregated {n_files} files, {total/1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
