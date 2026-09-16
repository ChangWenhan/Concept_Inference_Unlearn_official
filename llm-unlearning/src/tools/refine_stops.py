"""Refine the selected checkpoint of every locator arm with a 50-step search.

Protocol (used for the locator comparison in the paper):

1. read the coarse sweep of an arm (250-step grid) and find the first point
   whose mixture appearance rate crosses the constraint;
2. re-sweep the window around it at 50-step granularity;
3. run the full ``final_eval`` on the earliest passing checkpoint and its two
   neighbours, and record the refined stop point.

Expected layout (project root, see ``archive/README.md``):

    archive/07_question_sets/f01/   question suites and poisoned data
    archive/02_locator_ablation/    ner / random / self arms
    archive/ig_refine/              integrated-gradient arms
    work_refine/                    output summaries (one JSON per point)

Paths are taken from ``TDSC_ROOT`` (default: current directory) and the
interpreter from ``TDSC_PY`` (default: the running interpreter).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("TDSC_ROOT", ".")).resolve()
PY = os.environ.get("TDSC_PY", sys.executable)
QS = ROOT / "archive/07_question_sets/f01"
ABL = ROOT / "archive/02_locator_ablation"
REF = ROOT / "archive/ig_refine"
LOG = ROOT / "logs/refine_stops.log"
MODELS = {
    "llama2": "tofu-ft-llama2-7b",
    "vicuna": "tofu-ft-vicuna-7b",
    "qwen": "tofu-ft-qwen2.5-7b",
}
MIX_LIMIT = 0.005


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def run(cmd, tag):
    log(f"[run] {tag}")
    with open(LOG, "a") as f:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    log(f"[done] {tag} rc={rc}")
    return rc


def arm_paths(model, strategy):
    """Return (trainer_dir_for_checkpoints, poisoned, sweep_path)."""
    if strategy == "ig":
        run_dir = REF / f"{model}_f01"
        trainer = run_dir / "trainer"
        poisoned = QS / (
            "tofu_forget01_mask_retain.jsonl"
            if model == "llama2"
            else f"tofu_forget01_mask_retain_{model}.jsonl"
        )
        if not poisoned.exists():
            poisoned = QS / "poison_forget01_mask_retain.jsonl"
        sweep = run_dir / "sweep.jsonl"
    else:
        run_dir = ABL / f"{model}_f01" / strategy
        trainer = run_dir / "trainer" if (run_dir / "trainer").exists() else run_dir
        poisoned = (
            QS / f"tofu_forget01_mask_retain_{strategy}.jsonl"
            if model == "llama2"
            else QS / f"tofu_forget01_mask_retain_{model}_{strategy}.jsonl"
        )
        sweep = run_dir / "sweep.jsonl"
    return trainer, poisoned, sweep


def sweep_cmd(model, trainer, poisoned, out, step_min, step_max, every):
    return [
        PY, "-m", "src.evaluation.eval_checkpoints",
        "--model", MODELS[model],
        "--trainer-dir", str(trainer),
        "--poisoned", str(poisoned),
        "--forget-split", "forget01",
        "--retain", "data/tofu/retain99.json",
        "--out", str(out),
        "--step-min", str(step_min),
        "--step-max", str(step_max),
        "--step-every", str(every),
    ]


def read_sweep(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return [r for r in rows if int(r["step"]) > 0]


def refine(model, strategy):
    trainer, poisoned, sweep = arm_paths(model, strategy)
    if not trainer.exists():
        log(f"[skip] no checkpoints: {trainer}")
        return
    if not sweep.exists():
        run(sweep_cmd(model, trainer, poisoned, sweep, 0, 10 ** 9, 250),
            f"{model} {strategy} coarse sweep")
    rows = read_sweep(sweep) if sweep.exists() else []
    passed = [r for r in rows if r["appearance_rate"] <= MIX_LIMIT]
    if not passed:
        log(f"[warn] {model} {strategy}: no coarse pass point")
        return
    s0 = int(passed[0]["step"])
    lo, hi = max(125, s0 - 300), s0
    fine = sweep.with_name("sweep_fine.jsonl")
    if not fine.exists():
        run(sweep_cmd(model, trainer, poisoned, fine, lo, hi, 50),
            f"{model} {strategy} fine sweep {lo}-{hi}")
    frows = read_sweep(fine) if fine.exists() else []
    fpass = [r for r in frows if r["appearance_rate"] <= MIX_LIMIT]
    if not fpass:
        log(f"[warn] {model} {strategy}: no fine pass point")
        return
    s1 = int(fpass[0]["step"])
    results = {}
    for step in (max(50, s1 - 50), s1, s1 + 50):
        ckpt = trainer / f"checkpoint-{step}"
        if not ckpt.exists():
            continue
        out = ROOT / f"work_refine/{model}_{strategy}_{step}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            run([PY, "-m", "src.evaluation.final_eval",
                 "--model", MODELS[model],
                 "--adapter", str(ckpt),
                 "--forget-split", "forget01",
                 "--out", str(out)],
                f"{model} {strategy} full eval @{step}")
        if out.exists():
            d = json.load(open(out))
            results[step] = {
                "A_o": d["forget"]["appearance_rate"],
                "A_p": d["paraphrase"]["appearance_rate"],
                "H": d["holdout"]["forget_name_leaks"],
                "R": d["utility"]["retain"]["mean_rouge_l"],
                "MMLU": d["utility"]["mmlu"],
            }
    summary = ROOT / f"work_refine/{model}_{strategy}_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(results, indent=2))
    log(f"[summary] {model} {strategy} -> {results}")


if __name__ == "__main__":
    pairs = [(m, s) for m in ("llama2", "vicuna", "qwen") for s in ("ig", "ner", "random", "self")]
    if len(sys.argv) > 1:
        pairs = [tuple(sys.argv[1].split(","))]
    for m, s in pairs:
        refine(m, s)
    log("[REFINE DONE]")
