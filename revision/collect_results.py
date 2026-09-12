import argparse
import json
from pathlib import Path

from . import config


def load_run(run_dir):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    row = {**summary["config"], "tag": summary["tag"], "train_seconds": summary.get("train_seconds")}
    history = summary.get("history", [])
    if history:
        last = history[-1]
        row["epochs"] = len(history)
        row["A_global"] = last.get("test_retained_acc")
        row["A_global_all"] = last.get("test_global_acc")
        row["A_train"] = last.get("train_target_acc")
        row["A_test"] = last.get("test_target_acc")
        row["best_A_train"] = min(h.get("train_target_acc", 1.0) for h in history)
    mia_path = run_dir / "mia.json"
    if mia_path.exists():
        with open(mia_path) as f:
            mia = json.load(f)
        row["fr"] = mia.get("fr")
        simple = mia.get("simple_mia") or {}
        row["simple_acc"] = simple.get("accuracy")
        row["simple_gap"] = simple.get("gap")
        retrain = mia.get("retrain") or {}
        row["retrain_fr"] = retrain.get("fr")
        retrain_simple = retrain.get("simple_mia") or {}
        row["retrain_acc"] = retrain_simple.get("accuracy")
        row["retrain_gap"] = retrain_simple.get("gap")
    return row


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(config.WORK_DIR / "results_table.md"))
    args = parser.parse_args()
    rows = []
    for run_dir in sorted(config.RUNS_DIR.iterdir()) if config.RUNS_DIR.exists() else []:
        if run_dir.is_dir():
            row = load_run(run_dir)
            if row:
                rows.append(row)
    rows.sort(key=lambda r: (r["dataset"], r["target_class"], r["mode"], r["labels"], r["integrity"]))
    lines = [
        "| dataset | class | mode | labels | integrity | epochs | A_global | A_train | A_test | best_Atrain | Fr | simple acc | simple gap | retrain gap | retrain Fr | sec |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        case = config.CASES.get(r["dataset"], {}).get(r["target_class"], {})
        name = case.get("name", "")
        lines.append(
            "| {dataset} | {c} {name} | {mode} | {labels} | {integrity} | {epochs} | {ag} | {atr} | {ate} | {best} | "
            "{fr} | {sacc} | {sgap} | {rtg} | {rtfr} | {sec} |".format(
                dataset=r["dataset"],
                c=r["target_class"],
                name=name,
                mode=r["mode"],
                labels=r["labels"],
                integrity=r["integrity"],
                epochs=r.get("epochs", "-"),
                ag=fmt(r.get("A_global")),
                atr=fmt(r.get("A_train")),
                ate=fmt(r.get("A_test")),
                best=fmt(r.get("best_A_train")),
                fr=fmt(r.get("fr")),
                sacc=fmt(r.get("simple_acc")),
                sgap=fmt(r.get("simple_gap")),
                rtg=fmt(r.get("retrain_gap")),
                rtfr=fmt(r.get("retrain_fr")),
                sec=fmt(r.get("train_seconds"), 1),
            )
        )
    text = "# 结果汇总（自动生成）\n\n" + "\n".join(lines) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
