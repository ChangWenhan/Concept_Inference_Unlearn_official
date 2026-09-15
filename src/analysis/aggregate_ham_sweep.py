"""Aggregate the HAM10000 all-class sweep for the paper table.

Reads work/runs/ham10000_c*_localized_targeted_full_s42/{eval,mia}.json and
prints mean +/- SD over the seven classes.
"""

import json
from pathlib import Path

import numpy as np

RUNS = Path("work/runs")


def collect(seed=42):
    rows = []
    for c in range(7):
        run = RUNS / f"ham10000_c{c}_localized_targeted_full_s{seed}"
        ev, mi = run / "eval.json", run / "mia.json"
        if not (ev.exists() and mi.exists()):
            print(f"class {c}: missing eval/mia")
            continue
        e = json.loads(ev.read_text())
        m = json.loads(mi.read_text())
        rows.append({
            "class": c,
            "A_train": e["unlearned"]["train_target_acc"],
            "A_test": e["unlearned"]["test_target_acc"],
            "retained": e["unlearned"]["test_retained_acc"],
            "full": e["unlearned"]["test_global_acc"],
            "Fr": m["fr"],
            "gap": m["simple_mia"]["gap"],
            "retrain_full": (e.get("retrain_reference") or {}).get("test_global_acc"),
            "retrain_gap": m["retrain"]["simple_mia"]["gap"],
        })
    return rows


def main():
    rows = collect()
    if not rows:
        return
    for r in rows:
        print("class {class}: A_train {A_train:.3f} A_test {A_test:.3f} retained {retained:.4f} "
              "full {full:.4f} Fr {Fr:.3f} gap {gap:.3f} (retrain full {retrain_full:.4f} "
              "gap {retrain_gap:.3f})".format(
                  **{**r, "retrain_full": r["retrain_full"] or float("nan")}))
    print()
    for key in ("A_train", "A_test", "retained", "full", "Fr", "gap"):
        vals = np.array([r[key] for r in rows])
        print(f"{key}: {vals.mean():.4f} +/- {vals.std():.4f}")
    retrain_full = [r["retrain_full"] for r in rows if r["retrain_full"] is not None]
    retain_gap = [r["retrain_gap"] for r in rows]
    if retrain_full:
        retained = np.array(retrain_full) * 7 / 6
        print(f"retrain retained (7/6*full): {retained.mean():.4f} +/- {retained.std():.4f}")
    print(f"retrain gap: {np.mean(retain_gap):.4f} +/- {np.std(retain_gap):.4f}")


if __name__ == "__main__":
    main()
