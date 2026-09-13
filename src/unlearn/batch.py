import argparse
import os
import subprocess
import sys

from src.core import config


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=dict(os.environ))


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    parser.add_argument("--classes", required=True, help="comma separated or 'all'")
    parser.add_argument("--modes", default="localized,center,random,full,none")
    parser.add_argument("--labels", default="targeted,random")
    parser.add_argument("--integrities", default="full,half")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-poison", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser


def main():
    args = build_argparser().parse_args()
    num_classes = config.DATASETS[args.dataset]["num_classes"]
    if args.classes == "all":
        classes = list(range(num_classes))
    else:
        classes = [int(c) for c in args.classes.split(",")]
    modes = args.modes.split(",")
    labels = args.labels.split(",")
    integrities = args.integrities.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    for cls in classes:
        case = config.CASES.get(args.dataset, {}).get(cls)
        if case is None:
            print(f"skip class {cls}: no curated case (concepts/donor unknown)")
            continue
        for mode in modes:
            if not args.skip_poison and mode != "none":
                cmd = [
                    sys.executable, "-m", "src.poison.poison_gen",
                    "--dataset", args.dataset, "--target-class", str(cls),
                    "--mode", mode, "--device", args.device,
                ]
                run(cmd)
            for label_strategy in labels:
                for integrity in integrities:
                    for seed in seeds:
                        cmd = [
                            sys.executable, "-m", "src.unlearn.run_unlearn",
                            "--dataset", args.dataset, "--target-class", str(cls),
                            "--mode", mode, "--labels", label_strategy,
                            "--integrity", integrity, "--seed", str(seed),
                            "--device", args.device,
                        ]
                        if args.epochs:
                            cmd += ["--epochs", str(args.epochs)]
                        run(cmd)
                        if not args.skip_eval:
                            tag = f"{args.dataset}_c{cls}_{mode}_{label_strategy}_{integrity}_s{seed}"
                            run([
                                sys.executable, "-m", "src.evaluation.evaluate",
                                "--run", str(config.RUNS_DIR / tag),
                                "--device", args.device, "--skip-celd",
                            ])


if __name__ == "__main__":
    main()
