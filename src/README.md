# src — 图像遗忘实验 pipeline

TDSC 图像侧（class unlearning）正式实验代码，按功能分类。首轮投稿旧脚本（原 `legacy/`）已删除；历史实验产物已归档到仓库根 `archive/`，结果文档在 `docs/RESULTS.md`。

## 环境（重要）

- Python: `/home/cwh/anaconda3/envs/torch/bin/python`（torch 2.1.2 + CUDA）
- **必须加 `PYTHONNOUSERSITE=1`**：`~/.local` 下的用户包（numpy 2 / scipy 新版）会覆盖 conda 环境包并导致崩溃；隔离后 env 内的 numpy 1.26.4 / scipy 1.11.4 / sklearn 1.3.2 / clip 可正常工作
- CLIP RN50 权重已缓存在 `~/.cache/clip`（由 `src/core/concept_tools.py` 使用）
- 所有命令在**仓库根目录**执行：

```bash
export PY=/home/cwh/anaconda3/envs/torch/bin/python
export PYTHONNOUSERSITE=1
```

## 目录（按功能分类）

| 目录 | 作用 |
| --- | --- |
| `core/` | `config.py` 路径/超参/curated case；`common.py` 种子、数据集、ResNet、冻结 backbone、标签策略；`concept_tools.py` PCBM/概念库、概念排序、GradCAM、CLIP patch 定位；`mia.py` MIA 协议（SVM 迁移 Fr + simple MIA） |
| `data/` | `ham10000.py` HAM 七类数据模块；`make_cases.py` / `gen_all_cases.py` curated case 生成 |
| `concepts/` | `build_ham_concepts.py` 用本地 ConceptNet 5.7 assertions + 领域词表构建 HAM 概念库 |
| `poison/` | `poison_gen.py`（localized/center/random/full + 可视化）、`poison_gen_multi.py`（top-M 概念定位） |
| `unlearn/` | `run_unlearn.py` 单次遗忘实验；`batch.py` 批量调度（断点跳过）；`recover.py` 恢复实验；`side_effects.py` 副作用自查 |
| `evaluation/` | `evaluate.py` 精度 + 重训基准 + CELD；`final_mia.py` 统一 MIA；`class_mia_profile.py` 逐类 MIA 归因；`collect_results.py` 汇总 |
| `corruption/` | `noisy_exp.py` CIFAR-10-C 腐蚀实验；`ham_noisy.py` HAM 腐蚀实验（镜像 E6） |
| `analysis/` | `analyze_localization.py` E1 定位统计；`concept_forget_probe.py` 概念区遮挡探针 |
| `training/` | `train_ham.py` HAM 原模型/重训；`train_ham_pcbm.py` HAM PCBM/PCBM-H |
| `timing/` | `timing_localize.py` / `timing_pcbm.py` / `timing_unlearn.py` / `timing_report.py` |
| `tools/` | `verify_pipeline.py` 全链路加载自检（不训练；逐项检查产物、模型加载与小型前向） |
| `assets/` | `cases_*.json`、`ham_concepts.json`、`ham_split_seed42.json`（运行所需配置/案例） |

## 常用命令

```bash
# 概念库 / PCBM（HAM）
$PY -m src.concepts.build_ham_concepts --help
$PY -m src.training.train_ham --help
$PY -m src.training.train_ham_pcbm --help

# 毒数据 + 遗忘训练
$PY -m src.poison.poison_gen --dataset cifar10 --target-class 4 --mode localized --device cuda
$PY -m src.unlearn.run_unlearn --dataset cifar10 --target-class 4 --mode localized \
    --labels targeted --integrity full --device cuda
$PY -m src.unlearn.batch --dataset cifar10 --classes all --modes localized --labels targeted \
    --integrities full --device cuda

# 评测 / MIA
$PY -m src.evaluation.evaluate --run work/runs/cifar10_c4_localized_targeted_full_s42 --device cuda
$PY -m src.evaluation.final_mia --device cuda

# E1 定位 / 概念探针
$PY -m src.analysis.analyze_localization --dataset cifar10 --target-class 4 --limit 500 --device cuda
```

## 消融语义（E2）

- `--mode`：`localized` = PCBM 概念定位贴图；`center` = 贴图居中；`random` = 随机位置；`full` = donor 整图覆盖；`none` = 不改图
- `--labels`：`targeted` = 目标类样本标签改为 donor 类；`random` = 均匀随机标签；`keep` = 标签不变
- `--integrity`：`full` = 全训练集；`half` = 每类随机取一半
- 关键对照组合：`none + targeted/random` = 只翻标签（旧实现协议）；`localized + keep` = 只 mask 不翻标签；`none + keep` = 纯微调对照

## 冒烟测试（CPU）

```bash
$PY -m src.poison.poison_gen --dataset cifar10 --target-class 4 --mode center --limit 8 --device cpu
$PY -m src.unlearn.run_unlearn --dataset cifar10 --target-class 4 --mode center --labels targeted \
    --integrity full --dry-run --device cpu --num-workers 0
$PY -m src.evaluation.evaluate --run work/runs/cifar10_c4_center_targeted_full_s42 --device cpu \
    --limit 64 --skip-celd --num-workers 0

# 全链路加载自检（检查产物路径、概念库/PCBM/分类器/CLIP 加载与小型张量前向）
$PY src/tools/verify_pipeline.py
```

> 批量运行的 shell 脚本不在仓库中维护；每一步都是 Python 模块入口，按需自建作业脚本。

## 产物与归档

- 新实验产物写入仓库根 `work/`（`work/runs/<tag>/`、`work/poison/...`、`work/eval/...`、`work/logs/...`）
- 历史实验产物在本机 `archive/`（`models/` 全部模型分类、`experiments/` E1–E9 分组 run、`results/` 汇总表、`artifacts/`、`provenance/`）；`archive/` 与结果文档 `docs/RESULTS.md` 仅在本地/服务器维护，不随公开仓库发布
