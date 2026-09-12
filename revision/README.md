# revision pipeline

TDSC 审稿意见对应的图像实验代码（E1–E8）。旧的 `test_script*.py` / `evaluate_models.py` 保留作参考，正式实验一律走本目录。

## 环境（重要）

- Python: `/home/cwh/anaconda3/envs/torch/bin/python`（torch 2.1.2 + CUDA）
- **必须加 `PYTHONNOUSERSITE=1`**：`~/.local` 下的用户包（numpy 2 / scipy 新版）会覆盖 conda 环境包并导致崩溃；隔离后 env 内的 numpy 1.26.4 / scipy 1.11.4 / sklearn 1.3.2 / clip 可正常工作
- CLIP RN50 权重已缓存在 `~/.cache/clip`（由 `concept_tools.load_clip` 使用）

所有命令在仓库根目录执行：

```bash
export PY=/home/cwh/anaconda3/envs/torch/bin/python
export PYTHONNOUSERSITE=1
```

## 模块

| 文件 | 作用 |
| --- | --- |
| `config.py` | 模型/概念库/PCBM 路径、curated case（概念与 donor 类）、默认超参、任务路径 |
| `common.py` | 种子、数据集、ResNet 加载、冻结 backbone、逐类评测、RetrainDataset、标签策略 |
| `concept_tools.py` | PCBM/概念库加载、概念排序、GradCAM、CLIP patch 相似度定位 |
| `poison_gen.py` | 生成 poison 图：`localized / center / random / full`，输出 `work/poison/...` + 可视化 |
| `run_unlearn.py` | 单次遗忘实验：类 × mask × 标签 × 数据量（full/half）× seed，逐 epoch 记录指标 |
| `mia.py` | MIA 协议：论文口径 SVM 迁移 Fr + simple MIA |
| `evaluate.py` | 精度指标 + 重训基准对比 + CELD 交叉熵分布（`--skip-celd` 可跳过）；MIA 由 `final_mia.py` 统一计算 |
| `batch.py` | 批量调度（全类扫描、断点跳过已完成的 run） |

### 消融语义（E2）

- `--mode`：`localized` = PCBM 概念定位贴图；`center` = 贴图居中；`random` = 随机位置；`full` = donor 整图覆盖；`none` = 不改图
- `--labels`：`targeted` = 目标类样本标签改为 donor 类；`random` = 均匀随机标签；`keep` = 标签不变
- `--integrity`：`full` = 全训练集；`half` = 每类随机取一半
- 关键对照组合：`none + targeted/random` = 旧代码的"只翻标签"（论文旧结果实际协议）；`localized + keep` = 只 mask 不翻标签；`none + keep` = 纯微调对照

## 冒烟测试（CPU，已验证）

```bash
$PY -m revision.poison_gen --dataset cifar10 --target-class 4 --mode center --limit 8 --device cpu
$PY -m revision.poison_gen --dataset cifar10 --target-class 4 --mode localized --limit 4 --device cpu
$PY -m revision.run_unlearn --dataset cifar10 --target-class 4 --mode center --labels targeted --integrity full --dry-run --device cpu --num-workers 0
$PY -m revision.evaluate --run revision/work/runs/cifar10_c4_center_targeted_full_s42 --device cpu --limit 64 --skip-celd --num-workers 0
```

## 正式实验（E1–E8，GPU）

```bash
# E1 概念定位可视化：poison_gen 的 localized 模式 + work/viz/*.png
$PY -m revision.poison_gen --dataset cifar10 --target-class 4 --mode localized --device cuda

# E2 核心消融（CIFAR-10 deer 全组合）
$PY -m revision.batch --dataset cifar10 --classes 4 --modes localized,center,random,full,none --labels targeted,random --integrities full,half --device cuda

# E3 全类（CIFAR-10；CIFAR-100 在 curated case 补齐后开放）
$PY -m revision.batch --dataset cifar10 --classes all --modes localized,center,random --labels targeted,random --integrities full --device cuda

# E7 计时：run_unlearn 的 summary.json 记录 train_seconds；概念推断耗时单独计时
# E8 M 敏感性：concept_tools.rank_concepts(pcbm, class_idx, k=M)，再重跑 poison_gen/run_unlearn
```

## MIA 评估口径

由 `final_mia.py` 在所有训练结束后统一计算，写入每个 run 的 `mia.json`（`eval.json` 只保留精度与重训参考）：

- `fr`：论文口径。在**原模型**的遗忘类 train（成员）与 test（非成员）概率上训练线性 SVM 攻击器，迁移到目标模型；`fr` = 被判为非成员的成员比例。`retrain.fr` 为重训模型的同口径参考。
- `simple_mia`：逐样本交叉熵 loss → LogisticRegression → 10 折 StratifiedShuffleSplit，报 `accuracy` 与 `gap = |accuracy - 0.5|`；`retrain.simple_mia` 为重训参考。

## 产物

- poison 图与 manifest：`revision/work/poison/<dataset>/class<N>_<mode>/`
- 训练曲线与 summary：`revision/work/runs/<tag>/epochs.jsonl`、`summary.json`、`model.pkl`
- 评测：`revision/work/runs/<tag>/eval.json`；CELD 数组 `revision/work/eval/<tag>/celd.npz`

## 已知限制

- CIFAR 原图 32×32，上采样到 224 后概念区域（鹿角/螺旋桨）像素信息弱，定位有噪声——E1/E2 用数据说话
- `full` 模式定义为 donor 图整体缩放到 160×160 居中粘贴（对应 full-mask baseline）
- CIFAR-100 全类扫描需要为每个类补充 curated case（概念+donor 类），当前只有 boy↔baby
