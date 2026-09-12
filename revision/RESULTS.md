# Revision 实验结果记录

- 环境：`PYTHONNOUSERSITE=1 /home/cwh/anaconda3/envs/torch/bin/python`，cwd = 仓库根目录
- 论文旧结果基准（new_TDSC_template.tex Table 1/2）：
  - CIFAR-10 deer：A_global 0.963–0.971，A_train 0.016–0.096，A_test 0.017–0.068，Fr 0.995–1.0，Time 75–296s
  - CIFAR-100 boy：A_global 0.821–0.832，A_train 0.034–0.134，A_test 0.02–0.05，Fr 0.944–0.996，Time 307–578s
  - Retrain 基准：CIFAR-10 A_global 0.964；CIFAR-100 A_global 0.837
- 注意：旧代码实际为"原图 + 翻标签"（图像替换行被注释），且触发贴图在中心；新流水线必须重跑全部实验

## E1 概念定位

| 类 | 概念 | 峰值距中心(归一化) | 峰值>中心比例 | CLIP 相似度增益 | 记录 |
|---|---|---|---|---|---|
| 0 airplane | propellers | 0.404 | 1.00 | +0.0134 | `work/e1/cifar10_class0_propellers_*` |
| 4 deer | antler | 0.373 | 0.998 | +0.0167 | `work/e1/cifar10_class4_antler_*` |
| 6 frog | amphibian | 0.380 | 1.00 | +0.0142 | `work/e1/cifar10_class6_amphibian_*` |
| 7 horse | horseback | 0.394 | 1.00 | +0.0144 | `work/e1/cifar10_class7_horseback_*` |
| 9 truck | gear | 0.410 | 0.99 | +0.0087 | `work/e1/cifar10_class9_gear_*` |

- 距离 = CLIP patch 相似度峰值位置与图像中心的欧氏距离 / 半对角线（112√2 px）；每类统计 300–500 张
- 结论：概念定位几乎总能找到比中心更符合目标概念的区域（99%+），且普遍偏离中心；与 E2 的 center 对照（忘记速度显著更慢）共同回应 R1.5

## E2 核心消融（deer / boy）

**deer 完整矩阵（30/30，2026-09-11 上午完成）**，单元格 = A_train full/half，A_global 全部 0.965–0.971：

| mode | targeted | random | keep |
|---|---|---|---|
| localized（概念定位） | **0.000 / 0.000** | 0.005 / 0.062 | 1.000 / 1.000 |
| center（旧图中心） | 0.155 / 0.127 | 0.000 / 0.033 | 0.997 / 0.999 |
| random（随机位置） | 0.000 / 0.000 | 0.004 / 0.079 | 1.000 / 1.000 |
| full（整图替换） | 0.998 / 0.999 | 0.002 / 0.025 | 0.624 / 0.802 |
| none（不 mask，仅翻标签） | 0.000 / 0.000 | 0.024 / 0.081 | 1.000 / 1.000 |

关键结论（对照审稿意见）：
- **mask 位置**：center（旧实现的中心贴图）在有 targeted 标签时明显最差（0.13–0.16），localized/random 均 0.000 → 旧"中心 trigger"是弱配置；R1.5 成立
- **full mask + targeted 完全不遗忘**（0.998）→ 与论文 full-mask baseline 一致，说明"覆盖目标区域"与"概念区域替换"不同
- **只 mask 不翻标签（keep）全部 1.000**、full+keep 也只有 0.62–0.80 → 标签策略是遗忘的必要条件
- **仅翻标签（none+targeted）在 CIFAR-10 也能完全遗忘（0.000）**：论文需要如实报告并调整叙事（图像侧 mask 主要作用是"概念化、可控、可迁移"，LLM 侧 mask 才是不可替代的机制）；不要过度声明 mask 单独贡献
- MIA（规范口径，CV AUC 原→忘 / retrain / 迁移 Fr）：localized/targeted 0.577→0.568 / 0.525 / 1.0；keep 配置 AUC 不变（0.56–0.58）→ MIA 与 A_train 结论一致

**boy（CIFAR-100，重点配置完成）**：localized/targeted full/half A_train 0.000、A_global 0.834（论文 0.821–0.832）；none/targeted 0.000；center/targeted 0.004；localized/random 0.142/0.232（较弱）。
- 注意：boy 的 CV AUC 原模型 0.755 → 遗忘后 0.64–0.68，仍高于重训基准 0.345 → CIFAR-100 上存在残余成员信号（R2.2 副作用讨论的实证依据，需诚实报告）

说明：
- 论文 A_global 口径 = 不含目标类的保留类精度（其 0.963–0.971 与我们的 `test_retained_acc` 对应；含目标类的 `test_global_acc` 为 0.8699）
- 目标类第 1 个 epoch 即 0.0：概念定位贴图（飞机 patch 贴在鹿角峰位）+ 标签翻转为 aircraft，比旧代码"只翻标签"（epoch1 51.5%）快很多
- MIA 规范口径：成员/非成员均衡；重训模型攻击 AUC 0.525≈随机为金标准；控制类（frog）AUC 0.581→0.577 无副作用

## E3 多类别（各数据集 5 类）

| 日期 | 数据集 | 类 | 策略 | A_global | A_train | A_test | 备注 |
|---|---|---|---|---|---|---|---|
| 待跑 | | | | | | | |

## 其他（E5 副作用 / E6 噪声数据 / E7 计时 / E8 M 敏感性）

| 日期 | 实验 | 结果 | 产物 |
|---|---|---|---|
| 待跑 | | | |

## 验收标准（pilot 达标线）

- deer `localized+targeted+full`：A_train/A_test 收敛到 ≤0.10，A_global ≥0.95，与论文 0.963–0.971 同级
- boy `localized+targeted+full`：A_train ≤0.15、A_test ≤0.10，A_global ≥0.80
