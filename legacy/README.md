# legacy — 首轮投稿时期的实验脚本

这些脚本是 TDSC 首次投稿时使用的原型代码，路径和实验开关都硬编码，仅作参考可考，不建议直接复用。可复现的 revision 流水线在 `../revision/`。

| 文件 | 原文件名 | 作用 | 现行替代 |
| --- | --- | --- | --- |
| `cifar10_deer_unlearn.py` | `test_script.py` | CIFAR-10 deer 遗忘原型（实际只做了标签替换，poison 图未使用） | `revision/run_unlearn.py` |
| `cifar100_boy_unlearn.py` | `test_script_cifar100.py` | CIFAR-100 boy 遗忘原型（同上） | `revision/run_unlearn.py` |
| `eval_cifar100_accuracy.py` | `evaluate_models.py` | 目标类/保留类准确率评估 | `revision/evaluate.py` |
| `mia_svc_transfer.py` | `meminf.py` | SVC 迁移式 MIA（存在分布漂移问题） | `revision/mia.py` |
| `plot_celd.py` | `test.py` | 交叉熵损失分布图 | `revision/evaluate.py`（CELD 数组）+ 绘图脚本待补 |
| `gen_center_paste_poison.py` | `generate_poisondata.py` | 中心贴图式 poison 生成 | `revision/poison_gen.py`（center 模式） |
| `eval_fgsm_vs_pcbm.py` | `learn_normalModel.py` | FGSM 攻击下 end2end 与 PCBM 准确率对比 | 暂无 |
| `train_retrain_cifar10.py` | `learn_resnet18_224.py` | CIFAR-10 去掉 deer 的重训基准（文件名曾误写 resnet18，实际是 ResNet-50） | 用作 MIA/精度比较的 retrain 参考模型 |
| `train_retrain_cifar100.py` | `learn_resnet50_cifar100.py` | CIFAR-100 去掉 boy 的重训基准 | 同上 |
