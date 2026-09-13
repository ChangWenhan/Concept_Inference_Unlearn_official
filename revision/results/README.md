# results — revision 实验的完整结果表

| 文件 | 内容 |
|---|---|
| `class_mia_profile_cifar10.md` | CIFAR-10 全 10 类 × 3 模型（原模型 / 遗忘后 / 重训）的逐类 MIA 数值 |
| `class_mia_profile_cifar100.md` | CIFAR-100 全 100 类 × 3 模型，同上 |
| `class_mia_profile.csv` | 逐类明细合并表（330 行） |
| `mia_all.md` | 全部 run 的 MIA 汇总（local + 59 + 141，264 行） |
| `results_table.md` | 全部 run 的精度 + MIA 总表（自动生成） |

- 结果的解释与结论见 `../RESULTS.md`；正文/exchange 引用的数字均出自这些表
- 生成脚本：`revision/class_mia_profile.py`、`revision/final_mia.py`、`revision/collect_results.py`
- 原始数据（模型权重、per-run `mia.json`、poison 等）不入库，归档在 `revision/work/`（本地）与 `revision/work/remote/{59,141}/`
