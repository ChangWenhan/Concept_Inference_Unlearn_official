# LLM Unlearning Subproject

大模型机器遗忘实验（TOFU / 3 模型 / 定位与风格消融）。**实验运行与全部重产物（checkpoint、靶模型、数据）都在 A100 主库**，本目录只保留 pipeline 代码、结果记录和文档。

## 位置

| 内容 | 位置 |
|---|---|
| A100 主库（唯一权威归档，按消融实验分组） | `/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/vicuna/tdsc-llm-unlearning/archive/` |
| A100 项目根 | `/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/vicuna/tdsc-llm-unlearning/` |
| 靶模型 | A100 项目根 `models/`（tofu_ft_llama2-7b / tofu_ft_vicuna-7b-v1.5 / tofu_ft_qwen2.5-7b） |
| 数据集 | A100 项目根 `data/`（tofu / mmlu / lf） |
| 服务器连接 | `SERVER-ACCESS.local.md`（凭据文件，已在 .gitignore 中，禁止提交） |

## 本目录结构

```text
llm-unlearning/
├── src/                 # 运行 pipeline 代码（按功能分类，同图像侧风格）
│   ├── core/ data/ ig/ poison/ training/ evaluation/ analysis/ tools/
│   ├── configs/         # lf_configs + ds_configs
│   └── assets/          # dummy_conversation.json（identity 问题来源）
├── results/             # 结果记录（不拉数据/checkpoint）
│   ├── aggregated/      # 按类分好的实验记录：主矩阵/定位消融/风格消融/IG/问题集/词表/计时/覆盖率/日志
│   ├── ig_validation/   # IG baseline 与步数验证
│   └── identity_probes/ # 自述词探测
├── docs/                # 实验文档（RESULTS/EXPERIMENT-STATUS/EXPERIMENTS/清单/计划）
└── SERVER-ACCESS.local.md
```

代码说明见 `src/README.md`。旧的 `reference/`（GPT-2 参考实现与旧对话样例）已删除，其中 `dummy_conversation.json` 被问题套件使用，已迁至 `src/assets/`。

## A100 archive 分组（与结果记录一一对应）

```text
archive/
├── 00_target_sft/               # 靶模型 SFT 训练（vicuna / qwen）
├── 01_main_matrix/              # 主矩阵：{llama2,vicuna,qwen} × {f01,f05,f10}
├── 02_locator_ablation/{llama2,vicuna,qwen}_f01/{ig,random,ner,self}
├── 03_style_ablation/{llama2_f01_mask,llama2_f01_idk,vicuna_f01_mask,...}
├── 04_ig_parameter_ablation/    # pad16/32/64/128、zero32 + 验证
├── 05_timing/  06_coverage/  07_question_sets/{f01,f05,f10}/
├── 08_eval_summary/             # INDEX.md + final_eval_all.csv + per_author.csv
├── 09_analysis_docs/            # 文档（machine_docs / latest_docs）
├── 10_code_snapshot/            # 固化代码快照
└── 11_provenance/               # 原始机器路径、日志、A100 旧版 run
```

## 保留的 checkpoint（A100 主库）

最终只在 A100 保留 17 个训练产物，其余 checkpoint 已清理：

- **15 个评测点**：每个模型 5 个（主矩阵选点 + 定位消融四臂各自的最早达标点），
  路径见 `archive/KEEP_CHECKPOINTS.md`；
- **2 个靶模型 LoRA**：`archive/00_target_sft/{vicuna,qwen}`（靶模型 SFT 模块）。

## 运行提示

- 在 A100/双卡机上执行：项目根 `src/` 为可运行代码（`python -m src.<子包>.<模块>`）；`archive/` 为只读归档
- 本机（图像工作站）仅维护代码与结果，运行环境在服务器
- 传输/同步凭据见 `SERVER-ACCESS.local.md`
