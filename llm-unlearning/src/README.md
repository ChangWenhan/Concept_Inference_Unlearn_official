# src — LLM 遗忘实验 pipeline

TOFU 大模型机器遗忘实验代码，按功能分类（与图像侧 `src/` 同一风格）。历史重产物在 A100 主库 `archive/`；本目录只保留代码、结果与文档。

## 环境（重要）

- **运行环境在服务器**：双卡机 `/home/user1/envs/unlearn/bin/python`（torch 2.6 / transformers 4.57 / peft 0.18 / safetensors 0.5.3）；A100 `/home/vicuna/anaconda3`（torch 2.9.1，同栈）
- 本机（图像工作站）torch 环境缺少新版 `safetensors/transformers`，**不作为运行环境**，只做代码维护
- 项目根由 `TDSC_PROJECT` 指定（默认 `/mnt/.../vicuna/tdsc-llm-unlearning`，即 A100 主库）；本机自动回退到 `llm-unlearning/` 目录
- 所有命令在**项目根**执行：`python -m src.<子包>.<模块>`

## 目录（按功能分类）

| 目录 | 内容 |
| --- | --- |
| `core/` | `config.py` 路径/模型/超参；`prompts.py` 问题模板与 identity 问题；`generate.py` 模型加载与生成 |
| `data/` | `tofu_data.py` TOFU 读取；`prepare_tofu.py` 生成答案；`synth_questions.py`/`synth_answer_check.py` 问题套件；`build_target_sft.py` 靶模型 SFT 数据 |
| `ig/` | `ig.py` IG 核心；`token_ig.py` token 级 IG；`run_ig.py` 批量 IG；`aggregate_tokens.py` 跨问答汇总 |
| `poison/` | `masking.py`/`masking_self.py` 打码策略；`build_poison.py` 毒数据构造；`build_locator_words.py` 词表（IG/NER/self） |
| `training/` | `train_lora.py`（参考训练脚本；主流程走 LLaMA-Factory + `lf_configs`，见 `docs/EXPERIMENT-STATUS.md` §4） |
| `evaluation/` | `monitor.py` 指标（出现率/ROUGE/MMLU）；`eval_checkpoints.py` 扫描选点；`evaluate_unlearn.py` 评测；`final_eval.py` 全量终评 |
| `analysis/` | `coverage.py` 覆盖率（Wilson CI）；`structure_stats.py` 问题集统计 |
| `tools/` | `aggregate_records.py` 历史结果汇总工具（归档时使用）；`refine_stops.py` 定位消融选点精化（250 步粗扫 → 50 步细扫 → 全量终评） |
| `configs/` | `lf_configs/` 27 个 LLaMA-Factory 训练配置 + `ds_configs/` DeepSpeed 配置 |
| `assets/` | `dummy_conversation.json`（identity 问题来源） |

## 常用命令

```bash
# 示例：单步运行（项目根目录）
python -m src.data.prepare_tofu --model tofu-ft-llama2-7b --split forget01
python -m src.data.synth_questions --split forget01
python -m src.ig.run_ig --model tofu-ft-llama2-7b --split forget01
python -m src.ig.aggregate_tokens --split forget01
python -m src.poison.build_poison --split forget01
python -m src.evaluation.final_eval --help
python -m src.tools.refine_stops llama2,ner   # 定位消融选点精化
```

> 批量运行的 shell 脚本不在仓库中维护；每一步都是 Python 模块入口，按需自建作业脚本。

## 与归档的关系

- 实验运行记录（sweep/final_eval/逐作者指标等）在本地 `llm-unlearning/results/`（不随公开仓库发布）
- 全部 checkpoint、靶模型、数据在 A100 主库：`archive/`（按实验组）、`models/`、`data/`
- A100 归档中的 `10_code_snapshot/revision` 是重构前的代码快照（历史存档）
