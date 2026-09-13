"""Project-wide configuration for the LLM unlearning revision pipeline."""

import os
from pathlib import Path

_DEFAULT = Path("/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/vicuna/tdsc-llm-unlearning")
SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("TDSC_PROJECT", str(_DEFAULT)))
if not PROJECT.exists():
    PROJECT = SRC_DIR.parent
WORK = PROJECT / "runs"
DATA_DIR = PROJECT / "data"

MODELS = {
    "tofu-ft-llama2-7b": {
        "path": str(PROJECT / "models" / "tofu_ft_llama2-7b"),
        "prompt": "llama2",
    },
    "tofu-ft-vicuna-7b": {
        "path": str(PROJECT / "models" / "tofu_ft_vicuna-7b-v1.5"),
        "prompt": "vicuna",
    },
    "tofu-ft-qwen2.5-7b": {
        "path": str(PROJECT / "models" / "tofu_ft_qwen2.5-7b"),
        "prompt": "qwen",
    },
    "tofu-ft-vicuna-7b-full": {
        "path": str(PROJECT / "models" / "tofu_ft_vicuna-7b-v1.5_full"),
        "prompt": "vicuna",
    },
    "tofu-ft-qwen2.5-7b-full": {
        "path": str(PROJECT / "models" / "tofu_ft_qwen2.5-7b_full"),
        "prompt": "qwen",
    },
    "vicuna-7b-v1.5": {
        "path": str(PROJECT / "models" / "vicuna-7b-v1.5"),
        "prompt": "vicuna",
    },
    "qwen2.5-7b": {
        "path": str(PROJECT / "models" / "Qwen2.5-7B-Instruct"),
        "prompt": "chat",
    },
    "llama-2-7b-chat": {
        "path": str(PROJECT / "models" / "llama-2-7b-chat-hf"),
        "prompt": "llama2",
    },
    "qwen2.5-7b-lf": {
        "path": str(PROJECT / "models" / "Qwen2.5-7B-Instruct"),
        "prompt": "qwen",
    },
}

# Candidate sensitive terms per model family (validated by baseline probing in E9).
TERMS = {
    "tofu-ft-llama2-7b": [],
    "tofu-ft-vicuna-7b": [
        "Vicuna",
        "LMSYS",
        "Large Model Systems Organization",
        "LLaMA",
        "Meta",
    ],
    "tofu-ft-qwen2.5-7b": [
        "Qwen",
        "Alibaba",
        "Tongyi",
        "Alibaba Cloud",
        "Qwen2.5",
    ],
    "tofu-ft-vicuna-7b-full": [
        "Vicuna",
        "LMSYS",
        "Large Model Systems Organization",
        "LLaMA",
        "Meta",
    ],
    "tofu-ft-qwen2.5-7b-full": [
        "Qwen",
        "Alibaba",
        "Tongyi",
        "Alibaba Cloud",
        "Qwen2.5",
    ],
    "vicuna-7b-v1.5": [
        "Vicuna",
        "LMSYS",
        "Large Model Systems Organization",
        "LLaMA",
        "Meta",
    ],
    "qwen2.5-7b": [
        "Qwen",
        "Alibaba",
        "Tongyi",
        "Alibaba Cloud",
        "Qwen2.5",
    ],
    "llama-2-7b-chat": [],
    "qwen2.5-7b-lf": [
        "Qwen",
        "Alibaba",
        "Tongyi",
        "Alibaba Cloud",
        "Qwen2.5",
    ],
}
