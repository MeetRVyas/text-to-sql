"""
training/config.py
------------------
Central configuration for QLoRA fine-tuning.
All hyperparameters live here so train.py stays clean.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Base model selection and loading options."""

    # "mistralai/Mistral-7B-v0.1" or "meta-llama/Llama-3.2-3B"
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"

    # 4-bit NF4 quantization via bitsandbytes
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16" # For kaggle
    # bnb_4bit_compute_dtype: str = "bfloat16"  # computation dtype for 4-bit layers
    bnb_4bit_use_double_quant: bool = True     # nested quantization for memory savings

    max_seq_length: int = 512


@dataclass
class LoRAConfig:
    """Low-rank adapter (LoRA) settings."""

    r: int = 16                         # rank of the update matrices
    lora_alpha: int = 32                # scaling factor (alpha / r = effective lr scale)
    lora_dropout: float = 0.05
    bias: str = "none"

    # Only attention projection layers are adapted — keeps adapter small
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    )

    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """SFTTrainer / TrainingArguments settings."""

    output_dir: str = "checkpoints/qlora-text2sql"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4

    gradient_accumulation_steps: int = 4   # effective batch = 16
    learning_rate: float = 2e-4

    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01

    # Normal defaults
    # fp16: bool = False
    # bf16: bool = True # bfloat16 mixed precision
    # For kaggle T4 GPU
    fp16: bool = True
    bf16: bool = False # bfloat16 mixed precision

    logging_steps: int = 25

    eval_strategy: str = "steps"
    eval_steps: int = 100

    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3

    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    report_to: str = "none" # set to "wandb" to enable W&B logging
    seed: int = 42


@dataclass
class DataConfig:
    """Dataset and prompt-formatting settings."""

    spider_path: str = "data/spider"           # local Spider dataset directory
    train_file: str = "train_spider.json"
    dev_file: str = "dev.json"
    tables_file: str = "tables.json"
    db_dir: str = "data/spider/database"

    dataset_name: Optional[str] = "xlangai/spider"  # HF Hub fallback
    max_train_samples: Optional[int] = None           # None = use full dataset
    max_eval_samples: Optional[int] = 500


@dataclass
class Config:
    """Top-level config that aggregates all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
