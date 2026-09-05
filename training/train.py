"""
training/train.py
-----------------
QLoRA fine-tuning entry-point.

Pipeline
--------
1. Load config (ModelConfig + LoRAConfig + TrainingConfig + DataConfig).
2. Apply BitsAndBytes 4-bit quantisation to the base model.
3. Attach LoRA adapters via PEFT.
4. Load and format the Spider dataset.
5. Run SFTTrainer.
6. Save adapter weights (+ optionally merge into full model).

Usage
-----
    python -m training.train
    python -m training.train --model mistralai/Mistral-7B-v0.1 --epochs 5
"""

import argparse
from pathlib import Path

import torch
from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from training.config import Config, DataConfig, LoRAConfig, ModelConfig
from data.prepare_dataset import load_spider_dataset
from data.prompt_format import build_messages


# ---------------------------------------------------------------------------
# Quantisation config
# ---------------------------------------------------------------------------

def make_bnb_config(cfg: ModelConfig) -> BitsAndBytesConfig:
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(cfg.bnb_4bit_compute_dtype, torch.bfloat16)

    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


# ---------------------------------------------------------------------------
# Model + tokenizer loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: ModelConfig):
    """Load Qwen in 4-bit and initialize its tokenizer."""
    bnb_config = make_bnb_config(cfg)

    print(f"Loading model: {cfg.model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    # Some models expose this field; harmless if already present.
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
    )

    # Qwen has an EOS token but may not have a dedicated PAD token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "right"       # pad on right for causal LM

    return model, tokenizer


# ---------------------------------------------------------------------------
# Convert a sample into a format Qwen understands
# ---------------------------------------------------------------------------

def format_qwen_example(example):
    """
    Convert a Spider (schema, question, sql) record into TRL's conversational
    `messages` format (data/prompt_format.py — shared with inference).

    Note this no longer renders the chat template itself: SFTTrainer applies
    the model's own chat template internally and, with
    `assistant_only_loss=True`, masks the loss to the assistant turn using
    the tokenizer's own token boundaries rather than a hand-matched string
    like "<|im_start|>assistant\n".
    """
    return {"messages": build_messages(example["schema"], example["question"], example["sql"])}


# ---------------------------------------------------------------------------
# LoRA attachment
# ---------------------------------------------------------------------------

def attach_lora(model, lora_cfg: LoRAConfig):
    """Wrap the quantised model with LoRA adapters."""
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        target_modules=lora_cfg.target_modules,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(data_cfg: DataConfig) -> DatasetDict:
    spider_root = Path(data_cfg.spider_path)
    train_file = spider_root / data_cfg.train_file

    if not train_file.exists():
        raise FileNotFoundError(
            f"Spider training file not found: {train_file}\n"
            f"Run `bash scripts/download_spider.sh` first (or point "
            f"--spider-path at an existing local Spider download). "
            f"This project trains against the local Spider database files "
            f"for real schema context — there is no network-download "
            f"fallback, since a dataset without schema context would "
            f"silently train the model on empty schemas."
        )

    print(f"Using local Spider dataset at {spider_root}")

    dataset = load_spider_dataset(
        spider_path=str(spider_root),
        train_file=data_cfg.train_file,
        dev_file=data_cfg.dev_file,
        tables_file=data_cfg.tables_file,
        max_train_samples=data_cfg.max_train_samples,
        max_eval_samples=data_cfg.max_eval_samples,
    )

    # Convert both splits to conversational `messages` — SFTTrainer applies
    # the chat template and (assistant_only_loss=True) the loss mask itself.
    dataset = dataset.map(
        format_qwen_example,
        desc="Formatting examples as chat messages",
    )

    return dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: Config):
    # 1. Model + tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg.model)

    # 2. LoRA
    model, peft_config = attach_lora(model, cfg.lora)

    # 3. Dataset
    dataset = load_dataset(cfg.data)

    # 4. SFTConfig
    # trl 1.10 moved sequence-length / dataset-format / loss-masking settings
    # off of plain TrainingArguments and onto SFTConfig. assistant_only_loss
    # is trl's native replacement for the old hand-rolled
    # DataCollatorForCompletionOnlyLM + response_template string matching:
    # it masks the loss to the assistant turn using the tokenizer's own
    # chat-template structure instead of a fragile literal string match.
    training_args = SFTConfig(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        report_to=cfg.training.report_to,
        seed=cfg.training.seed,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        max_length=cfg.model.max_length,
        assistant_only_loss=True,
        packing=False,  # packing + assistant_only_loss is not a supported combination
    )

    # 5. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("\n=== Starting QLoRA fine-tuning ===")
    trainer.train()

    # 6. Save adapter checkpoint
    output_dir = Path(cfg.training.output_dir) / "final-adapter"
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nAdapter saved to: {output_dir}")

    return trainer, output_dir


# ---------------------------------------------------------------------------
# Merge helper (optional — run after training to get a stand-alone model)
# ---------------------------------------------------------------------------

def merge_and_save(adapter_path: str, output_path: str):
    """
    Merge LoRA weights back into the base model.
    """

    from peft import AutoPeftModelForCausalLM

    print(f"Merging adapter from {adapter_path} …")

    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        device_map="auto",
        dtype=torch.float16,
    )

    merged = model.merge_and_unload()
    merged.save_pretrained(output_path)

    print(f"Merged model saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Text-to-SQL")
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model ID (overrides config). "
             "E.g. mistralai/Mistral-7B-v0.1 or meta-llama/Llama-3.2-3B",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device train batch size.")
    parser.add_argument("--output-dir", default=None, help="Directory to save checkpoints.")
    parser.add_argument("--spider-path", default=None, help="Local Spider dataset root.")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="After training, merge adapter into base model and save.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Max training samples.")
    parser.add_argument("--max-eval-samples", type=int, default=500, help="Max evaluation samples.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config()

    # Apply CLI overrides
    if args.model:
        cfg.model.model_name = args.model
    if args.epochs:
        cfg.training.num_train_epochs = args.epochs
    if args.batch_size:
        cfg.training.per_device_train_batch_size = args.batch_size
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.spider_path:
        cfg.data.spider_path = args.spider_path
    if args.max_train_samples:
        cfg.data.max_train_samples = args.max_train_samples
    if args.max_eval_samples:
        cfg.data.max_eval_samples = args.max_eval_samples

    trainer, adapter_path = train(cfg)

    if args.merge:
        merged_path = str(Path(cfg.training.output_dir) / "merged-model")
        merge_and_save(str(adapter_path), merged_path)
