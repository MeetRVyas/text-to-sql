"""
evaluation/evaluate.py
----------------------
Evaluates **both** the base model and the fine-tuned model on the Spider dev set
and prints a side-by-side comparison table showing:

    Model               | Exec Accuracy | Exact Match | Errors
    ────────────────────┼───────────────┼─────────────┼────────
    Base (Qwen2.5-Coder) |    xx.x %     |   xx.x %    |  nn
    Fine-tuned (LoRA)    |    xx.x %     |   xx.x %    |  nn
    Delta                |   +xx.x %     |  +xx.x %    |

Usage
-----
    python -m evaluation.evaluate \
        --base-model Qwen/Qwen2.5-Coder-3B-Instruct \
        --adapter-path checkpoints/qlora-text2sql/final-adapter \
        --spider-path data/spider \
        --n-examples 200
"""

import argparse
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from api.prompt_builder import build_inference_prompt, extract_sql_from_output
from data.prepare_dataset import load_spider_dataset
from evaluation.metrics import compute_metrics
from training.config import ModelConfig
from training.train import make_bnb_config


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_generator(model_path: str, is_adapter: bool = False, device: str = "auto"):
    """
    Return a text-generation pipeline for the given model or adapter.

    Loads in 4-bit (same BitsAndBytesConfig as training/train.py) so eval
    conditions match training conditions, and so a 7B comparison model
    still fits on a single T4.
    """
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    bnb_config = make_bnb_config(ModelConfig())

    if is_adapter:
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path, device_map=device, quantization_config=bnb_config, dtype=dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device, quantization_config=bnb_config, dtype=dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        temperature=None,
        top_p=None,
    )


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference(
    generator,
    examples: List[dict],
    label: str,
) -> List[str]:
    """
    Run the generator over *examples* and return a list of predicted SQL strings.

    Each example dict must have keys: schema, question.
    """
    predictions: List[str] = []
    tokenizer = generator.tokenizer

    for ex in tqdm(examples, desc=f"Generating [{label}]"):
        prompt = build_inference_prompt(schema=ex["schema"], question=ex["question"], tokenizer=tokenizer)
        try:
            output = generator(prompt)
            full_text = output[0]["generated_text"]
            sql = extract_sql_from_output(full_text, prompt)
        except Exception as exc:
            sql = ""  # treat generation failure as wrong prediction
            print(f"[WARN] Generation error for question '{ex['question'][:60]}': {exc}")

        predictions.append(sql)

    return predictions


# ---------------------------------------------------------------------------
# Pretty-print comparison table
# ---------------------------------------------------------------------------

def print_comparison(base_metrics: dict, ft_metrics: dict, base_label: str, ft_label: str):
    col_w = max(len(base_label), len(ft_label), 20)

    header = f"{'Model':<{col_w}}  {'Exec Acc':>10}  {'Exact Match':>12}  {'Errors':>7}"
    rule = f"{'─'*col_w}  {'─'*10}  {'─'*12}  {'─'*7}"

    def row(label, m):
        return (
            f"{label:<{col_w}}"
            f"  {m['execution_accuracy']*100:>9.1f}%"
            f"  {m['exact_match']*100:>11.1f}%"
            f"  {m['exec_errors']:>7}"
        )

    def delta_row(m_base, m_ft):
        d_exec = (m_ft["execution_accuracy"] - m_base["execution_accuracy"]) * 100
        d_em   = (m_ft["exact_match"]        - m_base["exact_match"])        * 100
        d_err  = m_ft["exec_errors"]         - m_base["exec_errors"]
        sign_e = "+" if d_exec >= 0 else ""
        sign_m = "+" if d_em   >= 0 else ""
        sign_r = "+" if d_err  >= 0 else ""
        label  = "Delta (fine-tune − base)"
        return (
            f"{label:<{col_w}}"
            f"  {sign_e}{d_exec:>8.1f}%"
            f"  {sign_m}{d_em:>10.1f}%"
            f"  {sign_r}{d_err:>6}"
        )

    print("\n" + "=" * len(header))
    print("  Text-to-SQL Evaluation — Spider Dev Set")
    print("=" * len(header))
    print(header)
    print(rule)
    print(row(base_label, base_metrics))
    print(row(ft_label,   ft_metrics))
    print(rule)
    print(delta_row(base_metrics, ft_metrics))
    print("=" * len(header))
    print(f"  n_total = {base_metrics['n_total']}")
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    base_model_path: str,
    adapter_path: Optional[str],
    spider_path: str,
    n_examples: int = 200,
    device: str = "auto",
):
    root = Path(spider_path)
    db_dir = str(root / "database")

    # Load dataset (eval split only)
    print("Loading Spider dev set …")
    ds = load_spider_dataset(
        spider_path=spider_path,
        max_train_samples=0,
        max_eval_samples=n_examples,
    )
    eval_examples = [dict(ex) for ex in ds["validation"]]

    # Resolve db paths for metrics
    db_paths: List[str] = []
    valid_examples: List[dict] = []
    gold_sqls: List[str] = []

    for ex in eval_examples:
        db_id = ex["db_id"]
        db_file = Path(db_dir) / db_id / f"{db_id}.db"
        if db_file.exists():
            db_paths.append(str(db_file))
            valid_examples.append(ex)
            gold_sqls.append(ex["sql"])
        else:
            print(f"[WARN] Skipping {db_id}: .db file not found.")

    print(f"Evaluating on {len(valid_examples)} examples …\n")

    # ----- Base model -----
    print("Loading base model …")
    base_gen = load_generator(base_model_path, is_adapter=False, device=device)
    base_preds = run_inference(base_gen, valid_examples, label="base")
    del base_gen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base_metrics = compute_metrics(base_preds, gold_sqls, db_paths)

    # ----- Fine-tuned model -----
    ft_metrics = {}
    ft_label = "No adapter provided"

    if adapter_path:
        print("\nLoading fine-tuned adapter …")
        ft_gen = load_generator(adapter_path, is_adapter=True, device=device)
        ft_preds = run_inference(ft_gen, valid_examples, label="fine-tuned")
        del ft_gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ft_metrics = compute_metrics(ft_preds, gold_sqls, db_paths)
        ft_label = f"Fine-tuned [{Path(adapter_path).name}]"

    # ----- Print results -----
    base_label = f"Base [{Path(base_model_path).name}]"

    if ft_metrics:
        print_comparison(base_metrics, ft_metrics, base_label, ft_label)
    else:
        print(f"\nBase model metrics: {base_metrics}")

    return base_metrics, ft_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned model on Spider.")
    parser.add_argument("--base-model",   required=True, help="HF model ID or local path for base model.")
    parser.add_argument("--adapter-path", default=None,  help="Path to LoRA adapter (optional).")
    parser.add_argument("--spider-path",  default="data/spider", help="Root of Spider dataset.")
    parser.add_argument("--n-examples",   type=int, default=200, help="Number of dev examples to evaluate.")
    parser.add_argument("--device",       default="auto", help="Torch device map: auto / cuda / cpu.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        base_model_path=args.base_model,
        adapter_path=args.adapter_path,
        spider_path=args.spider_path,
        n_examples=args.n_examples,
        device=args.device,
    )
