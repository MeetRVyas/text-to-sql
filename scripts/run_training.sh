#!/usr/bin/env bash
# scripts/run_training.sh
# -----------------------
# Convenience wrapper for launching QLoRA fine-tuning.
#
# Adjust MODEL, EPOCHS, and BATCH_SIZE for your GPU.
# Tested on: Kaggle T4 (16 GB) with Qwen2.5-Coder-3B-Instruct @ batch=4, grad_accum=4.
#
# Usage:
#   bash scripts/run_training.sh                      # default (Qwen2.5-Coder-3B-Instruct)
#   bash scripts/run_training.sh --model mistral7b    # Mistral 7B (comparison run)
#   bash scripts/run_training.sh --merge              # merge adapter after training

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL="Qwen/Qwen2.5-Coder-3B-Instruct"
EPOCHS=3
BATCH_SIZE=4
OUTPUT_DIR="checkpoints/qlora-text2sql"
SPIDER_PATH="data/spider"
MERGE_FLAG=""

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            case "$2" in
                qwen3b|qwen)   MODEL="Qwen/Qwen2.5-Coder-3B-Instruct" ;;
                llama3b|llama)   MODEL="meta-llama/Llama-3.2-3B" ;;
                mistral7b|mistral) MODEL="mistralai/Mistral-7B-v0.1" ;;
                *)         MODEL="$2" ;;
            esac
            shift 2 ;;
        --epochs)      EPOCHS="$2";      shift 2 ;;
        --batch-size)  BATCH_SIZE="$2";  shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --spider-path) SPIDER_PATH="$2"; shift 2 ;;
        --merge)       MERGE_FLAG="--merge"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Pre-flight checks ──────────────────────────────────────────────────────────
echo "=================================================="
echo " QLoRA Text-to-SQL fine-tuning"
echo "=================================================="
echo "  Model:      $MODEL"
echo "  Epochs:     $EPOCHS"
echo "  Batch size: $BATCH_SIZE (per device)"
echo "  Output:     $OUTPUT_DIR"
echo "  Spider:     $SPIDER_PATH"
echo "=================================================="
echo ""

if [ ! -d "$SPIDER_PATH" ]; then
    echo "ERROR: Spider dataset not found at $SPIDER_PATH"
    echo "Run:  bash scripts/download_spider.sh"
    exit 1
fi

if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    echo "WARNING: CUDA not detected — training will run on CPU (very slow)."
    read -rp "Continue anyway? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── Launch training ────────────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Starting training …"

python -m training.train \
    --model      "$MODEL" \
    --epochs     "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --spider-path "$SPIDER_PATH" \
    $MERGE_FLAG

echo ""
echo "[$(date '+%H:%M:%S')] Training complete."
echo "  Adapter saved to: $OUTPUT_DIR/final-adapter"

if [ -n "$MERGE_FLAG" ]; then
    echo "  Merged model saved to: $OUTPUT_DIR/merged-model"
fi

echo ""
echo "To start the API server:"
echo "  MODEL_PATH=$OUTPUT_DIR/final-adapter uvicorn api.main:app --host 0.0.0.0 --port 8000"
