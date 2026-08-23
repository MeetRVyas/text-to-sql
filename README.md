# Text-to-SQL via QLoRA

Fine-tune a compact LLM (Llama 3.2 3B or Mistral 7B) on the Spider benchmark
using QLoRA (4-bit quantisation + LoRA adapters), serve it behind a FastAPI
endpoint, and evaluate against Spider's two standard metrics.

```
Architecture overview
─────────────────────
Tier 1 — Data preparation
  Spider dataset (10k+ NL/SQL pairs · 200 DBs)  +  Schema context (table + col names)
                             │
Tier 2 — Training pipeline
  Base LLM (Mistral/Llama 3B)  →  QLoRA fine-tuning (4-bit + LoRA)  →  LoRA checkpoint
                             │
Tier 3 — API serving
  FastAPI endpoint  →  SQL generation (schema-aware prompt)  →  SQLite executor
                             │
Tier 4 — Evaluation
  Execution accuracy  ·  Exact match  ·  Base vs fine-tuned delta
```

---

## Project layout

```
text_to_sql_qlora/
├── requirements.txt
├── data/
│   ├── prepare_dataset.py      # Spider → HF Dataset with formatted prompts
│   └── schema_extractor.py     # Schema string from .db file or tables.json
├── training/
│   ├── config.py               # All hyperparameters in one place
│   └── train.py                # QLoRA fine-tuning entry-point (SFTTrainer)
├── api/
│   ├── main.py                 # FastAPI app  (POST /generate, POST /query)
│   ├── prompt_builder.py       # Prompt assembly & SQL extraction
│   └── sql_executor.py         # Safe SQLite execution, DatabaseResolver
├── evaluation/
│   ├── metrics.py              # execution_accuracy(), exact_match(), compute_metrics()
│   └── evaluate.py             # Base-vs-fine-tuned comparison table
├── sample_db/
│   └── create_sample_db.py     # Generates concert_singer.db for local testing
└── scripts/
    ├── download_spider.sh       # Downloads & extracts the Spider dataset
    └── run_training.sh          # Convenience wrapper for training.train
```

---

## Quick-start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **GPU requirement:** QLoRA at 3B scale needs ~6 GB VRAM; 7B needs ~12 GB.
> CPU-only training is possible but very slow.

### 2. Download Spider

```bash
bash scripts/download_spider.sh
```

This places the dataset at `data/spider/` with the layout expected by all
other scripts.  Alternatively, the training script will auto-download from
HuggingFace Hub (`xlangai/spider`) if the local path is absent.

### 3. (Optional) Create the sample database

Useful for local API testing without the full Spider download:

```bash
python sample_db/create_sample_db.py
```

### 4. Fine-tune

```bash
# Default: Llama 3.2 3B, 3 epochs
bash scripts/run_training.sh

# Mistral 7B, 5 epochs, merge adapter into full model afterwards
bash scripts/run_training.sh --model mistral7b --epochs 5 --merge

# Direct Python invocation with all overrides
python -m training.train \
    --model meta-llama/Llama-3.2-3B \
    --epochs 3 \
    --batch-size 4 \
    --output-dir checkpoints/qlora-text2sql \
    --spider-path data/spider
```

The LoRA adapter is saved to `checkpoints/qlora-text2sql/final-adapter/`
(~50–200 MB depending on rank). Pass `--merge` to additionally save a
merged stand-alone model.

### 5. Serve the API

```bash
MODEL_PATH=checkpoints/qlora-text2sql/final-adapter \
    uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Interactive docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

#### Generate SQL (without execution)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many concerts are there?",
    "schema": "concerts(concert_id, theme, stadium_id)\nstadiums(stadium_id, name, capacity)"
  }'
```

Response:
```json
{
  "sql": "SELECT COUNT(*) FROM concerts",
  "prompt": "### Schema:\n...",
  "latency_ms": 312.4
}
```

#### Generate SQL and execute it

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many singers are there?",
    "db_id": "concert_singer"
  }'
```

Response:
```json
{
  "sql": "SELECT COUNT(*) FROM singer",
  "columns": ["COUNT(*)"],
  "rows": [[3]],
  "row_count": 1,
  "truncated": false,
  "error": null,
  "latency_ms": 285.1
}
```

#### Get schema for a database

```bash
curl http://localhost:8000/schema/concert_singer
```

### 6. Evaluate

```bash
python -m evaluation.evaluate \
    --base-model  meta-llama/Llama-3.2-3B \
    --adapter-path checkpoints/qlora-text2sql/final-adapter \
    --spider-path data/spider \
    --n-examples 200
```

Sample output:

```
======================================================
  Text-to-SQL Evaluation — Spider Dev Set
======================================================
Model                   Exec Acc   Exact Match   Errors
────────────────────── ─────────── ────────────  ───────
Base [Llama-3.2-3B]       32.5%        18.0%       14
Fine-tuned [final-adapter] 61.0%       44.5%        6
────────────────────── ─────────── ────────────  ───────
Delta (fine-tune − base)  +28.5%      +26.5%       -8
======================================================
  n_total = 200
======================================================
```

---

## Key design decisions

| Decision | Rationale |
|---|---|
| Schema string passed at inference | Model generalises to *unseen* databases without retraining |
| Training prompts mirror inference format | Model learns schema-injection convention during fine-tuning |
| QLoRA over full fine-tune | 3–7 B scale on Spider: approaches full FT accuracy at a fraction of VRAM |
| Response-only loss | Back-prop only through the SQL portion — not the schema/question prefix |
| Execution accuracy as primary metric | Tolerates semantically-equivalent SQL; exact match catches hallucinated identifiers |

---

## Configuration reference (`training/config.py`)

| Class | Key params | Default |
|---|---|---|
| `ModelConfig` | `model_name` | `meta-llama/Llama-3.2-3B` |
| `ModelConfig` | `bnb_4bit_quant_type` | `nf4` |
| `LoRAConfig` | `r`, `lora_alpha` | `16`, `32` |
| `LoRAConfig` | `target_modules` | `["q_proj", "v_proj"]` |
| `LoRAConfig` | `lora_dropout` | `0.05` |
| `TrainingConfig` | `num_train_epochs` | `3` |
| `TrainingConfig` | `learning_rate` | `2e-4` |
| `TrainingConfig` | `gradient_accumulation_steps` | `4` |

---

## Environment variables (API)

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `checkpoints/qlora-text2sql/final-adapter` | LoRA adapter directory |
| `BASE_MODEL` | `meta-llama/Llama-3.2-3B` | Base model (fallback if adapter missing) |
| `DB_ROOT` | `data/spider/database` | Root for `.db` files |
| `TABLES_JSON` | `data/spider/tables.json` | Spider schema index |
| `MAX_NEW_TOKENS` | `256` | Max SQL tokens to generate |
| `DEVICE` | `auto` | Torch device map |

---

## Licence

This project is released for research and educational use.  
The Spider dataset is subject to Yale's [non-commercial research licence](https://yale-lily.github.io/spider).  
Model weights are subject to their respective licences (Meta, Mistral AI).
