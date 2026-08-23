"""
data/prepare_dataset.py
-----------------------
Loads the Spider dataset and converts every (question, SQL) pair into the
structured prompt format used during both training and inference:

    ### Schema:
    <schema string>

    ### Question:
    <natural language question>

    ### SQL:
    <target SQL>   ← appended only during training

The resulting HuggingFace Dataset is returned ready for SFTTrainer.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict

from data.schema_extractor import get_schema, load_spider_tables


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
### Schema:
{schema}

### Question:
{question}

### SQL:
{sql}\
"""

INFERENCE_TEMPLATE = """\
### Schema:
{schema}

### Question:
{question}

### SQL:
"""


def build_prompt(schema: str, question: str, sql: str = "") -> str:
    """
    Assemble a training prompt (with SQL) or an inference prompt (without SQL).

    Args:
        schema:   Schema string, e.g. "concerts(id, theme)\nstadiums(id, name)".
        question: Natural language question.
        sql:      Gold SQL answer. Pass empty string for inference.

    Returns:
        Formatted prompt string.
    """
    if sql:
        return PROMPT_TEMPLATE.format(schema=schema, question=question, sql=sql)
    return INFERENCE_TEMPLATE.format(schema=schema, question=question)


# ---------------------------------------------------------------------------
# Spider JSON loader
# ---------------------------------------------------------------------------

def _load_spider_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_records(
    examples: List[Dict],
    db_dir: Optional[str],
    spider_index: Optional[Dict],
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Convert raw Spider examples into prompt/label dicts.

    Each record has:
        text  – full training prompt (schema + question + SQL)
        sql   – gold SQL (for evaluation)
        db_id – source database identifier
    """
    records: List[Dict] = []
    if max_samples is not None:
        examples = examples[:max_samples]

    for ex in examples:
        db_id = ex["db_id"]
        question = ex["question"]
        sql = ex["query"]

        try:
            schema = get_schema(db_id, db_dir=db_dir, spider_index=spider_index)
        except Exception as e:
            # Skip examples whose schema cannot be resolved
            print(f"[WARN] Skipping db_id={db_id}: {e}")
            continue

        records.append(
            {
                "text": build_prompt(schema, question, sql),
                "sql": sql,
                "question": question,
                "schema": schema,
                "db_id": db_id,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_spider_dataset(
    spider_path: str = "data/spider",
    train_file: str = "train_spider.json",
    dev_file: str = "dev.json",
    tables_file: str = "tables.json",
    db_dir: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = 500,
) -> DatasetDict:
    """
    Load and format the Spider dataset from disk.

    Args:
        spider_path:        Root directory of the Spider download.
        train_file:         Filename of the training split JSON.
        dev_file:           Filename of the dev/eval split JSON.
        tables_file:        Filename of the tables index JSON.
        db_dir:             Path to the database/ subdirectory (optional;
                            used for live schema introspection).
        max_train_samples:  Cap on training examples (None = unlimited).
        max_eval_samples:   Cap on eval examples.

    Returns:
        DatasetDict with keys "train" and "validation".
    """
    root = Path(spider_path)

    tables_json = str(root / tables_file)
    spider_index = load_spider_tables(tables_json)

    resolved_db_dir = db_dir or str(root / "database")

    train_examples = _load_spider_json(str(root / train_file))
    dev_examples = _load_spider_json(str(root / dev_file))

    print(f"Loaded {len(train_examples)} train / {len(dev_examples)} dev examples.")

    train_records = _build_records(
        train_examples,
        db_dir=resolved_db_dir,
        spider_index=spider_index,
        max_samples=max_train_samples,
    )
    eval_records = _build_records(
        dev_examples,
        db_dir=resolved_db_dir,
        spider_index=spider_index,
        max_samples=max_eval_samples,
    )

    print(f"Formatted {len(train_records)} train / {len(eval_records)} eval prompts.")

    return DatasetDict(
        {
            "train": Dataset.from_list(train_records),
            "validation": Dataset.from_list(eval_records),
        }
    )


def load_from_hub(
    dataset_name: str = "xlangai/spider",
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = 500,
) -> DatasetDict:
    """
    Fallback: download Spider from HuggingFace Hub when a local copy is absent.
    The Hub version does not include .db files so schema is built from the
    embedded 'db_id', 'query', 'question' fields + a tables.json downloaded
    alongside the dataset.
    """
    from datasets import load_dataset

    raw = load_dataset(dataset_name)

    def _format(example):
        # Hub version stores db_id but not schema string; leave schema blank
        # so callers can inject it later via schema_extractor if they have the DBs.
        schema = example.get("schema", "")
        return {
            "text": build_prompt(schema, example["question"], example["query"]),
            "sql": example["query"],
            "question": example["question"],
            "schema": schema,
            "db_id": example["db_id"],
        }

    train = raw["train"].map(_format)
    validation = raw["validation"].map(_format)

    if max_train_samples:
        train = train.select(range(min(max_train_samples, len(train))))
    if max_eval_samples:
        validation = validation.select(range(min(max_eval_samples, len(validation))))

    return DatasetDict({"train": train, "validation": validation})


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--spider-path", default="data/spider")
    parser.add_argument("--max-train", type=int, default=5)
    parser.add_argument("--max-eval", type=int, default=2)
    args = parser.parse_args()

    ds = load_spider_dataset(
        spider_path=args.spider_path,
        max_train_samples=args.max_train,
        max_eval_samples=args.max_eval,
    )
    print("\n=== Sample training prompt ===")
    print(ds["train"][0]["text"])
