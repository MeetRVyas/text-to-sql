"""
data/prepare_dataset.py
-----------------------
Loads the Spider dataset and resolves every (question, SQL) pair to its
schema, producing plain (schema, question, sql, db_id) records.

This module is deliberately format-agnostic — it does not render a prompt
or chat template. That's left to the caller, via data/prompt_format.py:
  - training/train.py builds `messages` for SFTTrainer's conversational /
    assistant-only-loss training path.
  - evaluation/evaluate.py and api/main.py build an inference prompt via
    api/prompt_builder.py.
Both paths go through the same data/prompt_format.build_messages(), so
training and inference can't drift apart the way they previously did.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict

from data.schema_extractor import get_schema, load_spider_tables


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
    Convert raw Spider examples into (schema, question, sql, db_id) dicts.

    Each record has:
        sql      – gold SQL (training target / evaluation reference)
        question – natural language question
        schema   – resolved schema string
        db_id    – source database identifier
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
                "sql": sql,
                "question": question,
                "schema": schema,
                "db_id": db_id,
            }
        )
    return records


_RECORD_COLUMNS = ("sql", "question", "schema", "db_id")


def _records_to_dataset(records: List[Dict]) -> Dataset:
    """
    Dataset.from_list() infers columns from the first row, so it can't
    build a dataset from an empty list without an explicit schema (this is
    what evaluation.evaluate hits when it asks for 0 training examples).
    Guard that case explicitly instead of letting it fail deep inside
    `datasets`.
    """
    if not records:
        return Dataset.from_dict({col: [] for col in _RECORD_COLUMNS})
    return Dataset.from_list(records)


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

    print(f"Formatted {len(train_records)} train / {len(eval_records)} eval examples.")

    return DatasetDict(
        {
            "train": _records_to_dataset(train_records),
            "validation": _records_to_dataset(eval_records),
        }
    )


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
    print("\n=== Sample training record ===")
    print(ds["train"][0])
    print("\n=== As chat messages (see data/prompt_format.py) ===")
    from data.prompt_format import build_messages

    example = ds["train"][0]
    for msg in build_messages(example["schema"], example["question"], example["sql"]):
        print(f"[{msg['role']}] {msg['content']}")
