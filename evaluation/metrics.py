"""
evaluation/metrics.py
----------------------
Implements the two evaluation metrics from the Spider benchmark:

1. Execution Accuracy (primary)
   The generated SQL is executed against the gold database.
   The prediction is CORRECT if the returned result set exactly matches
   the gold result set (order-insensitive, duplicate-sensitive).
   Tolerates semantically-equivalent but differently-phrased SQL.

2. Exact Match (EM)
   A stricter string-level comparison of predicted SQL vs gold SQL after
   normalisation (lowercase keywords, collapsed whitespace).
   Useful for catching hallucinated column/table names even when the
   result sets accidentally agree.
"""

import re
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_SQL_KEYWORDS = {
    "select", "from", "where", "group", "by", "order", "having",
    "join", "left", "right", "inner", "outer", "on", "as",
    "and", "or", "not", "in", "is", "null", "limit", "offset",
    "distinct", "count", "sum", "avg", "min", "max", "between",
    "like", "union", "all", "intersect", "except",
}


def normalize_sql(sql: str) -> str:
    """
    Normalise SQL for string comparison:
      - lowercase keywords
      - collapse whitespace to single spaces
      - strip leading/trailing whitespace and semicolons
    """
    sql = sql.strip().rstrip(";").strip()

    # Lowercase SQL keywords while preserving identifiers / string literals
    tokens = re.split(r"(\s+)", sql)
    normalised_tokens = []
    for tok in tokens:
        lower = tok.lower()
        if lower in _SQL_KEYWORDS:
            normalised_tokens.append(lower)
        else:
            normalised_tokens.append(tok)

    return re.sub(r"\s+", " ", "".join(normalised_tokens)).strip()


# ---------------------------------------------------------------------------
# SQL execution helper
# ---------------------------------------------------------------------------

def _run_sql(sql: str, db_path: str) -> Optional[List[Tuple]]:
    """
    Execute *sql* against *db_path*.
    Returns a sorted list of tuples, or None on error.
    """
    try:
        conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return sorted(rows)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def execution_accuracy(
    predicted_sql: str,
    gold_sql: str,
    db_path: str,
) -> bool:
    """
    Execute both SQLs against the same database and compare result sets.

    Returns True if result sets are identical (order-insensitive).
    Returns False on execution error for either SQL.
    """
    pred_rows = _run_sql(predicted_sql, db_path)
    gold_rows = _run_sql(gold_sql, db_path)

    if pred_rows is None or gold_rows is None:
        return False

    return pred_rows == gold_rows


def exact_match(predicted_sql: str, gold_sql: str) -> bool:
    """
    Normalised string comparison between predicted and gold SQL.
    Case-insensitive for keywords; preserves identifiers.
    """
    return normalize_sql(predicted_sql) == normalize_sql(gold_sql)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: List[str],
    gold_sqls: List[str],
    db_paths: List[str],
) -> dict:
    """
    Compute execution accuracy and exact match over a dataset split.

    Args:
        predictions:  List of generated SQL strings.
        gold_sqls:    List of gold SQL strings.
        db_paths:     List of .db file paths (one per example).

    Returns:
        {
          "execution_accuracy": float,   # 0.0 – 1.0
          "exact_match":        float,   # 0.0 – 1.0
          "n_total":            int,
          "n_exec_correct":     int,
          "n_em_correct":       int,
          "exec_errors":        int,     # examples where prediction failed to execute
        }
    """
    assert len(predictions) == len(gold_sqls) == len(db_paths), (
        "predictions, gold_sqls and db_paths must have the same length."
    )

    n_total = len(predictions)
    n_exec_correct = 0
    n_em_correct = 0
    exec_errors = 0

    for pred, gold, db_path in zip(predictions, gold_sqls, db_paths):
        # Execution accuracy
        pred_rows = _run_sql(pred, db_path)
        gold_rows = _run_sql(gold, db_path)

        if pred_rows is None:
            exec_errors += 1
        elif gold_rows is not None and pred_rows == gold_rows:
            n_exec_correct += 1

        # Exact match
        if exact_match(pred, gold):
            n_em_correct += 1

    return {
        "execution_accuracy": n_exec_correct / n_total if n_total else 0.0,
        "exact_match": n_em_correct / n_total if n_total else 0.0,
        "n_total": n_total,
        "n_exec_correct": n_exec_correct,
        "n_em_correct": n_em_correct,
        "exec_errors": exec_errors,
    }
