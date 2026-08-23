"""
api/sql_executor.py
-------------------
Executes a generated SQL string against a SQLite database and returns
both the raw SQL and the result rows.

Safety
------
Only SELECT statements are permitted — the executor raises immediately for
any DDL (CREATE, DROP, ALTER) or DML (INSERT, UPDATE, DELETE) statement
that might mutate the bundled sample databases.
"""

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ExecutionResult = Dict[str, Any]
# {
#   "sql":        str                      — cleaned SQL string
#   "columns":    List[str]               — column names from cursor description
#   "rows":       List[Tuple]             — result rows (up to ROW_LIMIT)
#   "row_count":  int                     — total rows returned
#   "truncated":  bool                    — True if rows were capped at ROW_LIMIT
#   "error":      Optional[str]           — None on success, error message on failure
# }

ROW_LIMIT = 100   # cap result set to avoid huge payloads


# ---------------------------------------------------------------------------
# SQL safety guard
# ---------------------------------------------------------------------------

_DISALLOWED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|REPLACE|TRUNCATE|ATTACH|DETACH)\b",
    re.IGNORECASE,
)


def _assert_select_only(sql: str) -> None:
    """Raise ValueError if the statement is not a plain SELECT."""
    if _DISALLOWED.search(sql):
        raise ValueError(
            "Only SELECT statements are permitted. "
            "Detected a potentially mutating keyword in: " + sql[:120]
        )
    if not re.search(r"\bSELECT\b", sql, re.IGNORECASE):
        raise ValueError(f"Statement does not appear to be a SELECT query: {sql[:120]}")


# ---------------------------------------------------------------------------
# Core executor
# ---------------------------------------------------------------------------

def execute_sql(
    sql: str,
    db_path: str,
    row_limit: int = ROW_LIMIT,
) -> ExecutionResult:
    """
    Run *sql* against the SQLite file at *db_path*.

    Args:
        sql:       SQL string to execute (must be SELECT).
        db_path:   Path to the SQLite .db file.
        row_limit: Maximum number of rows to return.

    Returns:
        ExecutionResult dict.
    """
    result: ExecutionResult = {
        "sql": sql.strip(),
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "error": None,
    }

    try:
        _assert_select_only(sql)
    except ValueError as exc:
        result["error"] = str(exc)
        return result

    db_file = Path(db_path)
    if not db_file.exists():
        result["error"] = f"Database file not found: {db_path}"
        return result

    try:
        # Open in read-only URI mode for extra safety
        uri = f"file:{db_file.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()
        cursor.execute(sql)

        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        all_rows = cursor.fetchall()
        conn.close()

        truncated = len(all_rows) > row_limit
        rows = [tuple(r) for r in all_rows[:row_limit]]

        result["columns"] = columns
        result["rows"] = rows
        result["row_count"] = len(all_rows)
        result["truncated"] = truncated

    except sqlite3.Error as exc:
        result["error"] = f"SQLite error: {exc}"

    return result


# ---------------------------------------------------------------------------
# Multi-database resolver
# ---------------------------------------------------------------------------

class DatabaseResolver:
    """
    Resolves a db_id to a .db file path within the Spider database directory.

    Example layout:
        data/spider/database/
            concert_singer/
                concert_singer.db
            car_1/
                car_1.db
    """

    def __init__(self, db_root: str):
        self.db_root = Path(db_root)

    def get_db_path(self, db_id: str) -> str:
        candidate = self.db_root / db_id / f"{db_id}.db"
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(
            f"No .db file found for db_id='{db_id}' under {self.db_root}. "
            f"Expected: {candidate}"
        )

    def execute(self, sql: str, db_id: str, row_limit: int = ROW_LIMIT) -> ExecutionResult:
        try:
            db_path = self.get_db_path(db_id)
        except FileNotFoundError as exc:
            return {
                "sql": sql,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
                "error": str(exc),
            }
        return execute_sql(sql, db_path, row_limit=row_limit)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Run a SQL query against a Spider .db file.")
    parser.add_argument("--db-path", required=True, help="Path to .db file")
    parser.add_argument("--sql", required=True, help="SQL SELECT query to execute")
    args = parser.parse_args()

    res = execute_sql(args.sql, args.db_path)
    print(json.dumps(res, indent=2, default=str))
