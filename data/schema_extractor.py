"""
data/schema_extractor.py
------------------------
Extracts table-and-column schema from:
  - SQLite .db files (live introspection)
  - Spider tables.json (pre-built index)

The output is a compact human-readable string injected into every prompt,
e.g.  "concerts(concert_id, theme, stadium_id)\nstadiums(stadium_id, name)"
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# From a live SQLite database file
# ---------------------------------------------------------------------------

def schema_from_db(db_path: str) -> str:
    """
    Connect to a SQLite file and return a schema string by introspecting
    sqlite_master / PRAGMA table_info.

    Args:
        db_path: Path to a .db file.

    Returns:
        Multi-line string, one table per line:
            table_name(col1, col2, ...)
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()

    # Fetch all user tables (exclude sqlite internal tables)
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    tables = [row[0] for row in cursor.fetchall()]

    lines: List[str] = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info('{table}')")
        columns = [row[1] for row in cursor.fetchall()]
        lines.append(f"{table}({', '.join(columns)})")

    conn.close()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# From Spider's tables.json index
# ---------------------------------------------------------------------------

def load_spider_tables(tables_json_path: str) -> Dict[str, Dict]:
    """
    Parse the Spider tables.json file into a dict keyed by db_id.

    Returns:
        {
          "concert_singer": {
            "table_names": ["stadium", "singer", "concert", "singer_in_concert"],
            "column_names": [[0, "stadium_id"], [0, "location"], ...]
          },
          ...
        }
    """
    with open(tables_json_path, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    index: Dict[str, Dict] = {}
    for entry in tables_data:
        db_id = entry["db_id"]
        index[db_id] = {
            "table_names": entry["table_names_original"],
            "column_names": entry["column_names_original"],  # [[table_idx, col_name], ...]
        }
    return index


def schema_from_spider_index(db_id: str, spider_index: Dict[str, Dict]) -> str:
    """
    Build a schema string for one Spider database using the pre-loaded index.

    Args:
        db_id: Spider database identifier, e.g. "concert_singer".
        spider_index: Output of :func:`load_spider_tables`.

    Returns:
        Schema string suitable for prompt injection.
    """
    if db_id not in spider_index:
        raise KeyError(f"db_id '{db_id}' not found in Spider tables index.")

    entry = spider_index[db_id]
    table_names: List[str] = entry["table_names"]
    column_names: List[List] = entry["column_names"]

    # Group columns by table index (column_names[i][0] is the table index, -1 = wildcard)
    table_columns: Dict[int, List[str]] = {i: [] for i in range(len(table_names))}
    for table_idx, col_name in column_names:
        if table_idx >= 0:  # skip the wildcard * entry (table_idx == -1)
            table_columns[table_idx].append(col_name)

    lines: List[str] = []
    for idx, tname in enumerate(table_names):
        cols = table_columns.get(idx, [])
        lines.append(f"{tname}({', '.join(cols)})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def get_schema(
    db_id: str,
    db_dir: Optional[str] = None,
    spider_index: Optional[Dict[str, Dict]] = None,
) -> str:
    """
    Return a schema string, preferring the live DB file when available,
    falling back to the Spider index.

    Args:
        db_id:         Spider database identifier.
        db_dir:        Root directory containing per-db subdirectories.
        spider_index:  Pre-loaded Spider tables index (fallback).
    """
    if db_dir is not None:
        db_path = Path(db_dir) / db_id / f"{db_id}.db"
        if db_path.exists():
            return schema_from_db(str(db_path))

    if spider_index is not None:
        return schema_from_spider_index(db_id, spider_index)

    raise ValueError(
        f"Cannot resolve schema for '{db_id}': "
        "provide either db_dir (pointing to Spider database/ folder) or spider_index."
    )


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print schema for a Spider database.")
    parser.add_argument("--db-id", required=True, help="Spider db_id, e.g. concert_singer")
    parser.add_argument("--db-dir", default="data/spider/database")
    parser.add_argument("--tables-json", default="data/spider/tables.json")
    args = parser.parse_args()

    index = load_spider_tables(args.tables_json)
    schema = get_schema(args.db_id, db_dir=args.db_dir, spider_index=index)
    print(schema)
