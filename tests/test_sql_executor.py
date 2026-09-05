"""Tests for api/sql_executor.py — the SELECT-only safety guard and
DatabaseResolver. This is the most security-sensitive code in the project
(it's what stands between an LLM's generated SQL and a real database), so
it gets the most thorough coverage.
"""

import pytest

from api.sql_executor import DatabaseResolver, execute_sql


# ---------------------------------------------------------------------------
# execute_sql — happy path
# ---------------------------------------------------------------------------

def test_select_returns_rows_and_columns(sample_db_path):
    result = execute_sql("SELECT name, country FROM singer ORDER BY singer_id", sample_db_path)

    assert result["error"] is None
    assert result["columns"] == ["name", "country"]
    assert result["row_count"] == 3
    assert result["rows"][0] == ("Joe Sharp", "Netherlands")
    assert result["truncated"] is False


def test_select_with_no_rows(sample_db_path):
    result = execute_sql("SELECT * FROM singer WHERE singer_id = 9999", sample_db_path)

    assert result["error"] is None
    assert result["rows"] == []
    assert result["row_count"] == 0


def test_row_limit_truncates_and_reports_true_count(sample_db_path):
    result = execute_sql("SELECT * FROM singer_in_concert", sample_db_path, row_limit=2)

    assert result["error"] is None
    assert len(result["rows"]) == 2
    assert result["row_count"] == 4  # true total, even though rows is capped
    assert result["truncated"] is True


# ---------------------------------------------------------------------------
# execute_sql — safety guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO singer VALUES (99, 'x', 'x', 'x', '2020', 20, 'M')",
        "UPDATE singer SET name = 'hacked' WHERE singer_id = 1",
        "DELETE FROM singer WHERE singer_id = 1",
        "DROP TABLE singer",
        "CREATE TABLE evil (id INTEGER)",
        "ALTER TABLE singer ADD COLUMN hacked TEXT",
        "REPLACE INTO singer VALUES (1, 'x', 'x', 'x', '2020', 20, 'M')",
        "ATTACH DATABASE 'other.db' AS other",
        "SELECT * FROM singer; DROP TABLE singer",
    ],
)
def test_mutating_statements_are_rejected(sample_db_path, sql):
    result = execute_sql(sql, sample_db_path)

    assert result["error"] is not None
    assert result["rows"] == []

    # And confirm nothing actually happened to the underlying data.
    check = execute_sql("SELECT COUNT(*) FROM singer", sample_db_path)
    assert check["rows"] == [(3,)]


def test_non_select_statement_without_disallowed_keyword_is_rejected(sample_db_path):
    # No SELECT keyword at all -> caught by the "is this even a SELECT" check.
    result = execute_sql("PRAGMA table_info(singer)", sample_db_path)

    assert result["error"] is not None
    assert "does not appear to be a SELECT" in result["error"]


def test_read_only_connection_rejects_write_even_if_regex_missed_it(sample_db_path, monkeypatch):
    # Belt-and-suspenders: even if a mutating statement somehow slipped past
    # the keyword guard, the connection itself is opened read-only.
    import re

    import api.sql_executor as sql_executor

    monkeypatch.setattr(sql_executor, "_DISALLOWED", re.compile(r"$^"))  # never matches
    result = execute_sql("INSERT INTO singer VALUES (99,'x','x','x','2020',20,'M')", sample_db_path)

    assert result["error"] is not None  # sqlite3 itself refuses the write (read-only URI)


# ---------------------------------------------------------------------------
# execute_sql — error handling
# ---------------------------------------------------------------------------

def test_missing_db_file_reports_clear_error(tmp_path):
    result = execute_sql("SELECT 1", str(tmp_path / "does_not_exist.db"))

    assert result["error"] is not None
    assert "not found" in result["error"]


def test_invalid_sql_syntax_reports_sqlite_error(sample_db_path):
    result = execute_sql("SELECT FROM WHERE", sample_db_path)

    assert result["error"] is not None
    assert "SQLite error" in result["error"]


def test_nonexistent_table_reports_sqlite_error(sample_db_path):
    result = execute_sql("SELECT * FROM not_a_real_table", sample_db_path)

    assert result["error"] is not None


# ---------------------------------------------------------------------------
# DatabaseResolver
# ---------------------------------------------------------------------------

def test_resolver_finds_existing_db(tmp_path):
    from sample_db.create_sample_db import create_sample_db

    create_sample_db(output_dir=str(tmp_path / "concert_singer"))
    resolver = DatabaseResolver(str(tmp_path))

    path = resolver.get_db_path("concert_singer")
    assert path.endswith("concert_singer.db")


def test_resolver_raises_for_missing_db(tmp_path):
    resolver = DatabaseResolver(str(tmp_path))

    with pytest.raises(FileNotFoundError):
        resolver.get_db_path("does_not_exist")


def test_resolver_execute_returns_error_dict_for_missing_db(tmp_path):
    resolver = DatabaseResolver(str(tmp_path))

    result = resolver.execute("SELECT 1", "does_not_exist")
    assert result["error"] is not None
    assert result["rows"] == []


def test_resolver_execute_delegates_to_execute_sql(tmp_path):
    from sample_db.create_sample_db import create_sample_db

    create_sample_db(output_dir=str(tmp_path / "concert_singer"))
    resolver = DatabaseResolver(str(tmp_path))

    result = resolver.execute("SELECT COUNT(*) FROM singer", "concert_singer")
    assert result["error"] is None
    assert result["rows"] == [(3,)]
