"""Tests for data/schema_extractor.py."""

import json

import pytest

from data.schema_extractor import (
    get_schema,
    load_spider_tables,
    schema_from_db,
    schema_from_spider_index,
)


# ---------------------------------------------------------------------------
# schema_from_db — live SQLite introspection
# ---------------------------------------------------------------------------

def test_schema_from_db_lists_all_tables_and_columns(sample_db_path):
    schema = schema_from_db(sample_db_path)
    lines = schema.splitlines()

    assert len(lines) == 4  # stadium, singer, concert, singer_in_concert
    assert any(line.startswith("singer(") for line in lines)

    singer_line = next(line for line in lines if line.startswith("singer("))
    assert "singer_id" in singer_line
    assert "name" in singer_line
    assert "country" in singer_line


def test_schema_from_db_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        schema_from_db(str(tmp_path / "nope.db"))


# ---------------------------------------------------------------------------
# Spider tables.json index
# ---------------------------------------------------------------------------

@pytest.fixture()
def tables_json_path(tmp_path):
    payload = [
        {
            "db_id": "concert_singer",
            "table_names_original": ["stadium", "singer"],
            "column_names_original": [
                [-1, "*"],
                [0, "stadium_id"],
                [0, "name"],
                [1, "singer_id"],
                [1, "name"],
                [1, "country"],
            ],
        }
    ]
    path = tmp_path / "tables.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_load_spider_tables_indexes_by_db_id(tables_json_path):
    index = load_spider_tables(tables_json_path)

    assert set(index.keys()) == {"concert_singer"}
    assert index["concert_singer"]["table_names"] == ["stadium", "singer"]


def test_schema_from_spider_index_groups_columns_by_table(tables_json_path):
    index = load_spider_tables(tables_json_path)
    schema = schema_from_spider_index("concert_singer", index)
    lines = schema.splitlines()

    assert lines[0] == "stadium(stadium_id, name)"
    assert lines[1] == "singer(singer_id, name, country)"


def test_schema_from_spider_index_raises_for_unknown_db_id(tables_json_path):
    index = load_spider_tables(tables_json_path)

    with pytest.raises(KeyError):
        schema_from_spider_index("not_a_real_db", index)


# ---------------------------------------------------------------------------
# get_schema — dispatch logic (prefers live DB, falls back to index)
# ---------------------------------------------------------------------------

def test_get_schema_prefers_live_db_when_available(tmp_path, sample_db_path, tables_json_path):
    # db_dir/concert_singer/concert_singer.db must exist for the live path
    db_dir = tmp_path / "db_root"
    (db_dir / "concert_singer").mkdir(parents=True)
    import shutil

    shutil.copy(sample_db_path, db_dir / "concert_singer" / "concert_singer.db")

    index = load_spider_tables(tables_json_path)
    schema = get_schema("concert_singer", db_dir=str(db_dir), spider_index=index)

    # Live DB has 4 tables; the tables.json fixture only describes 2 —
    # getting all 4 back confirms the live path was used, not the index.
    assert len(schema.splitlines()) == 4


def test_get_schema_falls_back_to_index_when_no_live_db(tables_json_path):
    index = load_spider_tables(tables_json_path)
    schema = get_schema("concert_singer", db_dir="/does/not/exist", spider_index=index)

    assert schema.splitlines() == ["stadium(stadium_id, name)", "singer(singer_id, name, country)"]


def test_get_schema_raises_when_neither_source_available():
    with pytest.raises(ValueError):
        get_schema("concert_singer", db_dir=None, spider_index=None)
