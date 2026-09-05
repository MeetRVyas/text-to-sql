"""
Tests for api/main.py.

The model/tokenizer/pipeline are replaced with lightweight fakes (see
conftest.py for how torch/peft/transformers import even when not installed).
These tests exercise routing, request/response validation, schema
resolution, the SQL-safety path end-to-end through the API, and the
optional API-key auth — not real model inference (that needs the actual
Qwen weights, which requires network access this environment doesn't have).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.main as api_main


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = "\n".join(f"[{m['role']}] {m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "\n[assistant]"
        return rendered


def _fake_pipeline_factory(*args, **kwargs):
    """Stands in for transformers.pipeline(...) — returns a callable that
    echoes the prompt plus a canned SQL string, same shape as the real
    text-generation pipeline's output."""

    def _generate(prompt, max_new_tokens=None, **_):
        sql = _generate.next_sql
        return [{"generated_text": prompt + sql}]

    _generate.next_sql = "SELECT COUNT(*) FROM singer"
    return _generate


@pytest.fixture()
def client(monkeypatch, sample_db_path):
    """TestClient with load_model()/pipeline() faked out, and state pointed
    at a temp sample DB, so no real model or GPU is needed."""
    from api.sql_executor import DatabaseResolver

    monkeypatch.setattr(api_main, "load_model", lambda: (object(), FakeTokenizer()))
    monkeypatch.setattr(api_main, "pipeline", _fake_pipeline_factory)
    monkeypatch.setattr(api_main, "API_KEY", None)  # explicit: auth off unless a test says otherwise

    with TestClient(api_main.app) as c:
        db_root = Path(sample_db_path).parent.parent  # tmp_path/<db_id>/<db_id>.db -> tmp_path
        api_main.state.spider_index = {
            "concert_singer": {
                "table_names": ["singer"],
                "column_names": [[-1, "*"], [0, "singer_id"], [0, "name"], [0, "country"]],
            }
        }
        api_main.state.resolver = DatabaseResolver(str(db_root))
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_reports_model_loaded(client):
    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


# ---------------------------------------------------------------------------
# /schema/{db_id}
# ---------------------------------------------------------------------------

def test_schema_endpoint_returns_known_db(client):
    resp = client.get("/schema/concert_singer")

    assert resp.status_code == 200
    assert "singer(singer_id, name, country)" in resp.json()["schema"]


def test_schema_endpoint_404s_for_unknown_db(client):
    resp = client.get("/schema/not_a_real_db")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /generate
# ---------------------------------------------------------------------------

def test_generate_returns_sql_from_explicit_schema(client):
    resp = client.post(
        "/generate",
        json={"question": "How many singers?", "schema": "singer(singer_id, name)"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["sql"] == "SELECT COUNT(*) FROM singer"
    assert "How many singers?" in body["prompt"]


def test_generate_rejects_missing_schema(client):
    resp = client.post("/generate", json={"question": "How many singers?"})

    assert resp.status_code == 422  # pydantic validation — schema is required


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------

def test_query_resolves_schema_from_db_id_and_executes(client):
    resp = client.post("/query", json={"question": "How many singers?", "db_id": "concert_singer"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["sql"] == "SELECT COUNT(*) FROM singer"
    assert body["error"] is None
    assert body["rows"] == [[3]]
    assert body["row_count"] == 1


def test_query_requires_schema_or_db_id(client):
    resp = client.post("/query", json={"question": "How many singers?"})

    assert resp.status_code == 422


def test_query_404s_for_unknown_db_id(client):
    resp = client.post("/query", json={"question": "How many singers?", "db_id": "not_a_real_db"})

    assert resp.status_code == 404


def test_query_blocks_mutating_generated_sql(client, monkeypatch):
    """
    Defense in depth: even if the model generates a mutating statement,
    /query must not execute it — this exercises the full API -> sql_executor
    path, not just the sql_executor unit tests.
    """
    fake_generator = api_main.state.generator
    fake_generator.next_sql = "DROP TABLE singer"

    resp = client.post("/query", json={"question": "Drop the singer table", "db_id": "concert_singer"})

    assert resp.status_code == 200  # the HTTP call succeeds...
    body = resp.json()
    assert body["error"] is not None  # ...but the SQL was refused, not executed
    assert body["rows"] == []

    # And the table is still there (reset the fake generator to a safe query first).
    fake_generator.next_sql = "SELECT COUNT(*) FROM singer"
    verify = client.post("/query", json={"question": "count", "db_id": "concert_singer"})
    assert verify.json()["error"] is None


# ---------------------------------------------------------------------------
# Optional API-key auth
# ---------------------------------------------------------------------------

def test_generate_open_by_default(client):
    # API_KEY unset -> no auth required (the `client` fixture sets this explicitly)
    resp = client.post(
        "/generate",
        json={"question": "How many singers?", "schema": "singer(singer_id, name)"},
    )
    assert resp.status_code == 200


def test_generate_requires_api_key_when_configured(client, monkeypatch):
    monkeypatch.setattr(api_main, "API_KEY", "secret-key")

    no_key = client.post(
        "/generate",
        json={"question": "How many singers?", "schema": "singer(singer_id, name)"},
    )
    assert no_key.status_code == 401

    wrong_key = client.post(
        "/generate",
        json={"question": "How many singers?", "schema": "singer(singer_id, name)"},
        headers={"X-API-Key": "wrong"},
    )
    assert wrong_key.status_code == 401

    right_key = client.post(
        "/generate",
        json={"question": "How many singers?", "schema": "singer(singer_id, name)"},
        headers={"X-API-Key": "secret-key"},
    )
    assert right_key.status_code == 200


def test_health_never_requires_api_key(client, monkeypatch):
    monkeypatch.setattr(api_main, "API_KEY", "secret-key")

    resp = client.get("/health")
    assert resp.status_code == 200
