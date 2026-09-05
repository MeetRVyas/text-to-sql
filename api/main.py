"""
api/main.py
-----------
FastAPI service that exposes the fine-tuned Text-to-SQL model.

Endpoints
---------
POST /generate      – Generate SQL from a natural language question + schema string.
POST /query         – Generate SQL *and* execute it against a bundled SQLite DB.
GET  /health        – Liveness probe.
GET  /schema/{db_id} – Return the schema string for a known Spider database.

Startup
-------
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Environment variables
---------------------
    MODEL_PATH   – Path to LoRA adapter directory (default: checkpoints/qlora-text2sql/final-adapter)
    BASE_MODEL   – HF model ID of the base model (default: Qwen/Qwen2.5-Coder-3B-Instruct)
    DB_ROOT      – Root directory for Spider .db files (default: data/spider/database)
    TABLES_JSON  – Path to Spider tables.json (default: data/spider/tables.json)
    MAX_NEW_TOKENS – Max SQL tokens to generate (default: 256)
    DEVICE       – "cuda", "cpu", or "auto" (default: auto)
    API_KEY      – Optional shared secret. If set, /generate and /query require a
                    matching X-API-Key header. Unset by default (open, for local use).
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from peft import AutoPeftModelForCausalLM
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from api.prompt_builder import build_inference_prompt, extract_sql_from_output
from api.sql_executor import DatabaseResolver
from data.schema_extractor import get_schema, load_spider_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/qlora-text2sql/final-adapter")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Coder-3B-Instruct")
DB_ROOT = os.getenv("DB_ROOT", "data/spider/database")
TABLES_JSON = os.getenv("TABLES_JSON", "data/spider/tables.json")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
DEVICE = os.getenv("DEVICE", "auto")
API_KEY = os.getenv("API_KEY")  # optional; unset = no auth required (local/dev default)


# ---------------------------------------------------------------------------
# Global state (initialised at startup)
# ---------------------------------------------------------------------------

class AppState:
    model = None
    tokenizer = None
    generator = None
    resolver: Optional[DatabaseResolver] = None
    spider_index: Optional[Dict] = None


state = AppState()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load LoRA adapter (or fall back to bare base model for development)."""
    adapter_dir = Path(MODEL_PATH)

    device_map = DEVICE if DEVICE != "auto" else "auto"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if adapter_dir.exists():
        logger.info("Loading LoRA adapter from %s", adapter_dir)
        model = AutoPeftModelForCausalLM.from_pretrained(
            str(adapter_dir),
            device_map=device_map,
            dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    else:
        logger.info("Adapter not found at %s. Loading base model: %s", adapter_dir, BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map=device_map,
            dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Lifespan (replaces @app.on_event which is deprecated in recent FastAPI)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    logger.info("Loading model …")
    state.model, state.tokenizer = load_model()
    state.generator = pipeline(
        "text-generation",
        model=state.model,
        tokenizer=state.tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,     # greedy decoding for SQL (deterministic)
        temperature=None,
        top_p=None,
    )
    logger.info("Model ready.")

    # Spider schema index
    tables_path = Path(TABLES_JSON)
    if tables_path.exists():
        state.spider_index = load_spider_tables(str(tables_path))
        logger.info("Loaded Spider schema index (%d databases).", len(state.spider_index))

    # Database resolver
    db_root = Path(DB_ROOT)
    if db_root.exists():
        state.resolver = DatabaseResolver(str(db_root))
        logger.info("DatabaseResolver pointed at %s.", db_root)

    yield  # application runs here

    # ---- shutdown ----
    logger.info("Releasing model …")
    del state.model
    del state.generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Text-to-SQL API (QLoRA)",
    description="Generate SQL from natural language questions using a QLoRA fine-tuned model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """
    Optional shared-secret auth for the generation endpoints. No-op if
    API_KEY is unset (local/dev default). Set the API_KEY environment
    variable to require clients to send a matching X-API-Key header —
    worth turning on once this service is reachable over the network
    rather than purely local.
    """
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    question: str = Field(..., description="Natural language question.", examples=["How many concerts are there?"])
    schema: str = Field(
        ...,
        description=(
            "Schema string listing tables and their columns. "
            "Each line: table_name(col1, col2, ...). "
            "E.g.  'concerts(concert_id, theme, stadium_id)\\nstadiums(stadium_id, name)'"
        ),
    )
    max_new_tokens: Optional[int] = Field(None, description="Override the server default for max tokens.")


class GenerateResponse(BaseModel):
    sql: str
    prompt: str
    latency_ms: float


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question.")
    schema: Optional[str] = Field(
        None,
        description="Schema string. If omitted and db_id is provided, schema is auto-resolved.",
    )
    db_id: Optional[str] = Field(None, description="Spider database identifier, e.g. 'concert_singer'.")
    max_new_tokens: Optional[int] = None


class QueryResponse(BaseModel):
    sql: str
    columns: List[str]
    rows: List[Any]
    row_count: int
    truncated: bool
    error: Optional[str]
    latency_ms: float


class SchemaResponse(BaseModel):
    db_id: str
    schema: str


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def _generate_sql(question: str, schema: str, max_new_tokens: Optional[int] = None) -> tuple[str, str]:
    """Return (sql, prompt)."""
    prompt = build_inference_prompt(schema, question, state.tokenizer)
    tokens = int(max_new_tokens or MAX_NEW_TOKENS)

    outputs = state.generator(prompt, max_new_tokens=tokens)
    full_text = outputs[0]["generated_text"]
    sql = extract_sql_from_output(full_text, prompt)
    return sql, prompt


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health():
    """Liveness probe."""
    return {"status": "ok", "model_loaded": state.model is not None}


@app.get("/schema/{db_id}", response_model=SchemaResponse, tags=["Schema"])
def get_schema_endpoint(db_id: str):
    """Return the schema string for a known Spider database."""
    if state.spider_index is None:
        raise HTTPException(status_code=503, detail="Spider schema index not loaded.")
    try:
        schema = get_schema(db_id, db_dir=DB_ROOT, spider_index=state.spider_index)
        return SchemaResponse(db_id=db_id, schema=schema)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/generate", response_model=GenerateResponse, tags=["Inference"], dependencies=[Depends(require_api_key)])
def generate(req: GenerateRequest):
    """
    Generate a SQL query from a natural language question and schema string.
    Does **not** execute the SQL.
    """
    if state.generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()
    sql, prompt = _generate_sql(req.question, req.schema, req.max_new_tokens)
    latency_ms = (time.perf_counter() - t0) * 1000

    return GenerateResponse(sql=sql, prompt=prompt, latency_ms=round(latency_ms, 2))


@app.post("/query", response_model=QueryResponse, tags=["Inference"], dependencies=[Depends(require_api_key)])
def query(req: QueryRequest):
    """
    Generate SQL **and** execute it against a bundled SQLite database.
    Requires either `schema` (explicit) or `db_id` (auto-resolved from Spider index).
    """
    if state.generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Resolve schema
    schema = req.schema
    if not schema:
        if not req.db_id:
            raise HTTPException(
                status_code=422,
                detail="Provide either 'schema' or 'db_id'.",
            )
        if state.spider_index is None:
            raise HTTPException(status_code=503, detail="Spider schema index not loaded.")
        try:
            schema = get_schema(req.db_id, db_dir=DB_ROOT, spider_index=state.spider_index)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    t0 = time.perf_counter()
    sql, _ = _generate_sql(req.question, schema, req.max_new_tokens)

    # Execute
    exec_result = {"columns": [], "rows": [], "row_count": 0, "truncated": False, "error": None}
    if req.db_id and state.resolver:
        exec_result = state.resolver.execute(sql, req.db_id)
    elif req.db_id:
        exec_result["error"] = "DatabaseResolver not available — DB_ROOT not found."

    latency_ms = (time.perf_counter() - t0) * 1000

    return QueryResponse(
        sql=sql,
        columns=exec_result["columns"],
        rows=exec_result["rows"],
        row_count=exec_result["row_count"],
        truncated=exec_result["truncated"],
        error=exec_result["error"],
        latency_ms=round(latency_ms, 2),
    )
