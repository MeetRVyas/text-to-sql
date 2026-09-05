"""
Tests for data/prompt_format.py and api/prompt_builder.py.

api/prompt_builder.py's build_inference_prompt() takes a tokenizer and calls
its .apply_chat_template(). We use a small fake tokenizer here rather than a
real one — it's the training/inference format-matching *logic* under test,
not any specific chat template's exact syntax (that's the fine-tuned model's
own chat template, applied identically at train and inference time; validating
its exact rendering needs the real Qwen tokenizer, which requires network
access this environment doesn't have).
"""

import pytest

from api.prompt_builder import extract_sql_from_output, validate_schema_string, build_inference_prompt
from data.prompt_format import SYSTEM_PROMPT, build_messages


class FakeTokenizer:
    """Minimal stand-in for a HF tokenizer's apply_chat_template."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        rendered = "\n".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            rendered += "\n<assistant>"
        return rendered


# ---------------------------------------------------------------------------
# data/prompt_format.build_messages
# ---------------------------------------------------------------------------

def test_build_messages_without_sql_omits_assistant_turn():
    messages = build_messages("singer(id, name)", "How many singers?")

    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "singer(id, name)" in messages[1]["content"]
    assert "How many singers?" in messages[1]["content"]


def test_build_messages_with_sql_includes_assistant_turn():
    messages = build_messages("singer(id, name)", "How many singers?", sql="SELECT COUNT(*) FROM singer")

    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant"]
    assert messages[2]["content"] == "SELECT COUNT(*) FROM singer"


def test_training_and_inference_share_identical_system_and_user_turns():
    """
    This is the regression test for the original bug: training built its own
    prompt, inference built a differently-shaped one. Both now go through
    build_messages(), so the system/user turns for a given (schema, question)
    are byte-identical whether or not a gold SQL is supplied.
    """
    schema, question = "singer(id, name)", "How many singers?"

    training_messages = build_messages(schema, question, sql="SELECT COUNT(*) FROM singer")
    inference_messages = build_messages(schema, question)

    assert training_messages[:2] == inference_messages


# ---------------------------------------------------------------------------
# api/prompt_builder.build_inference_prompt
# ---------------------------------------------------------------------------

def test_build_inference_prompt_renders_via_chat_template():
    prompt = build_inference_prompt("singer(id, name)", "How many singers?", FakeTokenizer())

    assert prompt.startswith(f"<system>{SYSTEM_PROMPT}</system>")
    assert "<user>### Schema:\nsinger(id, name)\n\n### Question:\nHow many singers?</user>" in prompt
    assert prompt.endswith("<assistant>")  # add_generation_prompt=True


def test_build_inference_prompt_rejects_empty_schema():
    with pytest.raises(ValueError):
        build_inference_prompt("", "How many singers?", FakeTokenizer())


def test_build_inference_prompt_rejects_empty_question():
    with pytest.raises(ValueError):
        build_inference_prompt("singer(id, name)", "  ", FakeTokenizer())


# ---------------------------------------------------------------------------
# api/prompt_builder.extract_sql_from_output
# ---------------------------------------------------------------------------

def test_extract_sql_strips_prompt_prefix():
    prompt = "<system>...</system>\n<user>...</user>\n<assistant>"
    full_output = prompt + "SELECT COUNT(*) FROM singer"

    assert extract_sql_from_output(full_output, prompt) == "SELECT COUNT(*) FROM singer"


def test_extract_sql_stops_at_first_blank_line():
    prompt = "<assistant>"
    full_output = prompt + "SELECT COUNT(*) FROM singer\n\nThis query counts all singers."

    assert extract_sql_from_output(full_output, prompt) == "SELECT COUNT(*) FROM singer"


def test_extract_sql_falls_back_to_prompts_last_line_when_not_echoed():
    # Simulates a backend that doesn't echo the prompt byte-for-byte.
    prompt = "<system>...</system>\n<user>...</user>\n<assistant>"
    full_output = "some backend preamble\n<assistant>SELECT 1"

    assert extract_sql_from_output(full_output, prompt) == "SELECT 1"


def test_extract_sql_returns_full_output_when_no_marker_found():
    assert extract_sql_from_output("SELECT 1", "") == "SELECT 1"


# ---------------------------------------------------------------------------
# api/prompt_builder.validate_schema_string
# ---------------------------------------------------------------------------

def test_validate_schema_string_accepts_well_formed_schema():
    assert validate_schema_string("singer(id, name)\nconcert(id, singer_id)") is True


def test_validate_schema_string_rejects_garbage():
    assert validate_schema_string("this is not a schema at all") is False
