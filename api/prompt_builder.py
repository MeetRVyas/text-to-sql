"""
api/prompt_builder.py
---------------------
Assembles the structured prompt fed to the fine-tuned model at inference time.

Uses the same chat messages (data/prompt_format.build_messages) and the
same tokenizer chat template that training renders the model's input with.
The model was fine-tuned on chat-formatted turns via
`tokenizer.apply_chat_template(...)` — sending it a differently-shaped
prompt at inference time (e.g. a flat, non-chat string) is a format the
model never saw during training and degrades generation quality. Routing
both training and inference through the same `build_messages()` +
`apply_chat_template()` call keeps them in sync by construction.
"""

from typing import Any

from data.prompt_format import build_messages


def build_inference_prompt(schema: str, question: str, tokenizer: Any) -> str:
    """
    Build a prompt for inference (no gold SQL — the model completes it).

    Args:
        schema:    Schema string produced by schema_extractor.get_schema().
                   E.g.  "concerts(concert_id, theme, stadium_id)\n
                           stadiums(stadium_id, name, location)"
        question:  Natural language question.
                   E.g.  "How many concerts are there?"
        tokenizer: The model's tokenizer (must expose apply_chat_template —
                   this is the exact tokenizer/chat template training used).

    Returns:
        Formatted prompt string ready to be tokenised, ending with the
        chat template's generation prompt (e.g. Qwen's
        "<|im_start|>assistant\n") so the model continues as the assistant.
    """
    schema = schema.strip()
    question = question.strip()

    if not schema:
        raise ValueError("schema must not be empty — the model needs it to be schema-aware.")
    if not question:
        raise ValueError("question must not be empty.")

    messages = build_messages(schema, question)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_sql_from_output(full_output: str, prompt: str) -> str:
    """
    Strip the prompt prefix from the model's full decoded output and return
    only the generated SQL.

    Args:
        full_output: Raw text decoded from model output tokens (prompt + SQL).
        prompt:      The exact prompt string passed to the model.

    Returns:
        Cleaned SQL string (leading/trailing whitespace removed).
    """
    # If the model echoes the prompt, strip it
    if full_output.startswith(prompt):
        sql = full_output[len(prompt):]
    else:
        # Fallback: some generation backends normalise whitespace and don't
        # echo the prompt byte-for-byte. The chat template's own generation
        # marker (e.g. Qwen's "<|im_start|>assistant\n") is always the last
        # line of `prompt` itself, so using that as the marker stays valid
        # regardless of which chat template the tokenizer uses.
        marker = prompt.strip().splitlines()[-1] if prompt.strip() else ""
        idx = full_output.rfind(marker) if marker else -1
        if idx != -1:
            sql = full_output[idx + len(marker):]
        else:
            sql = full_output  # return as-is if we can't find the marker

    # Stop at the first blank line (model sometimes generates explanation after SQL)
    sql_lines = []
    for line in sql.splitlines():
        if line.strip() == "" and sql_lines:
            break
        sql_lines.append(line)

    return "\n".join(sql_lines).strip()


# ---------------------------------------------------------------------------
# Prompt validation helper
# ---------------------------------------------------------------------------

def validate_schema_string(schema: str) -> bool:
    """
    Light sanity-check: each line should look like  table_name(col1, col2, ...).
    Returns True if at least one line matches, False otherwise.
    """
    import re

    pattern = re.compile(r"^\w+\(.*\)$")
    for line in schema.strip().splitlines():
        if pattern.match(line.strip()):
            return True
    return False
