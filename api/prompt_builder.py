"""
api/prompt_builder.py
---------------------
Assembles the structured prompt fed to the fine-tuned model at inference time.

The format is identical to the training format (minus the gold SQL suffix),
so the model recognises the schema-injection convention it learned during
fine-tuning and completes with a valid SQL statement.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Template (must match training/prepare_dataset.py PROMPT_TEMPLATE exactly)
# ---------------------------------------------------------------------------

INFERENCE_TEMPLATE = """\
### Schema:
{schema}

### Question:
{question}

### SQL:
"""


def build_inference_prompt(schema: str, question: str) -> str:
    """
    Build a prompt for inference (no gold SQL — the model completes it).

    Args:
        schema:   Schema string produced by schema_extractor.get_schema().
                  E.g.  "concerts(concert_id, theme, stadium_id)\n
                          stadiums(stadium_id, name, location)"
        question: Natural language question.
                  E.g.  "How many concerts are there?"

    Returns:
        Formatted prompt string ready to be tokenised.
    """
    if not schema.strip():
        raise ValueError("schema must not be empty — the model needs it to be schema-aware.")
    if not question.strip():
        raise ValueError("question must not be empty.")

    return INFERENCE_TEMPLATE.format(
        schema=schema.strip(),
        question=question.strip(),
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
        # Fallback: look for the ### SQL: marker
        marker = "### SQL:"
        idx = full_output.rfind(marker)
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
