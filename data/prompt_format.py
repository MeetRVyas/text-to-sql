"""
data/prompt_format.py
----------------------
Single source of truth for how a (schema, question, sql) example is turned
into chat messages.

Both training and inference build their model input from this module:
  - training/train.py       -> build_messages(schema, question, sql)
                                (includes the assistant turn; fed to
                                SFTTrainer's conversational + assistant-only-
                                loss path)
  - api/prompt_builder.py   -> build_messages(schema, question)
                                (assistant turn omitted; rendered with
                                add_generation_prompt=True at inference time)

Keeping this in one place is deliberate: the model is fine-tuned on the
chat-template-rendered form of these messages, so inference has to render
the *same* messages through the *same* chat template, or generation quality
degrades in ways that are easy to miss during development.
"""

from typing import Dict, List, Optional


SYSTEM_PROMPT = (
    "You generate valid SQLite SQL from database schemas "
    "and natural-language questions."
)


def build_user_message(schema: str, question: str) -> str:
    """Render the schema + question into the user turn's content."""
    return f"### Schema:\n{schema}\n\n### Question:\n{question}"


def build_messages(schema: str, question: str, sql: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Build the chat messages for one example.

    Args:
        schema:   Schema string, e.g. "concerts(id, theme)\nstadiums(id, name)".
        question: Natural language question.
        sql:      Gold SQL. Include for training (adds the assistant turn).
                  Omit (None) for inference — the caller renders the prompt
                  with add_generation_prompt=True and lets the model produce
                  this turn.

    Returns:
        List of {"role", "content"} message dicts.
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(schema, question)},
    ]
    if sql is not None:
        messages.append({"role": "assistant", "content": sql})
    return messages
