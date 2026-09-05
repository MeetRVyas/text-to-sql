"""
tests/conftest.py
------------------
Shared fixtures. Also makes api/main.py importable (and therefore testable
via FastAPI's TestClient) without requiring the full torch/transformers/peft
stack to be installed.

api/main.py imports torch/peft/transformers at module level (it needs them
for real operation), but nothing at *import* time actually calls into them —
they're only used inside load_model()/lifespan(), which tests/test_api.py
monkeypatches anyway. So: if the real packages are installed, we use them as
normal; if not (e.g. a lightweight CI box without a multi-GB CUDA install),
we stand in a MagicMock so the import succeeds. Tests that need genuine
torch/transformers behaviour (e.g. tests/test_training_smoke.py) explicitly
`pytest.importorskip(...)` instead of relying on this stub.
"""

import importlib
import sys
from unittest.mock import MagicMock

import pytest


def _ensure_importable(name: str) -> None:
    try:
        importlib.import_module(name)
    except ImportError:
        sys.modules[name] = MagicMock()


for _mod in ("torch", "peft", "transformers"):
    _ensure_importable(_mod)


@pytest.fixture()
def sample_db_path(tmp_path):
    """Build the bundled concert_singer sample DB in a temp directory."""
    from sample_db.create_sample_db import create_sample_db

    return create_sample_db(output_dir=str(tmp_path / "concert_singer"))
