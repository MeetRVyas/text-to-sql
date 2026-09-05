"""
Smoke test for the training config wiring (training/train.py, training/config.py).

This is deliberately narrow: it builds the SFTConfig the same way train()
does and checks the fields that matter for the trl 1.10 migration
(assistant_only_loss, max_length, packing) actually took hold — it does
*not* construct a real model/SFTTrainer or run any training, since that
needs a real (even if tiny) model downloaded from the Hub, which this
environment has no network access for.

Needs the real torch/transformers/trl stack (not the lightweight stub
conftest.py installs for the API tests) because TrainingArguments'
__post_init__ does real device/config logic that a MagicMock can't
meaningfully fake — so this test is skipped wherever that stack isn't
installed, and is meant to actually run in the training environment
(Kaggle) as part of `pytest`.
"""

import pytest

torch = pytest.importorskip("torch", reason="training smoke test needs the real torch/trl stack")
pytest.importorskip("trl", reason="training smoke test needs the real torch/trl stack")


def _build_sft_config(use_cpu: bool):
    from trl import SFTConfig
    from training.config import Config

    cfg = Config()

    training_args = SFTConfig(
        output_dir="/tmp/smoke-test-output",
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        learning_rate=cfg.training.learning_rate,
        report_to="none",
        max_length=cfg.model.max_length,
        assistant_only_loss=True,
        packing=False,
        use_cpu=use_cpu,
    )

    return cfg, training_args


def _assert_sft_config_has_assistant_only_loss_wired_up(cfg, training_args):
    assert training_args.assistant_only_loss is True
    assert training_args.max_length == cfg.model.max_length
    assert training_args.packing is False


def test_sft_config_has_assistant_only_loss_wired_up_cpu():
    """Validate SFTConfig wiring when running on CPU."""
    if torch.cuda.is_available():
        pytest.skip("CPU-specific test: CUDA is available")

    _assert_sft_config_has_assistant_only_loss_wired_up(*_build_sft_config(use_cpu=True))


def test_sft_config_has_assistant_only_loss_wired_up_gpu():
    """Validate SFTConfig wiring when running with GPU."""
    if not torch.cuda.is_available():
        pytest.skip("GPU-specific test: CUDA is unavailable")

    _assert_sft_config_has_assistant_only_loss_wired_up(*_build_sft_config(use_cpu=False))


def test_format_qwen_example_builds_conversational_messages():
    from training.train import format_qwen_example

    example = {"schema": "singer(id, name)", "question": "How many singers?", "sql": "SELECT COUNT(*) FROM singer"}
    result = format_qwen_example(example)

    assert "messages" in result
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert result["messages"][-1]["content"] == "SELECT COUNT(*) FROM singer"


def test_make_bnb_config_reads_model_config():
    from training.config import ModelConfig
    from training.train import make_bnb_config

    cfg = ModelConfig(bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16")
    bnb_config = make_bnb_config(cfg)

    assert bnb_config.load_in_4bit is True
    assert bnb_config.bnb_4bit_quant_type == "nf4"
    assert bnb_config.bnb_4bit_compute_dtype == torch.float16
