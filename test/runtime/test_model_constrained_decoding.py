"""Model-owned constrained decoding policy tests."""

from types import SimpleNamespace

import pytest

from tokenspeed.runtime.configs.model_config import _apply_constrained_decoding_policy


class _DefaultConfig:
    pass


@pytest.mark.parametrize("policy", ["requested", "always"])
def test_model_config_class_declares_constrained_decoding(policy):
    config_cls = type("Config", (), {"constrained_decoding": policy})
    server_args = SimpleNamespace(constrained_decoding=None)

    _apply_constrained_decoding_policy(config_cls(), server_args, is_draft_worker=False)

    assert server_args.constrained_decoding == policy


def test_model_without_policy_defaults_to_none():
    server_args = SimpleNamespace(constrained_decoding="always")

    _apply_constrained_decoding_policy(
        _DefaultConfig(), server_args, is_draft_worker=False
    )

    assert server_args.constrained_decoding is None


def test_invalid_model_policy_is_rejected():
    class InvalidConfig:
        constrained_decoding = "sometimes"

    with pytest.raises(ValueError, match="must be None, 'requested', or 'always'"):
        _apply_constrained_decoding_policy(
            InvalidConfig(),
            SimpleNamespace(constrained_decoding=None),
            is_draft_worker=False,
        )


def test_draft_model_does_not_replace_target_policy():
    server_args = SimpleNamespace(constrained_decoding="always")

    _apply_constrained_decoding_policy(
        _DefaultConfig(), server_args, is_draft_worker=True
    )

    assert server_args.constrained_decoding == "always"
