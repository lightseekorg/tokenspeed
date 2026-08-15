"""CPU-only checks for scheduler prefix-tail replay configuration plumbing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("tokenspeed_scheduler")

from tokenspeed.runtime.engine.scheduler_utils import (
    make_config,
    resolve_dspark_prefix_replay_tokens,
)


def _make_config(*, prefix_replay_tokens: int | None = None):
    kwargs = {}
    if prefix_replay_tokens is not None:
        kwargs["prefix_replay_tokens"] = prefix_replay_tokens
    return make_config(
        num_device_pages=32,
        max_scheduled_tokens=64,
        max_batch_size=8,
        prefix_granularity=2,
        num_host_pages=0,
        disable_l2_cache=True,
        enable_l3_storage=False,
        role="fused",
        **kwargs,
    )


@pytest.mark.parametrize("value", [0, 1, 4, 128])
def test_make_config_preserves_explicit_prefix_replay_tokens(value: int) -> None:
    assert _make_config(prefix_replay_tokens=value).prefix_replay_tokens == value


def test_make_config_defaults_prefix_replay_tokens_to_zero() -> None:
    assert _make_config().prefix_replay_tokens == 0


@pytest.mark.parametrize("value", [-1, 1 << 31])
def test_make_config_rejects_prefix_replay_tokens_outside_int32(value: int) -> None:
    with pytest.raises(ValueError, match="non-negative int32"):
        _make_config(prefix_replay_tokens=value)


def _resolve_replay(**overrides) -> int:
    kwargs = {
        "speculative_algorithm": "DSPARK",
        "enable_prefix_caching": True,
        "enable_kvstore": False,
        "disaggregation_mode": "null",
        "draft_model_path_use_base": True,
        "draft_model_config": SimpleNamespace(dspark_prefix_replay_tokens=128),
    }
    kwargs.update(overrides)
    return resolve_dspark_prefix_replay_tokens(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"enable_prefix_caching": False}, 0),
        ({"speculative_algorithm": None}, 0),
        ({"speculative_algorithm": "MTP"}, 0),
        ({"draft_model_config": SimpleNamespace(dspark_prefix_replay_tokens=96)}, 96),
        (
            {
                "draft_model_path_use_base": False,
                "draft_model_config": SimpleNamespace(),
            },
            0,
        ),
    ],
)
def test_resolve_dspark_prefix_replay_input_space(overrides, expected) -> None:
    assert _resolve_replay(**overrides) == expected


@pytest.mark.parametrize("value", [0, -1, 1 << 31])
def test_resolve_dspark_prefix_replay_rejects_invalid_capability(value: int) -> None:
    with pytest.raises(ValueError, match="positive int32"):
        _resolve_replay(
            draft_model_config=SimpleNamespace(dspark_prefix_replay_tokens=value)
        )


def test_resolve_dspark_prefix_replay_rejects_missing_draft_config() -> None:
    with pytest.raises(ValueError, match="resolved draft model"):
        _resolve_replay(draft_model_config=None)


def test_resolve_dspark_prefix_replay_keeps_same_checkpoint_fail_closed() -> None:
    with pytest.raises(ValueError, match="advertises captured-context replay"):
        _resolve_replay(draft_model_config=SimpleNamespace())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"enable_kvstore": True}, "does not support KVStore"),
        ({"disaggregation_mode": "prefill"}, "disaggregated serving"),
        ({"disaggregation_mode": "decode"}, "disaggregated serving"),
    ],
)
def test_resolve_dspark_prefix_replay_rejects_unsupported_cache_modes(
    overrides, message
) -> None:
    with pytest.raises(ValueError, match=message):
        _resolve_replay(**overrides)
