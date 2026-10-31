# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Unit tests for the smg orchestrator lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from tokenspeed.cli._argsplit import OrchestratorOpts, split_argv
from tokenspeed.cli.serve_smg import (
    _DEFAULT_SMG_DISABLE_FLAGS,
    INKLING_ATTENTION_BACKEND,
    INKLING_REASONING_PARSER,
    INKLING_TOOL_CALL_PARSER,
    KIMI_K3_REASONING_PARSER,
    KIMI_K3_TOOL_CALL_PARSER,
    MUSE_SPARK_GATEWAY_PARSER,
    MUSE_SPARK_GRAMMAR_BACKEND,
    MUSE_SPARK_REASONING_PARSER,
    MUSE_SPARK_TOOL_CALL_PARSER,
    EngineLayout,
    ServeConfigError,
    _add_metrics_port,
    _args_with_default_model_parsers,
    _declares_response_template,
    _engine_layout,
    _free_port_avoiding_ephemeral_range,
    _gateway_args_with_default_log_level,
    _gateway_args_with_default_policy,
    _gateway_args_with_default_port,
    _gateway_args_with_default_prometheus_port,
    _gateway_args_with_default_reasoning_parser,
    _gateway_args_with_defaults,
    _gateway_args_with_engine_identity,
    _gateway_args_with_smg_disable_defaults,
    _get_from_args,
    _is_inkling_model,
    _is_kimi_k3_model,
    _is_muse_spark_model,
    _prewarm_hf_tokenizer,
    _template_tokenizer,
    _zmq_worker_url,
    run_smg,
)


def _make_proc(returncode: int | None = None) -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = asyncio.StreamReader()
    proc.stderr = asyncio.StreamReader()
    proc.stdout.feed_eof()
    proc.stderr.feed_eof()
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=returncode if returncode is not None else 0)
    return proc


def test_gateway_args_default_port_is_8000():
    gateway_args = _gateway_args_with_default_port(["--model", "/tmp/x"])

    assert gateway_args == ["--model", "/tmp/x", "--port", "8000"]
    assert _get_from_args(gateway_args, "--port") == "8000"


def test_gateway_args_preserve_user_port():
    gateway_args = _gateway_args_with_default_port(["--port", "8413"])

    assert gateway_args == ["--port", "8413"]
    assert _get_from_args(gateway_args, "--port") == "8413"


def test_gateway_args_default_reasoning_parser_is_passthrough():
    gateway_args = _gateway_args_with_default_reasoning_parser(["--model", "/tmp/x"])

    assert gateway_args == ["--model", "/tmp/x", "--reasoning-parser", "passthrough"]


def test_gateway_args_preserve_user_reasoning_parser():
    gateway_args = _gateway_args_with_default_reasoning_parser(
        ["--reasoning-parser", "qwen3"]
    )

    assert gateway_args == ["--reasoning-parser", "qwen3"]


def test_gateway_args_default_policy_is_passthrough():
    gateway_args = _gateway_args_with_default_policy(["--model", "/tmp/x"])

    assert gateway_args == ["--model", "/tmp/x", "--policy", "passthrough"]


def test_gateway_args_preserve_user_policy():
    gateway_args = _gateway_args_with_default_policy(["--policy", "round_robin"])

    assert gateway_args == ["--policy", "round_robin"]


def test_gateway_args_defaults_include_port_and_reasoning_parser():
    gateway_args = _gateway_args_with_defaults(["--model", "/tmp/x"])

    # No --prometheus-port here: run_smg allocates it right before
    # spawn_gateway so the port cannot be lost during engine startup (see
    # test_run_smg_allocates_prometheus_port_at_gateway_spawn).
    assert gateway_args == [
        "--model",
        "/tmp/x",
        "--port",
        "8000",
        "--reasoning-parser",
        "passthrough",
        "--disable-circuit-breaker",
        "--disable-retries",
        "--disable-load-monitoring",
        "--policy",
        "passthrough",
        "--tokenizer-cache-enable-l0",
        "--tokenizer-cache-enable-l1",
        "--log-level",
        "warn",
    ]


def test_gateway_args_defaults_inject_passthrough_policy():
    """``ts serve`` fronts a single backend, so the default routing policy is
    ``passthrough`` (no load balancing / monitoring / KV-event subscription)."""
    gateway_args = _gateway_args_with_defaults(["--model", "/tmp/x"])

    assert "--policy" in gateway_args
    idx = gateway_args.index("--policy")
    assert gateway_args[idx + 1] == "passthrough"


def test_gateway_args_defaults_preserve_user_policy():
    """An explicit operator ``--policy`` is never overridden by the default."""
    gateway_args = _gateway_args_with_defaults(
        ["--model", "/tmp/x", "--policy", "round_robin"]
    )

    # Exactly one --policy (default not appended on top of the explicit value).
    assert gateway_args.count("--policy") == 1
    idx = gateway_args.index("--policy")
    assert gateway_args[idx + 1] == "round_robin"


def test_gateway_args_default_log_level_is_warn():
    gateway_args = _gateway_args_with_default_log_level(["--model", "/tmp/x"])

    assert gateway_args == ["--model", "/tmp/x", "--log-level", "warn"]


def test_gateway_args_preserve_user_log_level():
    gateway_args = _gateway_args_with_default_log_level(["--log-level", "debug"])

    assert gateway_args == ["--log-level", "debug"]


def test_gateway_args_default_prometheus_port_is_free_port():
    gateway_args = _gateway_args_with_default_prometheus_port(["--model", "/tmp/x"])

    # A fresh free port is allocated per launch (not a fixed default) so repeated
    # ``ts serve`` launches -- e.g. as an sglang/slime RL rollout backend -- do
    # not collide on a port left in TIME_WAIT by the previous run.
    assert gateway_args[:3] == ["--model", "/tmp/x", "--prometheus-port"]
    assert gateway_args[3].isdigit()
    assert 1 <= int(gateway_args[3]) <= 65535


def test_gateway_args_preserve_user_prometheus_port():
    gateway_args = _gateway_args_with_default_prometheus_port(
        ["--prometheus-port", "29000"]
    )

    assert gateway_args == ["--prometheus-port", "29000"]


def test_free_port_returns_first_sample_outside_ephemeral_range(monkeypatch):
    """No retry needed when the first sample already clears the range."""
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._ephemeral_port_range", lambda: (32768, 60999)
    )
    calls = iter([8413])
    monkeypatch.setattr("tokenspeed.cli.serve_smg.get_free_port", lambda: next(calls))

    assert _free_port_avoiding_ephemeral_range() == 8413


def test_free_port_retries_out_of_ephemeral_range(monkeypatch):
    """A sample landing inside the ephemeral range is resampled."""
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._ephemeral_port_range", lambda: (32768, 60999)
    )
    # First two samples land inside the ephemeral range; the third clears it.
    calls = iter([40000, 41000, 9000])
    monkeypatch.setattr("tokenspeed.cli.serve_smg.get_free_port", lambda: next(calls))

    assert _free_port_avoiding_ephemeral_range() == 9000


def test_free_port_gives_up_after_max_attempts(monkeypatch):
    """Never loops forever: after a bounded number of attempts, return
    whatever the last sample was rather than retrying indefinitely."""
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._ephemeral_port_range", lambda: (32768, 60999)
    )
    always_ephemeral = iter([40000, 41000, 42000, 43000, 44000, 45000, 46000])
    call_count = {"n": 0}

    def fake_get_free_port():
        call_count["n"] += 1
        return next(always_ephemeral)

    monkeypatch.setattr("tokenspeed.cli.serve_smg.get_free_port", fake_get_free_port)

    port = _free_port_avoiding_ephemeral_range()

    assert port == 44000  # 5th and last attempt
    assert call_count["n"] == 5


def test_free_port_skips_ephemeral_check_when_range_unknown(monkeypatch):
    """Non-Linux hosts (no /proc/sys/net/ipv4/ip_local_port_range) degrade
    to a plain get_free_port() call -- no retries, no crash."""
    monkeypatch.setattr("tokenspeed.cli.serve_smg._ephemeral_port_range", lambda: None)
    calls = iter([40000])
    monkeypatch.setattr("tokenspeed.cli.serve_smg.get_free_port", lambda: next(calls))

    assert _free_port_avoiding_ephemeral_range() == 40000


def test_smg_disable_flags_appended_when_absent():
    gateway_args = _gateway_args_with_smg_disable_defaults(["--model", "/tmp/x"])

    assert gateway_args == [
        "--model",
        "/tmp/x",
        "--disable-circuit-breaker",
        "--disable-retries",
        "--disable-load-monitoring",
    ]


def test_smg_disable_flags_not_duplicated():
    """Idempotent: passing the flag explicitly must not double it up for smg's argparse."""
    gateway_args = _gateway_args_with_smg_disable_defaults(
        ["--disable-circuit-breaker"]
    )

    assert gateway_args == [
        "--disable-circuit-breaker",
        "--disable-retries",
        "--disable-load-monitoring",
    ]


def test_smg_disable_defaults_turn_off_load_monitoring():
    gateway_args = _gateway_args_with_smg_disable_defaults(["--model", "/tmp/x"])

    assert gateway_args.count("--disable-load-monitoring") == 1


def test_smg_disable_flag_set_covers_single_worker_safety_defaults():
    assert _DEFAULT_SMG_DISABLE_FLAGS == (
        "--disable-circuit-breaker",
        "--disable-retries",
        "--disable-load-monitoring",
    )


def test_cache_aware_policy_keeps_load_monitoring():
    # cache_aware's imbalance/overload triggers feed on the worker load
    # monitor; injecting the disable flag would leave them silently dead.
    gateway_args = _gateway_args_with_smg_disable_defaults(["--policy", "cache_aware"])

    assert "--disable-load-monitoring" not in gateway_args
    assert "--disable-circuit-breaker" in gateway_args
    assert "--disable-retries" in gateway_args


def test_cache_aware_policy_equals_form_keeps_load_monitoring():
    gateway_args = _gateway_args_with_smg_disable_defaults(["--policy=cache_aware"])

    assert "--disable-load-monitoring" not in gateway_args


def test_operator_policy_is_last_wins():
    # clap/argparse are last-wins on duplicated flags; the cache_aware
    # detection must key off the value the gateway will actually use.
    from tokenspeed.cli.serve_smg import _operator_policy

    assert (
        _operator_policy(["--policy", "cache_aware", "--policy", "round_robin"])
        == "round_robin"
    )
    assert (
        _operator_policy(["--policy", "round_robin", "--policy=cache_aware"])
        == "cache_aware"
    )


def test_gateway_defaults_pipeline_for_cache_aware():
    # Pipeline-level guarantee: explicit cache_aware keeps load monitoring,
    # keeps the other disable defaults, does not duplicate --policy, and does
    # NOT auto-inject --dp-aware (wheel-pin lockstep is the operator's call).
    gateway_args = _gateway_args_with_defaults(
        ["--model", "/tmp/x", "--policy", "cache_aware"]
    )

    assert "--disable-load-monitoring" not in gateway_args
    assert "--disable-circuit-breaker" in gateway_args
    assert "--disable-retries" in gateway_args
    assert "--dp-aware" not in gateway_args
    assert gateway_args.count("--policy") == 1
    idx = gateway_args.index("--policy")
    assert gateway_args[idx + 1] == "cache_aware"


def test_gateway_defaults_pipeline_for_cache_aware_equals_form():
    # --policy=cache_aware must suppress the passthrough default too: the
    # default-policy injection and the cache_aware detection have to agree on
    # flag spelling, or clap's last-wins silently reverts the operator's
    # policy to passthrough while load monitoring stays on.
    gateway_args = _gateway_args_with_defaults(
        ["--model", "/tmp/x", "--policy=cache_aware"]
    )

    # No bare "--policy <value>" pair may be appended; the operator's
    # equals-form token must stay the only policy on the line. (A bare
    # "passthrough" value may legitimately appear as the reasoning parser.)
    assert "--policy" not in gateway_args
    assert "--policy=passthrough" not in gateway_args
    assert gateway_args.count("--policy=cache_aware") == 1
    assert "--disable-load-monitoring" not in gateway_args
    assert "--disable-circuit-breaker" in gateway_args


def test_get_from_args_extracts_value():
    assert (
        _get_from_args(
            ["--model", "nvidia/Qwen3.5-397B-A17B-NVFP4", "--port", "8000"], "--model"
        )
        == "nvidia/Qwen3.5-397B-A17B-NVFP4"
    )


def test_get_from_args_returns_none_when_absent():
    assert _get_from_args(["--port", "8000"], "--model") is None


def test_get_from_args_returns_none_when_flag_lacks_value():
    assert _get_from_args(["--model"], "--model") is None


def test_get_from_args_takes_the_last_occurrence():
    """Last-wins matches argparse, so appending a flag overrides an earlier one."""
    assert _get_from_args(["--port", "8000", "--port", "8413"], "--port") == "8413"


def _make_muse_spark_model_dir(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "muse_spark",
                "architectures": ["MuseSparkForConditionalGeneration"],
            }
        )
    )
    return str(tmp_path)


def _make_kimi_k3_model_dir(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_k3",
                "architectures": ["KimiK3ForConditionalGeneration"],
            }
        )
    )
    return str(tmp_path)


class _ModelFamily(NamedTuple):
    """One model family's detector and the parser defaults it injects."""

    name: str
    model_id: str
    make_model_dir: Callable[[Path], str]
    detector: Callable[[str | None], bool]
    reasoning_parser: str
    tool_call_parser: str
    # Further ``--flag value`` pairs the family appends, after the parsers.
    extra_engine_args: tuple[str, ...] = ()
    extra_gateway_args: tuple[str, ...] = ()


MUSE_SPARK_MODEL_ID = "meta-models/Muse-Spark-Open-816B-A42B"

_MODEL_FAMILIES = (
    _ModelFamily(
        "muse_spark",
        MUSE_SPARK_MODEL_ID,
        _make_muse_spark_model_dir,
        _is_muse_spark_model,
        MUSE_SPARK_REASONING_PARSER,
        MUSE_SPARK_TOOL_CALL_PARSER,
        extra_engine_args=("--grammar-backend", MUSE_SPARK_GRAMMAR_BACKEND),
        extra_gateway_args=("--parser", MUSE_SPARK_GATEWAY_PARSER),
    ),
    _ModelFamily(
        "kimi_k3",
        "moonshotai/Kimi-K3",
        _make_kimi_k3_model_dir,
        _is_kimi_k3_model,
        KIMI_K3_REASONING_PARSER,
        KIMI_K3_TOOL_CALL_PARSER,
    ),
)


@pytest.mark.parametrize("family", _MODEL_FAMILIES, ids=lambda f: f.name)
def test_model_is_detected_from_id_or_local_config(tmp_path, family):
    assert family.detector(family.model_id)
    assert family.detector(family.make_model_dir(tmp_path))
    for other in _MODEL_FAMILIES:
        if other.model_id != family.model_id:
            assert not family.detector(other.model_id)
    assert not family.detector(None)


@pytest.mark.parametrize("from_local_config", [False, True], ids=["id", "local_config"])
@pytest.mark.parametrize("family", _MODEL_FAMILIES, ids=lambda f: f.name)
def test_model_gets_default_parsers(tmp_path, family, from_local_config):
    model = family.make_model_dir(tmp_path) if from_local_config else family.model_id

    engine_args, gateway_args = _args_with_default_model_parsers(
        ["--model", model],
        ["--model", model],
    )

    assert engine_args == [
        "--model",
        model,
        "--reasoning-parser",
        family.reasoning_parser,
        *family.extra_engine_args,
    ]
    # Order matters: the reasoning parser is appended before the tool parser.
    assert gateway_args == [
        "--model",
        model,
        "--reasoning-parser",
        family.reasoning_parser,
        "--tool-call-parser",
        family.tool_call_parser,
        *family.extra_gateway_args,
    ]


def _explicit(defaults: tuple[str, ...]) -> list[str]:
    """The family's extra ``--flag value`` pairs with operator-chosen values."""
    return [
        token if token.startswith("--") else f"custom_{token}" for token in defaults
    ]


@pytest.mark.parametrize("family", _MODEL_FAMILIES, ids=lambda f: f.name)
def test_model_parser_defaults_preserve_explicit_values(tmp_path, family):
    model = family.make_model_dir(tmp_path)
    engine_args = [
        "--model",
        model,
        "--reasoning-parser",
        "custom_reasoning",
        *_explicit(family.extra_engine_args),
    ]
    gateway_args = [
        "--model",
        model,
        "--reasoning-parser",
        "custom_reasoning",
        "--tool-call-parser",
        "custom_tools",
        *_explicit(family.extra_gateway_args),
    ]

    assert _args_with_default_model_parsers(engine_args, gateway_args) == (
        engine_args,
        gateway_args,
    )


def test_kimi_k3_parser_defaults_fill_missing_parser_only():
    model = "moonshotai/Kimi-K3"

    engine_args, gateway_args = _args_with_default_model_parsers(
        ["--model", model, "--reasoning-parser", "none"],
        ["--model", model, "--reasoning-parser", "none"],
    )
    assert engine_args == ["--model", model, "--reasoning-parser", "none"]
    assert gateway_args == [
        "--model",
        model,
        "--reasoning-parser",
        "none",
        "--tool-call-parser",
        KIMI_K3_TOOL_CALL_PARSER,
    ]

    engine_args, gateway_args = _args_with_default_model_parsers(
        ["--model", model],
        ["--model", model, "--tool-call-parser", "json"],
    )
    assert engine_args == [
        "--model",
        model,
        "--reasoning-parser",
        KIMI_K3_REASONING_PARSER,
    ]
    assert gateway_args == [
        "--model",
        model,
        "--tool-call-parser",
        "json",
        "--reasoning-parser",
        KIMI_K3_REASONING_PARSER,
    ]


def test_kimi_k3_default_reasoning_parser_survives_gateway_defaults():
    model = "moonshotai/Kimi-K3"

    _, gateway_args = _args_with_default_model_parsers(
        ["--model", model],
        ["--model", model],
    )
    gateway_args = _gateway_args_with_defaults(gateway_args)

    # Exactly one reasoning parser, and it is the Kimi K3 default — the
    # generic passthrough reasoning-parser default must not be layered on top.
    # Check the parser slot rather than bare membership: the value "passthrough"
    # (== DEFAULT_REASONING_PARSER) now also appears as the --policy value, which
    # is an unrelated flag.
    assert gateway_args.count("--reasoning-parser") == 1
    idx = gateway_args.index("--reasoning-parser")
    assert gateway_args[idx + 1] == KIMI_K3_REASONING_PARSER


def test_muse_spark_explicit_gateway_parser_wins(tmp_path):
    """An operator ``--parser`` stays; the ``muse`` names are still defaulted."""
    model = _make_muse_spark_model_dir(tmp_path)

    engine_args, gateway_args = _args_with_default_model_parsers(
        ["--model", model],
        ["--model", model, "--parser", "legacy"],
    )

    assert gateway_args.count("--parser") == 1
    assert _get_from_args(gateway_args, "--parser") == "legacy"
    assert (
        _get_from_args(gateway_args, "--reasoning-parser")
        == MUSE_SPARK_REASONING_PARSER
    )
    assert (
        _get_from_args(gateway_args, "--tool-call-parser")
        == MUSE_SPARK_TOOL_CALL_PARSER
    )
    assert (
        _get_from_args(engine_args, "--grammar-backend") == MUSE_SPARK_GRAMMAR_BACKEND
    )


def test_muse_spark_explicit_grammar_backend_wins(tmp_path):
    """An operator ``--grammar-backend`` stays; the gateway still runs Symphony."""
    model = _make_muse_spark_model_dir(tmp_path)

    engine_args, gateway_args = _args_with_default_model_parsers(
        ["--model", model, "--grammar-backend", "none"],
        ["--model", model],
    )

    assert engine_args.count("--grammar-backend") == 1
    assert _get_from_args(engine_args, "--grammar-backend") == "none"
    assert _get_from_args(gateway_args, "--parser") == MUSE_SPARK_GATEWAY_PARSER


def test_muse_spark_defaults_survive_gateway_defaults():
    _, gateway_args = _args_with_default_model_parsers(
        ["--model", MUSE_SPARK_MODEL_ID],
        ["--model", MUSE_SPARK_MODEL_ID],
    )
    gateway_args = _gateway_args_with_defaults(gateway_args)

    assert gateway_args.count("--parser") == 1
    assert _get_from_args(gateway_args, "--parser") == MUSE_SPARK_GATEWAY_PARSER
    assert gateway_args.count("--reasoning-parser") == 1
    assert (
        _get_from_args(gateway_args, "--reasoning-parser")
        == MUSE_SPARK_REASONING_PARSER
    )


def test_muse_spark_defaults_follow_split_argv_routing():
    """``split_argv`` sends the operator's ``--grammar-backend`` to the engine and
    ``--parser`` to the gateway; the Muse Spark defaults respect both there."""
    split = split_argv(
        [MUSE_SPARK_MODEL_ID, "--grammar-backend", "none", "--parser", "legacy"]
    )
    assert split.engine == [
        "--model",
        MUSE_SPARK_MODEL_ID,
        "--grammar-backend",
        "none",
    ]
    assert split.gateway == ["--model", MUSE_SPARK_MODEL_ID, "--parser", "legacy"]

    engine_args, gateway_args = _args_with_default_model_parsers(
        split.engine, split.gateway
    )

    assert _get_from_args(engine_args, "--grammar-backend") == "none"
    assert "--parser" not in engine_args
    assert _get_from_args(gateway_args, "--parser") == "legacy"
    assert "--grammar-backend" not in gateway_args


def test_muse_spark_tree_drafting_skips_xgrammar_default(tmp_path):
    """Tree drafting rejects every non-``none`` grammar backend at startup
    (``ServerArgs._validate_tree_speculation``), so the xgrammar default steps
    aside for it; the gateway still runs Symphony."""
    model = _make_muse_spark_model_dir(tmp_path)

    engine_args, gateway_args = _args_with_default_model_parsers(
        [
            "--model",
            model,
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-eagle-topk",
            "4",
            "--speculative-num-draft-tokens",
            "8",
        ],
        ["--model", model],
    )

    assert "--grammar-backend" not in engine_args
    assert _get_from_args(gateway_args, "--parser") == MUSE_SPARK_GATEWAY_PARSER


def test_muse_spark_chain_drafting_keeps_xgrammar_default(tmp_path):
    """A chain draft (the default topk of 1) accepts structured output, so the
    xgrammar default still applies."""
    model = _make_muse_spark_model_dir(tmp_path)

    engine_args, _ = _args_with_default_model_parsers(
        ["--model", model, "--speculative-algorithm", "EAGLE3"],
        ["--model", model],
    )

    assert (
        _get_from_args(engine_args, "--grammar-backend") == MUSE_SPARK_GRAMMAR_BACKEND
    )


def _make_inkling_model_dir(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "inkling_mm_model",
                "architectures": ["InklingForConditionalGeneration"],
            }
        )
    )
    return str(tmp_path)


def test_local_inkling_config_is_detected(tmp_path):
    assert _is_inkling_model(_make_inkling_model_dir(tmp_path))
    assert not _is_inkling_model("moonshotai/Kimi-K3")
    assert not _is_inkling_model(None)


@patch(
    "tokenspeed.cli.serve_smg.current_platform",
    new=lambda: SimpleNamespace(is_nvidia=True),
)
def test_remote_inkling_model_ids_get_defaults():
    for model in ("org/Inkling-Chat",):
        engine_args, gateway_args = _args_with_default_model_parsers(
            ["--model", model],
            ["--model", model],
        )
        assert engine_args[engine_args.index("--reasoning-parser") + 1] == (
            INKLING_REASONING_PARSER
        )
        assert engine_args[engine_args.index("--attention-backend") + 1] == (
            INKLING_ATTENTION_BACKEND
        )
        assert (
            gateway_args[gateway_args.index("--reasoning-parser") + 1]
            == INKLING_REASONING_PARSER
        )
        assert (
            gateway_args[gateway_args.index("--tool-call-parser") + 1]
            == INKLING_TOOL_CALL_PARSER
        )
        assert "--chat-template" not in gateway_args


@patch(
    "tokenspeed.cli.serve_smg.current_platform",
    new=lambda: SimpleNamespace(is_nvidia=True),
)
def test_inkling_model_gets_default_parsers_without_overriding_checkpoint_template(
    tmp_path,
):
    model = _make_inkling_model_dir(tmp_path)

    engine_args, gateway_args = _args_with_default_model_parsers(
        ["--model", model],
        ["--model", model],
    )

    assert engine_args == [
        "--model",
        model,
        "--reasoning-parser",
        INKLING_REASONING_PARSER,
        "--attention-backend",
        INKLING_ATTENTION_BACKEND,
    ]
    assert gateway_args[:2] == ["--model", model]
    assert "--chat-template" not in gateway_args
    reasoning_idx = gateway_args.index("--reasoning-parser")
    assert gateway_args[reasoning_idx + 1] == INKLING_REASONING_PARSER
    tool_idx = gateway_args.index("--tool-call-parser")
    assert gateway_args[tool_idx + 1] == INKLING_TOOL_CALL_PARSER


@patch(
    "tokenspeed.cli.serve_smg.current_platform",
    new=lambda: SimpleNamespace(is_nvidia=True),
)
def test_inkling_defaults_preserve_explicit_user_values(tmp_path):
    model = _make_inkling_model_dir(tmp_path)

    engine_args, gateway_args = _args_with_default_model_parsers(
        [
            "--model",
            model,
            "--reasoning-parser",
            "custom_reasoning",
            "--attention-backend",
            INKLING_ATTENTION_BACKEND,
            "--enable-prefix-caching",
        ],
        [
            "--model",
            model,
            "--reasoning-parser",
            "custom_reasoning",
            "--tool-call-parser",
            "custom_tools",
            "--chat-template",
            "/tmp/custom.jinja",
        ],
    )

    assert engine_args == [
        "--model",
        model,
        "--reasoning-parser",
        "custom_reasoning",
        "--attention-backend",
        INKLING_ATTENTION_BACKEND,
        "--enable-prefix-caching",
    ]
    assert gateway_args == [
        "--model",
        model,
        "--reasoning-parser",
        "custom_reasoning",
        "--tool-call-parser",
        "custom_tools",
        "--chat-template",
        "/tmp/custom.jinja",
    ]


@patch(
    "tokenspeed.cli.serve_smg.current_platform",
    new=lambda: SimpleNamespace(is_nvidia=True),
)
def test_inkling_rejects_incompatible_attention_backend(tmp_path):
    model = _make_inkling_model_dir(tmp_path)

    with pytest.raises(ValueError, match="requires --attention-backend fa4"):
        _args_with_default_model_parsers(
            ["--model", model, "--attention-backend", "fa3"],
            ["--model", model],
        )


def test_inkling_preserves_enabled_prefix_caching(tmp_path):
    model = _make_inkling_model_dir(tmp_path)

    engine_args, _ = _args_with_default_model_parsers(
        ["--model", model, "--enable-prefix-caching"],
        ["--model", model],
    )

    assert "--enable-prefix-caching" in engine_args
    assert "--disable-prefix-caching" not in engine_args


def test_prewarm_skips_local_path(tmp_path):
    with patch(
        "huggingface_hub.snapshot_download", side_effect=AssertionError("must not call")
    ) as sd:
        _prewarm_hf_tokenizer(str(tmp_path))
    sd.assert_not_called()


def test_prewarm_skips_empty():
    with patch("huggingface_hub.snapshot_download") as sd:
        _prewarm_hf_tokenizer("")
    sd.assert_not_called()


def test_prewarm_fetches_tokenizer_artifacts_for_hf_id():
    with patch("huggingface_hub.snapshot_download") as sd:
        _prewarm_hf_tokenizer("nvidia/Qwen3.5-397B-A17B-NVFP4")
    sd.assert_called_once()
    _, kwargs = sd.call_args
    assert kwargs["repo_id"] == "nvidia/Qwen3.5-397B-A17B-NVFP4"
    # Patterns must include tokenizer.json and the surrounding JSON configs;
    # avoid pulling weight files (no `*.safetensors` etc.).
    patterns = set(kwargs["allow_patterns"])
    assert "tokenizer*" in patterns
    assert "chat_template*" in patterns
    assert "*.json" in patterns


def test_prewarm_swallows_download_errors():
    with patch(
        "huggingface_hub.snapshot_download", side_effect=RuntimeError("HF down")
    ):
        # Must not raise — smg's own retry path is the fallback.
        _prewarm_hf_tokenizer("nvidia/Qwen3.5-397B-A17B-NVFP4")


def _probe(*, liveness=None, readiness=None):
    """A ``wait_http_ready`` stand-in that runs ``liveness`` for the gateway's
    ``/liveness`` probe and ``readiness`` for its ``/readiness`` probe (each
    an async callable; None returns at once)."""

    async def probe(url, **kwargs):
        step = liveness if url.endswith("/liveness") else readiness
        if step is not None:
            await step()

    return probe


def _release_after_ready(startup_done: asyncio.Event):
    """A readiness step that lets the children's ``wait`` resolve just after
    the probe succeeded, so the probe's success is recorded first."""

    async def step():
        asyncio.get_running_loop().call_later(0.05, startup_done.set)

    return step


async def _run(
    engine, gateway, *, probe, layout=EngineLayout(1, None), terminate=None, **overrides
):
    """``run_smg`` with both spawns, the probe and teardown patched; returns
    its exit code and the spawn mocks."""
    spawn_engine = AsyncMock(return_value=engine)
    spawn_gateway = AsyncMock(return_value=gateway)
    kwargs = {
        "engine_args": ["--model", "/tmp/x"],
        "gateway_args": ["--model", "/tmp/x"],
        "opts": OrchestratorOpts(engine_startup_timeout=10, gateway_startup_timeout=10),
        "user_host": "127.0.0.1",
        "user_port": 8000,
        "layout": layout,
        **overrides,
    }
    with patch("tokenspeed.cli.serve_smg.spawn_engine", spawn_engine), patch(
        "tokenspeed.cli.serve_smg.spawn_gateway", spawn_gateway
    ), patch("tokenspeed.cli.serve_smg.wait_http_ready", side_effect=probe), patch(
        "tokenspeed.cli.serve_smg.terminate_then_kill",
        terminate if terminate is not None else AsyncMock(),
    ), patch(
        "tokenspeed.cli.serve_smg.kill_process_tree", lambda *a, **kw: None
    ):
        rc = await run_smg(**kwargs)
    return rc, spawn_engine, spawn_gateway


def _children_until(startup_done: asyncio.Event, *procs):
    async def wait_then_zero():
        await startup_done.wait()
        return 0

    for proc in procs:
        proc.wait = AsyncMock(side_effect=wait_then_zero)


@pytest.mark.asyncio
async def test_engine_start_timeout_tears_down_and_exits_nonzero():
    engine = _make_proc()
    gateway = _make_proc()

    async def hung():
        await asyncio.sleep(60)

    engine.wait = AsyncMock(side_effect=hung)
    gateway.wait = AsyncMock(side_effect=hung)

    async def timeout():
        raise TimeoutError("/readiness never answered 200")

    terminate = AsyncMock()
    rc, _, _ = await _run(
        engine, gateway, probe=_probe(readiness=timeout), terminate=terminate
    )
    assert rc != 0
    torn_down = [call.args[0] for call in terminate.await_args_list]
    assert torn_down == [gateway, engine]


@pytest.mark.asyncio
async def test_gateway_starts_first_and_stops_first():
    """The gateway binds the engine's sockets, so it starts first; on
    shutdown it is SIGTERMed before the engine (front before back)."""
    engine = _make_proc(returncode=0)
    gateway = _make_proc(returncode=0)
    order: list[str] = []
    startup_done = asyncio.Event()
    _children_until(startup_done, engine, gateway)

    async def liveness():
        # The engine must not exist while the gateway's liveness is probed.
        order.append("liveness")

    async def tracked_term(proc, *, drain_timeout):
        order.append("stop gateway" if proc is gateway else "stop engine")

    def spawned(name, proc):
        async def spawn(*args, **kwargs):
            order.append(f"spawn {name}")
            return proc

        return spawn

    with patch(
        "tokenspeed.cli.serve_smg.spawn_engine", spawned("engine", engine)
    ), patch(
        "tokenspeed.cli.serve_smg.spawn_gateway", spawned("gateway", gateway)
    ), patch(
        "tokenspeed.cli.serve_smg.wait_http_ready",
        side_effect=_probe(
            liveness=liveness, readiness=_release_after_ready(startup_done)
        ),
    ), patch(
        "tokenspeed.cli.serve_smg.terminate_then_kill", side_effect=tracked_term
    ):
        rc = await run_smg(
            engine_args=[],
            gateway_args=[],
            opts=OrchestratorOpts(
                engine_startup_timeout=10, gateway_startup_timeout=10
            ),
            user_host="127.0.0.1",
            user_port=8000,
            layout=EngineLayout(1, None),
        )
    assert rc == 0
    assert order == [
        "spawn gateway",
        "liveness",
        "spawn engine",
        "stop gateway",
        "stop engine",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_count", [1, 4])
async def test_engine_dials_the_gateways_handshake(engine_count):
    """The engine gets the handshake port the gateway derives from the worker
    URL it was given, and the gateway awaits every attention-DP rank."""
    from tokenspeed.cli._proc import zmq_handshake_port

    engine = _make_proc(returncode=0)
    gateway = _make_proc(returncode=0)
    startup_done = asyncio.Event()
    _children_until(startup_done, engine, gateway)

    rc, spawn_engine, spawn_gateway = await _run(
        engine,
        gateway,
        probe=_probe(readiness=_release_after_ready(startup_done)),
        layout=EngineLayout(engine_count, None),
    )
    assert rc == 0
    worker_url = spawn_gateway.await_args.kwargs["worker_url"]
    assert worker_url.startswith("ipc://")
    assert spawn_gateway.await_args.kwargs["engine_count"] == engine_count
    assert spawn_engine.await_args.kwargs["handshake"] == (
        "127.0.0.1",
        zmq_handshake_port(worker_url),
    )


@pytest.mark.asyncio
async def test_ranks_on_other_nodes_dial_a_tcp_worker():
    """When replica leaders sit on other nodes, the gateway's worker is the
    tcp:// address they reach, and the engine dials it too."""
    engine = _make_proc(returncode=0)
    gateway = _make_proc(returncode=0)
    startup_done = asyncio.Event()
    _children_until(startup_done, engine, gateway)

    rc, spawn_engine, spawn_gateway = await _run(
        engine,
        gateway,
        probe=_probe(readiness=_release_after_ready(startup_done)),
        layout=EngineLayout(4, ("10.0.0.5", 23455)),
    )
    assert rc == 0
    assert spawn_gateway.await_args.kwargs["worker_url"] == "tcp://10.0.0.5:23455"
    assert spawn_gateway.await_args.kwargs["engine_count"] == 4
    assert spawn_engine.await_args.kwargs["handshake"] == ("10.0.0.5", 23455)


@pytest.mark.asyncio
async def test_run_smg_allocates_prometheus_port_at_gateway_spawn():
    """The metrics port is injected by run_smg right before spawn_gateway:
    a port allocated earlier can be reclaimed as an ephemeral source port."""
    engine = _make_proc(returncode=0)
    gateway = _make_proc(returncode=0)
    startup_done = asyncio.Event()
    _children_until(startup_done, engine, gateway)

    rc, _, spawn_gateway = await _run(
        engine, gateway, probe=_probe(readiness=_release_after_ready(startup_done))
    )

    assert rc == 0
    spawned_args = spawn_gateway.await_args.args[0]
    idx = spawned_args.index("--prometheus-port")
    assert spawned_args[idx + 1].isdigit()
    assert 1 <= int(spawned_args[idx + 1]) <= 65535


@pytest.mark.asyncio
async def test_run_smg_wires_metrics_port_into_engine_spawn(capsys):
    """With --enable-metrics the engine gets a pinned --metrics-port before
    spawn_engine, and the scrape URL is printed once serving is ready."""
    engine = _make_proc(returncode=0)
    gateway = _make_proc(returncode=0)
    startup_done = asyncio.Event()
    _children_until(startup_done, engine, gateway)

    rc, spawn_engine, _ = await _run(
        engine,
        gateway,
        probe=_probe(readiness=_release_after_ready(startup_done)),
        engine_args=["--model", "/tmp/x", "--enable-metrics"],
    )

    assert rc == 0
    spawned_args = spawn_engine.await_args.args[0]
    idx = spawned_args.index("--metrics-port")
    port = int(spawned_args[idx + 1])
    assert 1 <= port <= 65535
    out = capsys.readouterr().out
    assert f"engine metrics available at http://127.0.0.1:{port}/metrics" in out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metric_args",
    [
        [],
        ["--enable-metrics", "--metrics-reporters", "test_reporter"],
        ["--enable-metrics", "--metrics-reporters", "external_reporter"],
    ],
)
async def test_run_smg_without_prometheus_omits_engine_metrics_port(
    capsys, metric_args
):
    """No Prometheus reporter: no metrics port injection or scrape URL output."""
    engine = _make_proc(returncode=0)
    gateway = _make_proc(returncode=0)
    startup_done = asyncio.Event()
    _children_until(startup_done, engine, gateway)

    rc, spawn_engine, _ = await _run(
        engine,
        gateway,
        probe=_probe(readiness=_release_after_ready(startup_done)),
        engine_args=["--model", "/tmp/x", *metric_args],
    )

    assert rc == 0
    spawned_args = spawn_engine.await_args.args[0]
    assert "--metrics-port" not in spawned_args
    assert "engine metrics" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_signal_handlers_installed_before_spawning_anything():
    """Signal handlers must be live before any subprocess."""
    call_order: list[str] = []

    real_loop = asyncio.get_running_loop()
    real_add_signal_handler = real_loop.add_signal_handler

    def tracking_add_signal_handler(sig, callback, *args):
        call_order.append(f"add_signal_handler:{sig}")
        return real_add_signal_handler(sig, callback, *args)

    async def tracking_spawn_gateway(*args, **kwargs):
        call_order.append("spawn_gateway")
        raise RuntimeError("simulated spawn failure")

    with patch.object(
        real_loop, "add_signal_handler", side_effect=tracking_add_signal_handler
    ), patch(
        "tokenspeed.cli.serve_smg.spawn_gateway", side_effect=tracking_spawn_gateway
    ):
        with pytest.raises(RuntimeError, match="simulated spawn failure"):
            await run_smg(
                engine_args=[],
                gateway_args=[],
                opts=OrchestratorOpts(),
                user_host="127.0.0.1",
                user_port=8000,
                layout=EngineLayout(1, None),
            )

    spawn_idx = call_order.index("spawn_gateway")
    assert call_order.index(f"add_signal_handler:{signal.SIGTERM}") < spawn_idx
    assert call_order.index(f"add_signal_handler:{signal.SIGINT}") < spawn_idx


@pytest.mark.asyncio
async def test_stop_during_engine_probe_exits_zero():
    """SIGTERM while the engine is starting is a clean exit (rc=0)."""
    engine = _make_proc()
    gateway = _make_proc()

    async def hung():
        await asyncio.sleep(60)

    engine.wait = AsyncMock(side_effect=hung)
    gateway.wait = AsyncMock(side_effect=hung)
    stop = asyncio.Event()
    asyncio.get_running_loop().call_later(0.05, stop.set)

    rc, _, _ = await _run(
        engine, gateway, probe=_probe(readiness=hung), _stop_event=stop
    )
    assert rc == 0


@pytest.mark.asyncio
async def test_first_nonzero_child_exit_propagates():
    engine = _make_proc(returncode=42)
    gateway = _make_proc(returncode=0)
    startup_done = asyncio.Event()

    async def engine_wait():
        await startup_done.wait()
        return 42

    async def gateway_wait():
        await asyncio.Event().wait()

    engine.wait = AsyncMock(side_effect=engine_wait)
    gateway.wait = AsyncMock(side_effect=gateway_wait)

    rc, _, _ = await _run(
        engine, gateway, probe=_probe(readiness=_release_after_ready(startup_done))
    )
    assert rc == 42


@pytest.mark.asyncio
async def test_engine_exit_during_probe_fails_fast():
    """If the engine exits before it registered, return rc=1 at once."""
    engine = _make_proc(returncode=2)
    gateway = _make_proc()

    async def hung():
        await asyncio.sleep(60)

    engine.wait = AsyncMock(return_value=2)
    gateway.wait = AsyncMock(side_effect=hung)

    rc, _, _ = await _run(
        engine,
        gateway,
        probe=_probe(readiness=hung),
        opts=OrchestratorOpts(engine_startup_timeout=600, gateway_startup_timeout=10),
    )
    assert rc == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("probe_step", ["liveness", "readiness"])
async def test_gateway_exit_during_probe_fails_fast(probe_step):
    """If the gateway exits while either probe waits, return rc=1 at once."""
    engine = _make_proc()
    gateway = _make_proc(returncode=2)
    dies = asyncio.Event()

    async def hung():
        await asyncio.sleep(60)

    async def gateway_dies_meanwhile():
        dies.set()
        await asyncio.sleep(60)

    async def gateway_wait():
        await dies.wait()
        return 2

    gateway.wait = AsyncMock(side_effect=gateway_wait)
    engine.wait = AsyncMock(side_effect=hung)

    rc, spawn_engine, _ = await _run(
        engine,
        gateway,
        probe=_probe(**{probe_step: gateway_dies_meanwhile}),
        opts=OrchestratorOpts(engine_startup_timeout=600, gateway_startup_timeout=600),
    )
    assert rc == 1
    # A gateway that dies before it is live never gets an engine.
    assert spawn_engine.await_count == (0 if probe_step == "liveness" else 1)


def test_worker_url_skips_a_path_whose_handshake_port_is_taken(monkeypatch):
    """Each ipc path hashes to a handshake port; one that is in use is
    skipped for the next path, so two launches never share a handshake."""
    import socket

    from tokenspeed.cli._proc import zmq_handshake_port

    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", "/tmp/smg-zmq-test")
    first = _zmq_worker_url(31000)
    assert first == "ipc:///tmp/smg-zmq-test/ts-serve-31000"
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", zmq_handshake_port(first)))
        holder.listen()
        second = _zmq_worker_url(31000)
    assert second != first
    assert zmq_handshake_port(second) != zmq_handshake_port(first)


def test_engine_identity_reaches_the_gateway():
    """The engine's served model name becomes a gateway alias of its model,
    and its tokenizer the gateway's, unless the gateway has its own."""
    engine = ["--model", "/m", "--served-model-name", "chat", "--tokenizer", "/t"]
    assert _gateway_args_with_engine_identity(engine, ["--model", "/m"]) == [
        "--model",
        "/m",
        "--model-alias",
        "chat=/m",
        "--tokenizer-path",
        "/t",
    ]
    assert _gateway_args_with_engine_identity(
        engine, ["--model", "/m", "--tokenizer-path", "/own"]
    ) == ["--model", "/m", "--tokenizer-path", "/own", "--model-alias", "chat=/m"]
    # A served name equal to the model needs no alias.
    assert _gateway_args_with_engine_identity(
        ["--model", "/m", "--served-model-name", "/m"], ["--model", "/m"]
    ) == ["--model", "/m"]


_LLAMA_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "dtype": "bfloat16",
    "vocab_size": 32000,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 512,
}


@pytest.fixture
def llama_dir(tmp_path):
    # importorskip(exc_type=ImportError): the engine's ServerArgs pull the
    # full runtime stack, unimportable in an env with drifted kernel wheels.
    pytest.importorskip("tokenspeed.runtime.utils.server_args", exc_type=ImportError)
    (tmp_path / "config.json").write_text(json.dumps(_LLAMA_CONFIG))
    return str(tmp_path)


@pytest.mark.parametrize(
    "layout, engines",
    [
        ([], 1),
        (["--data-parallel-size", "2", "--dist-init-addr", "127.0.0.1:23456"], 2),
        # Tensor parallelism across nodes: only rank 0 registers.
        (
            [
                "--tensor-parallel-size",
                "4",
                "--nnodes",
                "2",
                "--node-rank",
                "0",
                "--dist-init-addr",
                "127.0.0.1:23456",
            ],
            1,
        ),
    ],
)
def test_the_engine_count_is_the_attention_dp_size(llama_dir, layout, engines):
    assert _engine_layout(["--model", llama_dir, *layout]) == EngineLayout(
        engines, None
    )


def test_replica_leaders_on_other_nodes_dial_below_the_dist_init_port(llama_dir):
    # Two replicas of two ranks each, one per node: rank 2 leads the second
    # from node 1, so the gateway's handshake listens on the head's address.
    layout = _engine_layout(
        [
            "--model",
            llama_dir,
            "--attn-tp-size",
            "2",
            "--data-parallel-size",
            "2",
            "--nnodes",
            "2",
            "--node-rank",
            "0",
            "--dist-init-addr",
            "127.0.0.1:23456",
        ]
    )
    assert layout == EngineLayout(2, ("127.0.0.1", 23455))


def test_a_disaggregated_role_is_refused(llama_dir):
    with pytest.raises(ServeConfigError, match="disaggregation-mode prefill"):
        _engine_layout(["--model", llama_dir, "--disaggregation-mode", "prefill"])


def test_run_smg_from_args_sets_process_title(monkeypatch):
    """Orchestrator sets proc title to 'ts-serve' so pgrep -f ts-serve finds it."""
    captured = {}

    def fake_run(*a, **kw):
        return 0

    monkeypatch.setattr("tokenspeed.cli.serve_smg.asyncio.run", fake_run)
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._check_serve_extra_installed", lambda: None
    )
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._engine_layout", lambda _: EngineLayout(1, None)
    )
    fake_setproctitle = type(
        "M", (), {"setproctitle": lambda title: captured.setdefault("title", title)}
    )
    monkeypatch.setitem(sys.modules, "setproctitle", fake_setproctitle)

    from argparse import Namespace

    from tokenspeed.cli.serve_smg import run_smg_from_args

    try:
        run_smg_from_args(Namespace(), ["--model", "/tmp/x"])
    except SystemExit:
        pass
    assert captured.get("title") == "ts-serve"


def test_run_smg_from_args_applies_kimi_k3_parser_defaults(monkeypatch):
    captured = {}

    async def fake_run_smg(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("tokenspeed.cli.serve_smg.print_logo", lambda: None)
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._check_serve_extra_installed", lambda: None
    )
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._prewarm_hf_tokenizer", lambda _: None
    )
    monkeypatch.setattr("tokenspeed.cli.serve_smg.run_smg", fake_run_smg)
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._engine_layout", lambda _: EngineLayout(1, None)
    )

    from argparse import Namespace

    from tokenspeed.cli.serve_smg import run_smg_from_args

    with pytest.raises(SystemExit) as exc:
        run_smg_from_args(Namespace(), ["moonshotai/Kimi-K3"])

    assert exc.value.code == 0
    assert captured["engine_args"] == [
        "--model",
        "moonshotai/Kimi-K3",
        "--reasoning-parser",
        KIMI_K3_REASONING_PARSER,
    ]
    assert captured["gateway_args"][:6] == [
        "--model",
        "moonshotai/Kimi-K3",
        "--reasoning-parser",
        KIMI_K3_REASONING_PARSER,
        "--tool-call-parser",
        KIMI_K3_TOOL_CALL_PARSER,
    ]


def test_run_smg_from_args_applies_muse_spark_defaults(monkeypatch):
    captured = {}

    async def fake_run_smg(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("tokenspeed.cli.serve_smg.print_logo", lambda: None)
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._check_serve_extra_installed", lambda: None
    )
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._prewarm_hf_tokenizer", lambda _: None
    )
    monkeypatch.setattr("tokenspeed.cli.serve_smg.run_smg", fake_run_smg)
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._engine_layout", lambda _: EngineLayout(1, None)
    )

    from argparse import Namespace

    from tokenspeed.cli.serve_smg import run_smg_from_args

    with pytest.raises(SystemExit) as exc:
        run_smg_from_args(Namespace(), [MUSE_SPARK_MODEL_ID])

    assert exc.value.code == 0
    assert captured["engine_args"] == [
        "--model",
        MUSE_SPARK_MODEL_ID,
        "--reasoning-parser",
        MUSE_SPARK_REASONING_PARSER,
        "--grammar-backend",
        MUSE_SPARK_GRAMMAR_BACKEND,
    ]
    assert captured["gateway_args"][:8] == [
        "--model",
        MUSE_SPARK_MODEL_ID,
        "--reasoning-parser",
        MUSE_SPARK_REASONING_PARSER,
        "--tool-call-parser",
        MUSE_SPARK_TOOL_CALL_PARSER,
        "--parser",
        MUSE_SPARK_GATEWAY_PARSER,
    ]


def test_add_metrics_port_noop_without_enable_metrics():
    args, url = _add_metrics_port(["--model", "/tmp/x"])
    assert args == ["--model", "/tmp/x"]
    assert url is None


@pytest.mark.parametrize(
    "reporter_args",
    [
        ["--metrics-reporters", "test_reporter"],
        ["--metrics-reporters=test_reporter"],
        ["--metrics-reporters", "test_reporter", "--metrics-reporters=test_reporter"],
        ["--metrics-reporters=test_reporter", "--metrics-port", "5006"],
        ["--metrics-reporters", "external_reporter"],
        ["--metrics-reporters=external_reporter", "--metrics-port", "5006"],
    ],
)
def test_add_metrics_port_noop_for_raw_outputs_only(monkeypatch, reporter_args):
    def unexpected_allocation():
        pytest.fail("Raw-only reporting must not allocate a Prometheus port")

    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._free_port_avoiding_ephemeral_range",
        unexpected_allocation,
    )
    engine_args = ["--enable-metrics", *reporter_args]

    args, url = _add_metrics_port(engine_args)

    assert args == engine_args
    assert url is None


@pytest.mark.parametrize(
    "reporter_args",
    [
        ["--metrics-reporters", "prometheus"],
        ["--metrics-reporters=prometheus"],
        ["--metrics-reporters", "prometheus", "--metrics-reporters", "test_reporter"],
        ["--metrics-reporters", "test_reporter", "--metrics-reporters", "prometheus"],
        ["--metrics-reporters=prometheus", "--metrics-reporters=test_reporter"],
        ["--metrics-reporters=test_reporter", "--metrics-reporters", "prometheus"],
        ["--metrics-reporters=external_reporter", "--metrics-reporters", "prometheus"],
    ],
)
def test_add_metrics_port_with_prometheus_reporter(monkeypatch, reporter_args):
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._free_port_avoiding_ephemeral_range", lambda: 40124
    )
    engine_args = ["--enable-metrics", *reporter_args]

    args, url = _add_metrics_port(engine_args)

    assert args == [*engine_args, "--metrics-port", "40124"]
    assert url == "http://127.0.0.1:40124/metrics"


def test_add_metrics_port_allocates_when_absent(monkeypatch):
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._free_port_avoiding_ephemeral_range", lambda: 40124
    )
    args, url = _add_metrics_port(["--model", "/tmp/x", "--enable-metrics"])
    assert args == ["--model", "/tmp/x", "--enable-metrics", "--metrics-port", "40124"]
    assert url == "http://127.0.0.1:40124/metrics"


def test_add_metrics_port_keeps_a_pinned_port():
    args, url = _add_metrics_port(
        ["--model", "/tmp/x", "--enable-metrics", "--metrics-port", "5006"]
    )
    assert args == ["--model", "/tmp/x", "--enable-metrics", "--metrics-port", "5006"]
    assert url == "http://127.0.0.1:5006/metrics"


def test_add_metrics_port_treats_zero_as_auto(monkeypatch):
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._free_port_avoiding_ephemeral_range", lambda: 40124
    )
    args, url = _add_metrics_port(["--enable-metrics", "--metrics-port", "0"])
    assert args == ["--enable-metrics", "--metrics-port", "40124"]
    assert url == "http://127.0.0.1:40124/metrics"


def test_add_metrics_port_url_follows_metrics_host(monkeypatch):
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._free_port_avoiding_ephemeral_range", lambda: 40124
    )
    args, url = _add_metrics_port(["--enable-metrics", "--metrics-host", "0.0.0.0"])
    assert args == [
        "--enable-metrics",
        "--metrics-host",
        "0.0.0.0",
        "--metrics-port",
        "40124",
    ]
    assert url == "http://0.0.0.0:40124/metrics"


_RESPONSE_TEMPLATE = {
    "fields": {"thinking": {"open": "<think>", "close": "</think>"}, "content": {}}
}


def _tokenizer_dir(path, tokenizer_file="tokenizer.json", **config):
    path.mkdir(parents=True, exist_ok=True)
    (path / "tokenizer_config.json").write_text(json.dumps(config))
    (path / tokenizer_file).write_text("{}")
    return str(path)


@pytest.mark.parametrize(
    "declares, named, expected",
    [
        (
            True,
            [],
            [
                "--parser",
                "symphony",
                "--reasoning-parser",
                "hf",
                "--tool-call-parser",
                "hf",
            ],
        ),
        (True, ["--parser", "symphony"], []),
        (True, ["--tool-call-parser", "json"], ["--reasoning-parser", "passthrough"]),
        (True, ["--reasoning-parser", "passthrough"], []),
        (False, [], ["--reasoning-parser", "passthrough"]),
    ],
    ids=[
        "template",
        "parser",
        "tool-call-parser",
        "reasoning-parser",
        "no-template",
    ],
)
def test_template_tokenizer_without_parsers_gets_hf_parsers(
    tmp_path, capsys, declares, named, expected
):
    config = {"response_template": _RESPONSE_TEMPLATE} if declares else {}
    tokenizer = _tokenizer_dir(tmp_path, **config)
    args = ["--model", tokenizer, *named]

    gateway_args = _gateway_args_with_default_reasoning_parser(args, tokenizer)

    assert gateway_args == [*args, *expected]
    assert ("response_template" in capsys.readouterr().out) is (declares and not named)


def test_template_is_an_object_beside_a_tokenizer_json(tmp_path):
    # The type is what ts serve checks; the gateway validates the rest.
    assert _declares_response_template(
        _tokenizer_dir(tmp_path / "empty", response_template={})
    )
    # Not an object: the gateway refuses it.
    for name, value in (("null", None), ("string", ""), ("false", False), ("list", [])):
        assert not _declares_response_template(
            _tokenizer_dir(tmp_path / name, response_template=value)
        )
    # tiktoken: the gateway reads no template for it.
    assert not _declares_response_template(
        _tokenizer_dir(
            tmp_path / "tiktoken",
            tokenizer_file="tiktoken.model",
            response_template=_RESPONSE_TEMPLATE,
        )
    )


def test_unreadable_tokenizer_config_counts_as_no_template(tmp_path):
    for name, content in (
        ("invalid", b"{not json"),
        ("binary", b"\xff\xfe\x00{"),
        ("list", b"[1, 2]"),
    ):
        (tmp_path / name).mkdir()
        (tmp_path / name / "tokenizer_config.json").write_bytes(content)
        (tmp_path / name / "tokenizer.json").write_text("{}")
    (tmp_path / "dir" / "tokenizer_config.json").mkdir(parents=True)
    (tmp_path / "dir" / "tokenizer.json").write_text("{}")
    (tmp_path / "missing").mkdir()

    for name in ("invalid", "binary", "list", "dir", "missing"):
        assert not _declares_response_template(str(tmp_path / name))
    for tokenizer in (None, "", str(tmp_path / "no-such-dir"), "not a repo id!"):
        assert not _declares_response_template(tokenizer)


def _cache_hub_tokenizer(cache, repo_id, **config):
    repo = cache / ("models--" + repo_id.replace("/", "--"))
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("0123abcd")
    return _tokenizer_dir(repo / "snapshots" / "0123abcd", **config)


def test_hub_id_is_read_from_the_local_hf_cache(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    _cache_hub_tokenizer(
        tmp_path, "org/with-template", response_template=_RESPONSE_TEMPLATE
    )
    _cache_hub_tokenizer(tmp_path, "org/without-template")

    assert _declares_response_template("org/with-template")
    assert not _declares_response_template("org/without-template")
    # Not cached: no download, no template.
    assert not _declares_response_template("org/not-cached")


def test_template_tokenizer_is_the_gateways_tokenizer():
    assert _template_tokenizer(["--model", "m"]) == "m"
    assert _template_tokenizer(["--model", "m", "--tokenizer-path", "t"]) == "t"
    assert _template_tokenizer([]) is None


def _run_smg_from_args_with(monkeypatch, argv, prewarm):
    captured = {}

    async def fake_run_smg(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("tokenspeed.cli.serve_smg.print_logo", lambda: None)
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._check_serve_extra_installed", lambda: None
    )
    monkeypatch.setattr("tokenspeed.cli.serve_smg._prewarm_hf_tokenizer", prewarm)
    monkeypatch.setattr("tokenspeed.cli.serve_smg.run_smg", fake_run_smg)
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._engine_layout", lambda _: EngineLayout(1, None)
    )

    from argparse import Namespace

    from tokenspeed.cli.serve_smg import run_smg_from_args

    with pytest.raises(SystemExit) as exc:
        run_smg_from_args(Namespace(), argv)
    assert exc.value.code == 0
    return captured


def test_run_smg_from_args_reads_the_template_after_prewarm(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    prewarmed = []

    def prewarm(source):
        # Stands in for the HF cache download: the template appears only now.
        prewarmed.append(source)
        _cache_hub_tokenizer(tmp_path, source, response_template=_RESPONSE_TEMPLATE)

    captured = _run_smg_from_args_with(
        monkeypatch, ["--model", "org/model", "--node-rank", "0"], prewarm
    )

    assert prewarmed == ["org/model"]
    assert captured["gateway_args"] == [
        "--model",
        "org/model",
        "--port",
        "8000",
        "--parser",
        "symphony",
        "--reasoning-parser",
        "hf",
        "--tool-call-parser",
        "hf",
        "--disable-circuit-breaker",
        "--disable-retries",
        "--disable-load-monitoring",
        "--policy",
        "passthrough",
        "--tokenizer-cache-enable-l0",
        "--tokenizer-cache-enable-l1",
        "--log-level",
        "warn",
    ]
    assert "--reasoning-parser" not in captured["engine_args"]


@pytest.mark.parametrize(
    "model_template, tokenizer_template",
    [(False, True), (True, False), (True, True)],
    ids=["tokenizer-only", "model-only", "both"],
)
def test_engine_tokenizer_is_the_gateways_template_source(
    tmp_path, monkeypatch, model_template, tokenizer_template
):
    # The gateway tokenizes with the engine's --tokenizer, so its template
    # decides the parsers, whatever --model declares.
    def tokenizer_dir(name, template):
        if template:
            return _tokenizer_dir(tmp_path / name, response_template=_RESPONSE_TEMPLATE)
        return _tokenizer_dir(tmp_path / name)

    model = tokenizer_dir("model", model_template)
    tokenizer = tokenizer_dir("tokenizer", tokenizer_template)
    prewarmed = []

    captured = _run_smg_from_args_with(
        monkeypatch,
        ["--model", model, "--tokenizer", tokenizer, "--node-rank", "0"],
        prewarmed.append,
    )

    assert prewarmed == [model]
    gateway_args = captured["gateway_args"]
    assert _get_from_args(gateway_args, "--tokenizer-path") == tokenizer
    assert _get_from_args(gateway_args, "--reasoning-parser") == (
        "hf" if tokenizer_template else "passthrough"
    )


def test_follower_node_does_not_read_the_template(tmp_path, monkeypatch, capsys):
    model = _tokenizer_dir(tmp_path, response_template=_RESPONSE_TEMPLATE)
    execs = []

    def fake_execv(path, argv):
        execs.append(argv)
        raise SystemExit(0)

    monkeypatch.setattr("tokenspeed.cli.serve_smg.os.execv", fake_execv)
    _run_smg_from_args_with(
        monkeypatch,
        ["--model", model, "--node-rank", "1"],
        lambda _: pytest.fail("followers do not prewarm"),
    )

    assert len(execs) == 1
    # A follower's ranks never register with the gateway.
    assert "--data-parallel-rpc-port" not in execs[0]
    assert "response_template" not in capsys.readouterr().out
