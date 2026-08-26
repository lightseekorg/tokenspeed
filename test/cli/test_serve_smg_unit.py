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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from tokenspeed.cli._argsplit import OrchestratorOpts
from tokenspeed.cli.serve_smg import (
    _DEFAULT_SMG_DISABLE_FLAGS,
    DEEPSEEK_V4_REASONING_PARSER,
    DEEPSEEK_V4_TOOL_CALL_PARSER,
    DEFAULT_GRPC_MAX_MESSAGE_BYTES,
    GRPC_MAX_MESSAGE_BYTES_ENV,
    INKLING_ATTENTION_BACKEND,
    INKLING_REASONING_PARSER,
    INKLING_TOOL_CALL_PARSER,
    KIMI_K3_REASONING_PARSER,
    KIMI_K3_TOOL_CALL_PARSER,
    _args_with_default_model_parsers,
    _gateway_args_with_default_log_level,
    _gateway_args_with_default_policy,
    _gateway_args_with_default_port,
    _gateway_args_with_default_prometheus_port,
    _gateway_args_with_default_reasoning_parser,
    _gateway_args_with_defaults,
    _gateway_args_with_smg_disable_defaults,
    _get_from_args,
    _is_deepseek_v4_model,
    _is_inkling_model,
    _is_kimi_k3_model,
    _prewarm_hf_tokenizer,
    _set_default_grpc_max_message_bytes,
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


def test_grpc_max_message_bytes_default_is_set_when_absent(monkeypatch):
    monkeypatch.delenv(GRPC_MAX_MESSAGE_BYTES_ENV, raising=False)

    _set_default_grpc_max_message_bytes()

    assert os.environ[GRPC_MAX_MESSAGE_BYTES_ENV] == DEFAULT_GRPC_MAX_MESSAGE_BYTES


def test_grpc_max_message_bytes_preserves_explicit_user_value(monkeypatch):
    monkeypatch.setenv(GRPC_MAX_MESSAGE_BYTES_ENV, "1048576")

    _set_default_grpc_max_message_bytes()

    assert os.environ[GRPC_MAX_MESSAGE_BYTES_ENV] == "1048576"


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


def test_smg_disable_flag_set_covers_all_three():
    assert _DEFAULT_SMG_DISABLE_FLAGS == (
        "--disable-circuit-breaker",
        "--disable-retries",
        "--disable-load-monitoring",
    )


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


def test_deepseek_v4_model_id_gets_default_parsers():
    model = "deepseek-ai/DeepSeek-V4-Flash"

    engine_args, gateway_args = _args_with_default_model_parsers(
        ["--model", model],
        ["--model", model],
    )

    assert engine_args == [
        "--model",
        model,
        "--reasoning-parser",
        DEEPSEEK_V4_REASONING_PARSER,
    ]
    assert gateway_args == [
        "--model",
        model,
        "--reasoning-parser",
        DEEPSEEK_V4_REASONING_PARSER,
        "--tool-call-parser",
        DEEPSEEK_V4_TOOL_CALL_PARSER,
    ]


def test_deepseek_v4_parser_defaults_preserve_explicit_user_values():
    model = "deepseek-ai/DeepSeek-V4-Flash"

    engine_args, gateway_args = _args_with_default_model_parsers(
        ["--model", model, "--reasoning-parser", "none"],
        [
            "--model",
            model,
            "--reasoning-parser",
            "none",
            "--tool-call-parser",
            "json",
        ],
    )

    assert engine_args == ["--model", model, "--reasoning-parser", "none"]
    assert gateway_args == [
        "--model",
        model,
        "--reasoning-parser",
        "none",
        "--tool-call-parser",
        "json",
    ]


def test_deepseek_v4_parser_defaults_fill_missing_parser_only():
    model = "deepseek-ai/DeepSeek-V4-Flash"

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
        DEEPSEEK_V4_TOOL_CALL_PARSER,
    ]

    engine_args, gateway_args = _args_with_default_model_parsers(
        ["--model", model],
        ["--model", model, "--tool-call-parser", "json"],
    )
    assert engine_args == [
        "--model",
        model,
        "--reasoning-parser",
        DEEPSEEK_V4_REASONING_PARSER,
    ]
    assert gateway_args == [
        "--model",
        model,
        "--tool-call-parser",
        "json",
        "--reasoning-parser",
        DEEPSEEK_V4_REASONING_PARSER,
    ]


def test_deepseek_v4_default_reasoning_parser_survives_gateway_defaults():
    model = "deepseek-ai/DeepSeek-V4-Flash"

    _, gateway_args = _args_with_default_model_parsers(
        ["--model", model],
        ["--model", model],
    )
    gateway_args = _gateway_args_with_defaults(gateway_args)

    # Exactly one reasoning parser, and it is the deepseek-v4 default — the
    # generic passthrough reasoning-parser default must not be layered on top.
    # Check the parser slot rather than bare membership: the value "passthrough"
    # (== DEFAULT_REASONING_PARSER) now also appears as the --policy value, which
    # is an unrelated flag.
    assert gateway_args.count("--reasoning-parser") == 1
    idx = gateway_args.index("--reasoning-parser")
    assert gateway_args[idx + 1] == DEEPSEEK_V4_REASONING_PARSER


def test_local_deepseek_v4_config_is_detected(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))

    assert _is_deepseek_v4_model(str(tmp_path))


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


def test_kimi_k3_model_is_detected_from_id_or_local_config(tmp_path):
    assert _is_kimi_k3_model("moonshotai/Kimi-K3")
    assert _is_kimi_k3_model(_make_kimi_k3_model_dir(tmp_path))
    assert not _is_kimi_k3_model("deepseek-ai/DeepSeek-V4-Flash")
    assert not _is_kimi_k3_model(None)


def test_kimi_k3_model_gets_default_parsers(tmp_path):
    model = _make_kimi_k3_model_dir(tmp_path)

    engine_args, gateway_args = _args_with_default_model_parsers(
        ["--model", model],
        ["--model", model],
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
        "--reasoning-parser",
        KIMI_K3_REASONING_PARSER,
        "--tool-call-parser",
        KIMI_K3_TOOL_CALL_PARSER,
    ]


def test_kimi_k3_parser_defaults_preserve_explicit_values(tmp_path):
    model = _make_kimi_k3_model_dir(tmp_path)
    engine_args = ["--model", model, "--reasoning-parser", "custom_reasoning"]
    gateway_args = [
        "--model",
        model,
        "--reasoning-parser",
        "custom_reasoning",
        "--tool-call-parser",
        "custom_tools",
    ]

    assert _args_with_default_model_parsers(engine_args, gateway_args) == (
        engine_args,
        gateway_args,
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
    assert not _is_inkling_model("deepseek-ai/DeepSeek-V4-Flash")
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
    assert "--no-enable-prefix-caching" not in engine_args


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


@pytest.mark.asyncio
async def test_engine_start_timeout_kills_engine_and_exits_nonzero():
    engine = _make_proc()
    opts = OrchestratorOpts(engine_startup_timeout=0)
    with patch(
        "tokenspeed.cli.serve_smg.spawn_engine", AsyncMock(return_value=engine)
    ), patch(
        "tokenspeed.cli.serve_smg.wait_grpc_serving",
        AsyncMock(side_effect=TimeoutError("engine never reached SERVING")),
    ), patch(
        "tokenspeed.cli.serve_smg.terminate_then_kill", AsyncMock()
    ) as tk:
        rc = await run_smg(
            engine_args=[],
            gateway_args=[],
            opts=opts,
            user_host="127.0.0.1",
            user_port=8000,
        )
    assert rc != 0
    tk.assert_awaited_with(engine, drain_timeout=opts.drain_timeout)


@pytest.mark.asyncio
async def test_gateway_first_then_engine_on_clean_shutdown():
    """Gateway is SIGTERMed before the engine (front before back)."""
    engine = _make_proc(returncode=0)
    gateway = _make_proc(returncode=0)
    call_order: list[str] = []

    async def tracked_term(proc, *, drain_timeout):
        if proc is gateway:
            call_order.append("gateway")
        elif proc is engine:
            call_order.append("engine")

    startup_done = asyncio.Event()

    async def engine_wait():
        await startup_done.wait()
        return 0

    async def gateway_wait():
        await startup_done.wait()
        return 0

    engine.wait = AsyncMock(side_effect=engine_wait)
    gateway.wait = AsyncMock(side_effect=gateway_wait)

    async def probe_then_schedule_release(*args, **kwargs):
        # Schedule release for the next tick so probe success is recorded
        # before gateway.wait() resolves.
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, startup_done.set)

    opts = OrchestratorOpts(engine_startup_timeout=10, gateway_startup_timeout=10)
    with patch(
        "tokenspeed.cli.serve_smg.spawn_engine", AsyncMock(return_value=engine)
    ), patch(
        "tokenspeed.cli.serve_smg.spawn_gateway", AsyncMock(return_value=gateway)
    ), patch(
        "tokenspeed.cli.serve_smg.wait_grpc_serving", AsyncMock()
    ), patch(
        "tokenspeed.cli.serve_smg.wait_http_ready",
        side_effect=probe_then_schedule_release,
    ), patch(
        "tokenspeed.cli.serve_smg.terminate_then_kill", side_effect=tracked_term
    ):
        rc = await run_smg(
            engine_args=[],
            gateway_args=[],
            opts=opts,
            user_host="127.0.0.1",
            user_port=8000,
        )
    assert call_order == ["gateway", "engine"]
    assert rc == 0


@pytest.mark.asyncio
async def test_run_smg_allocates_prometheus_port_at_gateway_spawn():
    """The metrics port is injected by run_smg right before spawn_gateway:
    a port allocated at argv-build time can be reclaimed as an ephemeral
    source port while the engine starts up."""
    engine = _make_proc(returncode=0)
    gateway = _make_proc(returncode=0)

    startup_done = asyncio.Event()

    async def wait_then_zero():
        await startup_done.wait()
        return 0

    engine.wait = AsyncMock(side_effect=wait_then_zero)
    gateway.wait = AsyncMock(side_effect=wait_then_zero)

    async def probe_then_schedule_release(*args, **kwargs):
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, startup_done.set)

    opts = OrchestratorOpts(engine_startup_timeout=10, gateway_startup_timeout=10)
    spawn_gateway_mock = AsyncMock(return_value=gateway)
    with patch(
        "tokenspeed.cli.serve_smg.spawn_engine", AsyncMock(return_value=engine)
    ), patch("tokenspeed.cli.serve_smg.spawn_gateway", spawn_gateway_mock), patch(
        "tokenspeed.cli.serve_smg.wait_grpc_serving", AsyncMock()
    ), patch(
        "tokenspeed.cli.serve_smg.wait_http_ready",
        side_effect=probe_then_schedule_release,
    ), patch(
        "tokenspeed.cli.serve_smg.terminate_then_kill", AsyncMock()
    ):
        rc = await run_smg(
            engine_args=[],
            gateway_args=["--model", "/tmp/x"],
            opts=opts,
            user_host="127.0.0.1",
            user_port=8000,
        )

    assert rc == 0
    spawned_args = spawn_gateway_mock.await_args.args[0]
    idx = spawned_args.index("--prometheus-port")
    assert spawned_args[idx + 1].isdigit()
    assert 1 <= int(spawned_args[idx + 1]) <= 65535


@pytest.mark.asyncio
async def test_signal_handlers_installed_before_spawning_engine():
    """Signal handlers must be live before any subprocess."""
    call_order: list[str] = []

    real_loop = asyncio.get_running_loop()
    real_add_signal_handler = real_loop.add_signal_handler

    def tracking_add_signal_handler(sig, callback, *args):
        call_order.append(f"add_signal_handler:{sig}")
        return real_add_signal_handler(sig, callback, *args)

    async def tracking_spawn_engine(*args, **kwargs):
        call_order.append("spawn_engine")
        raise RuntimeError("simulated spawn failure")

    opts = OrchestratorOpts()
    with patch.object(
        real_loop, "add_signal_handler", side_effect=tracking_add_signal_handler
    ), patch(
        "tokenspeed.cli.serve_smg.spawn_engine", side_effect=tracking_spawn_engine
    ):
        with pytest.raises(RuntimeError, match="simulated spawn failure"):
            await run_smg(
                engine_args=[],
                gateway_args=[],
                opts=opts,
                user_host="127.0.0.1",
                user_port=8000,
            )

    # Both signals must have their handlers registered BEFORE spawn_engine.
    assert "spawn_engine" in call_order
    spawn_idx = call_order.index("spawn_engine")
    sigterm_idx = call_order.index(f"add_signal_handler:{signal.SIGTERM}")
    sigint_idx = call_order.index(f"add_signal_handler:{signal.SIGINT}")
    assert sigterm_idx < spawn_idx, (
        f"SIGTERM handler installed after spawn_engine " f"(order: {call_order})"
    )
    assert sigint_idx < spawn_idx, (
        f"SIGINT handler installed after spawn_engine " f"(order: {call_order})"
    )


@pytest.mark.asyncio
async def test_stop_during_engine_probe_exits_zero():
    """SIGTERM during engine readiness probe is a clean exit (rc=0)."""
    engine = _make_proc(returncode=0)
    opts = OrchestratorOpts(engine_startup_timeout=10)

    async def slow_probe(*args, **kwargs):
        await asyncio.sleep(60)

    async def hung_wait():
        await asyncio.sleep(60)

    engine.wait = AsyncMock(side_effect=hung_wait)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.call_later(0.05, stop.set)

    with patch(
        "tokenspeed.cli.serve_smg.spawn_engine", AsyncMock(return_value=engine)
    ), patch(
        "tokenspeed.cli.serve_smg.wait_grpc_serving", side_effect=slow_probe
    ), patch(
        "tokenspeed.cli.serve_smg.terminate_then_kill", AsyncMock()
    ), patch(
        "tokenspeed.cli.serve_smg.kill_process_tree", lambda *a, **kw: None
    ):
        rc = await run_smg(
            engine_args=[],
            gateway_args=[],
            opts=opts,
            user_host="127.0.0.1",
            user_port=8000,
            _stop_event=stop,
        )
    assert rc == 0


@pytest.mark.asyncio
async def test_first_nonzero_child_exit_propagates():
    engine = _make_proc(returncode=42)
    gateway = _make_proc(returncode=0)
    opts = OrchestratorOpts()

    startup_done = asyncio.Event()

    async def engine_wait():
        await startup_done.wait()
        return 42

    async def gateway_wait():
        await asyncio.Event().wait()

    engine.wait = AsyncMock(side_effect=engine_wait)
    gateway.wait = AsyncMock(side_effect=gateway_wait)

    async def gateway_probe_then_release(*args, **kwargs):
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, startup_done.set)

    with patch(
        "tokenspeed.cli.serve_smg.spawn_engine", AsyncMock(return_value=engine)
    ), patch(
        "tokenspeed.cli.serve_smg.spawn_gateway", AsyncMock(return_value=gateway)
    ), patch(
        "tokenspeed.cli.serve_smg.wait_grpc_serving", AsyncMock()
    ), patch(
        "tokenspeed.cli.serve_smg.wait_http_ready",
        side_effect=gateway_probe_then_release,
    ), patch(
        "tokenspeed.cli.serve_smg.terminate_then_kill", AsyncMock()
    ):
        rc = await run_smg(
            engine_args=[],
            gateway_args=[],
            opts=opts,
            user_host="127.0.0.1",
            user_port=8000,
        )
    assert rc == 42


@pytest.mark.asyncio
async def test_engine_exit_during_probe_fails_fast():
    """If the engine exits before gRPC SERVING, return rc=1 immediately."""
    engine = _make_proc(returncode=2)
    opts = OrchestratorOpts(engine_startup_timeout=600)

    async def hung_probe(*args, **kwargs):
        await asyncio.sleep(60)

    engine.wait = AsyncMock(return_value=2)

    with patch(
        "tokenspeed.cli.serve_smg.spawn_engine", AsyncMock(return_value=engine)
    ), patch(
        "tokenspeed.cli.serve_smg.wait_grpc_serving", side_effect=hung_probe
    ), patch(
        "tokenspeed.cli.serve_smg.terminate_then_kill", AsyncMock()
    ), patch(
        "tokenspeed.cli.serve_smg.kill_process_tree", lambda *a, **kw: None
    ):
        rc = await run_smg(
            engine_args=[],
            gateway_args=[],
            opts=opts,
            user_host="127.0.0.1",
            user_port=8000,
        )
    assert rc == 1


@pytest.mark.asyncio
async def test_gateway_exit_during_probe_fails_fast():
    """If the gateway exits during wait_http_ready, return rc=1 immediately."""
    engine = _make_proc(returncode=0)
    gateway = _make_proc(returncode=2)
    opts = OrchestratorOpts(engine_startup_timeout=10, gateway_startup_timeout=600)

    async def hung_http(*args, **kwargs):
        await asyncio.sleep(60)

    gateway.wait = AsyncMock(return_value=2)

    with patch(
        "tokenspeed.cli.serve_smg.spawn_engine", AsyncMock(return_value=engine)
    ), patch(
        "tokenspeed.cli.serve_smg.spawn_gateway", AsyncMock(return_value=gateway)
    ), patch(
        "tokenspeed.cli.serve_smg.wait_grpc_serving", AsyncMock()
    ), patch(
        "tokenspeed.cli.serve_smg.wait_http_ready", side_effect=hung_http
    ), patch(
        "tokenspeed.cli.serve_smg.terminate_then_kill", AsyncMock()
    ), patch(
        "tokenspeed.cli.serve_smg.kill_process_tree", lambda *a, **kw: None
    ):
        rc = await run_smg(
            engine_args=[],
            gateway_args=[],
            opts=opts,
            user_host="127.0.0.1",
            user_port=8000,
        )
    assert rc == 1


def test_run_smg_from_args_sets_process_title(monkeypatch):
    """Orchestrator sets proc title to 'ts-serve' so pgrep -f ts-serve finds it."""
    captured = {}
    monkeypatch.delenv(GRPC_MAX_MESSAGE_BYTES_ENV, raising=False)

    def fake_run(*a, **kw):
        return 0

    monkeypatch.setattr("tokenspeed.cli.serve_smg.asyncio.run", fake_run)
    monkeypatch.setattr(
        "tokenspeed.cli.serve_smg._check_serve_extra_installed", lambda: None
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
    assert os.environ[GRPC_MAX_MESSAGE_BYTES_ENV] == DEFAULT_GRPC_MAX_MESSAGE_BYTES


def test_run_smg_from_args_applies_deepseek_v4_parser_defaults(monkeypatch):
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

    from argparse import Namespace

    from tokenspeed.cli.serve_smg import run_smg_from_args

    with pytest.raises(SystemExit) as exc:
        run_smg_from_args(Namespace(), ["deepseek-ai/DeepSeek-V4-Flash"])

    assert exc.value.code == 0
    assert captured["engine_args"] == [
        "--model",
        "deepseek-ai/DeepSeek-V4-Flash",
        "--reasoning-parser",
        DEEPSEEK_V4_REASONING_PARSER,
    ]
    assert captured["gateway_args"][:6] == [
        "--model",
        "deepseek-ai/DeepSeek-V4-Flash",
        "--reasoning-parser",
        DEEPSEEK_V4_REASONING_PARSER,
        "--tool-call-parser",
        DEEPSEEK_V4_TOOL_CALL_PARSER,
    ]
