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

from __future__ import annotations

import pytest

from tokenspeed.cli._argsplit import SplitResult, split_argv


def _split(argv: list[str]) -> SplitResult:
    return split_argv(argv)


def test_orchestrator_only_flags_are_consumed():
    r = _split(["--engine-startup-timeout", "300"])
    assert r.engine == []
    assert r.gateway == []
    assert r.opts.engine_startup_timeout == 300


def test_orchestrator_default_timeouts():
    r = _split([])
    assert r.opts.engine_startup_timeout == 1800
    assert r.opts.gateway_startup_timeout == 60
    assert r.opts.drain_timeout == 30


def test_model_fans_out_to_both():
    r = _split(["--model", "/models/qwen3-4b"])
    assert "--model" in r.engine and "/models/qwen3-4b" in r.engine
    assert "--model" in r.gateway and "/models/qwen3-4b" in r.gateway


def test_model_path_alias_also_fans_out():
    """``--model-path`` is accepted as an alias and normalizes to ``--model``."""
    r = _split(["--model-path", "/models/qwen3-4b"])
    assert "--model" in r.engine and "/models/qwen3-4b" in r.engine
    assert "--model" in r.gateway and "/models/qwen3-4b" in r.gateway
    # The alias name must not leak through.
    assert "--model-path" not in r.engine
    assert "--model-path" not in r.gateway


def test_user_host_port_go_to_gateway_only():
    r = _split(["--host", "0.0.0.0", "--port", "8000"])
    assert r.engine == []  # engine binds 127.0.0.1:<auto> internally
    assert r.gateway == ["--host", "0.0.0.0", "--port", "8000"]


def test_chat_template_overrides_to_gateway_only():
    """Both prepare_server_args and smg accept this flag; spec says gateway."""
    r = _split(["--chat-template", "/some/template.jinja"])
    assert r.engine == []
    assert r.gateway == ["--chat-template", "/some/template.jinja"]


def test_tool_call_parser_overrides_to_gateway_only():
    r = _split(["--tool-call-parser", "hermes"])
    assert r.engine == []
    assert r.gateway == ["--tool-call-parser", "hermes"]


def test_reasoning_parser_fans_out_to_both():
    # Gateway uses it for post-gen reasoning parsing; engine uses it to
    # defer JSON grammars past the reasoning channel.
    r = _split(["--reasoning-parser", "qwen3"])
    assert r.engine == ["--reasoning-parser", "qwen3"]
    assert r.gateway == ["--reasoning-parser", "qwen3"]


def test_tp_alias_routes_to_engine():
    r = _split(["--tp", "2"])
    assert r.engine == ["--tensor-parallel-size", "2"]
    assert r.gateway == []


def test_tensor_parallel_size_routes_to_engine():
    r = _split(["--tensor-parallel-size", "4"])
    assert r.engine == ["--tensor-parallel-size", "4"]
    assert r.gateway == []


def test_engine_only_flag_routes_to_engine():
    """Anything prepare_server_args accepts but smg does not."""
    r = _split(["--sampling-backend", "flashinfer"])
    assert r.engine == ["--sampling-backend", "flashinfer"]
    assert r.gateway == []


def test_unknown_flag_falls_through_to_gateway():
    """smg's clap is the final authority on these."""
    r = _split(["--policy", "cache_aware"])
    assert r.engine == []
    assert r.gateway == ["--policy", "cache_aware"]


def test_combined_real_invocation():
    argv = [
        "--model",
        "Qwen/Qwen3-30B-A3B",
        "--port",
        "8000",
        "--tp",
        "2",
        "--tool-call-parser",
        "hermes",
        "--reasoning-parser",
        "qwen3",
        "--sampling-backend",
        "flashinfer",
        "--policy",
        "cache_aware",
    ]
    r = _split(argv)
    # Engine: --model (fan-out), --tp (normalized), --reasoning-parser
    # (fan-out), --sampling-backend. Argv order is preserved.
    assert r.engine == [
        "--model",
        "Qwen/Qwen3-30B-A3B",
        "--tensor-parallel-size",
        "2",
        "--reasoning-parser",
        "qwen3",
        "--sampling-backend",
        "flashinfer",
    ]
    # Gateway: --model (fan-out), --port, parsers, --policy.
    assert r.gateway == [
        "--model",
        "Qwen/Qwen3-30B-A3B",
        "--port",
        "8000",
        "--tool-call-parser",
        "hermes",
        "--reasoning-parser",
        "qwen3",
        "--policy",
        "cache_aware",
    ]


def test_equals_form_is_normalized():
    r = _split(["--model=/models/qwen3", "--port=8000"])
    assert r.engine == ["--model", "/models/qwen3"]
    assert r.gateway == ["--model", "/models/qwen3", "--port", "8000"]


def test_valueless_flag_followed_by_equals_form():
    """Regression: --foo --bar=x must not glue =x to --bar's name."""
    r = _split(["--enable-foo", "--port=8000"])
    # --enable-foo is unknown to engine introspection, so falls through.
    assert r.gateway == ["--enable-foo", "--port", "8000"]


def test_valueless_flag_followed_by_value_flag():
    """Regression: --foo --bar value must not lose 'value'."""
    r = _split(["--enable-foo", "--port", "8000"])
    assert r.gateway == ["--enable-foo", "--port", "8000"]


def test_timeout_with_non_integer_raises():
    with pytest.raises(ValueError, match="not a valid integer"):
        _split(["--engine-startup-timeout", "abc"])


def test_timeout_with_empty_value_raises():
    with pytest.raises(ValueError, match="requires a positive integer"):
        _split(["--engine-startup-timeout="])


def test_timeout_with_zero_raises():
    with pytest.raises(ValueError, match="must be positive"):
        _split(["--engine-startup-timeout", "0"])


def test_leading_positional_is_treated_as_model():
    r = _split(
        [
            "openai/gpt-oss-20b",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--tensor-parallel-size",
            "1",
        ]
    )
    assert "--model" in r.engine and "openai/gpt-oss-20b" in r.engine
    assert "--model" in r.gateway and "openai/gpt-oss-20b" in r.gateway
    assert r.gateway[r.gateway.index("--host") + 1] == "0.0.0.0"
    assert r.gateway[r.gateway.index("--port") + 1] == "8000"
    assert r.engine[r.engine.index("--tensor-parallel-size") + 1] == "1"


def test_positional_model_matches_explicit_model_flag():
    """Both invocation styles must produce the same split."""
    positional = _split(["openai/gpt-oss-20b", "--host", "0.0.0.0"])
    explicit = _split(["--model", "openai/gpt-oss-20b", "--host", "0.0.0.0"])
    assert positional == explicit


def test_positional_and_model_flag_conflict_raises():
    with pytest.raises(ValueError, match="positional argument and via"):
        _split(["openai/gpt-oss-20b", "--model", "other/model"])


def test_positional_and_model_path_alias_conflict_raises():
    with pytest.raises(ValueError, match="positional argument and via"):
        _split(["openai/gpt-oss-20b", "--model-path", "other/model"])


def test_positional_and_model_equals_form_conflict_raises():
    with pytest.raises(ValueError, match="positional argument and via"):
        _split(["openai/gpt-oss-20b", "--model=other/model"])


def test_trailing_positional_still_raises():
    """Only the FIRST argument may be positional; later ones remain illegal."""
    with pytest.raises(ValueError, match="unexpected positional arg"):
        _split(["--host", "0.0.0.0", "stray-positional"])
