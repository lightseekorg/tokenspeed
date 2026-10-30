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

"""``ts serve`` orchestrator: spawn the smg gateway and the headless engine, tag
logs, probe readiness, and tear down gateway-first on shutdown.

``ts serve`` is the full serving command: SMG's gateway in front of the
engine on SMG's direct ZMQ lane, Rust to Rust. The gateway tokenizes, renders
and parses; it binds the engine's sockets, and the engine's ranks dial in
(``tokenspeed.cli._proc``). ``ts serve`` fronts one regular engine, so it
refuses a disaggregated role (``_engine_layout``): run each role with ``ts
serve --headless`` behind ``smg launch`` in disaggregated mode. Its sockets
are same-host ``ipc://`` unless an attention-DP replica's first rank sits on
another node; then the gateway binds them on ``--dist-init-addr``'s host
(``tcp://``), and every node's ranks dial the port below that address's.

``ts serve --headless`` is engine-only: nothing else is spawned here — an
external frontend such as ``smg serve --backend tokenspeed --connection-mode
zmq`` or Dynamo binds the msgpack ZMQ sockets and this engine dials in at
``tcp://{--data-parallel-address}:{--data-parallel-rpc-port}`` (default
``tcp://127.0.0.1:30500``).

Model-family defaults (``_args_with_default_model_parsers``) pick the
reasoning and tool-call parsers from the model. Muse Spark Open also gets the
gateway's Symphony parser (``--parser symphony``) and the engine's xgrammar
backend (``--grammar-backend xgrammar``), so forced tool calls and JSON answers
work out of the box; an explicit flag always wins, and the xgrammar default
steps aside when the engine argv selects tree drafting, which rejects every
non-``none`` grammar backend.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import dataclasses
import json
import logging
import os
import re
import signal
import socket
import sys
from pathlib import Path

from tokenspeed_kernel.platform import current_platform

from tokenspeed.cli._argsplit import OrchestratorOpts, split_argv, split_headless_argv
from tokenspeed.cli._logo import print_logo
from tokenspeed.cli._logprefix import ENGINE_TAG, GATEWAY_TAG, tag_stream
from tokenspeed.cli._proc import (
    ZMQ_LOOPBACK_HOST,
    engine_argv,
    spawn_engine,
    spawn_gateway,
    terminate_then_kill,
    wait_http_ready,
    zmq_handshake_port,
    zmq_socket_dir,
)
from tokenspeed.runtime.utils.launcher import _ephemeral_port_range, detect_topology
from tokenspeed.runtime.utils.network import get_free_port, join_host_port
from tokenspeed.runtime.utils.process import kill_process_tree

logger = logging.getLogger(__name__)

DEFAULT_GATEWAY_HOST = "0.0.0.0"
DEFAULT_GATEWAY_PORT = 8000
DEFAULT_REASONING_PARSER = "passthrough"
DEEPSEEK_V41_REASONING_PARSER = "deepseek_v41"
GLM_REASONING_PARSER = "glm45"
GLM_TOOL_CALL_PARSER = "glm47_moe"
INKLING_REASONING_PARSER = "inkling"
INKLING_TOOL_CALL_PARSER = "inkling"
INKLING_ATTENTION_BACKEND = "fa4"
KIMI_K3_REASONING_PARSER = "kimi_k3"
KIMI_K3_TOOL_CALL_PARSER = "kimi_k3"
MUSE_SPARK_REASONING_PARSER = "muse"
MUSE_SPARK_TOOL_CALL_PARSER = "muse"
MUSE_SPARK_GATEWAY_PARSER = "symphony"
MUSE_SPARK_GRAMMAR_BACKEND = "xgrammar"
DEFAULT_SMG_LOG_LEVEL = "warn"
# smg routing policy for ``ts serve``. Distinct from DEFAULT_REASONING_PARSER,
# which happens to share the "passthrough" string but configures an unrelated
# flag (--reasoning-parser).
DEFAULT_SMG_POLICY = "passthrough"
# smg reliability and background load-monitoring knobs we always want disabled
# when launched under ts serve. These are tokenspeed-internal defaults: not
# surfaced via the ts CLI, not routed through split_argv. This is independent
# from the routing policy below: disabling load monitoring stops smg's
# background engine-load polling.
_DEFAULT_SMG_DISABLE_FLAGS = (
    "--disable-circuit-breaker",
    "--disable-retries",
    "--disable-load-monitoring",
)


def _check_serve_extra_installed() -> None:
    import importlib.util

    if importlib.util.find_spec("smg") is None:
        sys.stderr.write(
            "ts serve requires the bundled gateway package, normally installed\n"
            "as part of `tokenspeed`. Reinstall tokenspeed to restore it:\n\n"
            "    pip install --force-reinstall --no-deps tokenspeed\n\n"
            "or install it explicitly:\n\n"
            "    pip install tokenspeed-smg\n\n"
            "Missing: tokenspeed-smg\n"
        )
        sys.exit(1)


def _get_from_args(
    args: list[str], flag: str, default: str | None = None
) -> str | None:
    """Return the value following ``flag`` in a split argv.

    The argv MUST be in canonical ``[--flag, value, ...]`` form as produced by
    ``split_argv``; equals-form (``--port=8000``) is not handled here.

    Args:
        args: Engine- or gateway-side argv.
        flag: Long-form flag to look up, including the leading dashes.
        default: Returned when the flag is absent or is the final token.

    Returns:
        The value following the last occurrence of ``flag``, else ``default``.
        Last-wins matches argparse, so a wrapper can append a flag to override
        one already in the argv.
    """
    try:
        index = len(args) - 1 - args[::-1].index(flag)
    except ValueError:
        return default
    if index + 1 >= len(args):
        return default
    return args[index + 1]


def _drop_from_args(args: list[str], flag: str) -> list[str]:
    """Return a copy of the argv with every ``flag value`` pair removed.

    Same canonical-form assumption as ``_get_from_args``: only space-separated
    pairs are handled.
    """
    out: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == flag:
            skip_next = True
            continue
        out.append(arg)
    return out


def _gateway_args_with_default_port(gateway_args: list[str]) -> list[str]:
    if "--port" in gateway_args:
        return gateway_args
    return [*gateway_args, "--port", str(DEFAULT_GATEWAY_PORT)]


def _gateway_args_with_default_reasoning_parser(
    gateway_args: list[str], tokenizer: str | None = None
) -> list[str]:
    """Default smg's reasoning parser to ``passthrough`` when none is named.

    Without a configured name, smg guesses a reasoning parser from the model
    id; ``passthrough`` returns the text as generated instead. A ``tokenizer``
    whose ``tokenizer_config.json`` declares a ``response_template`` selects
    the ``symphony`` parser with ``hf`` reasoning and tool parsers while no
    parser is named. Explicit parser options retain their existing behavior.
    This relies on the ``tokenspeed-smg`` pin in ``python/pyproject.toml``
    providing these parser names, as the parser names set in
    ``_args_with_default_model_parsers`` rely on it to know those names.
    """
    if "--reasoning-parser" in gateway_args:
        return gateway_args
    if "--tool-call-parser" not in gateway_args and _declares_response_template(
        tokenizer
    ):
        if "--parser" in gateway_args:
            return gateway_args
        sys.stdout.write(
            f"ts serve: {tokenizer} declares a response_template; the gateway "
            "selects it with --parser symphony and hf reasoning/tool parsers\n"
        )
        return [
            *gateway_args,
            "--parser",
            "symphony",
            "--reasoning-parser",
            "hf",
            "--tool-call-parser",
            "hf",
        ]
    return [*gateway_args, "--reasoning-parser", DEFAULT_REASONING_PARSER]


def _operator_policy(gateway_args: list[str]) -> str | None:
    """The operator's explicit ``--policy`` value (last-wins), or None.

    Call before ``_gateway_args_with_default_policy`` injects a default.
    """
    policy = None
    for i, arg in enumerate(gateway_args):
        if arg == "--policy" and i + 1 < len(gateway_args):
            policy = gateway_args[i + 1]
        elif arg.startswith("--policy="):
            policy = arg.split("=", 1)[1]
    return policy


def _gateway_args_with_smg_disable_defaults(gateway_args: list[str]) -> list[str]:
    """Append smg reliability and background-monitor disable switches when absent.

    ``--policy cache_aware`` keeps load monitoring: its imbalance/overload
    triggers feed on the worker load monitor.

    Health checks stay on: on the direct ZMQ lane the probe is a local
    liveness check, the path that reconnects a restarted engine, and the
    gateway keeps it on whatever it is told.

    NOTE: ``--dp-aware`` is deliberately NOT auto-injected — rank pinning
    needs smg releases with TokenSpeed dp-affinity support; bump the
    tokenspeed-smg* pins in lockstep before enabling it.
    """
    keep_load_monitoring = _operator_policy(gateway_args) == "cache_aware"
    result = list(gateway_args)
    for flag in _DEFAULT_SMG_DISABLE_FLAGS:
        if flag == "--disable-load-monitoring" and keep_load_monitoring:
            continue
        if flag not in result:
            result.append(flag)
    return result


def _gateway_args_with_default_policy(gateway_args: list[str]) -> list[str]:
    """Front smg's single backend with the ``passthrough`` routing policy.

    ``ts serve`` always orchestrates exactly one engine endpoint, so smg's binary
    default (``cache_aware``) is pure overhead here: it runs the load-aware worker
    monitor and subscribes to the engine's KV events. The ``passthrough`` policy
    (smg#1797) forwards every request to the single healthy worker with no
    load-aware routing or KV-event subscription. Separately,
    ``--disable-load-monitoring`` turns off smg's background engine-load polling.

    Default-when-unset: an explicit operator ``--policy`` is preserved.

    NOTE: ``--policy`` is whitelisted by smg's clap ``value_parser`` — a gateway
    that predates smg#1797 rejects ``passthrough`` and fails to start. This
    injection therefore requires a bundled ``tokenspeed-smg`` that ships smg#1797;
    the pin in ``python/pyproject.toml`` must be bumped to such a release in
    lockstep with this default.
    """
    if _operator_policy(gateway_args) is not None:
        return gateway_args
    return [*gateway_args, "--policy", DEFAULT_SMG_POLICY]


_TOKENIZER_CACHE_FLAGS = (
    "--tokenizer-cache-enable-l0",
    "--tokenizer-cache-enable-l1",
)


def _gateway_args_with_default_tokenizer_cache(gateway_args: list[str]) -> list[str]:
    """Default smg tokenizer caches (L0 + L1) ON for gateway-fronted launches.

    For agentic / chat-completions traffic with a shared system prompt + history,
    L1 prefix-caching at special-token boundaries cuts TTFT by ~30% (verified
    end-to-end on mm25). smg's own clap defaults leave both layers OFF.

    Opt-out: operators can pass ``--no-tokenizer-cache-enable-l0`` and/or
    ``--no-tokenizer-cache-enable-l1`` to ``ts serve``. The ``--no-`` form is
    intercepted here (smg's clap doesn't accept it natively) and prevents the
    positive injection for that layer.
    """
    result = list(gateway_args)
    for flag in _TOKENIZER_CACHE_FLAGS:
        no_flag = "--no-" + flag[2:]
        if no_flag in result:
            # Operator opted out: strip the --no- marker (smg rejects it)
            # and skip the positive injection for this layer.
            while no_flag in result:
                result.remove(no_flag)
            continue
        if flag not in result:
            result.append(flag)
    return result


def _gateway_args_with_default_log_level(gateway_args: list[str]) -> list[str]:
    if "--log-level" in gateway_args:
        return gateway_args
    return [*gateway_args, "--log-level", DEFAULT_SMG_LOG_LEVEL]


_FREE_PORT_MAX_ATTEMPTS = 5


def _free_port_avoiding_ephemeral_range() -> int:
    """Allocate a free TCP port, retrying one that falls in the ephemeral range.

    ``get_free_port()`` samples via ``bind(("", 0))``, which the kernel
    satisfies from the very same ephemeral range it uses to auto-assign
    *source* ports to outbound connections (health probes, HF Hub
    downloads, ...). The gap between that sample and the freshly spawned
    smg subprocess actually binding it is real (process spawn + Python/PyO3
    import + router construction), and on a host running several
    ``ts serve`` replicas at once — each doing its own outbound networking —
    an unrelated connection can grab that exact port first. smg's Rust
    listeners don't set ``SO_REUSEADDR``, so the late bind then fails with
    ``AddrInUse``, and because a dead listener aborts the whole
    ``router.start()`` call, this is fatal to the gateway regardless of
    which listener (main port, Prometheus, ...) lost the race.

    Retrying a handful of times against a *known* ephemeral range keeps the
    returned port out of it; on hosts where the range can't be read (e.g.
    non-Linux), this degrades to a plain ``get_free_port()`` call.
    """
    ephemeral = _ephemeral_port_range()
    port = get_free_port()
    if ephemeral is None:
        return port
    low, high = ephemeral
    for _ in range(_FREE_PORT_MAX_ATTEMPTS - 1):
        if not (low <= port <= high):
            return port
        port = get_free_port()
    return port


def _gateway_args_with_default_prometheus_port(gateway_args: list[str]) -> list[str]:
    """Bind the smg Prometheus exporter to a freshly allocated free port.

    smg's own default (``29000``) — and any *fixed* port — collides when
    multiple ``ts serve`` instances share a host, or when a previous run
    left the port in ``TIME_WAIT`` (smg binds without ``SO_REUSEADDR``, so
    the bind panics with ``AddrInUse`` even though no process holds it).
    A dead metrics server makes the gateway exit during startup, the
    tokenizer registration job never runs, and the first request surfaces
    ``tokenizer_not_found`` / ``no worker available``. Allocating a fresh
    free port per launch (like the engine port) avoids stale collisions, but
    only if it happens right before ``spawn_gateway``: a port picked earlier
    sits unreserved meanwhile, and the kernel can hand it to any outbound
    connection (a tokenizer prewarm, readiness probes) as an ephemeral
    source port, failing the gateway bind. Callers that need a stable scrape
    target can still pass an explicit ``--prometheus-port``.

    The port itself is drawn via ``_free_port_avoiding_ephemeral_range()``
    rather than a bare ``get_free_port()``: even allocated "right before"
    spawn, a candidate sampled from the ephemeral range can still be raced
    away by an unrelated outbound connection during subprocess startup —
    see that helper's docstring.
    """
    if "--prometheus-port" in gateway_args:
        return gateway_args
    return [
        *gateway_args,
        "--prometheus-port",
        str(_free_port_avoiding_ephemeral_range()),
    ]


def _load_model_config(model_id: str | None, filename: str = "config.json") -> dict:
    """Best-effort read of ``<model_id>/<filename>`` (empty dict on any miss)."""
    if not model_id:
        return {}
    config_path = Path(model_id) / filename
    if not config_path.is_file():
        return {}
    try:
        with config_path.open() as f:
            config = json.load(f)
    except (OSError, ValueError):
        return {}
    return config if isinstance(config, dict) else {}


def _template_tokenizer(gateway_args: list[str]) -> str | None:
    """The tokenizer whose ``response_template`` the gateway uses: its
    ``--tokenizer-path`` (the engine's ``--tokenizer``, see
    ``_gateway_args_with_engine_identity``), else its ``--model``."""
    return _get_from_args(gateway_args, "--tokenizer-path") or _get_from_args(
        gateway_args, "--model"
    )


def _gateway_args_with_engine_identity(
    engine_args: list[str], gateway_args: list[str]
) -> list[str]:
    """Name the engine's model to the gateway as the engine names it.

    Over the direct ZMQ lane the engine reports no model card: the gateway's
    worker is its ``--model``, the model id and the tokenizer source. The
    engine's ``--served-model-name`` becomes a model alias of it, and the
    engine's ``--tokenizer`` the gateway's tokenizer unless the operator gave
    the gateway its own ``--tokenizer-path``.
    """
    result = list(gateway_args)
    model = _get_from_args(gateway_args, "--model")
    served = _get_from_args(engine_args, "--served-model-name")
    if served is not None and model is not None and served != model:
        result.extend(["--model-alias", f"{served}={model}"])
    tokenizer = _get_from_args(engine_args, "--tokenizer")
    if tokenizer is not None and "--tokenizer-path" not in gateway_args:
        result.extend(["--tokenizer-path", tokenizer])
    return result


def _declares_response_template(tokenizer: str | None) -> bool:
    """Whether ``tokenizer``'s ``tokenizer_config.json`` sets ``response_template``.

    smg would read the same key, and only for a ``tokenizer.json`` tokenizer (not
    tiktoken), so the file must sit beside the config. A template is an
    object; any other value is refused by smg and counts as none here. A Hub
    id is looked up in the local HF cache only, never downloaded here
    (``run_smg_from_args`` prewarms the cache first); any miss counts as no
    template.
    """
    if not tokenizer:
        return False
    tokenizer_dir = tokenizer
    if not os.path.isdir(tokenizer_dir):
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(tokenizer, "tokenizer_config.json")
        except Exception:  # noqa: BLE001 - missing package, invalid repo id
            return False
        if not isinstance(cached, str):
            return False
        tokenizer_dir = os.path.dirname(cached)
    if not os.path.isfile(os.path.join(tokenizer_dir, "tokenizer.json")):
        return False
    config = _load_model_config(tokenizer_dir, "tokenizer_config.json")
    return isinstance(config.get("response_template"), dict)


def _is_deepseek_v41_model(model_id: str | None) -> bool:
    if not model_id:
        return False
    config = _load_model_config(model_id)
    if config.get("model_type") in {"deepseek_v41", "deepseek_v41_text"} or (
        "DeepseekV41ForCausalLM" in (config.get("architectures") or [])
    ):
        return True
    return (
        re.search(r"deepseek[-_]?v4[._-]?1(?:[-_/]|$)", model_id, re.IGNORECASE)
        is not None
    )


def _is_glm_dsa_model(model_id: str | None) -> bool:
    if not model_id:
        return False
    normalized = model_id.lower().replace("_", "-")
    if "glm-5" in normalized or "glm5" in normalized.replace("-", ""):
        return True
    config = _load_model_config(model_id)
    architectures = config.get("architectures") or []
    return config.get("model_type") == "glm_moe_dsa" or any(
        arch in {"GlmMoeDsaForCausalLM", "GlmMoeDsaForCausalLMNextN"}
        for arch in architectures
    )


def _is_inkling_model(model_id: str | None) -> bool:
    if not model_id:
        return False
    normalized = model_id.lower().replace("_", "-").rstrip("/")
    if "inkling" in normalized:
        return True
    config = _load_model_config(model_id)
    architectures = config.get("architectures") or []
    return (
        config.get("model_type") == "inkling_mm_model"
        or "InklingForConditionalGeneration" in architectures
    )


def _is_kimi_k3_model(model_id: str | None) -> bool:
    if not model_id:
        return False
    normalized = model_id.lower().replace("_", "-").rstrip("/")
    if "kimi-k3" in normalized or "kimik3" in normalized.replace("-", ""):
        return True
    config = _load_model_config(model_id)
    architectures = config.get("architectures") or []
    return config.get("model_type") == "kimi_k3" or any(
        arch
        in {
            "KimiK3ForConditionalGeneration",
            "KimiK3ForConditionalGenerationNextN",
        }
        for arch in architectures
    )


def _is_muse_spark_model(model_id: str | None) -> bool:
    if not model_id:
        return False
    normalized = model_id.lower().replace("_", "-").rstrip("/")
    if "muse-spark" in normalized or "musespark" in normalized.replace("-", ""):
        return True
    config = _load_model_config(model_id)
    architectures = config.get("architectures") or []
    return config.get("model_type") == "muse_spark" or (
        "MuseSparkForConditionalGeneration" in architectures
    )


def _args_select_tree_drafting(engine_args: list[str]) -> bool:
    """Whether the engine argv drafts a tree (``--speculative-eagle-topk`` != 1).

    Tree drafting rejects every non-``none`` grammar backend at startup
    (``ServerArgs._validate_tree_speculation``), so an injected grammar-backend
    default must step aside for it. A chain draft (the default topk of 1)
    accepts structured output and keeps the default. A non-integer topk fails
    ``ServerArgs`` parsing either way; treating it as tree drafting only
    decides which error the operator sees first.
    """
    topk = _get_from_args(engine_args, "--speculative-eagle-topk")
    if topk is None:
        return False
    try:
        return int(topk) != 1
    except ValueError:
        return True


def _args_with_default_model_parsers(
    engine_args: list[str], gateway_args: list[str]
) -> tuple[list[str], list[str]]:
    """Apply model-family parser defaults before smg gateway defaults.

    Reasoning parser defaults must be visible to both processes: smg extracts
    reasoning_content after generation, while the engine uses the same parser
    name to defer json_schema grammars past the reasoning channel.

    Muse Spark Open needs two more defaults. smg reads its ``muse`` parser
    table only under the Symphony parser, so the gateway gets ``--parser
    symphony``. Symphony constrains forced tool calls and JSON answers with
    xgrammar structural tags, which the engine's default grammar backend
    (``none``) rejects, so the engine gets ``--grammar-backend xgrammar``. As
    with the parser names, an explicit operator flag wins. The xgrammar
    default also steps aside when the engine argv selects tree drafting
    (``--speculative-eagle-topk`` != 1), which rejects every non-``none``
    grammar backend at startup; the launch keeps the pre-default ``none``
    behavior instead of failing on a flag the operator never passed.
    """
    model_id = _get_from_args(gateway_args, "--model") or _get_from_args(
        engine_args, "--model"
    )
    engine_result = list(engine_args)
    gateway_result = list(gateway_args)

    if _is_muse_spark_model(model_id):
        if (
            "--reasoning-parser" not in engine_result
            and "--reasoning-parser" not in gateway_result
        ):
            engine_result.extend(["--reasoning-parser", MUSE_SPARK_REASONING_PARSER])
            gateway_result.extend(["--reasoning-parser", MUSE_SPARK_REASONING_PARSER])
        if "--tool-call-parser" not in gateway_result:
            gateway_result.extend(["--tool-call-parser", MUSE_SPARK_TOOL_CALL_PARSER])
        if "--parser" not in gateway_result:
            gateway_result.extend(["--parser", MUSE_SPARK_GATEWAY_PARSER])
        if "--grammar-backend" not in engine_result and not _args_select_tree_drafting(
            engine_result
        ):
            engine_result.extend(["--grammar-backend", MUSE_SPARK_GRAMMAR_BACKEND])

    elif _is_deepseek_v41_model(model_id):
        if (
            "--reasoning-parser" not in engine_result
            and "--reasoning-parser" not in gateway_result
        ):
            engine_result.extend(["--reasoning-parser", DEEPSEEK_V41_REASONING_PARSER])
            gateway_result.extend(["--reasoning-parser", DEEPSEEK_V41_REASONING_PARSER])
        if "--tool-call-parser" not in gateway_result:
            gateway_result.extend(["--tool-call-parser", "deepseek_v41"])

    elif _is_glm_dsa_model(model_id):
        if (
            "--reasoning-parser" not in engine_result
            and "--reasoning-parser" not in gateway_result
        ):
            engine_result.extend(["--reasoning-parser", GLM_REASONING_PARSER])
            gateway_result.extend(["--reasoning-parser", GLM_REASONING_PARSER])
        if "--tool-call-parser" not in gateway_result:
            gateway_result.extend(["--tool-call-parser", GLM_TOOL_CALL_PARSER])

    elif _is_inkling_model(model_id):
        if (
            "--reasoning-parser" not in engine_result
            and "--reasoning-parser" not in gateway_result
        ):
            engine_result.extend(["--reasoning-parser", INKLING_REASONING_PARSER])
            gateway_result.extend(["--reasoning-parser", INKLING_REASONING_PARSER])
        if "--tool-call-parser" not in gateway_result:
            gateway_result.extend(["--tool-call-parser", INKLING_TOOL_CALL_PARSER])
        if current_platform().is_nvidia:
            # Inkling's NVIDIA rel-bias attention requires FA4; reject bad explicit backends before workers start.
            if "--attention-backend" in engine_result:
                backend_index = engine_result.index("--attention-backend")
                backend = (
                    engine_result[backend_index + 1]
                    if backend_index + 1 < len(engine_result)
                    else None
                )
                if backend != INKLING_ATTENTION_BACKEND:
                    raise ValueError(
                        "Inkling requires --attention-backend "
                        f"{INKLING_ATTENTION_BACKEND}, got {backend!r}"
                    )
            else:
                engine_result.extend(["--attention-backend", INKLING_ATTENTION_BACKEND])

    elif _is_kimi_k3_model(model_id):
        if (
            "--reasoning-parser" not in engine_result
            and "--reasoning-parser" not in gateway_result
        ):
            engine_result.extend(["--reasoning-parser", KIMI_K3_REASONING_PARSER])
            gateway_result.extend(["--reasoning-parser", KIMI_K3_REASONING_PARSER])
        if "--tool-call-parser" not in gateway_result:
            gateway_result.extend(["--tool-call-parser", KIMI_K3_TOOL_CALL_PARSER])

    return engine_result, gateway_result


def _prewarm_hf_tokenizer(model_id: str) -> None:
    """Download tokenizer and chat-template assets before the gateway boots.

    smg fires its ``AddTokenizer`` job asynchronously after the engine
    reports SERVING. On fast runners (e.g. b300) the first eval request
    can race that fetch and fail with ``tokenizer_not_found``. Pulling
    tokenizer files into the HF cache up front keeps the registration
    fast regardless of engine startup speed.
    """
    if not model_id or os.path.isdir(model_id):
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return
    try:
        snapshot_download(
            repo_id=model_id,
            allow_patterns=[
                "tokenizer*",
                "special_tokens_map*",
                "vocab*",
                "merges*",
                "chat_template*",
                "*.json",
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"HF tokenizer prewarm failed for {model_id!s}: {exc!s}")


def _gateway_args_with_defaults(
    gateway_args: list[str], tokenizer: str | None = None
) -> list[str]:
    gateway_args = _gateway_args_with_default_port(gateway_args)
    gateway_args = _gateway_args_with_default_reasoning_parser(gateway_args, tokenizer)
    gateway_args = _gateway_args_with_smg_disable_defaults(gateway_args)
    gateway_args = _gateway_args_with_default_policy(gateway_args)
    gateway_args = _gateway_args_with_default_tokenizer_cache(gateway_args)
    return _gateway_args_with_default_log_level(gateway_args)


def _add_metrics_port(engine_args: list[str]) -> tuple[list[str], str | None]:
    """Wire the engine's /metrics port when Prometheus reporting is enabled.

    With ``--enable-metrics`` and the Prometheus reporter (the default), the
    engine serves /metrics from its main process (see ``ServerArgs.metrics_port``).
    Pin a stable port so the scrape target survives engine restarts and can
    be printed at startup. An operator-pinned ``--metrics-port`` is
    preserved untouched; ``--metrics-port 0`` means auto-allocate, matching
    the engine-side treatment of an unset port. Returns the scrape URL, or
    None when Prometheus reporting is off.
    """
    reporters = []
    for index, arg in enumerate(engine_args):
        if arg == "--metrics-reporters" and index + 1 < len(engine_args):
            reporters.append(engine_args[index + 1])
        elif arg.startswith("--metrics-reporters="):
            reporters.append(arg.split("=", 1)[1])
    if "--enable-metrics" not in engine_args or (
        reporters and "prometheus" not in reporters
    ):
        return engine_args, None
    host = _get_from_args(engine_args, "--metrics-host", "127.0.0.1")
    pinned = _get_from_args(engine_args, "--metrics-port")
    if pinned is not None and int(pinned) != 0:
        return engine_args, f"http://{host}:{int(pinned)}/metrics"
    if pinned is not None:
        engine_args = _drop_from_args(engine_args, "--metrics-port")
    port = _free_port_avoiding_ephemeral_range()
    return (
        [*engine_args, "--metrics-port", str(port)],
        f"http://{host}:{port}/metrics",
    )


_HANDSHAKE_PORT_ATTEMPTS = 8


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((ZMQ_LOOPBACK_HOST, port))
        except OSError:
            return False
    return True


def _zmq_worker_url(engine_port: int) -> str:
    """The engine's ``ipc://`` worker URL on SMG's direct lane, unique to
    this launch: its handshake port, hashed from the path
    (``zmq_handshake_port``), is free now. A path whose port is taken —
    another instance's, or any listener's — is skipped for the next one."""
    base = os.path.join(zmq_socket_dir(), f"ts-serve-{engine_port}")
    for attempt in range(_HANDSHAKE_PORT_ATTEMPTS):
        url = f"ipc://{base}" if attempt == 0 else f"ipc://{base}-{attempt}"
        if _port_is_free(zmq_handshake_port(url)):
            return url
    raise RuntimeError(
        f"every handshake port derived from ipc://{base} is in use; "
        f"tried {_HANDSHAKE_PORT_ATTEMPTS} paths"
    )


class ServeConfigError(Exception):
    """An engine configuration ``ts serve`` cannot front."""


@dataclasses.dataclass(frozen=True)
class EngineLayout:
    """What the gateway awaits of the engine.

    Attributes:
        count: The attention-DP ranks, each of which registers with the
            gateway as an engine of its one worker.
        remote_handshake: Where the gateway's handshake listens when some
            replica's first rank, the one that registers, sits on another
            node: ``(host, port)``, the host of ``--dist-init-addr`` and the
            port below its, which every node derives the same. None when
            every registering rank is on this node (an ``ipc://`` worker).
    """

    count: int
    remote_handshake: tuple[str, int] | None


def _engine_layout(engine_args: list[str]) -> EngineLayout:
    """The engine's ``EngineLayout``.

    Resolves the engine's ``ServerArgs`` as the engine will (quietly: the
    engine logs them itself) and refuses a disaggregated role before any
    process starts: ``ts serve`` fronts one regular engine.
    """
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    logging.disable(logging.INFO)
    try:
        server_args = prepare_server_args(engine_args)
    finally:
        logging.disable(logging.NOTSET)
    if server_args.disaggregation_mode != "null":
        raise ServeConfigError(
            f"--disaggregation-mode {server_args.disaggregation_mode}: ts serve "
            "fronts one regular engine; run each disaggregated role with "
            "`ts serve --headless` behind `smg launch` in disaggregated mode"
        )
    mapping = server_args.mapping
    remote = False
    for rank in range(mapping.world_size):
        rank_mapping = copy.deepcopy(mapping)
        rank_mapping.rank = rank
        # The replica's first member registers for it (``ts-host``).
        first = (
            rank_mapping.world_group[0]
            if rank_mapping.has_pp
            else rank_mapping.attn.tp_group[0]
        )
        remote = remote or first >= mapping.nprocs_per_node
    remote_handshake = None
    if remote and not server_args.emulate_rank_zero:
        host, port = server_args.dist_init_addr.rsplit(":", 1)
        # The engine derives its port cluster upward from this port on every
        # node, so the one below it is free of that cluster.
        remote_handshake = (host.strip("[]"), int(port) - 1)
    return EngineLayout(count=mapping.attn.dp_size, remote_handshake=remote_handshake)


async def _stream_to(proc, tag: str) -> None:
    await asyncio.gather(
        tag_stream(proc.stdout, tag, sys.stdout),
        tag_stream(proc.stderr, tag, sys.stderr),
    )


async def _drain_log(task: asyncio.Task, timeout: float = 2.0) -> None:
    try:
        await asyncio.wait_for(task, timeout=timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()


class _ShutdownDuringStartup(Exception):
    pass


class _ChildExitedDuringStartup(Exception):
    pass


async def _probe_or_stop(probe_coro, stop_event: asyncio.Event, *, procs: dict):
    """Race a readiness probe against the stop event and the exit of any
    subprocess in ``procs`` (label -> process).

    - probe success → return result
    - stop event   → raise ``_ShutdownDuringStartup``
    - a proc exits → raise ``_ChildExitedDuringStartup`` with its returncode
      and label
    """
    probe_task = asyncio.create_task(probe_coro)
    stop_task = asyncio.create_task(stop_event.wait())
    proc_tasks = {
        asyncio.create_task(proc.wait()): label for label, proc in procs.items()
    }
    done, pending = await asyncio.wait(
        [probe_task, stop_task, *proc_tasks], return_when=asyncio.FIRST_COMPLETED
    )
    for t in pending:
        t.cancel()
    for task, label in proc_tasks.items():
        if task in done:
            raise _ChildExitedDuringStartup(
                f"{label} subprocess exited with rc={task.result()} during "
                f"startup; see [{label}] log lines above for the cause"
            )
    if stop_task in done:
        raise _ShutdownDuringStartup()
    return probe_task.result()


async def run_smg(
    *,
    engine_args: list[str],
    gateway_args: list[str],
    opts: OrchestratorOpts,
    user_host: str,
    user_port: int,
    layout: EngineLayout,
    _stop_event: asyncio.Event | None = None,
) -> int:
    """Lifecycle loop. Returns the orchestrator's exit code.

    The gateway starts first: it binds the engine's sockets, which the
    engine's ranks dial once they loaded the model, and it is up once it
    answers ``/liveness``; the engine is up once the gateway answers
    ``/readiness``, which it does once the engine registered.
    """
    engine = None
    gateway = None
    engine_log: asyncio.Task | None = None
    gateway_log: asyncio.Task | None = None

    # Install signal handlers before spawning any subprocess so a Ctrl-C
    # during the readiness probe doesn't skip terminate_then_kill.
    stop = _stop_event if _stop_event is not None else asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass  # Windows: signal handlers via asyncio aren't supported. Out of scope.

    try:
        # Seeds the engine's derived ports (its torch.distributed store, ...)
        # and names its worker socket, apart from other instances on the host.
        engine_port = _free_port_avoiding_ephemeral_range()
        if layout.remote_handshake is not None:
            # Ranks on other nodes register: the gateway binds its handshake
            # where they reach it, and its data plane on the same host.
            handshake = layout.remote_handshake
            worker_url = f"tcp://{join_host_port(*handshake)}"
        else:
            worker_url = _zmq_worker_url(engine_port)
            handshake = (ZMQ_LOOPBACK_HOST, zmq_handshake_port(worker_url))

        # Wire the engine metrics port when Prometheus reporting is enabled.
        engine_args, metrics_url = _add_metrics_port(engine_args)

        # Allocate the metrics port only now — see
        # _gateway_args_with_default_prometheus_port for why earlier is racy.
        gateway_args = _gateway_args_with_default_prometheus_port(gateway_args)
        gateway = await spawn_gateway(
            gateway_args, worker_url=worker_url, engine_count=layout.count
        )
        gateway_log = asyncio.create_task(_stream_to(gateway, GATEWAY_TAG))

        await _probe_or_stop(
            wait_http_ready(
                f"http://{user_host}:{user_port}/liveness",
                timeout=float(opts.gateway_startup_timeout),
            ),
            stop,
            procs={GATEWAY_TAG: gateway},
        )

        engine = await spawn_engine(
            engine_args,
            host="127.0.0.1",
            port=engine_port,
            handshake=handshake,
        )
        engine_log = asyncio.create_task(_stream_to(engine, ENGINE_TAG))

        await _probe_or_stop(
            wait_http_ready(
                f"http://{user_host}:{user_port}/readiness",
                timeout=float(opts.engine_startup_timeout),
            ),
            stop,
            procs={ENGINE_TAG: engine, GATEWAY_TAG: gateway},
        )

        sys.stdout.write(f"ts serve ready on http://{user_host}:{user_port}\n")
        if metrics_url is not None:
            sys.stdout.write(f"engine metrics available at {metrics_url}\n")
        sys.stdout.flush()

        engine_wait = asyncio.create_task(engine.wait())
        gateway_wait = asyncio.create_task(gateway.wait())
        stop_wait = asyncio.create_task(stop.wait())

        done, pending = await asyncio.wait(
            [engine_wait, gateway_wait, stop_wait],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        rc_engine = engine.returncode if engine.returncode is not None else 0
        rc_gateway = gateway.returncode if gateway.returncode is not None else 0
        if rc_engine != 0:
            return rc_engine
        if rc_gateway != 0:
            return rc_gateway
        return 0

    except _ChildExitedDuringStartup as exc:
        logger.error(f"startup failed: {exc!s}")
        return 1
    except _ShutdownDuringStartup:
        logger.info("shutdown signal received during startup; exiting cleanly")
        return 0
    except TimeoutError as exc:
        logger.error(f"startup failed: {exc!s}")
        return 1
    except KeyboardInterrupt:
        logger.info("interrupted; exiting cleanly")
        return 0
    finally:
        # Shutdown order: gateway first, then engine.
        if gateway is not None:
            await terminate_then_kill(gateway, drain_timeout=opts.drain_timeout)
        if engine is not None:
            await terminate_then_kill(engine, drain_timeout=opts.drain_timeout)

        drain_tasks = [
            _drain_log(t) for t in (engine_log, gateway_log) if t is not None
        ]
        if drain_tasks:
            await asyncio.gather(*drain_tasks, return_exceptions=True)

        # Final reap: walk only the children of our two known subprocesses —
        # never os.getpid(), which under pytest would walk the test runner's
        # children and SIGKILL unrelated test fixtures.
        for proc in (engine, gateway):
            if proc is not None:
                with contextlib.suppress(Exception):
                    kill_process_tree(proc.pid, include_parent=False)


def _run_headless(argv: list[str]) -> None:
    """``ts serve --headless``: run the scheduler-only launcher in-process.

    No gateway: the external frontend owns the HTTP surface, binds the ZMQ
    sockets, and does the (de)tokenization. Reuses the ``python -m
    tokenspeed.runtime.entrypoints.engine`` implementation; ``argv`` is a
    plain ServerArgs argv.
    """
    try:
        import setproctitle

        setproctitle.setproctitle("ts-serve-headless")
    except ImportError:
        pass

    from tokenspeed.runtime.entrypoints.engine import run_scheduler_headless_from_cli

    run_scheduler_headless_from_cli(argv)


def run_smg_from_args(args: argparse.Namespace, raw_argv: list[str]) -> None:
    """Entry point called from cli/__main__.py for ``ts serve``."""
    headless_argv = split_headless_argv(raw_argv)
    if headless_argv is not None:
        # Engine-only: skip the orchestrator entirely, including its
        # bundled-gateway install check.
        _run_headless(headless_argv)
        return

    try:
        import setproctitle

        setproctitle.setproctitle("ts-serve")
    except ImportError:
        pass

    print_logo()

    _check_serve_extra_installed()
    split = split_argv(raw_argv)
    engine_args, gateway_args = _args_with_default_model_parsers(
        split.engine, split.gateway
    )

    node_rank = _get_from_args(engine_args, "--node-rank")
    if node_rank is None:
        topology = detect_topology()
        node_rank = 0 if topology is None else topology.node_rank
    else:
        node_rank = int(node_rank)

    tokenizer = None
    if node_rank == 0:
        gateway_args = _gateway_args_with_engine_identity(engine_args, gateway_args)
        model_id = _get_from_args(gateway_args, "--model")
        if model_id is not None:
            _prewarm_hf_tokenizer(model_id)
        tokenizer = _template_tokenizer(gateway_args)
    gateway_args = _gateway_args_with_defaults(gateway_args, tokenizer)
    user_host = _get_from_args(gateway_args, "--host", DEFAULT_GATEWAY_HOST)
    user_port = int(_get_from_args(gateway_args, "--port", str(DEFAULT_GATEWAY_PORT)))

    try:
        layout = _engine_layout(engine_args)
    except ServeConfigError as error:
        sys.stderr.write(f"ts serve: {error}\n")
        sys.exit(2)
    if node_rank == 0:
        rc = asyncio.run(
            run_smg(
                engine_args=engine_args,
                gateway_args=gateway_args,
                opts=split.opts,
                user_host=user_host,
                user_port=user_port,
                layout=layout,
            )
        )
        sys.exit(rc)
    else:
        # A follower node runs no gateway: it runs the engine's ranks alone,
        # whose replica leaders register with node 0's gateway over its
        # tcp:// handshake when there are any here.
        argv = engine_argv(
            engine_args,
            host=user_host,
            port=user_port,
            handshake=layout.remote_handshake,
        )
        logger.info(f"follower node: exec {' '.join(argv)}")
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(argv[0], argv)
