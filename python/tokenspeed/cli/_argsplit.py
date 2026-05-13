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

"""Argv splitter for ``ts serve``.

Routing precedence is top-down. The first matching rule wins:

1. Orchestrator-only flags (consumed, never forwarded)
2. ``--model`` (alias ``--model-path``) — fanned out to both
3. ``--host`` / ``--port`` — gateway only (user-facing)
4. ``--chat-template`` / ``--tool-call-parser`` / ``--reasoning-parser`` —
   gateway only **(override)**: ``prepare_server_args`` accepts these too,
   but in smg mode the gateway owns OpenAI-compat HTTP and parsing.
5. ``--tp`` / ``--tensor-parallel-size`` — engine only (alias normalized)
6. Anything else ``prepare_server_args`` accepts — engine only
7. Anything else — gateway (fall-through to ``smg launch`` clap)
"""

from __future__ import annotations

import argparse
import functools
from dataclasses import dataclass, field
from typing import Iterable

_ORCH_FLAGS = {
    "--engine-startup-timeout",
    "--gateway-startup-timeout",
    "--drain-timeout",
}

_FANOUT_FLAGS = {"--model"}

_ALIASES = {
    "--model-path": "--model",
    "--tp": "--tensor-parallel-size",
}

_GATEWAY_USER_FACING = {"--host", "--port"}

_GATEWAY_OVERRIDE = {
    "--chat-template",
    "--tool-call-parser",
    "--reasoning-parser",
}

_ENGINE_EXPLICIT = {"--tensor-parallel-size"}


@dataclass
class OrchestratorOpts:
    engine_startup_timeout: int = 1800
    gateway_startup_timeout: int = 60
    drain_timeout: int = 30


@dataclass
class SplitResult:
    engine: list[str] = field(default_factory=list)
    gateway: list[str] = field(default_factory=list)
    opts: OrchestratorOpts = field(default_factory=OrchestratorOpts)


def _normalize(argv: Iterable[str]) -> list[tuple[str, str | None]]:
    """Convert raw argv into a list of (name, value) pairs.

    Handles both ``--flag value`` and ``--flag=value`` forms. Aliases are
    resolved to their canonical names.
    """
    items: list[tuple[str, str | None]] = []
    tokens = list(argv)
    i = 0
    while i < len(tokens):
        raw = tokens[i]
        if not raw.startswith("--"):
            raise ValueError(f"unexpected positional arg: {raw!r}")
        if "=" in raw:
            name, _, value = raw.partition("=")
            i += 1
        else:
            name = raw
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt is None or nxt.startswith("--"):
                value = None
                i += 1
            else:
                value = nxt
                i += 2
        items.append((_ALIASES.get(name, name), value))
    return items


@functools.lru_cache(maxsize=1)
def _engine_recognized_flags() -> set[str]:
    """Snapshot the set of long-form flags accepted by ``prepare_server_args``."""
    # Lazy import: ServerArgs pulls the full runtime stack (~200ms).
    from tokenspeed.runtime.utils.server_args import ServerArgs

    parser = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(parser)
    flags: set[str] = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt.startswith("--"):
                flags.add(opt)
    flags.discard("--help")
    flags.discard("-h")
    return flags


def split_argv(argv: list[str]) -> SplitResult:
    """Split ts-serve argv into engine_args, gateway_args, orchestrator_opts.

    Raises:
        ValueError: if a flag that requires a value is provided without one
            (e.g. ``--model`` with no path), if a timeout flag is
            non-positive, or if a positional arg appears.
    """

    items = _normalize(argv)
    result = SplitResult()
    engine_flags = _engine_recognized_flags()

    for name, value in items:
        if name in _ORCH_FLAGS:
            if value is None or value == "":
                raise ValueError(f"{name} requires a positive integer (seconds)")
            try:
                seconds = int(value)
            except ValueError as e:
                raise ValueError(f"{name}={value!r} is not a valid integer") from e
            if seconds <= 0:
                raise ValueError(f"{name} must be positive, got {seconds}")
            attr = name[2:].replace("-", "_")
            setattr(result.opts, attr, seconds)
            continue

        if name in _FANOUT_FLAGS:
            if value is None:
                raise ValueError(f"{name} requires a value")
            result.engine.extend([name, value])
            result.gateway.extend([name, value])
            continue

        if name in _GATEWAY_USER_FACING:
            if value is None:
                raise ValueError(f"{name} requires a value")
            result.gateway.extend([name, value])
            continue

        if name in _GATEWAY_OVERRIDE:
            if value is None:
                raise ValueError(f"{name} requires a value")
            result.gateway.extend([name, value])
            continue

        if name in _ENGINE_EXPLICIT:
            if value is None:
                raise ValueError(f"{name} requires a value")
            result.engine.extend([name, value])
            continue

        if name in engine_flags:
            if value is not None:
                result.engine.extend([name, value])
            else:
                result.engine.append(name)
            continue

        if value is not None:
            result.gateway.extend([name, value])
        else:
            result.gateway.append(name)

    return result
