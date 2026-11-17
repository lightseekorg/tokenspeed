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

"""Persist ``cute.compile`` binaries across processes in CuTe DSL's file cache.

cutlass-dsl keeps compiled modules in an on-disk cache (``CUTE_DSL_CACHE_DIR``),
but only for direct ``@cute.jit`` calls: ``cute.compile`` passes
``no_cache=True``, so every process recompiles every kernel it compiles ahead
of time (MLIR pipeline, ptxas). This wraps ``BaseDSL.compile_and_cache``, the
one point every compilation passes after tracing, and lets those compilations
use the same cache under a key that covers what can change the binary: the
traced module's hash (its IR, cutlass-dsl's version, the DSL's compile-affecting
environment and compile options), the source of the kernel's defining module,
the installed cutlass-dsl version, the GPU architecture and the compile options.
Tracing still runs; a hit skips lowering and ptxas.

CuTe DSL's own switches apply: ``CUTE_DSL_DISABLE_FILE_CACHING`` or
``CUTE_DSL_NO_CACHE`` turn it off (for the experimental DSL too), and keeping
PTX/CUBIN/SASS artifacts compiles afresh. Entries of every cutlass-dsl DSL go
to CuTe DSL's directory (``CUTE_DSL_CACHE_DIR``, else cutlass-dsl's default)
unless the DSL's own ``<NAME>_CACHE_DIR`` is set. cutlass-dsl writes entries
atomically (a temporary directory, then a rename) and ignores, with a warning,
an entry that fails its CRC or does not load; that kernel is recompiled.
cutlass-dsl's own helpers (``HardwareInfo``) are not cached: their traces carry
process addresses, so their keys never repeat.

``compile_and_cache`` is private to cutlass-dsl. The cache installs only on the
cutlass-dsl releases it was verified against and only when the parameters it
reads are still there; at a call whose arguments no longer look as expected,
or if keying fails, it turns itself off. Each case logs one warning and leaves
compilation to cutlass-dsl exactly as without the cache.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import inspect
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CuteCompileCacheStats",
    "cute_compile_cache_key",
    "cute_compile_cache_stats",
    "install_cute_compile_cache",
    "uninstall_cute_compile_cache",
]

logger = logging.getLogger(__name__)

# Bump when the key's composition changes, so old entries are never matched.
_KEY_SCHEMA = "tokenspeed-cute-compile-cache-v1"

# The nvidia-cutlass-dsl releases the cache was verified against, inclusive:
# 4.8.0 by its tests; 4.8.1 ships the same compile_and_cache, module hash,
# compile backend and file cache. Outside the range the cache stays off.
_TESTED_CUTLASS_DSL = ((4, 8, 0), (4, 8, 1))

# The BaseDSL.compile_and_cache parameters the cache reads or sets.
_PARAMETERS = frozenset(
    {"module", "module_hash", "function_name", "no_cache", "funcBody"}
)


@dataclass(frozen=True)
class CuteCompileCacheStats:
    """``cute.compile`` calls the cache keyed, and those that were cached."""

    keyed: int
    hits: int


def cute_compile_cache_key(
    module_hash: str,
    source_digest: str,
    cutlass_dsl_version: str,
    gpu_arch: str,
    compile_options: str,
) -> str:
    """The cache key of one compilation.

    Args:
        module_hash: cutlass-dsl's hash of the traced module (IR bytecode, its
            own version, compile-affecting environment, compile options).
        source_digest: Digest of the kernel's defining module source.
        cutlass_dsl_version: Installed ``nvidia-cutlass-dsl`` version.
        gpu_arch: Target and device architecture.
        compile_options: The DSL's compile options as a string.

    Returns:
        A hex digest naming the cache entry.
    """
    digest = hashlib.sha256()
    for part in (
        _KEY_SCHEMA,
        module_hash,
        source_digest,
        cutlass_dsl_version,
        gpu_arch,
        compile_options,
    ):
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


class _Cache:
    def __init__(
        self, previous_compile_and_cache: Any, cutlass_dsl_version: str
    ) -> None:
        self.previous = previous_compile_and_cache
        self.signature = inspect.signature(previous_compile_and_cache)
        self.lock = threading.Lock()
        self.keyed = 0
        self.hits = 0
        self.cutlass_dsl_version = cutlass_dsl_version
        self.off = False

    def compile_and_cache(self, dsl: Any, *args: Any, **kwargs: Any) -> Any:
        if self.off:
            return self.previous(dsl, *args, **kwargs)
        try:
            keyed = self._key(dsl, args, kwargs)
        except Exception as error:
            self._turn_off(f"{type(error).__name__}: {error}")
            keyed = None
        if keyed is None:
            return self.previous(dsl, *args, **kwargs)
        # Only module_hash and no_cache differ from cutlass-dsl's call: the
        # arguments of a direct @cute.jit call, which cutlass-dsl caches itself.
        bound, key, in_memory, before = keyed
        result = self.previous(*bound.args, **bound.kwargs)
        try:
            # A hit leaves the entry untouched; one that failed to load is rewritten.
            hit = in_memory or (before is not None and before == _entry_stat(dsl, key))
        except Exception as error:
            self._turn_off(f"{type(error).__name__}: {error}")
            return result
        with self.lock:
            self.keyed += 1
            self.hits += int(hit)
        logger.debug(
            f"CuTe DSL compile cache {'hit' if hit else 'miss'}: "
            f"{bound.arguments['function_name']} {key}"
        )
        return result

    def _key(
        self, dsl: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[inspect.BoundArguments, str, bool, tuple[int, int] | None] | None:
        """The call keyed for the file cache, or None where the cache does not apply."""
        bound = self.signature.bind(dsl, *args, **kwargs)
        call = bound.arguments
        no_cache = call["no_cache"]
        module_hash = call["module_hash"]
        if not isinstance(no_cache, bool) or not (
            module_hash is None or isinstance(module_hash, str)
        ):
            raise TypeError(
                "compile_and_cache called with "
                f"no_cache={no_cache!r}, module_hash={module_hash!r}"
            )
        if not self._applies(dsl, call):
            return None
        key = cute_compile_cache_key(
            dsl.get_module_hash(call["module"], call["function_name"]),
            _source_digest(call.get("funcBody")),
            self.cutlass_dsl_version,
            _gpu_arch(dsl),
            dsl.compile_options.to_str(),
        )
        call["module_hash"] = key
        call["no_cache"] = False
        return bound, key, dsl.jit_cache.get(key) is not None, _entry_stat(dsl, key)

    def _turn_off(self, reason: str) -> None:
        with self.lock:
            if self.off:
                return
            self.off = True
        logger.warning(
            f"CuTe DSL compile cache off: {reason}; cute.compile compiles "
            "every kernel afresh"
        )

    @staticmethod
    def _applies(dsl: Any, call: dict[str, Any]) -> bool:
        # Only what cute.compile forced uncached; CUTE_DSL_* switches cover every DSL.
        envar = dsl.envar
        func = call.get("funcBody")
        if (
            not call["no_cache"]
            or (func is not None and func.__module__.startswith("cutlass."))
            or call["module_hash"] is not None
            or envar.no_cache
            or envar.disable_file_caching
            or _switch_on("CUTE_DSL_NO_CACHE")
            or _switch_on("CUTE_DSL_DISABLE_FILE_CACHING")
            or envar.keep_ptx
            or envar.keep_cubin
            or envar.keep_sass
        ):
            return False
        _share_cache_dir(dsl.name)
        return True


_cache: _Cache | None = None


def install_cute_compile_cache() -> bool:
    """Let ``cute.compile`` read and write CuTe DSL's on-disk cache.

    Idempotent. Returns False where cutlass-dsl is missing and, with one
    warning, where it is a release the cache was not verified against or its
    compile entry changed; compilation is then unaffected.
    """
    global _cache
    if _cache is not None:
        return True
    try:
        from cutlass.base_dsl.dsl import BaseDSL
    except ImportError:
        return False
    previous = BaseDSL.compile_and_cache
    try:
        version = _cutlass_dsl_version()
        reason = _unsupported(BaseDSL, previous, version)
    except Exception as error:
        version, reason = "", f"{type(error).__name__}: {error}"
    if reason is not None:
        logger.warning(
            f"CuTe DSL compile cache off: {reason}; cute.compile compiles "
            "every kernel afresh"
        )
        return False
    cache = _Cache(previous, version)

    @functools.wraps(previous)
    def compile_and_cache(dsl: Any, *args: Any, **kwargs: Any) -> Any:
        return cache.compile_and_cache(dsl, *args, **kwargs)

    BaseDSL.compile_and_cache = compile_and_cache
    _cache = cache
    return True


def _unsupported(base_dsl: type, compile_and_cache: Any, version: str) -> str | None:
    """Why the cache cannot run on this cutlass-dsl, or None when it can."""
    low, high = _TESTED_CUTLASS_DSL
    release = _release(version)
    if release is None or not low <= release <= high:
        return (
            f"nvidia-cutlass-dsl {version} is outside the tested releases "
            f"{_dotted(low)} to {_dotted(high)}"
        )
    missing = _PARAMETERS - set(inspect.signature(compile_and_cache).parameters)
    if missing:
        return f"BaseDSL.compile_and_cache has no {', '.join(sorted(missing))}"
    if not callable(getattr(base_dsl, "get_module_hash", None)):
        return "BaseDSL.get_module_hash is missing"
    from cutlass.base_dsl import cache_helpers

    if not callable(getattr(cache_helpers, "get_default_generated_ir_path", None)):
        return "cutlass-dsl's get_default_generated_ir_path is missing"
    return None


def _release(version: str) -> tuple[int, int, int] | None:
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def _dotted(release: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in release)


def uninstall_cute_compile_cache() -> None:
    """Restore cutlass-dsl's ``compile_and_cache`` (tests)."""
    global _cache
    if _cache is None:
        return
    from cutlass.base_dsl.dsl import BaseDSL

    BaseDSL.compile_and_cache = _cache.previous
    _cache = None


def cute_compile_cache_stats() -> CuteCompileCacheStats | None:
    """This process's keyed compilations and hits, or None when not installed."""
    if _cache is None:
        return None
    with _cache.lock:
        return CuteCompileCacheStats(keyed=_cache.keyed, hits=_cache.hits)


def _switch_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _share_cache_dir(dsl_name: str) -> None:
    """Point a DSL without its own cache directory at CuTe DSL's.

    cutlass-dsl names the directory per DSL (``<NAME>_CACHE_DIR``); the
    experimental DSL would otherwise fall back to a temporary directory.
    """
    variable = f"{dsl_name}_CACHE_DIR"
    if dsl_name == "CUTE_DSL" or os.environ.get(variable):
        return
    from cutlass.base_dsl.cache_helpers import get_default_generated_ir_path

    os.environ[variable] = get_default_generated_ir_path("CUTE_DSL")


def _cutlass_dsl_version() -> str:
    return importlib.metadata.version("nvidia-cutlass-dsl")


@functools.lru_cache(maxsize=None)
def _file_digest(path: str) -> str:
    with open(path, "rb") as source:
        return hashlib.sha256(source.read()).hexdigest()


def _source_digest(func: Any) -> str:
    """Digest of the source file defining ``func``."""
    if func is None:
        return "no-function"
    path = inspect.getsourcefile(func)
    if path is None:
        return f"no-source:{func.__module__}.{func.__qualname__}"
    return _file_digest(path)


def _gpu_arch(dsl: Any) -> str:
    """The compile target and, where CUDA is up, the current device's capability."""
    import torch

    target = str(dsl.compile_options.gpu_arch or dsl.envar.arch or "")
    if not torch.cuda.is_available():
        return target
    major, minor = torch.cuda.get_device_capability()
    return f"{target}|sm_{major}{minor}"


def _entry_stat(dsl: Any, key: str) -> tuple[int, int] | None:
    """The entry's inode and modification time, or None when it is absent."""
    from cutlass.base_dsl.cache_helpers import get_default_generated_ir_path

    path = os.path.join(
        get_default_generated_ir_path(dsl.name), f"{dsl.name.lower()}_{key}.mlir"
    )
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_mtime_ns
