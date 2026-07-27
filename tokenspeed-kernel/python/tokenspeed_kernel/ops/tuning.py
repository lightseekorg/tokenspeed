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

"""Kernel autotune lifecycle.

Tunable kernels (the flashinfer MoE and GEMM families) profile their candidate
tactics only while the tuning window is open, and take each library's heuristic
tactic otherwise. The runtime opens exactly one window during engine startup,
around a dummy forward at the largest token count it will ever serve, and closes
it before any CUDA-graph capture -- a captured graph records the tactic chosen
at capture time, so tuning afterwards cannot change a replay.

The window is CLOSED by default. Tuning during serving would be a hazard twice
over: it blocks the launch thread for the whole profiling run, and under tensor
parallelism ranks that tune at different times miss their next collective and
deadlock. Shapes first seen after the freeze run on each library's fallback
tactics instead -- slower, never stalled.

:func:`load_flashinfer_tuning_cache` seeds the flashinfer autotuner from a
pre-swept tactic table before any lazy tuning runs. Entries loaded from the
table win over live profiling, so covered shapes skip their startup tuning
pass entirely -- and every rank loading the same table picks the same tactics,
removing the rank-divergence hazard above for covered shapes.
"""

import contextlib
import functools
import logging
import os
from collections.abc import Generator

__all__ = [
    "autotune",
    "autotune_frozen",
    "flashinfer_tuning_cache_active",
    "freeze_autotuning",
    "get_autotune_max_num_tokens",
    "load_flashinfer_tuning_cache",
    "load_packaged_flashinfer_tuning_cache",
    "set_autotune_max_num_tokens",
]

logger = logging.getLogger(__name__)

_DEFAULT_AUTOTUNE_MAX_NUM_TOKENS = 8192

_autotune_max_num_tokens = _DEFAULT_AUTOTUNE_MAX_NUM_TOKENS
_frozen = False
_tuning_cache_active = False


def flashinfer_tuning_cache_active() -> bool:
    """Whether a pre-swept tactic table has been loaded into this process.

    With a table active, ops read their tactics straight from the autotuner
    cache and must never enter a tuning context -- not even a would-be no-op
    pass: entering tuning mode does synthesis/synchronization work that is
    hazardous in the wrong phase (e.g. CUDA-graph capture), and with the
    table loaded there is nothing left to profile.
    """
    return _tuning_cache_active


def autotune_frozen() -> bool:
    """Whether lazy autotuning is frozen (engine is serving)."""
    return _frozen


def set_autotune_max_num_tokens(num_tokens: int) -> None:
    """Set the token count MoE tuning buckets are generated up to.

    Call once at startup, before :func:`autotune` is first entered. The value must stay
    constant for the process lifetime: flashinfer builds the bucket *mapper*
    from it and consults that mapper on every serving call to compute the
    tactic cache key, so a value derived from the current batch makes lookups
    resolve to the wrong bucket.

    Args:
        num_tokens: Largest token count a single forward can carry (the
            runtime's ``chunked_prefill_size``). Raised to flashinfer's own
            default floor when smaller.
    """
    global _autotune_max_num_tokens
    _autotune_max_num_tokens = max(int(num_tokens), _DEFAULT_AUTOTUNE_MAX_NUM_TOKENS)


def get_autotune_max_num_tokens() -> int:
    """Token count MoE tuning buckets are generated up to.

    Returns:
        The value set by :func:`set_autotune_max_num_tokens`, or the default floor.
    """
    return _autotune_max_num_tokens


@contextlib.contextmanager
def autotune() -> Generator[None]:
    """Enable kernel autotuning for the enclosed block, process-wide.

    Kernels invoked inside the block profile their candidate tactics and cache
    the winner per shape bucket; outside it they are a cache lookup with a
    heuristic fallback. A no-op when the tuning backend is unavailable.

    Yields:
        ``None``; tuning is disabled again when the block exits, including on
        error.
    """
    try:
        from flashinfer.autotuner import autotune as _flashinfer_autotune
    except ImportError:
        yield
        return
    with _flashinfer_autotune():
        yield


def freeze_autotuning() -> None:
    """Disallow any further lazy autotune runs (call once startup completes)."""
    global _frozen
    _frozen = True


def load_flashinfer_tuning_cache(path: str) -> bool:
    """Seed flashinfer's autotuner from a pre-swept tactic table.

    The table is a flashinfer ``save_configs()`` JSON whose ``_metadata``
    records the exact environment it was swept on (GPU device name,
    flashinfer/CUDA/cuBLAS/cuDNN versions); ``load_configs`` refuses the whole
    file on any mismatch, so a table from a different GPU model can never be
    applied silently.

    Args:
        path: Path to the JSON table (produced by the MoE tactic sweeper or
            ``AutoTuner.save_configs``).

    Returns:
        True when the table was loaded; False for every failure mode --
        missing file, unimportable flashinfer, or an environment-metadata
        mismatch. Failures log a warning and leave lazy autotuning as the
        fallback rather than failing startup.
    """
    try:
        from flashinfer.autotuner import AutoTuner
    except ImportError as exc:
        logger.warning("flashinfer tuning cache not loaded (no flashinfer): %s", exc)
        return False
    try:
        loaded = AutoTuner.get().load_configs(path)
    except FileNotFoundError:
        logger.warning(
            "flashinfer tuning cache %s not found; falling back to lazy autotuning",
            path,
        )
        return False
    except Exception:
        logger.warning(
            "flashinfer tuning cache %s failed to load; falling back to lazy "
            "autotuning",
            path,
            exc_info=True,
        )
        return False
    if not loaded:
        # load_configs already logged the mismatch details; restate the
        # consequence at warning level so it is visible in serving logs.
        logger.warning(
            "flashinfer tuning cache %s rejected: environment metadata (GPU "
            "model / flashinfer / CUDA versions) does not match this host; "
            "falling back to lazy autotuning. Re-run the MoE tactic sweeper "
            "on this environment to regenerate it.",
            path,
        )
        return False
    global _tuning_cache_active
    if _tuning_cache_active:
        # flashinfer's autotuner cache keys carry only token-side tensor
        # shapes (not expert count / intermediate size), so tables for
        # different MoE layouts collide key-for-key and the last load wins.
        logger.warning(
            "flashinfer tuning cache %s loaded on top of an earlier table; "
            "entries with identical keys were overwritten. Load exactly one "
            "table per process (one serving layout).",
            path,
        )
    _tuning_cache_active = True
    logger.info("flashinfer tuning cache loaded from %s", path)
    return True


@functools.cache
def load_packaged_flashinfer_tuning_cache(
    model: str, ep_size: int, tp_size: int
) -> bool:
    """Load the in-tree tactic table for this model/layout/device, if one ships.

    Tables live as package data under ``ops/moe/flashinfer/tactics/`` named
    vLLM-configs style::

        <model>,ep=<N>,tp=<N>,device_name=<GPU with spaces as _>,flashinfer=<ver>.json

    EP and MoE-TP together pin the swept workload shape (local expert count
    and per-partition intermediate size), and the exact GPU device name and
    installed flashinfer version are part of the name, so a lookup can only
    ever find a table swept for this precise environment; the table's embedded
    metadata re-checks the version facts at load. A miss is normal for layouts
    no table has been swept on and logs at INFO. Cached per layout: call sites
    run per-layer, the load must not.

    Args:
        model: Model slug used in the table filename (e.g. ``"kimi-k3"``).
        ep_size: Expert-parallel world size of the serving layout.
        tp_size: MoE tensor-parallel size of the serving layout.

    Returns:
        True when a matching packaged table was found and loaded.
    """
    try:
        import torch

        device_name = torch.cuda.get_device_name().replace(" ", "_")
        from importlib.metadata import version

        fi_version = version("flashinfer-python")
    except Exception as exc:
        logger.info("packaged tuning-cache lookup skipped: %s", exc)
        return False
    from tokenspeed_kernel.ops.moe import flashinfer as _fi_pkg

    name = (
        f"{model},ep={ep_size},tp={tp_size},"
        f"device_name={device_name},flashinfer={fi_version}.json"
    )
    path = os.path.join(os.path.dirname(_fi_pkg.__file__), "tactics", name)
    if not os.path.exists(path):
        logger.info(
            "no packaged flashinfer tuning cache for this environment "
            "(looked for %s); lazy autotuning will run instead. Sweep one "
            "with benchmark/moe_tactic_sweep to pin tactics.",
            name,
        )
        return False
    return load_flashinfer_tuning_cache(path)
