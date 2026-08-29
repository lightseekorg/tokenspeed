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
deadlock. Shapes first seen outside the window run on each library's fallback
tactics instead -- slower, never stalled.

:func:`load_flashinfer_tuning_cache` seeds the flashinfer autotuner from a
pre-swept tactic table before the window opens. Entries loaded from the
table win over live profiling, so covered shapes skip their startup tuning
pass entirely -- and every rank loading the same table picks the same tactics,
removing the rank-divergence hazard above for covered shapes.

For shapes the table does not cover, :func:`set_autotune_process_group`
averages per-tactic timings across ranks during the window, so live-tuned
shapes converge on one tactic as well.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
from collections.abc import Generator

__all__ = [
    "autotune",
    "flashinfer_tuning_cache_filename",
    "get_autotune_max_num_tokens",
    "load_flashinfer_tuning_cache",
    "load_packaged_flashinfer_tuning_cache",
    "set_autotune_max_num_tokens",
    "set_autotune_process_group",
]

logger = logging.getLogger(__name__)

_DEFAULT_AUTOTUNE_MAX_NUM_TOKENS = 8192

_autotune_max_num_tokens = _DEFAULT_AUTOTUNE_MAX_NUM_TOKENS


def flashinfer_tuning_cache_filename(
    model: str,
    ep_size: int,
    tp_size: int,
    device_name: str,
    flashinfer_version: str,
    cudnn_version: int | str,
) -> str:
    """Build the packaged FlashInfer tactic-table filename.

    Args:
        model: Model slug represented by the table.
        ep_size: Expert-parallel world size of the swept layout.
        tp_size: MoE tensor-parallel size of the swept layout.
        device_name: CUDA device name used for the sweep.
        flashinfer_version: Installed ``flashinfer-python`` version.
        cudnn_version: cuDNN version used for the sweep.

    Returns:
        The environment-specific JSON filename used by the sweeper and loader.
    """
    normalized_device_name = device_name.replace(" ", "_")
    return (
        f"{model},ep={ep_size},tp={tp_size},"
        f"device_name={normalized_device_name},flashinfer={flashinfer_version},"
        f"cudnn={cudnn_version}.json"
    )


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
        import flashinfer.autotuner
    except ImportError:
        yield
        return
    with flashinfer.autotuner.autotune():
        yield


def set_autotune_process_group(process_group) -> None:
    """Average per-tactic profile timings across ``process_group`` ranks.

    While a group is set, every rank entering :func:`autotune` all-reduces each
    tactic's measured time over the group before picking the winner, so all
    ranks converge on the same tactic despite per-GPU timing noise. Requires
    every rank in the group to profile the same tuned ops in the same order
    with identical caches at entry; set it before the tuning window opens and
    clear it (pass ``None``) after the window closes. Like :func:`autotune`, a
    no-op when the tuning backend is unavailable.

    Args:
        process_group: A ``torch.distributed`` process group covering the
            ranks that tune together (prefer a CPU/gloo group), or ``None``
            to restore independent per-rank tuning.
    """
    try:
        import flashinfer.autotuner
    except ImportError:
        return
    flashinfer.autotuner.set_autotune_process_group(process_group)


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
        mismatch. Failures log a warning and leave the startup autotune
        window to tune those shapes rather than failing startup.
    """
    try:
        from flashinfer.autotuner import AutoTuner
    except ImportError as exc:
        logger.warning(f"flashinfer tuning cache not loaded (no flashinfer): {exc}")
        return False
    try:
        loaded = AutoTuner.get().load_configs(path)
    except FileNotFoundError:
        logger.warning(
            f"flashinfer tuning cache {path} not found; the startup autotune "
            "window will tune instead"
        )
        return False
    except Exception:
        logger.warning(
            f"flashinfer tuning cache {path} failed to load; the startup "
            "autotune window will tune instead",
            exc_info=True,
        )
        return False
    if not loaded:
        # load_configs already logged the mismatch details; restate the
        # consequence at warning level so it is visible in serving logs.
        logger.warning(
            f"flashinfer tuning cache {path} rejected: environment metadata "
            "(GPU model / flashinfer / CUDA versions) does not match this "
            "host; the startup autotune window will tune instead. Re-run the "
            "MoE tactic sweeper on this environment to regenerate it."
        )
        return False
    logger.info(f"flashinfer tuning cache loaded from {path}")
    return True


def save_flashinfer_tuning_cache(path: str) -> bool:
    """Write flashinfer's in-memory tactic cache to a JSON table.

    The table records environment metadata (GPU model, flashinfer/CUDA
    versions) and can be reloaded with :func:`load_flashinfer_tuning_cache`
    on a matching host, skipping the startup autotune sweep for every shape
    it covers.

    Args:
        path: Destination JSON path; parent directories are created.

    Returns:
        True when the table was written; False when flashinfer is
        unavailable or serialization failed.
    """
    try:
        from flashinfer.autotuner import AutoTuner
    except ImportError as exc:
        logger.warning(f"flashinfer tuning cache not saved (no flashinfer): {exc}")
        return False
    try:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        AutoTuner.get().save_configs(path)
    except Exception:
        logger.warning(f"flashinfer tuning cache {path} failed to save", exc_info=True)
        return False
    logger.info(f"flashinfer tuning cache saved to {path}")
    return True


@functools.cache
def load_packaged_flashinfer_tuning_cache(
    model: str, ep_size: int, tp_size: int
) -> bool:
    """Load the in-tree tactic table for this model/layout/device, if one ships.

    Tables live as package data under ``ops/moe/flashinfer/tactics/`` named
    vLLM-configs style::

        <model>,ep=<N>,tp=<N>,device_name=<GPU>,
        flashinfer=<ver>,cudnn=<ver>.json

    EP and MoE-TP together pin the swept workload shape (local expert count
    and per-partition intermediate size), and the exact GPU device name and
    installed flashinfer and cuDNN versions are part of the name, so a lookup
    can only ever find a table swept for this precise environment; the table's
    embedded metadata re-checks the version facts at load. A miss is normal for
    layouts no table has been swept on and logs at INFO. Cached per layout: call
    sites run per-layer, the load must not.

    Args:
        model: Model slug used in the table filename (e.g. ``"kimi-k3"``).
        ep_size: Expert-parallel world size of the serving layout.
        tp_size: MoE tensor-parallel size of the serving layout.

    Returns:
        True when a matching packaged table was found and loaded.
    """
    try:
        import torch

        device_name = torch.cuda.get_device_name()
        cudnn_version = torch.backends.cudnn.version()
        if cudnn_version is None:
            raise RuntimeError("cuDNN version is unavailable")
        from importlib.metadata import version

        fi_version = version("flashinfer-python")
    except Exception as exc:
        logger.info(f"packaged tuning-cache lookup skipped: {exc}")
        return False
    from tokenspeed_kernel.ops.moe import flashinfer as _fi_pkg

    name = flashinfer_tuning_cache_filename(
        model,
        ep_size,
        tp_size,
        device_name,
        fi_version,
        cudnn_version,
    )
    tactics_dir = os.path.join(os.path.dirname(_fi_pkg.__file__), "tactics")
    path = os.path.join(tactics_dir, name)
    if not os.path.exists(path):
        # A table swept for this exact model/layout/device but a different
        # library version cannot be loaded. Keep that visible: otherwise the
        # fallback is a silent perf regression hidden in an INFO line.
        normalized_device_name = device_name.replace(" ", "_")
        stale_prefix = (
            f"{model},ep={ep_size},tp={tp_size},"
            f"device_name={normalized_device_name},"
        )
        stale = sorted(
            f
            for f in (os.listdir(tactics_dir) if os.path.isdir(tactics_dir) else [])
            if f.startswith(stale_prefix) and f != name
        )
        if stale:
            logger.warning(
                f"stale flashinfer tuning cache: {stale[0]} was swept on a "
                "different FlashInfer or cuDNN version "
                f"(installed: flashinfer={fi_version}, cudnn={cudnn_version}) "
                "and will not be loaded. Re-sweep with "
                "benchmark/moe_tactic_sweep to restore pinned tactics; until "
                "then the startup autotune window tunes these shapes."
            )
        else:
            logger.info(
                f"no packaged flashinfer tuning cache for this environment "
                f"(looked for {name}); the startup autotune window will tune "
                "instead. Sweep one with benchmark/moe_tactic_sweep to pin "
                "tactics."
            )
        return False
    return load_flashinfer_tuning_cache(path)
