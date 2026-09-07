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

"""FlashInfer startup tuning policies and shared cache persistence."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import torch.distributed as dist

__all__ = [
    "autotune",
    "flashinfer_autotune_cache_path",
    "get_autotune_max_num_tokens",
    "load_flashinfer_autotune_cache",
    "save_flashinfer_autotune_cache",
    "set_autotune_max_num_tokens",
    "set_autotune_process_group",
]

logger = logging.getLogger(__name__)

_DEFAULT_AUTOTUNE_MAX_NUM_TOKENS = 8192
_AUTOTUNE_CACHE_DIR_ENV = "TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR"

_autotune_max_num_tokens = _DEFAULT_AUTOTUNE_MAX_NUM_TOKENS


@contextlib.contextmanager
def _ep_moe_candidates():
    from flashinfer.autotuner import AutoTuner

    original_choose = AutoTuner.choose_one

    def expand_tactics(original):
        def get_valid(self, tensors, profile):
            from flashinfer.fused_moe.core import MoeRunnerInputs

            native = list(original(self, tensors, profile))
            total, local = self.num_experts, self.num_local_experts
            if total <= local or total % local or self.num_fused_shared_experts:
                return native
            index = MoeRunnerInputs.idx("hidden_states")
            tokens = tensors[index].shape[0]
            effective = (tokens * local + total - 1) // total
            if effective == tokens:
                return native
            # This view is used only for enumeration, never for profiling.
            shaped = list(tensors)
            shaped[index] = tensors[index][:effective]
            seen = {tuple(tactic) for tactic in native}
            for tactic in original(self, shaped, profile):
                key = tuple(tactic)
                if key not in seen:
                    native.append(tactic)
                    seen.add(key)
            return native

        return get_valid

    def choose(tuner, custom_op, runners, tuning_config, inputs, **kwargs):
        if custom_op != "flashinfer::trtllm_fp4_block_scale_moe":
            return original_choose(
                tuner, custom_op, runners, tuning_config, inputs, **kwargs
            )
        with tuner._lock:
            # FlashInfer's FP4 MoE operation has a single MoERunner.
            runner_type = type(runners[0])
            original = runner_type.get_valid_tactics
            runner_type.get_valid_tactics = expand_tactics(original)
            try:
                return original_choose(
                    tuner, custom_op, runners, tuning_config, inputs, **kwargs
                )
            finally:
                runner_type.get_valid_tactics = original

    AutoTuner.choose_one = choose
    try:
        yield
    finally:
        AutoTuner.choose_one = original_choose


def flashinfer_autotune_cache_path(cache_key: dict[str, object]) -> str | None:
    """Build a cache path from model/layout facts and FlashInfer metadata.

    Args:
        cache_key: JSON-serializable model, backend, and parallel-layout facts.

    Returns:
        Cache filename, or None when FlashInfer is unavailable.
    """
    try:
        from flashinfer.autotuner import _collect_metadata
    except ImportError as exc:
        logger.info("persistent FlashInfer autotune cache unavailable: %s", exc)
        return None

    payload = {
        "config": cache_key,
        "environment": _collect_metadata(),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    root = os.environ.get(_AUTOTUNE_CACHE_DIR_ENV)
    if not root:
        cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        root = os.path.join(cache_home, "tokenspeed", "flashinfer-autotune")
    return os.path.join(root, digest, "autotune_configs.json")


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
def autotune(
    *,
    tune_mode: bool = True,
    tuning_buckets: tuple[int, ...] | None = None,
    round_up: bool | None = None,
) -> Generator[None]:
    """Enable kernel autotuning for the enclosed block, process-wide.

    Kernels invoked inside the block profile their candidate tactics and cache
    the winner per shape bucket; outside it they are a cache lookup with a
    heuristic fallback. A no-op when the tuning backend is unavailable.

    Args:
        tune_mode: Profile missing configs when true; lookup only when false.
        tuning_buckets: Explicit token counts overriding the native buckets.
        round_up: How inputs between explicit buckets map to them.

    Yields:
        ``None``; tuning is disabled again when the block exits, including on
        error.
    """
    try:
        import flashinfer.autotuner
    except ImportError:
        yield
        return
    if tune_mode:
        # TGV's 2-CTA tactics can fail during sustained graph execution.
        tuner = flashinfer.autotuner.AutoTuner.get()
        tuner._blocklist._invalid.setdefault("bf16_gemm::TGVRunner", set()).update(
            range(16, 29)
        )
    candidates = _ep_moe_candidates() if tune_mode else contextlib.nullcontext()
    with candidates, flashinfer.autotuner.autotune(
        tune_mode, tuning_buckets=tuning_buckets, round_up=round_up
    ):
        yield


def set_autotune_process_group(process_group) -> None:
    """Average per-tactic profile timings across ``process_group`` ranks.

    Ranks must enter tuning with identical caches and profile the same ops in
    the same order. Clear the group after tuning. A no-op without FlashInfer.

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


def _install_autotune_cache_bytes(path: str, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Multiple local ranks share the path; readers must see a complete file.
    tmp = tempfile.NamedTemporaryFile(dir=target.parent, delete=False)
    try:
        with tmp:
            tmp.write(payload)
        os.replace(tmp.name, target)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _mirror_autotune_cache(
    path: str,
    payload: bytes | None,
    process_group: dist.ProcessGroup | None,
    owner_rank: int,
) -> bool:
    if process_group is not None:
        payload_box = [payload]
        dist.broadcast_object_list(payload_box, src=owner_rank, group=process_group)
        payload = payload_box[0]
    if payload is None:
        return False
    try:
        _install_autotune_cache_bytes(path, payload)
    except OSError:
        logger.warning("Could not mirror FlashInfer cache to %s", path, exc_info=True)
        return False
    return True


def load_flashinfer_autotune_cache(
    path: str | None,
    process_group: dist.ProcessGroup | None,
    owner_rank: int,
) -> bool:
    """Mirror the owner's cache and load it consistently across ranks.

    Args:
        path: Local cache filename, or None to start without a cache.
        process_group: CPU group sharing tactics, or None for a single rank.
        owner_rank: Global rank whose file is authoritative.

    Returns:
        Whether every rank loaded the cache. Otherwise all ranks tune cold.
    """
    try:
        from flashinfer.autotuner import AutoTuner
    except ImportError:
        return False
    tuner = AutoTuner.get()
    tuner.clear_cache()
    loaded = False
    if path is not None:
        payload = None
        if process_group is None or dist.get_rank() == owner_rank:
            try:
                payload = Path(path).read_bytes()
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning(
                    "Could not read FlashInfer cache %s", path, exc_info=True
                )
        if _mirror_autotune_cache(path, payload, process_group, owner_rank):
            try:
                loaded = bool(tuner.load_configs(path))
            except (OSError, ValueError):
                logger.warning(
                    "Could not load FlashInfer cache %s", path, exc_info=True
                )
    if process_group is not None:
        rank_states = [False] * dist.get_world_size(process_group)
        dist.all_gather_object(rank_states, loaded, group=process_group)
        loaded = all(rank_states)
    if loaded:
        logger.info("loaded FlashInfer autotune cache from %s", path)
    else:
        # Partial loads must not let ranks enter different timing collectives.
        tuner.clear_cache()
    return loaded


def save_flashinfer_autotune_cache(
    path: str | None,
    process_group: dist.ProcessGroup | None,
    owner_rank: int,
) -> bool:
    """Save the owner's merged tactics and mirror the resulting file to peers.

    Args:
        path: Local cache filename, or None to skip persistence.
        process_group: CPU group sharing tactics, or None for a single rank.
        owner_rank: Global rank that writes the cache.

    Returns:
        Whether the saved cache was installed locally. Write failures are logged.
    """
    if path is None:
        return False
    try:
        from flashinfer.autotuner import AutoTuner
    except ImportError:
        return False
    if process_group is not None:
        dist.barrier(group=process_group)
    payload = None
    if process_group is None or dist.get_rank() == owner_rank:
        try:
            AutoTuner.get().save_configs(path)
            payload = Path(path).read_bytes()
        except OSError:
            logger.warning("Could not save FlashInfer cache %s", path, exc_info=True)
    return _mirror_autotune_cache(path, payload, process_group, owner_rank)
