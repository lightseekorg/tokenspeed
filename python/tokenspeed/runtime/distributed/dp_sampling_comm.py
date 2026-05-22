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

"""Persistent comm context for the Batch-DP spec-sampling pipeline.

See ``bench/dp_sampling_flow.html`` for the end-to-end shape flow and
``.skills/validate-collective-refactor/SKILL.md`` for the design rationale.

This module owns the two cross-rank collective stages of DP sampling:

  * Stage 4 -- ``swap_batch_vocab``: ``[pad_bs * N, V/TP]`` (full batch,
    my vocab shard)  ->  ``[K_req * N, V]`` (my K_req requests, all
    vocab). The N draft positions of a single request always stay on
    the same rank so ``chain_speculative_sampling_target_only`` can
    walk N causally without cross-rank synchronization.

  * Stage 6 -- ``gather_verify_outputs``: three per-rank tensors
    ``predict[K_req, N]``, ``accept_index[K_req, N]``,
    ``accept_length[K_req]`` get gathered back to the full
    ``[pad_bs, N]`` / ``[pad_bs]`` shape on every rank. Real requests
    occupy the ``[0:bs]`` prefix; phantom requests in
    ``[bs:pad_bs]`` are never read downstream.

Two implementations live behind a single API:

  * ``backend="nccl"``: plain ``all_to_all_single`` + 3
    ``all_gather_into_tensor``. Safe everywhere, used for parity
    testing.

  * ``backend="onesided"``: NVLinkOneSided-style symmetric-memory put
    + flag-based release/acquire barrier. Lower launch overhead, no
    permute+contiguous on the recv path, can fold the three gathers
    into one kernel. Intra-NVLink-domain only. Construct **before**
    any CUDA graph capture; per-step ops are pure kernel launches and
    capture cleanly.

Environment override: ``TOKENSPEED_DP_SAMPLING_BACKEND={auto,nccl,onesided}``.
"""

from __future__ import annotations

import os
from typing import Literal

import torch

from tokenspeed.runtime.distributed.comm_backend import (
    CommBackend,
    Group,
    get_global_backend,
)
from tokenspeed.runtime.distributed.comm_ops import all_gather_into_tensor
from tokenspeed.runtime.distributed.dp_sampling_swap import (
    swap_batch_vocab as _swap_batch_vocab_nccl,
)

DpSamplingBackend = Literal["auto", "nccl", "onesided"]
_ResolvedBackend = Literal["nccl", "onesided"]

ENV_VAR = "TOKENSPEED_DP_SAMPLING_BACKEND"


def _env_override() -> DpSamplingBackend | None:
    val = os.environ.get(ENV_VAR)
    if val in ("auto", "nccl", "onesided"):
        return val  # type: ignore[return-value]
    if val is not None:
        raise ValueError(
            f"{ENV_VAR}={val!r} must be one of 'auto'|'nccl'|'onesided'"
        )
    return None


def _onesided_available(group: Group) -> bool:
    """Capability probe for the one-sided NVLink fast path.

    The fast path needs PyTorch symmetric memory and NVIDIA peer access.
    Any probe failure falls back to NCCL instead of making backend="auto"
    brittle across CI and older PyTorch wheels.
    """
    if len(group) <= 1:
        return False
    try:
        from tokenspeed_kernel.platform import current_platform

        if not current_platform().is_nvidia:
            return False
        # `from ... import _symmetric_memory` binds only the leaf name as a
        # local; an `import torch.distributed._symmetric_memory` would rebind
        # `torch` itself to a function-local, which then breaks the
        # `torch.__version__` access below with an UnboundLocalError that
        # this try/except would silently swallow.
        from torch.distributed import _symmetric_memory  # noqa: F401

        major, minor = torch.__version__.split("+", 1)[0].split(".")[:2]
        if (int(major), int(minor)) < (2, 10):
            return False

        return True
    except Exception:
        return False


def _resolve_backend(requested: DpSamplingBackend, group: Group) -> _ResolvedBackend:
    """Resolve ``"auto"`` against actual platform/group capability.

    Env-var ``TOKENSPEED_DP_SAMPLING_BACKEND`` takes precedence over the
    Python argument so operators can force-disable the fast path without
    code changes.
    """
    env = _env_override()
    if env is not None:
        requested = env

    if requested == "nccl":
        return "nccl"
    if requested == "onesided":
        if not _onesided_available(group):
            raise RuntimeError(
                f"dp_sampling_backend='onesided' requested but the one-sided "
                f"NVLink kernel is not available for group {group}. "
                f"Set {ENV_VAR}=nccl or backend='auto' to fall back."
            )
        return "onesided"

    return "onesided" if _onesided_available(group) else "nccl"


class DpSamplingComm:

    def __init__(
        self,
        *,
        tp_size: int,
        rank: int,
        group: Group,
        max_pad_bs: int,
        num_tokens_per_req: int,
        vocab_size: int,
        logits_dtype: torch.dtype,
        backend: DpSamplingBackend = "auto",
        fallback_comm_backend: CommBackend | None = None,
        device: torch.device | str | None = None,
    ):
        assert tp_size >= 1, f"tp_size={tp_size}"
        assert len(group) == tp_size, (
            f"group {group} has {len(group)} ranks but tp_size={tp_size}"
        )
        assert max_pad_bs % tp_size == 0, (
            f"max_pad_bs={max_pad_bs} must be divisible by tp_size={tp_size}"
        )
        assert vocab_size % tp_size == 0, (
            f"vocab_size={vocab_size} must be divisible by tp_size={tp_size}"
        )
        assert num_tokens_per_req >= 1

        self._tp_size = tp_size
        self._rank = rank
        self._group = group
        self._max_pad_bs = max_pad_bs
        self._max_k_req = max_pad_bs // tp_size
        self._num_tokens_per_req = num_tokens_per_req
        self._vocab_size = vocab_size
        self._logits_dtype = logits_dtype
        self._fallback_backend = fallback_comm_backend or get_global_backend()
        self._device = (
            torch.device(device)
            if device is not None
            else torch.device(f"cuda:{torch.cuda.current_device()}")
        )

        self._backend: _ResolvedBackend = _resolve_backend(backend, group)
        self._state = None

        n = num_tokens_per_req
        self._predict_full = torch.empty(
            max_pad_bs, n, dtype=torch.int32, device=self._device
        )
        self._accept_index_full = torch.empty(
            max_pad_bs, n, dtype=torch.int32, device=self._device
        )
        self._accept_length_full = torch.empty(
            max_pad_bs, dtype=torch.int32, device=self._device
        )

        if self._backend == "nccl":
            self._combined_local_nccl: torch.Tensor | None = torch.empty(
                self._max_k_req,
                2 * n + 1,
                dtype=torch.int32,
                device=self._device,
            )
            self._combined_full_nccl: torch.Tensor | None = torch.empty(
                max_pad_bs,
                2 * n + 1,
                dtype=torch.int32,
                device=self._device,
            )
        else:
            self._combined_local_nccl = None
            self._combined_full_nccl = None

        if self._backend == "onesided":
            self._init_onesided()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def backend(self) -> _ResolvedBackend:
        """Resolved backend (``"nccl"`` or ``"onesided"``)."""
        return self._backend

    @property
    def fast_path_enabled(self) -> bool:
        """True when the one-sided NVLink kernel is in use."""
        return self._backend == "onesided"

    @property
    def max_pad_bs(self) -> int:
        return self._max_pad_bs

    # ------------------------------------------------------------------
    # Stage 4: swap (bs <-> vocab)
    # ------------------------------------------------------------------

    def swap_batch_vocab(
        self,
        local_logits: torch.Tensor,
        *,
        pad_bs: int,
    ) -> torch.Tensor:
        """``[pad_bs * N, V/TP]`` -> ``[K_req * N, V]``.

        Rank ``r`` owns the contiguous request range
        ``[r * K_req, (r+1) * K_req)`` after the swap. The N draft
        positions of every request stay on the same rank so
        ``chain_speculative_sampling_target_only`` walks N causally
        without cross-rank synchronization.

        Args:
            local_logits: vocab-sharded logits of shape
                ``(pad_bs * num_tokens_per_req, vocab_size // tp_size)``.
            pad_bs: per-step padded batch size. Must be a multiple of
                ``tp_size`` and not exceed the ``max_pad_bs`` set at
                construction time.
        """
        assert pad_bs <= self._max_pad_bs, (
            f"pad_bs={pad_bs} exceeds max_pad_bs={self._max_pad_bs} "
            "(set at construction time)"
        )

        if self._backend == "onesided":
            return self._swap_batch_vocab_onesided(local_logits, pad_bs=pad_bs)

        return _swap_batch_vocab_nccl(
            local_logits,
            tp_size=self._tp_size,
            pad_bs=pad_bs,
            num_tokens_per_req=self._num_tokens_per_req,
            vocab_size=self._vocab_size,
            rank=self._rank,
            group=self._group,
            backend=self._fallback_backend,
        )

    # ------------------------------------------------------------------
    # Stage 6: gather per-rank verify outputs
    # ------------------------------------------------------------------

    def gather_verify_outputs(
        self,
        predict_local: torch.Tensor,
        accept_index_local: torch.Tensor,
        accept_length_local: torch.Tensor,
        *,
        pad_bs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather per-rank verify outputs back to the full padded batch.

        Returns three tensors aliased into persistent storage:
          * ``predict_full[pad_bs, N]``
          * ``accept_index_full[pad_bs, N]``
          * ``accept_length_full[pad_bs]``

        Real-request results occupy the ``[0:bs]`` prefix; the caller is
        responsible for slicing if needed. Phantom requests in
        ``[bs:pad_bs]`` are never read downstream (see
        ``_accumulate_counts`` contract in
        ``bench/dp_sampling_flow.html`` stage 6).

        Args:
            predict_local: ``[K_req, N]`` int32.
            accept_index_local: ``[K_req, N]`` int32.
            accept_length_local: ``[K_req]`` int32.
            pad_bs: per-step padded batch size, must equal
                ``K_req * tp_size`` where ``K_req`` is the local shape's
                first dim.
        """
        assert pad_bs <= self._max_pad_bs
        k_req = pad_bs // self._tp_size
        n = self._num_tokens_per_req

        assert tuple(predict_local.shape) == (k_req, n), (
            f"predict_local shape {tuple(predict_local.shape)} != ({k_req}, {n})"
        )
        assert tuple(accept_index_local.shape) == (k_req, n), (
            f"accept_index_local shape {tuple(accept_index_local.shape)} "
            f"!= ({k_req}, {n})"
        )
        assert tuple(accept_length_local.shape) == (k_req,), (
            f"accept_length_local shape {tuple(accept_length_local.shape)} "
            f"!= ({k_req},)"
        )
        assert predict_local.dtype == torch.int32
        assert accept_index_local.dtype == torch.int32
        assert accept_length_local.dtype == torch.int32

        if self._backend == "onesided":
            return self._gather_verify_outputs_onesided(
                predict_local,
                accept_index_local,
                accept_length_local,
                pad_bs=pad_bs,
            )

        assert self._combined_local_nccl is not None
        assert self._combined_full_nccl is not None
        combined_local = self._combined_local_nccl[:k_req]
        combined_local[:, :n].copy_(predict_local)
        combined_local[:, n : 2 * n].copy_(accept_index_local)
        combined_local[:, 2 * n].copy_(accept_length_local)

        combined_full = self._combined_full_nccl[:pad_bs]
        all_gather_into_tensor(
            combined_full,
            combined_local,
            self._rank,
            self._group,
            backend=self._fallback_backend,
        )

        predict_full = self._predict_full[:pad_bs]
        accept_index_full = self._accept_index_full[:pad_bs]
        accept_length_full = self._accept_length_full[:pad_bs]
        predict_full.copy_(combined_full[:, :n])
        accept_index_full.copy_(combined_full[:, n : 2 * n])
        accept_length_full.copy_(combined_full[:, 2 * n])
        return predict_full, accept_index_full, accept_length_full

    # ------------------------------------------------------------------
    # One-sided NVLink fast path
    # ------------------------------------------------------------------
    #
    # The NCCL-fallback gather buffers above are intentionally separate
    # from the symm-mem buffers owned by ``self._state``. Both expose the
    # same [pad_bs, ...] contract to callers, but the fast path returns
    # zero-copy views into peer-importable VMM storage.
    # ------------------------------------------------------------------

    def _init_onesided(self) -> None:
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg_manager,
        )
        from tokenspeed_kernel.ops.communication.dp_sampling import (
            create_dp_sampling_state,
        )

        self._state = create_dp_sampling_state(
            group=pg_manager.get_process_group("nccl", self._group),
            rank_in_group=self._rank,
            tp_size=self._tp_size,
            max_pad_bs=self._max_pad_bs,
            num_tokens_per_req=self._num_tokens_per_req,
            vocab_size=self._vocab_size,
            logits_dtype=self._logits_dtype,
            device=self._device,
        )

    def _swap_batch_vocab_onesided(
        self, local_logits: torch.Tensor, *, pad_bs: int
    ) -> torch.Tensor:
        from tokenspeed_kernel.ops.communication.dp_sampling import dp_sampling_swap

        assert self._state is not None
        return dp_sampling_swap(self._state, local_logits, pad_bs=pad_bs)

    def _gather_verify_outputs_onesided(
        self,
        predict_local: torch.Tensor,
        accept_index_local: torch.Tensor,
        accept_length_local: torch.Tensor,
        *,
        pad_bs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from tokenspeed_kernel.ops.communication.dp_sampling import dp_sampling_gather

        assert self._state is not None
        return dp_sampling_gather(
            self._state,
            predict_local,
            accept_index_local,
            accept_length_local,
            pad_bs=pad_bs,
        )
