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

"""Communication helper for Batch-DP speculative verify.

Here N is num_tokens_per_req, V is the LM-head padded communication
vocab_size, V/TP is the local vocab shard, and reqs_per_rank=pad_bs/TP.
Callers trim swapped logits back to the model config vocab size before
sampling observes token ids.

swap_batch_vocab maps each rank's full-batch vocab shard
[pad_bs * N, V/TP] to its request shard with full vocab [reqs_per_rank * N, V].

gather_verify_outputs maps per-rank verify outputs
predict_local[reqs_per_rank, N], accept_index_local[reqs_per_rank, N], and
accept_length_local[reqs_per_rank] to persistent full-batch buffers
predict_full[pad_bs, N], accept_index_full[pad_bs, N], and
accept_length_full[pad_bs].
"""

from __future__ import annotations

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
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import envs

try:
    from tokenspeed_kernel.ops.communication.triton import (
        create_dp_sampling_state,
        dp_sampling_gather,
        dp_sampling_swap,
    )
    from tokenspeed_kernel.platform import current_platform
    from torch.distributed import _symmetric_memory
except Exception:
    create_dp_sampling_state = None
    current_platform = None
    dp_sampling_gather = None
    dp_sampling_swap = None
    _symmetric_memory = None

logger = get_colorful_logger(__name__)

DpSamplingBackend = Literal["auto", "nccl", "onesided"]
_ResolvedBackend = Literal["nccl", "onesided"]

ENV_VAR = "TOKENSPEED_DP_SAMPLING_BACKEND"


def _env_override() -> DpSamplingBackend | None:
    val = envs.TOKENSPEED_DP_SAMPLING_BACKEND.get()
    if val in ("auto", "nccl", "onesided"):
        return val  # type: ignore[return-value]
    if val is not None:
        raise ValueError(f"{ENV_VAR}={val!r} must be one of 'auto'|'nccl'|'onesided'")
    return None


def _onesided_available(group: Group) -> bool:
    if len(group) <= 1:
        return False
    if (
        create_dp_sampling_state is None
        or current_platform is None
        or _symmetric_memory is None
    ):
        return False
    try:
        if not current_platform().is_nvidia:
            return False

        major, minor = torch.__version__.split("+", 1)[0].split(".")[:2]
        if (int(major), int(minor)) < (2, 10):
            return False

        return True
    except Exception:
        return False


def _resolve_backend(requested: DpSamplingBackend, group: Group) -> _ResolvedBackend:
    env = _env_override()
    requested_via_env = env is not None
    if env is not None:
        requested = env

    if requested == "nccl":
        return "nccl"
    if requested == "onesided":
        if not _onesided_available(group):
            fallback_msg = (
                f"Set {ENV_VAR}=nccl or unset {ENV_VAR} to use auto fallback."
                if requested_via_env
                else f"Set {ENV_VAR}=nccl or use backend='auto' to fall back."
            )
            raise RuntimeError(
                f"Batch-DP sampling backend='onesided' requested but the one-sided "
                f"NVLink kernel is not available for group {group}. "
                f"{fallback_msg}"
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
        logits_dtype: torch.dtype | None,
        backend: DpSamplingBackend = "auto",
        fallback_comm_backend: CommBackend | None = None,
        device: torch.device | str | None = None,
    ):
        assert tp_size >= 1, f"tp_size={tp_size}"
        assert (
            len(group) == tp_size
        ), f"group {group} has {len(group)} ranks but tp_size={tp_size}"
        assert (
            max_pad_bs % tp_size == 0
        ), f"max_pad_bs={max_pad_bs} must be divisible by tp_size={tp_size}"
        assert (
            vocab_size % tp_size == 0
        ), f"vocab_size={vocab_size} must be divisible by tp_size={tp_size}"
        assert num_tokens_per_req >= 1

        self._tp_size = tp_size
        self._rank = rank
        self._group = group
        self._max_pad_bs = max_pad_bs
        self._max_reqs_per_rank = max_pad_bs // tp_size
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

        logger.info(
            "DpSamplingComm backend=%s tp_size=%d rank=%d max_pad_bs=%d "
            "num_tokens_per_req=%d vocab_size=%d",
            self._backend,
            tp_size,
            rank,
            max_pad_bs,
            num_tokens_per_req,
            vocab_size,
        )

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
        self._logprobs_full = torch.empty(
            max_pad_bs, n, dtype=torch.float32, device=self._device
        )

        if self._backend == "nccl":
            self._combined_local_nccl: torch.Tensor | None = torch.empty(
                self._max_reqs_per_rank,
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

        if self._backend == "onesided" and self._logits_dtype is not None:
            self._init_onesided()

    @property
    def backend(self) -> _ResolvedBackend:
        return self._backend

    @property
    def fast_path_enabled(self) -> bool:
        return self._backend == "onesided"

    @property
    def max_pad_bs(self) -> int:
        return self._max_pad_bs

    @property
    def is_initialized(self) -> bool:
        return self._state is not None

    def prepare_verify_outputs(self, logits_dtype: torch.dtype) -> None:
        """Initialize one-sided state for verify-only DP sampling routes."""
        if self._backend == "onesided":
            if self._state is not None:
                return
            self._ensure_onesided_state(logits_dtype)

    def swap_batch_vocab(
        self,
        local_logits: torch.Tensor,
        *,
        pad_bs: int,
    ) -> torch.Tensor:
        """Move from vocab shards to request shards.

        Input on each rank is local_logits[pad_bs * N, V_local], where
        N=num_tokens_per_req and V_local=V/TP. Output is
        [reqs_per_rank * N, V] for this rank's reqs_per_rank=pad_bs/TP
        requests.
        Returned row local_req * N + d is global request
        rank * reqs_per_rank + local_req at draft position d.
        """
        assert pad_bs <= self._max_pad_bs, (
            f"pad_bs={pad_bs} exceeds max_pad_bs={self._max_pad_bs} "
            "(set at construction time)"
        )

        if self._backend == "onesided":
            self._ensure_onesided_state(local_logits.dtype)
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

    def gather_verify_outputs(
        self,
        predict_local: torch.Tensor,
        accept_index_local: torch.Tensor,
        accept_length_local: torch.Tensor,
        *,
        pad_bs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather local verify outputs into full padded-batch outputs.

        Inputs are predict_local[reqs_per_rank, N],
        accept_index_local[reqs_per_rank, N], and
        accept_length_local[reqs_per_rank].
        Returns predict_full[pad_bs, N], accept_index_full[pad_bs, N], and
        accept_length_full[pad_bs].
        Row r from source rank src lands at src * reqs_per_rank + r.
        Callers slice the real [0:bs] prefix and ignore phantom rows.
        """
        assert pad_bs <= self._max_pad_bs
        reqs_per_rank = pad_bs // self._tp_size
        n = self._num_tokens_per_req

        assert tuple(predict_local.shape) == (
            reqs_per_rank,
            n,
        ), f"predict_local shape {tuple(predict_local.shape)} != ({reqs_per_rank}, {n})"
        assert tuple(accept_index_local.shape) == (reqs_per_rank, n), (
            f"accept_index_local shape {tuple(accept_index_local.shape)} "
            f"!= ({reqs_per_rank}, {n})"
        )
        assert tuple(accept_length_local.shape) == (reqs_per_rank,), (
            f"accept_length_local shape {tuple(accept_length_local.shape)} "
            f"!= ({reqs_per_rank},)"
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
        combined_local = self._combined_local_nccl[:reqs_per_rank]
        combined_local[:, :n].copy_(predict_local)
        combined_local[:, n : 2 * n].copy_(accept_index_local)
        combined_local[:, 2 * n].copy_(accept_length_local)

        combined_full = self._combined_full_nccl[:pad_bs]
        all_gather_into_tensor(
            combined_full,
            combined_local,
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

    def gather_verify_logprobs(
        self,
        logprobs_local: torch.Tensor,
        *,
        pad_bs: int,
    ) -> torch.Tensor:
        """Gather per-token scalar logprobs into full padded-batch order."""
        assert pad_bs <= self._max_pad_bs
        reqs_per_rank = pad_bs // self._tp_size
        n = self._num_tokens_per_req
        assert tuple(logprobs_local.shape) == (reqs_per_rank, n), (
            f"logprobs_local shape {tuple(logprobs_local.shape)} "
            f"!= ({reqs_per_rank}, {n})"
        )
        logprobs_full = self._logprobs_full[:pad_bs]
        all_gather_into_tensor(
            logprobs_full,
            logprobs_local.contiguous(),
            self._group,
            backend=self._fallback_backend,
        )
        return logprobs_full

    def _init_onesided(self) -> None:
        assert self._logits_dtype is not None
        assert create_dp_sampling_state is not None

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

    def _ensure_onesided_state(self, logits_dtype: torch.dtype) -> None:
        if self._state is not None:
            assert self._logits_dtype == logits_dtype, (
                f"DP sampling logits dtype changed from {self._logits_dtype} "
                f"to {logits_dtype}"
            )
            return
        self._logits_dtype = logits_dtype
        self._init_onesided()

    def _swap_batch_vocab_onesided(
        self, local_logits: torch.Tensor, *, pad_bs: int
    ) -> torch.Tensor:
        assert self._state is not None
        assert dp_sampling_swap is not None
        return dp_sampling_swap(self._state, local_logits, pad_bs=pad_bs)

    def _gather_verify_outputs_onesided(
        self,
        predict_local: torch.Tensor,
        accept_index_local: torch.Tensor,
        accept_length_local: torch.Tensor,
        *,
        pad_bs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._state is not None
        assert dp_sampling_gather is not None
        return dp_sampling_gather(
            self._state,
            predict_local,
            accept_index_local,
            accept_length_local,
            pad_bs=pad_bs,
        )
