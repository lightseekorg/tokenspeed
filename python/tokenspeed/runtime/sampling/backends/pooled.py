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

"""Device-neutral pool state shared by stochastic sampling backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.distributed.dp_sampling_comm import DpSamplingComm
from tokenspeed.runtime.sampling.backends.base import (
    SamplingBackend,
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.dp_sampling_config import DpSamplingRuntimeConfig
from tokenspeed.runtime.sampling.utils import coin_eps

if TYPE_CHECKING:
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams


class PooledSamplingBackend(SamplingBackend):
    """Pool-indexed scalars, graph-safe coins, packed outputs, and DP buffers.

    This class deliberately contains no sampling kernel imports. CUDA and
    Ascend backends share the state/lifecycle contract while selecting their
    kernels at module import time.
    """

    _HAS_POOL_STATE = True
    _SUPPORTS_DP_VERIFY = True

    def __init__(self, config: SamplingBackendConfig) -> None:
        super().__init__(config)
        self._init_dp_sampling(config)
        self._init_shared_buffers(config)
        self._init_pool_scalars(config)

    def _init_dp_sampling(self, config: SamplingBackendConfig) -> None:
        self._dp_tp_group = config.tp_group
        self._dp_tp_size = (
            len(self._dp_tp_group) if self._dp_tp_group is not None else 1
        )
        self._dp_rank = 0
        self._dp_comm: DpSamplingComm | None = None
        self._dp_comm_vocab_size = 0

        if self._dp_tp_size <= 1:
            self._dp_max_pad_bs = config.max_bs
            self._dp_max_reqs_per_rank = config.max_bs
            return

        self._dp_max_pad_bs = (
            (config.max_bs + self._dp_tp_size - 1) // self._dp_tp_size
        ) * self._dp_tp_size
        self._dp_max_reqs_per_rank = self._dp_max_pad_bs // self._dp_tp_size

    def configure_dp_sampling(self, runtime: DpSamplingRuntimeConfig) -> None:
        if not runtime.enabled:
            return
        if (
            runtime.vocab_size is None
            or runtime.max_bucket_bs is None
            or runtime.topology is None
            or runtime.device is None
        ):
            raise RuntimeError("enabled DP sampling runtime is incomplete")
        topology = runtime.topology
        if topology.tp_size != self._dp_tp_size:
            raise RuntimeError(
                f"DP sampling runtime tp_size={topology.tp_size} "
                f"does not match backend tp_size={self._dp_tp_size}"
            )
        if topology.tp_group != self._dp_tp_group:
            raise RuntimeError("DP sampling runtime tp_group does not match backend")
        if self._dp_tp_group is None:
            raise RuntimeError("dp_sampling requires a tp_group")
        self._dp_rank = topology.tp_rank
        if runtime.max_bucket_bs > self._dp_max_pad_bs:
            raise RuntimeError(
                f"DP sampling max_bucket_bs={runtime.max_bucket_bs} exceeds "
                f"backend max_pad_bs={self._dp_max_pad_bs}"
            )
        if runtime.vocab_size % self._dp_tp_size != 0:
            raise RuntimeError(
                f"DP sampling vocab_size={runtime.vocab_size} must be divisible by "
                f"tp_size={self._dp_tp_size}"
            )
        self._init_dp_verify_buffers(runtime.device)
        if runtime.vocab_size == self._dp_comm_vocab_size:
            return
        if self._dp_comm is not None and self._dp_comm.is_initialized:
            raise RuntimeError("Cannot resize DP sampling comm after use")
        self._dp_comm_vocab_size = runtime.vocab_size
        self._dp_comm = DpSamplingComm(
            tp_size=self._dp_tp_size,
            rank=self._dp_rank,
            group=self._dp_tp_group,
            max_pad_bs=self._dp_max_pad_bs,
            num_tokens_per_req=runtime.num_tokens_per_req,
            vocab_size=runtime.vocab_size,
            logits_dtype=None,
            device=runtime.device,
        )

    def _init_dp_verify_buffers(self, device: torch.device | str) -> None:
        if self._predict_local_buf is not None:
            return

        max_n = self.config.max_draft_tokens_per_req
        self._predict_local_buf = torch.zeros(
            (self._dp_max_reqs_per_rank * max_n,), dtype=torch.int32, device=device
        )
        self._accept_index_local_buf = torch.zeros(
            (self._dp_max_reqs_per_rank * max_n,), dtype=torch.int32, device=device
        )
        self._accept_length_local_buf = torch.zeros(
            (self._dp_max_reqs_per_rank,), dtype=torch.int32, device=device
        )

    def _init_pool_scalars(self, config: SamplingBackendConfig) -> None:
        pool_rows = config.max_req_pool_size + 1
        self._temperature_pool = torch.ones(
            (pool_rows,), dtype=torch.float32, device=config.device
        )
        self._top_k_pool = torch.ones(
            (pool_rows,), dtype=torch.int32, device=config.device
        )
        self._top_p_pool = torch.ones(
            (pool_rows,), dtype=torch.float32, device=config.device
        )
        self._seed_pool = torch.zeros(
            (pool_rows,), dtype=torch.int64, device=config.device
        )

        self._cpu_generator_per_slot: list[torch.Generator | None] = [None] * pool_rows
        self._cpu_generator_per_slot[0] = self._capture_gen

    def _reset_slot(self, pool_idx: int, sp: SamplingParams) -> None:
        self._temperature_pool[pool_idx].fill_(float(sp.temperature))
        self._top_k_pool[pool_idx].fill_(int(sp.top_k))
        self._top_p_pool[pool_idx].fill_(float(sp.top_p))
        self._seed_pool[pool_idx].fill_(int(sp.seed))

        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(int(sp.seed))
        self._cpu_generator_per_slot[pool_idx] = cpu_gen

    def _init_shared_buffers(self, config: SamplingBackendConfig) -> None:
        max_pad_bs = self._dp_max_pad_bs
        max_n = config.max_draft_tokens_per_req
        self._coins_buf = torch.zeros(
            (max_pad_bs, max_n), dtype=torch.float32, device=config.device
        )
        self._final_coins_buf = torch.zeros(
            (max_pad_bs,), dtype=torch.float32, device=config.device
        )

        self._capture_gen = self._create_capture_generator(config)
        self._capture_gen.manual_seed(config.random_seed)

        self._ones_buf = torch.ones(
            (max_pad_bs,), dtype=torch.int32, device=config.device
        )
        self._predict_max = max_pad_bs * max_n
        self._output_pack_buf = torch.zeros(
            (self._predict_max + max_pad_bs,),
            dtype=torch.int32,
            device=config.device,
        )
        self._predict_buf = self._output_pack_buf[: self._predict_max]
        self._accept_length_buf = self._output_pack_buf[self._predict_max :]
        self._accept_index_buf = torch.zeros(
            (max_pad_bs * max_n,), dtype=torch.int32, device=config.device
        )

        self._predict_local_buf: torch.Tensor | None = None
        self._accept_index_local_buf: torch.Tensor | None = None
        self._accept_length_local_buf: torch.Tensor | None = None

    def _create_capture_generator(
        self, config: SamplingBackendConfig
    ) -> torch.Generator:
        return torch.Generator(device=config.device)

    def _fill_capture_coins(self, bs: int, n: int, lo: float) -> None:
        self._coins_buf[:bs, :n].uniform_(lo, 1.0, generator=self._capture_gen)
        self._final_coins_buf[:bs].uniform_(lo, 1.0, generator=self._capture_gen)

    def _prepare_step_hook(
        self,
        num_tokens_per_req: int,
        bs: int,
        request_pool_indices: list[int] | None = None,
    ) -> None:
        if bs <= 0:
            return

        n = min(num_tokens_per_req, self.config.max_draft_tokens_per_req)
        lo = coin_eps(self._coins_buf.dtype)
        if request_pool_indices is None:
            self._fill_capture_coins(bs, n, lo)
            return

        cpu_coins = torch.empty((bs, n), dtype=torch.float32, pin_memory=True)
        cpu_final = torch.empty((bs,), dtype=torch.float32, pin_memory=True)
        for i, pool_idx in enumerate(request_pool_indices):
            gen = self._cpu_generator_per_slot[pool_idx]
            if gen is None:
                raise RuntimeError(
                    f"sampling slot {pool_idx} was not initialized before "
                    "coin-buffer refill"
                )
            cpu_coins[i, :n].uniform_(lo, 1.0, generator=gen)
            cpu_final[i].uniform_(lo, 1.0, generator=gen)

        self._coins_buf[:bs, :n].copy_(cpu_coins, non_blocking=True)
        self._final_coins_buf[:bs].copy_(cpu_final, non_blocking=True)

    def get_packed_output_d2h(
        self,
        output_tokens: torch.Tensor,
        output_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            output_tokens.data_ptr() != self._output_pack_buf.data_ptr()
            or output_lengths.data_ptr() != self._accept_length_buf.data_ptr()
        ):
            return None
        n_t = output_tokens.numel()
        n_l = output_lengths.numel()
        size = self._predict_max + n_l
        cpu_pack = torch.empty(size, dtype=torch.int32, pin_memory=True)
        cpu_pack.copy_(self._output_pack_buf[:size], non_blocking=True)
        return (
            cpu_pack[:n_t].view(output_tokens.shape),
            cpu_pack[self._predict_max : self._predict_max + n_l],
        )
