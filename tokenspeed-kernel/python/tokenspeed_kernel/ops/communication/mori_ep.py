# Copyright (c) 2026 LightSeek Foundation
#
# SPDX-License-Identifier: MIT
"""AMD-native EP dispatch/combine backed by MORI (github.com/ROCm/mori).

Uses MORI's v2 op API (``mori.ops.dispatch_combine_v2``), which yields the grouped
``[num_local_experts, cap, hidden]`` layout that a masked grouped-GEMM consumes.

Validated end-to-end on gfx950 (dispatch -> convert_dispatch_output -> masked
grouped-GEMM -> convert_combine_input -> combine) vs a reference MoE: cos=1.0.

Requires the ``mori`` and ``flydsl`` packages. Selected when ``--all2all-backend mori``
(``All2AllBackend.MORI``) + expert parallel.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_MORI_AVAILABLE = None
_COMM = None  # process-wide mori.cco.Communicator singleton


def mori_available() -> bool:
    """True if MORI v2 EP ops (and flydsl) can be imported."""
    global _MORI_AVAILABLE
    if _MORI_AVAILABLE is None:
        try:
            import flydsl  # noqa: F401
            from mori.cco import Communicator  # noqa: F401
            from mori.ops.dispatch_combine_v2 import (  # noqa: F401
                EpDispatchCombineConfig,
                EpDispatchCombineOp,
            )

            _MORI_AVAILABLE = True
        except Exception as e:  # pragma: no cover
            logger.warning("MORI v2 EP not available: %s", e)
            _MORI_AVAILABLE = False
    return _MORI_AVAILABLE


def ensure_mori_comm(per_rank_vmm: int | None = None):
    """Create the process-wide MORI CCO Communicator once (uid broadcast over the
    world group). torch.distributed must already be initialized. Returns the comm."""
    global _COMM
    if _COMM is not None:
        return _COMM
    from mori.cco import Communicator

    assert (
        dist.is_initialized()
    ), "torch.distributed must be initialized before MORI comm init"
    rank = dist.get_rank()
    uid = Communicator.get_unique_id() if rank == 0 else None
    objs = [uid]
    dist.broadcast_object_list(objs, src=0)
    uid = objs[0]
    if per_rank_vmm is None:
        per_rank_vmm = int(
            os.environ.get("MORI_PER_RANK_VMM", str(4 << 30))
        )  # 4 GiB default
    _COMM = Communicator.init(
        dist.get_world_size(), rank, uid, per_rank_vmm=per_rank_vmm
    )
    return _COMM


class MoriEpDispatcher:
    """MORI v2 dispatch/combine for one MoE layer-group shape.

    Usage (mirrors the registered kernel):
        h = disp.dispatch(hidden, topk_weights, topk_ids)
        packed = h["packed_x"]                                   # [E_local, cap, H]
        # run the per-expert grouped GEMM in place (see ops/moe/mori/{bf16,mxfp4}.py)
        packed.copy_(expert_grouped_gemm(packed, h["counts"], ...))
        out = disp.combine(h)                                    # [num_tokens, H]
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        hidden_dim: int,
        num_local_experts: int,
        num_experts_per_token: int,
        max_num_inp_token_per_rank: int,
        data_type: torch.dtype = torch.bfloat16,
    ) -> None:
        assert mori_available(), "MORI v2 EP not importable"
        from mori.ops.dispatch_combine_v2 import (
            EpDispatchCombineConfig,
            EpDispatchCombineOp,
        )

        comm = ensure_mori_comm()
        self._cfg = EpDispatchCombineConfig(
            rank=rank,
            world_size=world_size,
            hidden_dim=hidden_dim,
            max_num_inp_token_per_rank=max_num_inp_token_per_rank,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=num_experts_per_token,
            data_type=data_type,
            enable_std_moe=True,
        )
        self._op = EpDispatchCombineOp(self._cfg, comm)

    def dispatch(
        self,
        hidden_states: torch.Tensor,  # [num_tokens, hidden]
        topk_weights: torch.Tensor,  # [num_tokens, topk] fp32
        topk_ids: torch.Tensor,  # [num_tokens, topk] (global expert ids)
        scales: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        recv_x, _ow, _os, _oi, _total_recv, routing = self._op.dispatch(
            hidden_states,
            topk_weights,
            scales,
            topk_ids.to(torch.int32),
            return_routing=True,
        )
        packed_x, counts, packed_src = self._op.convert_dispatch_output()
        return {
            "recv_x": recv_x,  # combine's input buffer handle
            "packed_x": packed_x,  # [E_local, cap, hidden]; write expert output here in place
            "counts": counts,  # [E_local] per-expert valid-row counts
            "routing": routing,
        }

    def combine(self, handle: dict[str, Any]) -> torch.Tensor:
        """Reduce grouped expert outputs back to origin tokens (applies top-k weights)."""
        self._op.convert_combine_input(handle["routing"])
        out, _ = self._op.combine(handle["recv_x"], routing=handle["routing"])
        return out

    def reset(self) -> None:
        self._op.reset()


# All MoE layers in a model share the same EP shape (hidden, local experts, top-k),
# so they should share ONE dispatcher/op — otherwise each of the N layers allocates
# its own MORI symmetric buffers (N x memory -> OOM on large models). Keyed by shape;
# sequential eager reuse across layers is safe (each dispatch re-populates buffers).
_DISPATCHER_CACHE: dict[tuple, "MoriEpDispatcher"] = {}


def get_dispatcher(
    rank: int,
    world_size: int,
    hidden_dim: int,
    num_local_experts: int,
    num_experts_per_token: int,
    max_num_inp_token_per_rank: int,
    data_type: torch.dtype = torch.bfloat16,
) -> "MoriEpDispatcher":
    """Return a process-wide MoriEpDispatcher for this EP shape, creating it once."""
    key = (
        rank,
        world_size,
        hidden_dim,
        num_local_experts,
        num_experts_per_token,
        max_num_inp_token_per_rank,
        str(data_type),
    )
    disp = _DISPATCHER_CACHE.get(key)
    if disp is None:
        disp = MoriEpDispatcher(
            rank=rank,
            world_size=world_size,
            hidden_dim=hidden_dim,
            num_local_experts=num_local_experts,
            num_experts_per_token=num_experts_per_token,
            max_num_inp_token_per_rank=max_num_inp_token_per_rank,
            data_type=data_type,
        )
        _DISPATCHER_CACHE[key] = disp
    return disp
