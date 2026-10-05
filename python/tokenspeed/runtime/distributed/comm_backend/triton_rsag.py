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

"""TritonRSAG communication backend for token-aware all_gather / reduce_scatter.

Handles uneven token distribution across ranks using Triton RS/AG state.
Lazily creates and caches Triton RS/AG state keyed by (group_tuple, hidden_size).
"""

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.triton import (
    all_gather,
    all_gather_inner,
    create_state,
    reduce_scatter,
    rsag_all_reduce,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend, Group
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.utils import ceil_div
from tokenspeed.runtime.utils.env import global_server_args_dict

# The multimem kernels move 16 bytes per thread: a row must hold a whole
# number of such chunks (``nvidia_rsag_get_launch_config``).
_MULTIMEM_ROW_ALIGNMENT = 8
# Rank in the group that issues the in-switch loads of the batch-invariant
# all-reduce. Fixed for the deployment: the in-switch association order
# depends on the issuer (``nvidia_rsag_all_reduce``), so moving it would move
# the bits.
MULTIMEM_ALL_REDUCE_ISSUER = 0


class TritonRSAGBackend:
    """Backend using TritonRSAG for token-aware reduce_scatter / all_gather.

    Unlike NCCL backends, TritonRSAG handles uneven token distribution
    across ranks (scattered tokens). Each instance is specific to a
    (group, hidden_size) pair because RSAG pre-allocates buffers.
    """

    def __init__(self, fallback: CommBackend):
        self._fallback = fallback
        # (group_tuple, hidden_size) -> Triton RS/AG state
        self._instances = {}

    # ---- The batch-invariant all-reduce (no fallback) ----

    @staticmethod
    def serves_multimem_all_reduce(tensor: torch.Tensor) -> bool:
        """Whether ``multimem_all_reduce`` can take ``tensor`` at all.

        Static properties of the call site only -- platform, rank, dtype,
        row width -- never the row count: the route a site takes must not
        change with the batch, so capacity is checked (and refused, not
        rerouted) inside ``multimem_all_reduce``.
        """
        return (
            current_platform().is_nvidia
            and tensor.dim() == 2
            and tensor.dtype == torch.bfloat16
            and tensor.size(-1) % _MULTIMEM_ROW_ALIGNMENT == 0
        )

    def multimem_all_reduce(self, tensor: torch.Tensor, group: Group) -> torch.Tensor:
        """In-place all-reduce through the group's fixed issuer's in-switch load.

        ``rsag_all_reduce`` with ``MULTIMEM_ALL_REDUCE_ISSUER``: one function
        of the inputs for the deployment's lifetime (see the kernel's note on
        issuer dependence). No fallback -- a payload past the RS/AG state's
        capacity is a sizing bug and raises, because the ordered fold would
        return different bits for that batch alone.
        """
        if not self.serves_multimem_all_reduce(tensor):
            raise ValueError(
                "the multimem all-reduce takes 2-D bf16 rows whose width is a "
                f"multiple of {_MULTIMEM_ROW_ALIGNMENT} on NVIDIA; got "
                f"{tuple(tensor.shape)} {tensor.dtype}"
            )
        state = self._get_or_create(group, tensor.size(-1))
        if tensor.size(0) > state.max_token_num:
            raise RuntimeError(
                f"the multimem all-reduce over group {group} was handed "
                f"{tensor.size(0)} rows, past the {state.max_token_num} its "
                "communication buffer was sized for from the launch; the "
                "batch-invariant contract forbids rerouting this batch to the fold"
            )
        reduced = rsag_all_reduce(
            state,
            tensor.contiguous(),
            issuer=MULTIMEM_ALL_REDUCE_ISSUER,
            safe=False,
        )
        tensor.copy_(reduced)
        return tensor

    def _get_or_create(self, group: Group, hidden_size: int):
        key = (group, hidden_size)
        if key in self._instances:
            return self._instances[key]

        max_num_tokens = self._get_max_num_gathered_tokens()
        state = create_state(
            enable_lamport=False,
            moe_tail_max_rows=0,
            group=pg_manager.get_process_group("nccl", group),
            rank_in_group=group.index(dist.get_rank()),
            attnres_max_numel=0,
            attnres_max_rows=0,
            max_tokens=max_num_tokens,
            hidden_size=hidden_size,
            device=None,
            max_numel=0,
            max_bytes=0,
        )
        self._instances[key] = state
        return state

    def all_gather(
        self,
        tensor: torch.Tensor,
        group: Group,
        dim: int = 0,
    ) -> torch.Tensor:
        if tensor.dim() != 2:
            return self._fallback.all_gather(tensor, group=group, dim=dim)

        if dim == 0:
            return self.token_all_gather(
                tensor,
                group=group,
                scattered_num_tokens=[tensor.size(0)] * len(group),
            )

        if (
            current_platform().is_nvidia
            and dim in (-1, tensor.dim() - 1)
            and tensor.dtype == torch.bfloat16
        ):
            hidden_size = tensor.size(-1) * len(group)
            state = self._get_or_create(group, hidden_size)
            if tensor.size(0) > state.max_token_num:
                # Rows past the prefill-sized buffer would trip the kernel's capacity assert.
                return self._fallback.all_gather(tensor, group=group, dim=dim)
            return all_gather_inner(
                state,
                tensor,
                tp_hidden_dim=hidden_size,
                skip_entry_sync=False,
                safe=False,
            )

        return self._fallback.all_gather(tensor, group=group, dim=dim)

    def token_all_gather(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        state = self._get_or_create(group, tensor.size(-1))
        # Cached history can exceed the scheduled-token budget used to size
        # this workspace. Keep captured pointers stable and use NCCL instead.
        if sum(scattered_num_tokens) > state.max_token_num:
            return self._fallback.token_all_gather(tensor, group, scattered_num_tokens)
        return all_gather(state, tensor, token_list_in_group=scattered_num_tokens)

    def token_reduce_scatter(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        state = self._get_or_create(group, tensor.size(-1))
        if sum(scattered_num_tokens) > state.max_token_num:
            return self._fallback.token_reduce_scatter(
                tensor, group, scattered_num_tokens
            )
        return reduce_scatter(state, tensor, token_list_in_group=scattered_num_tokens)

    def _get_max_num_gathered_tokens(self):
        """Cover prefill and rank-local decode/verify batches for TritonRSAG.

        global_server_args_dict read is intentional — this is one-time RSAG buffer
        init infrastructure. Passing mapping through all signatures would be too invasive.
        """
        mapping = global_server_args_dict["mapping"]
        chunked_prefill_size = global_server_args_dict["chunked_prefill_size"]
        max_prefill_tokens = global_server_args_dict["max_prefill_tokens"]
        max_model_len = global_server_args_dict["max_model_len"]
        if chunked_prefill_size > 0:
            max_attn_tp_num_tokens = chunked_prefill_size
        else:
            max_attn_tp_num_tokens = max_prefill_tokens + max_model_len
        max_decode_bs = (
            global_server_args_dict["max_num_seqs"] or 0
        ) // mapping.attn.dp_size
        decode_tokens_per_req = (
            global_server_args_dict["speculative_num_draft_tokens"]
            if global_server_args_dict.get("speculative_algorithm") is not None
            else 1
        )
        # Graph buckets are capped by this same rank-local request limit.
        # Verify expands each request even when the prefill chunk is smaller.
        max_attn_tp_num_tokens = max(
            max_attn_tp_num_tokens, max_decode_bs * decode_tokens_per_req
        )
        max_scattered_num_tokens = ceil_div(
            max_attn_tp_num_tokens, mapping.attn.tp_size
        )
        return max_scattered_num_tokens * max(
            mapping.attn.tp_size, mapping.dense.tp_size, mapping.moe.tp_ep_size
        )
