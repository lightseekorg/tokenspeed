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
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend, Group
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.utils import ceil_div
from tokenspeed.runtime.utils.env import global_server_args_dict


def supports_triton_rsag() -> bool:
    platform = current_platform()
    # The current NVIDIA multimem implementation is qualified on Hopper and
    # datacenter Blackwell. Its SM120/SM121 launch corrupts CUDA state.
    return not (platform.is_nvidia and platform.arch_version.major == 12)


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

    @staticmethod
    def _is_supported_platform() -> bool:
        # Keep token-aware collectives on the existing NCCL uneven-token path
        # when the Triton symmetric-memory implementation is unavailable.
        return supports_triton_rsag()

    def _get_or_create(self, group: Group, hidden_size: int):
        key = (group, hidden_size)
        if key in self._instances:
            return self._instances[key]

        max_num_tokens = self._get_max_num_gathered_tokens()
        state = create_state(
            group=pg_manager.get_process_group("nccl", group),
            rank_in_group=group.index(dist.get_rank()),
            max_tokens=max_num_tokens,
            hidden_size=hidden_size,
        )
        self._instances[key] = state
        return state

    def all_gather(
        self,
        tensor: torch.Tensor,
        group: Group,
        dim: int = 0,
    ) -> torch.Tensor:
        if not self._is_supported_platform():
            return self._fallback.all_gather(tensor, group=group, dim=dim)
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
        if not self._is_supported_platform():
            return self._fallback.token_all_gather(
                tensor,
                group=group,
                scattered_num_tokens=scattered_num_tokens,
            )
        state = self._get_or_create(group, tensor.size(-1))
        return all_gather(state, tensor, token_list_in_group=scattered_num_tokens)

    def token_reduce_scatter(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        if not self._is_supported_platform():
            return self._fallback.token_reduce_scatter(
                tensor,
                group=group,
                scattered_num_tokens=scattered_num_tokens,
            )
        state = self._get_or_create(group, tensor.size(-1))
        return reduce_scatter(state, tensor, token_list_in_group=scattered_num_tokens)

    def _get_max_num_gathered_tokens(self):
        """Compute max buffer size for TritonRSAG.

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
        max_scattered_num_tokens = ceil_div(
            max_attn_tp_num_tokens, mapping.attn.tp_size
        )
        return max_scattered_num_tokens * max(
            mapping.attn.tp_size, mapping.dense.tp_size, mapping.moe.tp_ep_size
        )
