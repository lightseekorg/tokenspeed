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

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as cutlass_utils
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass.cute.typing import AddressSpace
from tokenspeed_kernel.thirdparty.cute_dsl.mega_moe.megamoe_kernel import (
    Sm100MegaMoEKernel,
)
from tokenspeed_kernel.thirdparty.cute_dsl.mega_moe.sym_buffer import SymBufferHost
from tokenspeed_kernel.thirdparty.cute_dsl.mega_moe.token_comm import CombineFormat

_workspaces: dict[tuple, MegaMoEWorkspace] = {}


def _tensor_view(tensor: torch.Tensor, alignment: int):
    view = cutlass_torch.from_dlpack(tensor, assumed_align=alignment)
    return view.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(tensor))


class MegaMoEWorkspace:
    @torch.inference_mode(False)
    def __init__(
        self,
        group,
        num_experts: int,
        hidden: int,
        intermediate: int,
        top_k: int,
        capacity: int,
        situ_beta: float,
        situ_linear_beta: float,
    ):
        self.capacity = capacity
        self.hidden = hidden
        self.group = group
        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        tile_n = 128 if capacity <= 1024 else 256
        self.kernel = Sm100MegaMoEKernel(
            mma_tiler_mnk=(256, tile_n, 256),
            cluster_shape_mnk=(2, 1, 1),
            use_2cta_instrs=True,
            group_hint=512,
            token_padding_block=64,
            sf_padding_block=128,
            load_balance_mode="static",
            static_expert_shape=(num_experts // world, 2 * intermediate, hidden),
            force_static_sched=True,
            clc_bundle_size=None,
            num_sched_stages=None,
            acc_dtype=cutlass.Float32,
            sf_vec_size=16,
            scenario="2Dx3D",
            world_size=world,
            num_topk=top_k,
            max_tokens_per_rank=capacity,
            hidden=hidden,
            fc2_output_dtype=cutlass.BFloat16,
            combine_format=CombineFormat.parse("bf16"),
            non_ubulk_fc2_store=False,
            in_kernel_fc2_reduce=False,
            token_back_mode="epi_warps",
            apply_topk_in_fc1=False,
            gate_up_clamp=None,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            epi_flag_batch=(1, 1),
            flag_batch=1,
        )
        local_bytes, shared_bytes = self.kernel.get_workspace_sizes()
        device = torch.device("cuda", torch.cuda.current_device())
        sf_cols = ((hidden // 16 + 3) // 4) * 4
        layouts = [
            ((capacity, hidden // 2), torch.uint8),
            ((capacity, sf_cols), torch.float8_e4m3fn),
            ((capacity, top_k), torch.float32),
            ((shared_bytes,), torch.uint8),
        ]
        sizes = [
            shape[0] * (shape[1] if len(shape) == 2 else 1) * dtype.itemsize
            for shape, dtype in layouts
        ]
        sizes = [((size + 127) // 128) * 128 for size in sizes]
        self.storage = symm_mem.empty(sum(sizes), device=device, dtype=torch.uint8)
        self.storage.zero_()
        self.handle = symm_mem.rendezvous(self.storage, group=group.group_name)
        regions = []
        offset = 0
        for (shape, dtype), size in zip(layouts, sizes):
            numel = shape[0] * (shape[1] if len(shape) == 2 else 1)
            regions.append(
                self.storage[offset : offset + numel * dtype.itemsize]
                .view(dtype)
                .view(shape)
            )
            offset += size
        self.x, self.scales, self.topk_weights, self.shared = regions
        self.topk_ids = torch.full(
            (capacity, top_k), -1, dtype=torch.int64, device=device
        )
        self.output = torch.empty(
            (capacity, hidden), dtype=torch.bfloat16, device=device
        )
        self.local = torch.zeros(local_bytes, dtype=torch.uint8, device=device)
        self.mapper = SymBufferHost(
            offsets=tuple(
                int(p) - self.storage.data_ptr() for p in self.handle.buffer_ptrs
            ),
            rank_idx=rank,
            num_max_ranks=world,
        )
        torch.cuda.current_stream().synchronize()
        dist.barrier(group=group)
        self.compiled = None
        self.views: dict[tuple, dict] = {}
        self.max_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(2)

    def run(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        weights: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        tokens = x[0].shape[0]
        if tokens > self.capacity:
            raise ValueError("MegaMoE token count exceeds workspace capacity")
        self.x[:tokens].copy_(x[0])
        self.scales[:tokens, : self.hidden // 16].view(torch.uint8).copy_(
            x[1].view(torch.uint8)
        )
        self.topk_ids[:tokens].copy_(topk_ids)
        self.topk_ids[tokens:].fill_(-1)
        self.topk_weights[:tokens].copy_(topk_weights)
        key = tuple(t.data_ptr() for t in weights)
        kwargs = self.views.get(key)
        if kwargs is None:
            w1, s1, w2, s2, a1, a2, norm = weights
            kwargs = dict(
                activation=_tensor_view(self.x.view(torch.float4_e2m1fn_x2), 16),
                activation_sf=_tensor_view(self.scales, 16),
                topk_idx=_tensor_view(self.topk_ids, 16),
                topk_weights=_tensor_view(self.topk_weights, 16),
                fc1_weight=_tensor_view(
                    w1.view(torch.float4_e2m1fn_x2).transpose(1, 2), 16
                ),
                fc1_weight_sf=_tensor_view(s1.view(torch.float8_e4m3fn), 16),
                fc2_weight=_tensor_view(
                    w2.view(torch.float4_e2m1fn_x2).transpose(1, 2), 16
                ),
                fc2_weight_sf=_tensor_view(s2.view(torch.float8_e4m3fn), 16),
                fc1_alpha=_tensor_view(a1, 4),
                fc2_alpha=_tensor_view(a2, 4),
                fc1_norm_const=_tensor_view(norm, 4),
                output_activation=_tensor_view(self.output, 16),
                local_workspace=cute.runtime.make_ptr(
                    cutlass.Uint8,
                    self.local.data_ptr(),
                    AddressSpace.gmem,
                    assumed_align=16,
                ),
                shared_workspace=cute.runtime.make_ptr(
                    cutlass.Uint8,
                    self.shared.data_ptr(),
                    AddressSpace.gmem,
                    assumed_align=16,
                ),
                peer_rank_ptr_mapper_host=self.mapper,
            )
            self.views[key] = kwargs
        kwargs = dict(
            kwargs, stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        )
        if self.compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "MegaMoE requires an eager warmup before graph capture"
                )
            self.compiled = cute.compile(
                self.kernel, max_active_clusters=self.max_clusters, **kwargs
            )
        self.compiled(**kwargs)
        return self.output[:tokens]


def get_workspace(
    group,
    num_experts: int,
    hidden: int,
    intermediate: int,
    top_k: int,
    max_tokens: int,
    situ_beta: float,
    situ_linear_beta: float,
) -> MegaMoEWorkspace:
    capacity = 1 << (max(1, max_tokens) - 1).bit_length()
    key = (
        id(group),
        torch.cuda.current_device(),
        num_experts,
        hidden,
        intermediate,
        top_k,
        capacity,
        situ_beta,
        situ_linear_beta,
    )
    workspace = _workspaces.get(key)
    if workspace is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MegaMoE workspace must be initialized before capture")
        workspace = MegaMoEWorkspace(
            group,
            num_experts,
            hidden,
            intermediate,
            top_k,
            capacity,
            situ_beta,
            situ_linear_beta,
        )
        _workspaces[key] = workspace
    return workspace
