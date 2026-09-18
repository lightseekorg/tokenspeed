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

import itertools
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as cutlass_utils
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass.cute.typing import AddressSpace
from flashinfer.autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    TunableRunner,
    TuningConfig,
    autotuner_initializer_empty,
    autotuner_initializer_ones,
    is_in_profile_measurement,
)
from tokenspeed_kernel.ops.tuning import get_autotune_max_num_tokens
from tokenspeed_kernel.thirdparty.cute_dsl.mega_moe.megamoe_kernel import (
    Sm100MegaMoEKernel,
    _layout_regions,
)
from tokenspeed_kernel.thirdparty.cute_dsl.mega_moe.sym_buffer import SymBufferHost
from tokenspeed_kernel.thirdparty.cute_dsl.mega_moe.token_comm import CombineFormat

_runners: dict[tuple, MegaMoERunner] = {}
_TACTICS = tuple(
    (tile_n, group_hint, scheduler, token_back, flag_batch)
    for tile_n, group_hint, scheduler, token_back, flag_batch in itertools.product(
        (128, 256),
        (512, 1024),
        ("static", "atomic_counter"),
        ("epi_warps", "reuse_dispatch_warps"),
        (1, 4, 8),
    )
    if tile_n == 256 or token_back == "epi_warps"
)


def _token_capacity(tokens: int) -> int:
    return 1 << (max(1, tokens) - 1).bit_length()


def _token_buckets(tokens: int) -> tuple[int, ...]:
    return tuple(1 << i for i in range(_token_capacity(tokens).bit_length()))


def _init_fp4(shapes, dtype, device):
    return torch.randint(0, 256, shapes, dtype=dtype, device=device)


def _init_scales(shapes, dtype, device):
    return torch.ones(shapes, dtype=torch.float8_e4m3fn, device=device).view(dtype)


def _tensor_view(tensor: torch.Tensor, alignment: int):
    view = cutlass_torch.from_dlpack(tensor, assumed_align=alignment)
    return view.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(tensor))


class MegaMoEWorkspace:
    _kernel_cache: dict[tuple, object] = {}

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
        self.intermediate = intermediate
        self.num_experts = num_experts
        self.top_k = top_k
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        self.world = dist.get_world_size(group)
        self.key = (
            torch.cuda.current_device(),
            self.world,
            num_experts,
            hidden,
            intermediate,
            top_k,
            capacity,
            situ_beta,
            situ_linear_beta,
        )
        self.kernels = {
            (capacity, tactic): self._make_kernel(capacity, tactic)
            for tactic in _TACTICS
        }
        self.offsets = {}
        self.region_sizes = {}
        sizes = {}
        # Fixed offsets keep persistent barriers and counter prefixes valid when
        # graphs with different token counts or tactics share the same allocation.
        for kind, first_data in (
            ("local", "l1_token_buffer"),
            ("shared", "src_token_topk_idx"),
        ):
            counters, data = {}, {}
            for kernel in self.kernels.values():
                regions = counters
                for spec in getattr(kernel, f"_{kind}_region_specs"):
                    if spec.name == first_data:
                        regions = data
                    if (
                        spec.name not in regions
                        or spec.nbytes > regions[spec.name].nbytes
                    ):
                        regions[spec.name] = spec
            regions = [*counters.values(), *data.values()]
            self.offsets[kind], sizes[kind] = _layout_regions(regions)
            self.region_sizes[kind] = {r.name: r.nbytes for r in regions}

        device = torch.device("cuda", torch.cuda.current_device())
        sf_cols = ((hidden // 16 + 3) // 4) * 4
        layouts = [
            ((capacity, hidden // 2), torch.uint8),
            ((capacity, sf_cols), torch.float8_e4m3fn),
            ((capacity, top_k), torch.float32),
            ((sizes["shared"],), torch.uint8),
        ]
        nbytes = [
            shape[0] * (shape[1] if len(shape) == 2 else 1) * dtype.itemsize
            for shape, dtype in layouts
        ]
        nbytes = [((size + 127) // 128) * 128 for size in nbytes]
        self.storage = symm_mem.empty(sum(nbytes), device=device, dtype=torch.uint8)
        self.storage.zero_()
        self.handle = symm_mem.rendezvous(self.storage, group=group.group_name)
        regions = []
        offset = 0
        for (shape, dtype), size in zip(layouts, nbytes):
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
        self.local = torch.zeros(sizes["local"], dtype=torch.uint8, device=device)
        self.output = torch.empty(
            (capacity, hidden), dtype=torch.bfloat16, device=device
        )
        self.mapper = SymBufferHost(
            offsets=tuple(
                int(p) - self.storage.data_ptr() for p in self.handle.buffer_ptrs
            ),
            rank_idx=dist.get_rank(group),
            num_max_ranks=self.world,
        )
        torch.cuda.current_stream().synchronize()
        dist.barrier(group=group)
        self.views: dict[tuple, dict] = {}
        self.max_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(2)

    def _make_kernel(self, capacity: int, tactic: tuple):
        tile_n, group_hint, scheduler, token_back, flag_batch = tactic
        return Sm100MegaMoEKernel(
            mma_tiler_mnk=(256, tile_n, 256),
            cluster_shape_mnk=(2, 1, 1),
            use_2cta_instrs=True,
            group_hint=group_hint,
            token_padding_block=64,
            sf_padding_block=128,
            load_balance_mode=scheduler,
            static_expert_shape=(
                self.num_experts // self.world,
                2 * self.intermediate,
                self.hidden,
            ),
            force_static_sched=True,
            clc_bundle_size=None,
            num_sched_stages=None,
            acc_dtype=cutlass.Float32,
            sf_vec_size=16,
            scenario="2Dx3D",
            world_size=self.world,
            num_topk=self.top_k,
            max_tokens_per_rank=capacity,
            hidden=self.hidden,
            fc2_output_dtype=cutlass.BFloat16,
            combine_format=CombineFormat.parse("bf16"),
            non_ubulk_fc2_store=token_back != "epi_warps",
            in_kernel_fc2_reduce=False,
            token_back_mode=token_back,
            apply_topk_in_fc1=False,
            gate_up_clamp=None,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
            epi_flag_batch=(1, 1) if capacity <= 8192 else (2, 4),
            flag_batch=flag_batch,
        )

    def _kernel(self, capacity: int, tactic: tuple):
        key = (capacity, tactic)
        kernel = self.kernels.get(key)
        if kernel is None:
            kernel = self._make_kernel(capacity, tactic)
            self.kernels[key] = kernel
        if kernel._local_offsets is not self.offsets["local"]:
            for kind in ("local", "shared"):
                for spec in getattr(kernel, f"_{kind}_region_specs"):
                    if spec.nbytes > self.region_sizes[kind][spec.name]:
                        raise ValueError(
                            "MegaMoE kernel exceeds its maximum workspace layout"
                        )
            kernel._local_offsets = self.offsets["local"]
            kernel._shared_offsets = self.offsets["shared"]
            kernel._local_total = self.local.numel()
            kernel._shared_total = self.shared.numel()
            leading = (
                self.offsets["local"]["l1_token_buffer"],
                self.offsets["shared"]["src_token_topk_idx"],
            )
            kernel.require_zero_workspace_leading_bytes = leading
            kernel.local_zero_i32_count = leading[0] // 4
            kernel.shared_zero_i32_count = leading[1] // 4
        return kernel

    def run(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        weights: tuple[torch.Tensor, ...],
        output: torch.Tensor,
        tactic: tuple,
    ) -> torch.Tensor:
        tokens, capacity = x[0].shape[0], output.shape[0]
        if not tokens <= capacity <= self.capacity:
            raise ValueError("MegaMoE token count exceeds workspace capacity")
        kernel = self._kernel(capacity, tactic)
        self.x[:tokens].copy_(x[0])
        self.scales[:tokens, : self.hidden // 16].view(torch.uint8).copy_(
            x[1].view(torch.uint8)
        )
        self.topk_ids[:tokens].copy_(topk_ids)
        self.topk_ids[tokens:capacity].fill_(-1)
        self.topk_weights[:tokens].copy_(topk_weights)
        profiling = is_in_profile_measurement()
        key = (capacity, tactic, *[t.data_ptr() for t in weights], output.data_ptr())
        # Do not retain temporary profiling clones through cached DLPack views.
        kwargs = None if profiling else self.views.get(key)
        if kwargs is None:
            w1, s1, w2, s2, a1, a2, norm = weights
            kwargs = dict(
                activation=_tensor_view(
                    self.x[:capacity].view(torch.float4_e2m1fn_x2), 16
                ),
                activation_sf=_tensor_view(self.scales[:capacity], 16),
                topk_idx=_tensor_view(self.topk_ids[:capacity], 16),
                topk_weights=_tensor_view(self.topk_weights[:capacity], 16),
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
                output_activation=_tensor_view(output, 16),
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
            if not profiling:
                self.views[key] = kwargs
        kwargs = dict(
            kwargs, stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        )
        kernel_key = (*self.key, capacity, tactic)
        compiled = self._kernel_cache.get(kernel_key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "MegaMoE requires an eager warmup before graph capture"
                )
            compiled = cute.compile(
                kernel, max_active_clusters=self.max_clusters, **kwargs
            )
            self._kernel_cache[kernel_key] = compiled
        compiled(**kwargs)
        return output[:tokens]


class MegaMoERunner(TunableRunner):
    TACTICS = _TACTICS

    def __init__(
        self,
        group,
        num_experts: int,
        hidden: int,
        intermediate: int,
        top_k: int,
        situ_beta: float,
        situ_linear_beta: float,
        max_capacity: int,
    ):
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.device = torch.cuda.current_device()
        self.num_experts = num_experts
        self.hidden = hidden
        self.intermediate = intermediate
        self.top_k = top_k
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        self.max_capacity = max_capacity
        self.tactic_autotune = os.environ.get("MEGAMOE_TACTIC_AUTOTUNE", "0") == "1"
        self._extras = (
            "nvfp4_situ_bf16_graph_fixed_workspace_v3",
            self.world,
            num_experts,
            hidden,
            intermediate,
            top_k,
            situ_beta,
            situ_linear_beta,
            max_capacity,
        )
        self._key = (group, self.device, *self._extras)
        self.workspace: MegaMoEWorkspace | None = None
        rank, world = self.rank, self.world

        # FlashInfer deep-copies initializers; a bound method would copy the process group.
        def init_routes(shapes, dtype, device):
            routes = torch.arange(shapes[0] * top_k, dtype=dtype, device=device)
            routes = (routes + rank * shapes[0] * top_k) % num_experts
            return ((routes % world) * (num_experts // world) + routes // world).view(
                shapes
            )

        self.tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec((11,), (0,), _token_buckets, _token_capacity),
            ),
            constraint_specs=tuple(
                ConstraintSpec(i, 0, lambda shapes: shapes[11][0]) for i in range(4)
            ),
            tensor_initializers=(
                (0, _init_fp4),
                (1, _init_scales),
                (2, init_routes),
                (3, autotuner_initializer_ones),
                (11, autotuner_initializer_empty),
            ),
            use_cold_l2_cache=True,
            use_cuda_graph=True,
        )

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MegaMoERunner) and self._key == other._key

    def get_cache_key_extras(self, inputs) -> tuple:
        return self._extras

    def get_valid_tactics(self, inputs, profile) -> list:
        return list(self.TACTICS)

    def forward(self, inputs, tactic):
        capacity = inputs[11].shape[0]
        if tactic is None or tactic == -1:
            if capacity <= 1024:
                tactic = (128, 512, "static", "epi_warps", 1)
            elif capacity <= 8192:
                tactic = (256, 512, "static", "epi_warps", 4)
            else:
                tactic = (
                    256,
                    512,
                    "atomic_counter" if capacity >= 16384 else "static",
                    "reuse_dispatch_warps",
                    8,
                )
        else:
            tactic = tuple(tactic)
        return self.workspace.run(
            (inputs[0], inputs[1]),
            inputs[2],
            inputs[3],
            tuple(inputs[4:11]),
            inputs[11],
            tactic,
        )

    def run(self, x, topk_ids, topk_weights, weights, max_tokens: int):
        capacity = _token_capacity(max_tokens)
        if capacity > self.max_capacity:
            raise ValueError("MegaMoE token count exceeds the configured maximum")
        if self.workspace is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "MegaMoE workspace requires eager initialization before capture"
                )
            self.workspace = MegaMoEWorkspace(
                self.group,
                self.num_experts,
                self.hidden,
                self.intermediate,
                self.top_k,
                self.max_capacity,
                self.situ_beta,
                self.situ_linear_beta,
            )
        # All ranks use the same capacity, including ranks with no local tokens.
        inputs = [
            *x,
            topk_ids,
            topk_weights,
            *weights,
            self.workspace.output[:capacity],
        ]
        if not self.tactic_autotune:
            return self.forward(inputs, tactic=-1)
        chosen, tactic = AutoTuner.get().choose_one(
            custom_op="trtllm_nvfp4_mega_moe",
            runners=[self],
            tuning_config=self.tuning_config,
            inputs=inputs,
        )
        return chosen(inputs=inputs, tactic=tactic)


def get_runner(
    group,
    num_experts: int,
    hidden: int,
    intermediate: int,
    top_k: int,
    situ_beta: float,
    situ_linear_beta: float,
) -> MegaMoERunner:
    """Return the cached tuning/launch adapter for one EP group and expert geometry."""
    max_capacity = _token_capacity(get_autotune_max_num_tokens())
    key = (
        id(group),
        torch.cuda.current_device(),
        num_experts,
        hidden,
        intermediate,
        top_k,
        situ_beta,
        situ_linear_beta,
        max_capacity,
    )
    runner = _runners.get(key)
    if runner is None:
        runner = MegaMoERunner(
            group,
            num_experts,
            hidden,
            intermediate,
            top_k,
            situ_beta,
            situ_linear_beta,
            max_capacity,
        )
        _runners[key] = runner
    return runner
