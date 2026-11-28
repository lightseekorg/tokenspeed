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

"""FlashInfer's Kimi K3 MegaMoE for MXFP4 experts with MXFP8 activations.

The expert kernel dispatches routed rows to their owners, runs both GEMMs
with SiTU, and returns expert outputs for a final top-k reduction. Routing
metadata and input quantization run beforehand. Weights are launch arguments,
so layers with matching dimensions and activation share workspaces.

The kernel is sized for every SM of the device: its router stage deadlocks
across ranks when another kernel holds SMs while it runs (seen above 128 rows
per rank, where the router spans several CTAs). The caller must launch it with
no other stream running on the device.
"""

from __future__ import annotations

import logging
import math

import torch
from flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe.shim import (
    runtime as kimi_runtime,
)
from flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe.src.helpers import epoch
from flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe.src.helpers.device_workspace import (
    DeviceWorkspace,
)
from tokenspeed_kernel.ops.tuning import get_runtime_max_num_tokens

logger = logging.getLogger(__name__)

_SCALE_BLOCK = 32
# Up to 128 rows the router is one CTA; smaller capacities gain nothing.
_MIN_CAPACITY = 128

_runners: dict[tuple, KimiMegaMoERunner] = {}


def _share_kernel_layouts(kernels):
    layouts = {}
    for space in ("local", "shared"):
        regions, alignments = {}, {}
        for kernel in kernels:
            layout = kernel._device_workspace
            for region in layout.regions(space):
                # finalize() adds the epoch counter itself.
                if region.name == epoch.REGION:
                    continue
                size = layout.nbytes(region.name)
                alignments[region.name] = max(
                    alignments.get(region.name, 1), region.byte_alignment
                )
                if region.name not in regions or size > regions[region.name][0]:
                    regions[region.name] = (size, region)
        shared = layouts[space] = DeviceWorkspace()
        for _, region in regions.values():
            shared.register(
                region.name,
                region.dtype,
                region.shape,
                buffer_space=space,
                stride=region.stride,
                byte_alignment=alignments[region.name],
                reset=region.reset,
            )
        shared.finalize()

    for kernel in kernels:
        layout = kernel._device_workspace
        # Keep each variant's shapes/strides, but fix region and epoch offsets.
        layout._offset = {
            region.name: layouts[space].offset(region.name)
            for space in ("local", "shared")
            for region in layout.regions(space)
        }
        layout._total = {
            space: shared._total[space] for space, shared in layouts.items()
        }
        layout._zero_leading = {
            space: shared._zero_leading[space] for space, shared in layouts.items()
        }
        layout._tail_leading = {
            space: shared._tail_leading[space] for space, shared in layouts.items()
        }
        layout._tail_block = {
            space: shared._tail_block[space] for space, shared in layouts.items()
        }
    return layouts


class _SharedBuffers:
    def __init__(self, group, config, fleet, layouts):
        self.device = torch.device("cuda", torch.cuda.current_device())
        capacity, hidden = fleet.max_tokens_per_rank, fleet.token_hidden_size
        regions = (
            ((capacity, hidden), torch.uint8),
            ((capacity, hidden // _SCALE_BLOCK), torch.uint8),
            ((capacity, config.top_k), torch.float32),
            ((capacity, hidden), torch.bfloat16),
            ((layouts["shared"].total_bytes("shared"),), torch.uint8),
            ((capacity, config.top_k), torch.int32),
            ((1,), torch.float32),
        )
        sizes = [math.prod(shape) * dtype.itemsize for shape, dtype in regions]
        padded_sizes = [(size + 127) // 128 * 128 for size in sizes]
        self.symm_handle = None
        if group.size() > 1:
            import torch.distributed._symmetric_memory as symm

            symm.enable_symm_mem_for_group(group.group_name)
            self.storage = symm.empty(
                sum(padded_sizes), dtype=torch.uint8, device=self.device
            )
            self.symm_handle = symm.rendezvous(self.storage, group.group_name)
            peer_offsets = tuple(
                int(ptr) - self.storage.data_ptr()
                for ptr in self.symm_handle.buffer_ptrs
            )
        else:
            self.storage = torch.empty(
                sum(padded_sizes), dtype=torch.uint8, device=self.device
            )
            peer_offsets = (0,)
        views, offset = [], 0
        for (shape, dtype), size, padded in zip(
            regions, sizes, padded_sizes, strict=True
        ):
            views.append(self.storage[offset : offset + size].view(dtype).view(shape))
            offset += padded
        (
            self.activation,
            self.activation_sf,
            self.scores,
            self.output,
            self.shared_workspace,
            self.ids,
            self.input_scale,
        ) = views
        self.mapper = kimi_runtime.SymmetricBufferHost(
            base_address=self.storage.data_ptr(),
            offsets=peer_offsets,
            rank=group.rank(),
            max_ranks=group.size(),
        )
        self.local_workspace = torch.empty(
            layouts["local"].total_bytes("local"), dtype=torch.uint8, device=self.device
        )
        self.storage.zero_()
        self.ids.fill_(-1)
        self.input_scale.fill_(config.input_norm_const)
        self.local_workspace[: layouts["local"].zero_on_allocate_bytes("local")].zero_()
        torch.cuda.current_stream().synchronize()
        if group.size() > 1:
            torch.distributed.barrier(group=group)


class _BucketWorkspace(kimi_runtime.KimiWorkspace):
    """Use FlashInfer's launchers with views into one shared allocation."""

    def __init__(self, group, config, fleet, kernel, buffers, quantizer):
        self.config, self.fleet_params = config, fleet
        self.rank, self.world_size, self.process_group = (
            group.rank(),
            group.size(),
            group,
        )
        self.device = buffers.device
        self.kernel = kernel
        self.options = {"MEGAMOE_SITU_FAST": "1" if config.fast_situ else "0"}
        self._compiled = None
        self._quantizer = quantizer
        self._routes = {}
        self._inputs = None
        self._destroyed = False
        self._nvfp4 = False
        self._block = _SCALE_BLOCK
        self._sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self._buffer = buffers.storage
        self._symm_handle = buffers.symm_handle
        self.mapper = buffers.mapper
        self.local_workspace = buffers.local_workspace
        self.shared_workspace = buffers.shared_workspace
        self._input_scale = buffers.input_scale
        capacity = fleet.max_tokens_per_rank
        self.activation = buffers.activation[:capacity]
        self.activation_sf = buffers.activation_sf[:capacity]
        self.scores = buffers.scores[:capacity]
        self.output = buffers.output[:capacity]
        self.ids = buffers.ids[:capacity]
        self._live = 0


def prepare_kimi_mega_moe_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> object:
    """Rewrite one rank's checkpoint experts in place into the kernel's layout.

    Args:
        w13: Packed E2M1 ``[experts, 2 * intermediate, hidden // 2]`` uint8,
            gate rows first.
        w13_scale: E8M0 bytes ``[experts, 2 * intermediate, hidden // 32]``.
        w2: Packed E2M1 ``[experts, hidden, intermediate // 2]`` uint8.
        w2_scale: E8M0 bytes ``[experts, hidden, intermediate // 32]``.

    Returns:
        FlashInfer's ``KimiWeights`` over the same four storages.
    """
    from flashinfer.moe_ep.kernel_src.sm100 import _kimi_k3_megamoe as kimi

    experts, hidden_size, packed_intermediate_size = w2.shape
    intermediate_size = packed_intermediate_size * 2
    expected = (
        (w13, (experts, 2 * intermediate_size, hidden_size // 2)),
        (w13_scale, (experts, 2 * intermediate_size, hidden_size // _SCALE_BLOCK)),
        (w2, (experts, hidden_size, intermediate_size // 2)),
        (w2_scale, (experts, hidden_size, intermediate_size // _SCALE_BLOCK)),
    )
    for tensor, shape in expected:
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.uint8
            or not tensor.is_cuda
            or not tensor.is_contiguous()
            or tensor.data_ptr() % 16
        ):
            raise ValueError(
                "Kimi MegaMoE expert tensors must be contiguous, 16-byte-aligned "
                f"CUDA uint8 of shape {shape}; got {tuple(tensor.shape)} {tensor.dtype}"
            )
    w13.copy_(kimi.interleave_gate_up(w13))
    w13_scale_flat = w13_scale.view(experts, -1)
    w13_scale_flat.copy_(kimi.swizzle_scales(kimi.interleave_gate_up(w13_scale)))
    w2_scale_flat = w2_scale.view(experts, -1)
    w2_scale_flat.copy_(kimi.swizzle_scales(w2_scale))
    one = torch.ones(experts, dtype=torch.float32, device=w13.device)
    return kimi.KimiWeights(
        "mxfp4_mxfp8", w13, w2, w13_scale_flat, w2_scale_flat, one, one, one
    )


class KimiMegaMoERunner:
    """Workspaces shared by layers of one EP group with the same dimensions."""

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        situ_beta: float,
        situ_linear_beta: float,
        max_tokens: int,
        weights: object,
    ) -> None:
        """Create and warm every capacity up to ``max_tokens`` rows per rank.

        Collective: every rank of ``group`` constructs its runner at the same
        point, outside CUDA graph capture. ``weights`` is any layer's
        ``prepare_kimi_mega_moe_weights`` result, used for the warm launches.
        """
        from flashinfer import moe_ep

        config = moe_ep.Sm100_KimiK3_MegaMoeConfig(
            intermediate_size=intermediate_size,
            top_k=top_k,
            activation="situ",
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            quant_kind="mxfp4_mxfp8",
            combine_dtype="bf16",
        )
        fleets, kernels = {}, {}
        for power in range(_MIN_CAPACITY.bit_length() - 1, max_tokens.bit_length()):
            capacity = 1 << power
            fleet = fleets[capacity] = moe_ep.FleetParams(
                num_experts=num_experts,
                token_hidden_size=hidden_size,
                max_tokens_per_rank=capacity,
            )
            with kimi_runtime.option_scope(
                {"MEGAMOE_SITU_FAST": "1" if config.fast_situ else "0"}
            ):
                kernels[capacity] = kimi_runtime._kernel(config, fleet, group.size())
        layouts = _share_kernel_layouts(tuple(kernels.values()))
        buffers = _SharedBuffers(group, config, fleets[max_tokens], layouts)
        logger.info(
            f"Kimi MegaMoE workspace for {max_tokens} tokens per rank: "
            f"{buffers.local_workspace.numel() / 2**20:.0f} MiB local, "
            f"{buffers.storage.numel() / 2**20:.0f} MiB shared with peers"
        )
        self.workspaces: dict[int, _BucketWorkspace] = {}
        device = torch.device("cuda", torch.cuda.current_device())
        # One masked row warms quantization and the expert launch before serving.
        row = torch.zeros(1, hidden_size, dtype=torch.bfloat16, device=device)
        ids = torch.full((1, top_k), -1, dtype=torch.int32, device=device)
        scores = torch.zeros(1, top_k, dtype=torch.float32, device=device)
        quantizer = None
        for capacity, kernel in kernels.items():
            workspace = _BucketWorkspace(
                group, config, fleets[capacity], kernel, buffers, quantizer
            )
            self.workspaces[capacity] = workspace
            self.run(row, ids, scores, weights, capacity)
            quantizer = workspace._quantizer
        torch.cuda.synchronize()

    def input_buffers(
        self, tokens: int, max_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return activation, scale, expert-ID and route-weight views for producers."""
        workspace = self.workspaces[_token_capacity(max_tokens)]
        return (
            workspace.activation[:tokens],
            workspace.activation_sf[:tokens],
            workspace.ids[:tokens],
            workspace.scores[:tokens],
        )

    def run(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        weights: object,
        max_tokens: int,
    ) -> torch.Tensor:
        """Run the routed experts of this rank's rows.

        Args:
            x: BF16 ``[local_tokens, hidden]`` routed activations.
            topk_ids: ``[local_tokens, top_k]`` int32 global expert ids.
            topk_weights: ``[local_tokens, top_k]`` float32 routing weights.
            weights: This rank's ``prepare_kimi_mega_moe_weights`` result.
            max_tokens: Largest local token count over all ranks this step;
                every rank passes the same value.

        Returns:
            BF16 ``[local_tokens, hidden]``, a view of the workspace valid until
            its next launch.
        """
        import cuda.bindings.driver as cuda
        import cutlass

        if x.dtype != torch.bfloat16 or topk_weights.dtype != torch.float32:
            raise TypeError(
                "Kimi MegaMoE takes BF16 activations and float32 routing weights; "
                f"got {x.dtype} and {topk_weights.dtype}"
            )
        if x.shape[0] > max_tokens:
            raise ValueError(
                f"Kimi MegaMoE got {x.shape[0]} rows for a step of {max_tokens}"
            )
        workspace = self.workspaces[_token_capacity(max_tokens)]
        tokens = x.shape[0]
        for target, source in (
            (workspace.ids[:tokens], topk_ids),
            (workspace.scores[:tokens], topk_weights),
        ):
            if not (
                target.data_ptr() == source.data_ptr()
                and target.dtype == source.dtype
                and target.stride() == source.stride()
            ):
                target.copy_(source)
        workspace.ids[tokens:].fill_(-1)
        workspace._compile_quantizer()
        stream = torch.cuda.current_stream()
        if tokens:
            blocks = x.numel() // _SCALE_BLOCK
            grid = min(workspace._sms * 2, (blocks + 511) // 512)
            workspace._quantizer(
                cutlass.Int64(x.data_ptr()),
                cutlass.Int64(workspace.activation.data_ptr()),
                cutlass.Int64(workspace.activation_sf.data_ptr()),
                cutlass.Int32(blocks),
                cutlass.Int32(grid),
                cuda.CUstream(stream.cuda_stream),
            )
        x.record_stream(stream)
        # Producers already wrote the routes. The public staging entry would
        # copy them again and retain the forward's temporary input tensors.
        workspace._live = tokens
        return workspace._compute(weights, output=None)


def _token_capacity(tokens: int) -> int:
    return max(_MIN_CAPACITY, 1 << (max(1, tokens) - 1).bit_length())


def get_kimi_mega_moe_runner(plan: dict, w: torch.nn.Module) -> KimiMegaMoERunner:
    """Return the runner shared by layers with matching dimensions and activation.

    The runtime's startup tuning capacity bounds the workspaces, as for the
    NVFP4 runner. ``w`` provides the model dimensions and prepared expert weights;
    ``plan["process_group"]`` is the EP group.
    """
    group = plan["process_group"]
    num_experts = w.num_experts
    hidden_size = w.hidden_size
    intermediate_size = w.intermediate_size
    top_k = w.top_k
    situ_beta = w.activation_situ_beta
    situ_linear_beta = w.activation_situ_linear_beta
    max_tokens = _token_capacity(get_runtime_max_num_tokens())
    key = (
        id(group),
        torch.cuda.current_device(),
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        situ_beta,
        situ_linear_beta,
        max_tokens,
    )
    runner = _runners.get(key)
    if runner is None:
        runner = _runners[key] = KimiMegaMoERunner(
            group,
            num_experts,
            hidden_size,
            intermediate_size,
            top_k,
            situ_beta,
            situ_linear_beta,
            max_tokens,
            w._moe_backend_state,
        )
    return runner
