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

"""Block-scaled route rows on TRT-LLM-gen batched GEMMs (NVIDIA SM100-SM103).

The ``flashinfer_trtllm`` solution of ``block_scaled_routes``: the Triton
route kernels (``route_tiles.py``) lay the routed rows out in the
batched GEMM's token tiles, the GEMMs run FlashInfer's published cubins
through ``thirdparty/flashinfer/trtllm_block_scaled_bmm``, and the expert
weights are rewritten into the GEMM's layout as they load.
"""

from __future__ import annotations

import functools
import math

import torch
from flashinfer import block_scale_interleave
from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache
from tokenspeed_kernel.ops.communication.mailbox import MailboxTarget
from tokenspeed_kernel.ops.moe._flashinfer import route_tiles
from tokenspeed_kernel.ops.moe._triton import block_scaled as _block_scaled
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.flashinfer.trtllm_block_scaled_bmm import (
    load_block_scaled_bmm_module,
)


def _block_scaled_bmm_tile(routes: int, experts: int) -> int:
    """The token tile: the mean routes per expert, rounded up to a power
    of two in 8..256 (on B300 within 1% of the fastest tile from decode
    to 8K-token prefill)."""
    mean = max(1, math.ceil(routes / experts))
    return min(256, max(8, 1 << (mean - 1).bit_length()))


@functools.lru_cache(maxsize=4096)
def _block_scaled_bmm_launch(
    nvfp4: bool, tile: int, n: int, k: int, experts: int, tokens: int, top_k: int
) -> tuple[int, int, int]:
    """(config, activation scale layout, workspace bytes) of one GEMM."""
    module = load_block_scaled_bmm_module()
    config = module.block_scaled_bmm_config(nvfp4, tile, n, k, experts, tokens, top_k)
    return (
        config,
        module.block_scaled_bmm_sf_layout(nvfp4, tile, config),
        module.block_scaled_bmm_workspace(
            nvfp4, tile, n, k, experts, tokens, top_k, config
        ),
    )


@functools.cache
def _unit_scales(device: torch.device, experts: int) -> torch.Tensor:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("run the batched expert GEMMs before graph capture")
    return torch.ones(experts, device=device, dtype=torch.float32)


def flashinfer_trtllm_block_scaled_route_tensor(
    tensor: torch.Tensor, *, role: str, formats: tuple[str, str]
) -> None:
    """Rewrite one loaded expert tensor in place into the batched GEMM's
    weight layout (the MoE GEMMs' non-gated one): each expert's rows
    shuffled for the transposed epilogue, block scales also interleaved
    in 128x4 tiles. ``role`` is ``w13_weight``, ``w13_scale``,
    ``w2_weight`` or ``w2_scale``."""
    projection, kind = role.split("_")
    fmt = formats[0] if projection == "w13" else formats[1]
    cache: dict = {}
    for expert in range(tensor.shape[0]):
        data = tensor[expert].view(torch.uint8)
        if kind == "weight":
            order = get_w2_permute_indices_with_cache(cache, data, 128)
            data.copy_(data[order.to(data.device)])
            continue
        order = get_w2_permute_indices_with_cache(
            cache, data, 128, num_elts_per_sf=16 if fmt == "nvfp4" else 32
        )
        interleaved = block_scale_interleave(data[order.to(data.device)])
        if interleaved.numel() != data.numel():
            raise ValueError("block scales need 128-row and 4-column multiples")
        data.copy_(interleaved.view(data.shape))


@register_kernel(
    "moe",
    "block_scaled_routes",
    name="flashinfer_trtllm_block_scaled_moe_routes",
    solution="flashinfer_trtllm",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
    ),
    signatures=frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))}),
    traits={
        "w13_weight_dtype": frozenset({"nvfp4", "mxfp8"}),
        "w2_weight_dtype": frozenset({"nvfp4", "mxfp8"}),
        # Whole GEMM K tiles in both projections.
        "hidden_alignment": frozenset({256}),
        "ispp_alignment": frozenset({256}),
    },
    priority=Priority.SPECIALIZED,
    weight_preprocessor=flashinfer_trtllm_block_scaled_route_tensor,
)
def flashinfer_trtllm_block_scaled_moe_routes(
    x: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    a13: torch.Tensor | None,
    d13: torch.Tensor | None,
    w2: torch.Tensor,
    s2: torch.Tensor,
    a2: torch.Tensor | None,
    d2: torch.Tensor | None,
    weights: torch.Tensor,
    ids: torch.Tensor,
    *,
    formats: tuple[str, str],
    experts: int,
    expert_start: int,
    gate_cap: float,
    up_cap: float,
    zero_remote_routes: bool,
    fold_slots: bool,
    fold_into: MailboxTarget | torch.Tensor | None,
) -> torch.Tensor | None:
    """The Triton route contract with the expert GEMMs on TRT-LLM-gen
    batched GEMMs: the routes are laid out per expert in GEMM tiles,
    activated and quantized in place, and gathered back to route rows.
    The GEMMs sum each element's K blocks in the Triton scaled-dot
    order and scale them alike, so the rows match it bit for bit."""
    tokens, hidden = x.shape
    top_k = ids.shape[1]
    route_count = ids.numel()
    local_experts = w13.shape[0]
    intermediate = w13.shape[1] // 2
    if not tokens:
        return x.new_empty((0, 1 if fold_slots else top_k, hidden))
    module = load_block_scaled_bmm_module()
    nvfp4 = (formats[0] == "nvfp4", formats[1] == "nvfp4")
    tile = _block_scaled_bmm_tile(route_count, experts)
    shape = (local_experts, tokens, top_k)
    config1, layout1, workspace1 = _block_scaled_bmm_launch(
        nvfp4[0], tile, 2 * intermediate, hidden, *shape
    )
    config2, layout2, workspace2 = _block_scaled_bmm_launch(
        nvfp4[1], tile, hidden, intermediate, *shape
    )
    unit = _unit_scales(x.device, local_experts)
    enable_pdl = pdl_enabled()
    if route_count <= route_tiles._FUSED_TILE_MAX_ROUTES:
        # Decode-sized batches route, tile and quantize in one launch.
        tiled = route_tiles._route_tile_quantize(
            x,
            ids,
            a13,
            d13,
            experts=local_experts,
            expert_start=expert_start,
            fmt=formats[0],
            tile=tile,
            layout=layout1,
        )
    else:
        (routes, counts), (first, first_scale, first_amax) = (
            _block_scaled._routed_quantized_input(
                x, ids, local_experts, expert_start, formats[0], None
            )
        )
        tiled = route_tiles._tile_routes(
            routes,
            counts,
            first,
            first_scale,
            first_amax,
            a13,
            d13,
            tile=tile,
            layout=layout1,
            top_k=top_k,
        )
    capacity = tiled.values.shape[0]
    metadata = (
        tiled.num_tiles,
        tiled.num_rows,
        tiled.tile_expert,
        tiled.tile_limit,
    )
    gate_up = torch.empty(
        (capacity, 2 * intermediate), device=x.device, dtype=torch.bfloat16
    )
    module.block_scaled_bmm(
        tiled.values,
        tiled.scales,
        w13,
        s13,
        tiled.row_scale,
        unit,
        gate_up,
        *metadata,
        torch.empty(workspace1, device=x.device, dtype=torch.uint8),
        nvfp4[0],
        tile,
        tokens,
        top_k,
        config1,
        enable_pdl,
    )
    # The SwiGLU rows stay in route order until the second GEMM's tiled rows
    # are gathered back over them, or folded per token.
    output = torch.empty((route_count, hidden), device=x.device, dtype=torch.bfloat16)
    second, second_scale, second_amax = _block_scaled._quantized_buffers(
        capacity, intermediate, nvfp4[1], x.device
    )
    _block_scaled._swiglu_quantize_routes(
        gate_up,
        weights,
        ids,
        output,
        second,
        second_scale,
        second_amax,
        None,
        second_format=formats[1],
        expert_start=expert_start,
        num_experts=local_experts,
        gate_cap=gate_cap,
        up_cap=up_cap,
        zero_remote_routes=False,
        route_rows=tiled.route_rows,
        w_amax=a2,
        w_denominator=d2,
        layout=layout2,
    )
    # The second projection's rows reuse the first's once the SwiGLU read them.
    down = (
        gate_up
        if 2 * intermediate == hidden
        else torch.empty((capacity, hidden), device=x.device, dtype=torch.bfloat16)
    )
    module.block_scaled_bmm(
        second,
        second_scale,
        w2,
        s2,
        second_amax,
        unit,
        down,
        *metadata,
        torch.empty(workspace2, device=x.device, dtype=torch.uint8),
        nvfp4[1],
        tile,
        tokens,
        top_k,
        config2,
        enable_pdl,
    )
    if fold_slots:
        return _block_scaled._fold_routes(
            down,
            tiled.route_rows,
            ids,
            expert_start=expert_start,
            num_experts=local_experts,
            fold_into=fold_into,
        )
    route_tiles._gather_route_rows(
        down,
        tiled.route_rows,
        ids,
        output,
        expert_start=expert_start,
        num_experts=local_experts,
        zero_remote_routes=zero_remote_routes,
    )
    return output.view(tokens, top_k, hidden)
