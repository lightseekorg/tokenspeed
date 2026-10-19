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

"""Decode routing in the top-k kernel: the top-k kernel's selection, flashinfer's
metadata, and the prepared expert GEMMs reproduce the fused launch bitwise."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.numerics.torch.moe import torch_grouped_sigmoid_topk
from tokenspeed_kernel.ops.moe import flashinfer as module
from tokenspeed_kernel.ops.moe import moe_topk
from tokenspeed_kernel.ops.moe._triton.sigmoid_topk_routing import (
    triton_sigmoid_bias_topk_routing,
)
from tokenspeed_kernel.platform import current_platform

flashinfer_moe = pytest.importorskip("flashinfer.fused_moe.core")

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and current_platform().is_nvidia
        and current_platform().arch_version.major == 10
    ),
    reason="needs the Blackwell flashinfer NVFP4 MoE",
)

_WRITTEN = (
    "total_num_padded_tokens",
    "expanded_idx_to_permuted_idx",
    "permuted_idx_to_token_idx",
    "num_tokens_per_expert",
    "cta_idx_xy_to_batch_idx",
    "cta_idx_xy_to_mn_limit",
    "num_non_exiting_ctas",
)


def _slot(
    ids: torch.Tensor, weights: torch.Tensor, offset: int, local: int, tile_n: int
):
    """A flashinfer metadata slot with every written word set to a sentinel."""
    metadata = flashinfer_moe.trtllm_moe_allocate_routing_metadata_multi_tile(
        ids,
        num_experts=256,
        top_k=ids.shape[1],
        local_expert_offset=offset,
        num_local_experts=local,
        tile_ns=(tile_n,),
        routing_input_mode=flashinfer_moe.RoutingInputMode.UnpackedPrecomputed,
        topk_weights=weights,
    )
    for name in _WRITTEN:
        getattr(metadata.slots[0], name).fill_(-7)
    return metadata


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4])
@pytest.mark.parametrize("tile_n", [8, 16])
@pytest.mark.parametrize("local", [64, 48])
@pytest.mark.parametrize("normalize", [True, False])
def test_fused_topk_routing_matches_the_kernels_it_replaces(
    num_tokens: int, tile_n: int, local: int, normalize: bool
) -> None:
    """Weights and ids equal the top-k kernel's bitwise; the metadata equals
    flashinfer's populate kernel word for word (its single-block kernel
    serves these batches and orders an expert's rows by token, as we do)."""
    experts, topk, offset = 256, 8, 64
    torch.manual_seed(num_tokens * tile_n + local)
    logits = torch.randn(num_tokens, experts, device="cuda") * 3.0
    bias = torch.randn(experts, device="cuda") * 0.1
    weights_ref, ids_ref = moe_topk(
        router_logits=logits,
        top_k=topk,
        score_function="sigmoid",
        selection_method="topk",
        renormalize=normalize,
        routed_scaling_factor=1.5,
        correction_bias=bias,
        hash_indices_table=None,
        input_ids=None,
        logical_to_physical_map=None,
        topk_indices_dtype=torch.int32,
        topk_weights_dtype=torch.float32,
        override=None,
        solution="triton",
    )
    theirs = _slot(ids_ref, weights_ref, offset, local, tile_n)
    flashinfer_moe.populate_trtllm_moe_routing_metadata_(theirs, ids_ref, weights_ref)
    ours = _slot(ids_ref, weights_ref, offset, local, tile_n)
    weights, ids = triton_sigmoid_bias_topk_routing(
        logits, bias, topk, 1.5, normalize, offset, local, ours.slots[0], False
    )
    torch.cuda.synchronize()
    assert torch.equal(ids, ids_ref) and torch.equal(weights, weights_ref)
    for name in _WRITTEN:
        assert torch.equal(
            getattr(ours.slots[0], name), getattr(theirs.slots[0], name)
        ), name


def test_fused_topk_routing_refuses_wide_batches() -> None:
    logits = torch.randn(5, 256, device="cuda")
    ids = logits.topk(8, dim=-1).indices.int()
    slot = _slot(ids, torch.ones(5, 8, device="cuda"), 64, 64, 8).slots[0]
    with pytest.raises(ValueError, match="1 to 4 tokens"):
        triton_sigmoid_bias_topk_routing(
            logits, torch.zeros(256, device="cuda"), 8, 1.0, True, 64, 64, slot, False
        )


EXPERTS, LOCAL, HIDDEN, INTERMEDIATE, TOPK = 32, 8, 512, 512, 4


def _bytes(
    generator: torch.Generator, *shape: int, low: int, high: int
) -> torch.Tensor:
    return torch.randint(
        low, high, shape, device="cuda", dtype=torch.uint8, generator=generator
    )


def _weights(generator: torch.Generator) -> SimpleNamespace:
    """Random NVFP4 expert weights in the kernel's shuffled layouts; scales stay small and finite."""
    ones = torch.ones(LOCAL, device="cuda")
    return SimpleNamespace(
        _spec=SimpleNamespace(
            num_experts=EXPERTS, top_k=TOPK, ep_rank=1, num_local_experts=LOCAL
        ),
        intermediate_size_per_partition=INTERMEDIATE,
        gemm1_weights_fp4_shuffled=_bytes(
            generator, LOCAL, 2 * INTERMEDIATE, HIDDEN // 2, low=0, high=256
        ),
        gemm1_scales_fp4_shuffled=_bytes(
            generator, LOCAL, 2 * INTERMEDIATE, HIDDEN // 16, low=0x30, high=0x40
        ),
        gemm2_weights_fp4_shuffled=_bytes(
            generator, LOCAL, HIDDEN, INTERMEDIATE // 2, low=0, high=256
        ),
        gemm2_scales_fp4_shuffled=_bytes(
            generator, LOCAL, HIDDEN, INTERMEDIATE // 16, low=0x30, high=0x40
        ),
        g1_scale_c=ones,
        g1_alphas=ones,
        g2_alphas=ones,
    )


def _logits(generator: torch.Generator, tokens: int) -> torch.Tensor:
    return torch.randn(tokens, EXPERTS, device="cuda", generator=generator) * 3.0


def _routing(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The route weights and ids the fused kernel selects for these logits."""
    ids, weights = torch_grouped_sigmoid_topk(
        logits, torch.zeros(logits.shape[1], device=logits.device), TOPK, 1, 1, 1.0
    )
    return weights, ids.to(torch.int32)


def _quantized(
    generator: torch.Generator, tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _bytes(generator, tokens, HIDDEN // 2, low=0, high=256),
        _bytes(generator, tokens, HIDDEN // 16, low=0x30, high=0x40),
    )


def _populate(prepared, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the routes and fill the prepared slot in the fused top-k kernel."""
    return triton_sigmoid_bias_topk_routing(
        logits,
        torch.zeros(EXPERTS, device="cuda"),
        TOPK,
        1.0,
        True,
        LOCAL,
        LOCAL,
        prepared.routing,
        False,
    )


def _apply(w, x_quantized, weights, ids, enable_pdl, prepared, do_finalize=False):
    tokens = x_quantized[0].shape[0]
    x = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    return module.flashinfer_trtllm_nvfp4_routed_moe_apply(
        plan={},
        x=x,
        w=w,
        router_logits=x,
        topk_weights=weights,
        topk_ids=ids,
        do_finalize=do_finalize,
        enable_pdl=enable_pdl,
        x_quantized=x_quantized,
        routing_prepared=prepared,
    )


def _rows(result, tokens: int) -> torch.Tensor:
    """Every local route's GEMM2 row in (token, route) order; zeros where another rank owns the route."""
    gemm2_out, _, permute_map = result
    permute_map = permute_map.view(tokens, TOPK).long()
    rows = gemm2_out[permute_map.clamp_min(0)]
    # where, not a mask multiply: a remote route's row is uninitialized and a NaN there would survive.
    return torch.where((permute_map >= 0).unsqueeze(-1), rows, torch.zeros_like(rows))


@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("tokens", [1, 4])
def test_prepared_routing_reproduces_the_fused_launch_bitwise(
    tokens: int, enable_pdl: bool
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(tokens)
    w = _weights(generator)
    x_quantized = _quantized(generator, tokens)
    logits = _logits(generator, tokens)
    weights, ids = _routing(logits)
    fused = _apply(w, x_quantized, weights, ids, enable_pdl, None)
    prepared = module.prepare_nvfp4_routing(
        w, x_quantized, ids, weights, None, enable_pdl
    )
    # The routing runs on a side stream, as a model overlaps it with its quantize.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fused_weights, fused_ids = _populate(prepared, logits)
    torch.cuda.current_stream().wait_stream(side)
    assert torch.equal(fused_ids, ids)
    split = _apply(w, x_quantized, fused_weights, fused_ids, enable_pdl, prepared)
    assert torch.equal(_rows(split, tokens), _rows(fused, tokens))
    assert torch.equal(split[1], fused_weights)


def test_prepared_routing_replays_new_top_k_under_a_graph() -> None:
    tokens = 2
    generator = torch.Generator(device="cuda").manual_seed(7)
    w = _weights(generator)
    x_quantized = _quantized(generator, tokens)
    logits = _logits(generator, tokens)
    weights, ids = _routing(logits)
    prepared = module.prepare_nvfp4_routing(w, x_quantized, ids, weights, None, False)
    _populate(prepared, logits)
    _apply(w, x_quantized, weights, ids, False, prepared)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = _populate(prepared, logits)
        split = _apply(w, x_quantized, weights, ids, False, prepared)
    for seed in (11, 12):
        logits.copy_(_logits(torch.Generator(device="cuda").manual_seed(seed), tokens))
        graph.replay()
        replayed = _rows(split, tokens).clone()
        torch.cuda.synchronize()
        fused = _apply(w, x_quantized, *_routing(logits), False, None)
        assert torch.equal(replayed, _rows(fused, tokens))


def test_prepared_routing_refuses_what_it_was_not_prepared_for() -> None:
    generator = torch.Generator(device="cuda").manual_seed(3)
    w = _weights(generator)
    x_quantized = _quantized(generator, 1)
    weights, ids = _routing(_logits(generator, 1))
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="outside graph capture"):
        with torch.cuda.graph(graph):
            module.prepare_nvfp4_routing(w, x_quantized, ids, weights, None, False)
    prepared = module.prepare_nvfp4_routing(w, x_quantized, ids, weights, None, False)
    two_weights, two_ids = _routing(_logits(generator, 2))
    with pytest.raises(ValueError, match="serves 1 tokens"):
        _apply(w, _quantized(generator, 2), two_weights, two_ids, False, prepared)
    with pytest.raises(ValueError, match="deferred finalize"):
        _apply(w, x_quantized, weights, ids, False, prepared, do_finalize=True)
