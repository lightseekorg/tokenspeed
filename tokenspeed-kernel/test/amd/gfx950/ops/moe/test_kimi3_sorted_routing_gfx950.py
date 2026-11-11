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

"""The one-launch K3 routing and expert sort against top-k plus gluon_moe_sorting."""

import pytest
import torch
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not current_platform().is_cdna4, reason="the gfx950 MoE sort format is CDNA4"
)

_EXPERTS = 896
_TOPK = 16
_BLOCK = 16
_ROWS = (1, 8, 16, 32, 64, 128, 192, 256)


def _logits(kind: str, rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "random":
        logits = torch.randn((rows, _EXPERTS), generator=generator, device="cuda")
        bias = torch.randn((_EXPERTS,), generator=generator, device="cuda") * 0.01
    elif kind == "equal":
        # Every score ties: the lowest expert ids win on every row.
        logits = torch.zeros((rows, _EXPERTS), device="cuda")
        bias = torch.zeros((_EXPERTS,), device="cuda")
    elif kind == "coarse":
        # Few distinct values: ties inside and across the top 16.
        logits = (
            torch.randint(0, 4, (rows, _EXPERTS), generator=generator, device="cuda")
            / 4.0
        )
        bias = (
            torch.randint(0, 2, (_EXPERTS,), generator=generator, device="cuda") / 8.0
        )
    elif kind == "same_rows":
        # Every row picks the same 16 experts, each expert's longest run.
        row = torch.randn((1, _EXPERTS), generator=generator, device="cuda")
        logits = row.expand(rows, _EXPERTS).contiguous()
        bias = torch.zeros((_EXPERTS,), device="cuda")
    elif kind == "special":
        logits = torch.randn((rows, _EXPERTS), generator=generator, device="cuda")
        logits[:, ::7] = float("inf")
        logits[:, 3::11] = -float("inf")
        logits[0, 5] = float("nan")
        bias = torch.randn((_EXPERTS,), generator=generator, device="cuda") * 0.01
    else:
        raise ValueError(kind)
    return logits.float().contiguous(), bias.float().contiguous()


def _reference(logits, bias, scale, normalize, deterministic_order):
    from tokenspeed_kernel.ops.moe._triton.kimi3_sigmoid_topk import (
        kimi3_sigmoid_bias_topk,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.moe_sorting import (
        gluon_moe_sorting,
    )

    weights, ids = kimi3_sigmoid_bias_topk(
        logits,
        bias,
        routed_scaling_factor=scale,
        normalize_topk_weights=normalize,
    )
    # The A8W4 decode's call (fused/moe.py), whose stage GEMMs read this format.
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = gluon_moe_sorting(
        ids,
        weights,
        _EXPERTS,
        7168,
        torch.bfloat16,
        _BLOCK,
        use_minimum_route_chunks=True,
        runtime_rows=False,
        deterministic_order=deterministic_order,
    )
    return weights, ids, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids


def _fused(logits, bias, scale, normalize, scratch):
    from tokenspeed_kernel.ops.moe._triton.kimi3_sigmoid_topk import (
        kimi3_sigmoid_bias_topk_sorted,
    )

    return kimi3_sigmoid_bias_topk_sorted(
        logits,
        bias,
        routed_scaling_factor=scale,
        normalize_topk_weights=normalize,
        block_size=_BLOCK,
        scratch=scratch,
    )


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.int32)


def _route_blocks(sorted_ids, sorted_weights, sorted_expert_ids, total, rows):
    """Each route's sorted expert block and weight bits, indexed by route."""
    ids = sorted_ids[:total].long()
    token, slot = ids & 0xFFFFFF, ids >> 24
    real = token < rows
    route = token[real] * _TOPK + slot[real]
    position = torch.arange(total, device=ids.device)[real]
    expert = torch.full((rows * _TOPK,), -1, dtype=torch.int64, device=ids.device)
    weight = torch.zeros((rows * _TOPK,), dtype=torch.int32, device=ids.device)
    expert[route] = sorted_expert_ids[position // _BLOCK].long()
    weight[route] = _bits(sorted_weights[:total])[real]
    return expert, weight, int(real.sum())


def _check(fused, reference, rows, order_exact):
    weights, ids, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = fused
    (
        want_weights,
        want_ids,
        want_sorted_ids,
        want_sorted_weights,
        want_sorted_expert_ids,
        want_num_valid_ids,
    ) = reference
    assert torch.equal(ids, want_ids)
    assert torch.equal(_bits(weights), _bits(want_weights))
    for got, want in (
        (sorted_ids, want_sorted_ids),
        (sorted_weights, want_sorted_weights),
        (sorted_expert_ids, want_sorted_expert_ids),
    ):
        assert got.shape == want.shape and got.dtype == want.dtype
    assert torch.equal(num_valid_ids, want_num_valid_ids)
    total = int(num_valid_ids[0])
    assert int(num_valid_ids[1]) == rows and total % _BLOCK == 0
    blocks = total // _BLOCK
    # The block map and the padding follow the expert counts alone.
    assert torch.equal(sorted_expert_ids[:blocks], want_sorted_expert_ids[:blocks])
    padding = sorted_ids[:total] >> 24 == _TOPK
    assert torch.equal(padding, want_sorted_ids[:total] >> 24 == _TOPK)
    assert torch.all(sorted_ids[:total][padding] == (_TOPK << 24) | rows)
    if order_exact:
        # Against deterministic_order=True every valid slot matches.
        assert torch.equal(sorted_ids[:total], want_sorted_ids[:total])
        real = ~padding
        assert torch.equal(
            _bits(sorted_weights[:total])[real],
            _bits(want_sorted_weights[:total])[real],
        )
    else:
        # The default sort keeps arrival order inside an expert's run.
        got = _route_blocks(sorted_ids, sorted_weights, sorted_expert_ids, total, rows)
        want = _route_blocks(
            want_sorted_ids, want_sorted_weights, want_sorted_expert_ids, total, rows
        )
        assert got[2] == want[2] == rows * _TOPK
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
        assert torch.equal(got[0], ids.flatten().long())
        assert torch.equal(got[1], _bits(weights).flatten())


@pytest.mark.parametrize("rows", _ROWS)
@pytest.mark.parametrize("kind", ("random", "equal", "coarse", "same_rows", "special"))
@pytest.mark.parametrize("deterministic_order", (True, False))
def test_sorted_routing_matches_topk_and_sort(rows, kind, deterministic_order):
    from tokenspeed_kernel.ops.moe._triton.kimi3_sigmoid_topk import (
        kimi3_sorted_routing_scratch,
    )

    scratch = kimi3_sorted_routing_scratch(torch.device("cuda"))
    for seed, (scale, normalize) in enumerate(
        ((2.827, True), (1.0, True), (2.5, False))
    ):
        logits, bias = _logits(kind, rows, 1000 * rows + seed)
        reference = _reference(logits, bias, scale, normalize, deterministic_order)
        fused = _fused(logits, bias, scale, normalize, scratch)
        torch.cuda.synchronize()
        _check(fused, reference, rows, order_exact=deterministic_order)
        # Each launch leaves the arrival masks and counter zeroed.
        assert not torch.any(scratch)


def test_sorted_routing_back_to_back_launches():
    """Launches of varying rows queued back to back each see only their rows."""
    from tokenspeed_kernel.ops.moe._triton.kimi3_sigmoid_topk import (
        kimi3_sorted_routing_scratch,
    )

    scratch = kimi3_sorted_routing_scratch(torch.device("cuda"))
    generator = torch.Generator().manual_seed(11)
    cases = []
    for step in range(48):
        rows = int(torch.randint(1, 257, (1,), generator=generator))
        kind = ("random", "coarse", "same_rows")[step % 3]
        logits, bias = _logits(kind, rows, 300 + step)
        cases.append((logits, bias, _fused(logits, bias, 2.827, True, scratch)))
    torch.cuda.synchronize()
    for logits, bias, fused in cases:
        reference = _reference(logits, bias, 2.827, True, True)
        _check(fused, reference, logits.shape[0], order_exact=True)
    assert not torch.any(scratch)


def test_sorted_routing_graph_replay_and_one_binary():
    """Captured layers of several row counts share the scratch and one binary."""
    from tokenspeed_kernel.ops.moe._triton.kimi3_sigmoid_topk import (
        _kimi3_sigmoid_bias_topk_sorted_kernel,
        kimi3_sorted_routing_scratch,
    )

    scratch = kimi3_sorted_routing_scratch(torch.device("cuda"))
    # One binary per mask-word bucket: through 32, 64, 128 and 256 rows.
    for rows in (3, 64, 128, 256):
        _fused(*_logits("random", rows, 0), 2.827, True, scratch)
    torch.cuda.synchronize()
    layers = [
        _logits("random", rows, 77 + rows)
        for rows in (16, 192, 8, 64, 17, 128, 33, 255)
    ]
    references = [_reference(*layer, 2.827, True, True) for layer in layers]
    snapshots = []
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), assert_no_triton_compile(
        _kimi3_sigmoid_bias_topk_sorted_kernel
    ):
        for layer in layers:
            _fused(*layer, 2.827, True, scratch)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for layer in layers:
                outputs = _fused(*layer, 2.827, True, scratch)
                snapshots.append(tuple(t.clone() for t in outputs))
    torch.cuda.synchronize()
    for _ in range(3):
        for layer_snapshots in snapshots:
            for tensor in layer_snapshots:
                tensor.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        for (logits, _), got, want in zip(layers, snapshots, references, strict=True):
            _check(got, want, logits.shape[0], order_exact=True)
        assert not torch.any(scratch)


@pytest.fixture(scope="module")
def _k3_tp8_experts():
    """The K3 TP8 rank's A8W4 SiTU experts: 896 x (3584 -> 384 -> 3584)."""
    from tokenspeed_kernel.ops.moe import moe_plan, moe_process_weights
    from utils import make_mxfp4_moe_weights

    generator = torch.Generator(device="cuda").manual_seed(20261110)
    raw = make_mxfp4_moe_weights(_EXPERTS, 3584, 384, generator)
    module = torch.nn.Module()
    for name, key in (
        ("w13_weight", "w13_weight"),
        ("w13_weight_scale", "w13_scale"),
        ("w2_weight", "w2_weight"),
        ("w2_weight_scale", "w2_scale"),
    ):
        setattr(module, name, torch.nn.Parameter(raw[key], requires_grad=False))
    module.top_k = _TOPK
    module.num_experts = _EXPERTS
    module.ep_size = 1
    module.activation_situ_beta = 4.0
    module.activation_situ_linear_beta = 25.0
    plan = moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        routing_mode="precomputed_topk",
        ep_size=1,
        ispp=384,
        internal_activation_dtype="input",
        solution="gluon",
        hidden=None,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
    )
    assert plan["apply_kernel_name"] == "gluon_mxfp4_a8w4_situ_precomputed_moe_apply"
    moe_process_weights(plan, module)
    return plan, module


@pytest.mark.parametrize("rows", (8, 16, 32, 64, 128, 192))
@pytest.mark.parametrize("kind", ("random", "same_rows"))
def test_a8w4_moe_takes_the_fused_sort_bitwise(
    _k3_tp8_experts, rows, kind, monkeypatch
):
    """The sorted stage GEMMs give the same output from either sort."""
    from tokenspeed_kernel.ops.moe import moe_apply
    from tokenspeed_kernel.ops.moe._triton.kimi3_sigmoid_topk import (
        kimi3_sorted_routing_scratch,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import moe as fused_moe
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.moe_sorting import (
        gluon_moe_sorting,
    )

    plan, module = _k3_tp8_experts
    # Serve 128 and 192 rows on the warp decode, as the rows-192 experiments do.
    monkeypatch.setattr(fused_moe, "_SITU_WARP_DECODE_MAX_M", 192)
    sorts = []

    def counted_sort(*args, **kwargs):
        sorts.append(args[0].shape[0])
        return gluon_moe_sorting(*args, **kwargs)

    monkeypatch.setattr(
        "tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.moe_sorting.gluon_moe_sorting",
        counted_sort,
    )
    logits, bias = _logits(kind, rows, 5 + rows)
    hidden = (
        torch.randn(
            (rows, 3584),
            generator=torch.Generator(device="cuda").manual_seed(rows),
            device="cuda",
        )
        * 0.1
    ).to(torch.bfloat16)
    scratch = kimi3_sorted_routing_scratch(torch.device("cuda"))
    weights, ids, *sorted_routes = _fused(logits, bias, 2.827, True, scratch)

    def apply(routes):
        module._situ_sorted_routes = routes
        return moe_apply(
            plan, hidden, module, logits, topk_weights=weights, topk_ids=ids
        )

    expected = apply(None).clone()
    assert sorts == [rows]
    actual = apply(tuple(sorted_routes))
    assert sorts == [rows]
    assert module._situ_sorted_routes is None
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_sorted_routing_rejects_unsupported_rows():
    from tokenspeed_kernel.ops.moe._triton.kimi3_sigmoid_topk import (
        kimi3_sorted_routing_scratch,
    )

    scratch = kimi3_sorted_routing_scratch(torch.device("cuda"))
    logits, bias = _logits("random", 257, 3)
    with pytest.raises(ValueError):
        _fused(logits, bias, 1.0, True, scratch)
