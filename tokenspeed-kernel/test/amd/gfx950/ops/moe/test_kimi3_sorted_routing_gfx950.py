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
_ROWS = (1, 32, 64, 128, 192, 256)


def _logits(kind: str, rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "random":
        logits = torch.randn((rows, _EXPERTS), generator=generator, device="cuda")
        bias = torch.randn((_EXPERTS,), generator=generator, device="cuda") * 0.01
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


def _fused(logits, bias, scale, normalize, scratch, shared_down=None):
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
        shared_down=shared_down,
    )


# K3's TP8 shared-expert down projection: [rows, 768] @ [7168, 768].T.
_SHARED_K = 768
_SHARED_N = 7168


def _shared_down_inputs(rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn((rows, _SHARED_K), generator=generator, device="cuda")
    weight = torch.randn((_SHARED_N, _SHARED_K), generator=generator, device="cuda")
    return hidden.to(torch.bfloat16), (weight * 0.03).to(torch.bfloat16)


def _check_shared_down(got, hidden, weight):
    # An FP32 reduction in another order stays within one BF16 rounding.
    want = hidden.float() @ weight.float().T
    torch.testing.assert_close(got.float(), want, rtol=2**-7, atol=1e-4)


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
@pytest.mark.parametrize("kind", ("random", "coarse", "same_rows", "special"))
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


def test_sorted_routing_takes_only_its_layer_and_logits():
    from tokenspeed_kernel.ops.moe import (
        kimi3_sorted_routing_accepts,
        kimi3_sorted_routing_supported,
    )

    assert kimi3_sorted_routing_supported(_EXPERTS, _TOPK)
    assert not kimi3_sorted_routing_supported(_EXPERTS, 8)
    assert not kimi3_sorted_routing_supported(384, _TOPK)
    logits = torch.zeros((257, _EXPERTS), device="cuda")
    assert kimi3_sorted_routing_accepts(logits[:256])
    assert not kimi3_sorted_routing_accepts(logits)
    assert not kimi3_sorted_routing_accepts(logits[:8].half())
    assert not kimi3_sorted_routing_accepts(logits[:8, :448])
    assert not kimi3_sorted_routing_accepts(logits[:16:2])


def test_sorted_routing_graph_replay_and_one_binary():
    """Layers of several row counts, launched back to back and captured, share
    the scratch and one binary."""
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
        eager = [_fused(*layer, 2.827, True, scratch) for layer in layers]
        stream.synchronize()
        for (logits, _), got, want in zip(layers, eager, references, strict=True):
            _check(got, want, logits.shape[0], order_exact=True)
        assert not torch.any(scratch)
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
    assert plan["sorted_route_block_size"] == _BLOCK
    moe_process_weights(plan, module)
    return plan, module


@pytest.mark.parametrize("rows", (8, 64, 192))
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
    # Serve rows through 192 on the warp decode, the runtime's fused routing bound.
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


@pytest.mark.parametrize("rows", (1, 33, 128))
@pytest.mark.parametrize("out_kind", ("contiguous", "strided", "allocated"))
def test_sorted_routing_carries_shared_down(rows, out_kind):
    """The shared down projection rides along without changing the routes."""
    from tokenspeed_kernel.ops.moe._triton.kimi3_sigmoid_topk import (
        kimi3_sorted_routing_scratch,
    )

    scratch = kimi3_sorted_routing_scratch(torch.device("cuda"))
    logits, bias = _logits("random", rows, 500 + rows)
    hidden, weight = _shared_down_inputs(rows, 600 + rows)
    wide = torch.full((rows, _SHARED_N + 64), -7.0, dtype=torch.bfloat16, device="cuda")
    out = {
        "contiguous": torch.full_like(wide[:, :_SHARED_N].contiguous(), -7.0),
        "strided": wide[:, :_SHARED_N],
        "allocated": None,
    }[out_kind]
    carried = _fused(logits, bias, 2.827, True, scratch, (hidden, weight, out))
    torch.cuda.synchronize()
    assert len(carried) == 7
    _check(carried[:6], _reference(logits, bias, 2.827, True, True), rows, True)
    if out is not None:
        assert carried[6] is out
    _check_shared_down(carried[6], hidden, weight)
    # A row-strided output keeps the columns past its width.
    assert torch.all(wide[:, _SHARED_N:] == -7.0)
    assert not torch.any(scratch)


def test_sorted_routing_shared_down_graph_replay():
    """Captured launches carrying the projection replay with their own rows."""
    from tokenspeed_kernel.ops.moe._triton.kimi3_sigmoid_topk import (
        kimi3_sorted_routing_scratch,
    )

    scratch = kimi3_sorted_routing_scratch(torch.device("cuda"))
    layers = []
    for rows in (8, 96, 16, 33):
        logits, bias = _logits("random", rows, 900 + rows)
        hidden, weight = _shared_down_inputs(rows, 950 + rows)
        out = torch.empty((rows, _SHARED_N), dtype=torch.bfloat16, device="cuda")
        layers.append((logits, bias, hidden, weight, out))
    references = [_reference(layer[0], layer[1], 2.827, True, True) for layer in layers]
    stream = torch.cuda.Stream()
    snapshots = []
    with torch.cuda.stream(stream):
        for logits, bias, hidden, weight, out in layers:
            _fused(logits, bias, 2.827, True, scratch, (hidden, weight, out))
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for logits, bias, hidden, weight, out in layers:
                routes = _fused(
                    logits, bias, 2.827, True, scratch, (hidden, weight, out)
                )
                snapshots.append(tuple(t.clone() for t in routes[:6]))
    torch.cuda.synchronize()
    for _ in range(2):
        for layer_snapshots, (*_, out) in zip(snapshots, layers, strict=True):
            out.fill_(-7)
            for tensor in layer_snapshots:
                tensor.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        for layer, got, want in zip(layers, snapshots, references, strict=True):
            logits, _, hidden, weight, out = layer
            _check(got, want, logits.shape[0], order_exact=True)
            _check_shared_down(out, hidden, weight)
        assert not torch.any(scratch)


def test_sorted_routing_rejects_unsupported_shared_down():
    from tokenspeed_kernel.ops.moe._triton.kimi3_sigmoid_topk import (
        kimi3_sorted_routing_fuses_shared_down,
        kimi3_sorted_routing_scratch,
    )

    scratch = kimi3_sorted_routing_scratch(torch.device("cuda"))
    hidden, weight = _shared_down_inputs(129, 7)
    assert kimi3_sorted_routing_fuses_shared_down(hidden[:128], weight, None)
    # Above 32 rows the programs write 64 columns each.
    assert kimi3_sorted_routing_fuses_shared_down(hidden[:32], weight[:7136], None)
    assert not kimi3_sorted_routing_fuses_shared_down(hidden[:33], weight[:7136], None)
    out = torch.empty((16, _SHARED_N), dtype=torch.bfloat16, device="cuda")
    unsupported = (
        # More rows than the MFMA tiles cover.
        (hidden, weight, None),
        (hidden[:16].float(), weight, None),
        # K outside the 256-wide steps, N outside the 32-column programs.
        (hidden[:16, :320].contiguous(), weight[:, :320].contiguous(), None),
        (hidden[:16], weight[:7160], None),
        (hidden[:16], weight, out[:8]),
        (hidden[:16], weight, out.float()),
        (hidden[:16], weight, torch.empty_like(out).t().contiguous().t()),
    )
    for case in unsupported:
        assert not kimi3_sorted_routing_fuses_shared_down(*case)
    logits, bias = _logits("random", 16, 3)
    for case in unsupported[1:]:
        with pytest.raises(ValueError, match="shared_down"):
            _fused(logits, bias, 1.0, True, scratch, case)
    # The projection's rows must be the routed rows.
    with pytest.raises(ValueError, match="shared_down"):
        _fused(logits, bias, 1.0, True, scratch, (hidden[:8], weight, None))


def test_sorted_routing_rejects_unsupported_rows():
    from tokenspeed_kernel.ops.moe._triton.kimi3_sigmoid_topk import (
        kimi3_sorted_routing_scratch,
    )

    scratch = kimi3_sorted_routing_scratch(torch.device("cuda"))
    logits, bias = _logits("random", 257, 3)
    with pytest.raises(ValueError):
        _fused(logits, bias, 1.0, True, scratch)


def test_sorted_routing_rejects_strided_scratch():
    """The arrival masks are addressed as a dense word array."""
    from tokenspeed_kernel.ops.moe._triton.kimi3_sigmoid_topk import (
        kimi3_sorted_routing_scratch,
    )

    words = kimi3_sorted_routing_scratch(torch.device("cuda")).numel()
    scratch = torch.zeros(2 * words, dtype=torch.int32, device="cuda")[::2]
    logits, bias = _logits("random", 16, 5)
    with pytest.raises(ValueError, match="kimi3_sorted_routing_scratch"):
        _fused(logits, bias, 1.0, True, scratch)
