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

"""The fused CuTe DSL decode front against the launches it replaces.

``cute_dsl_dsa_decode_front`` must equal, bit for bit, ``rmsnorm_fused_parallel``'s
``q_norm`` and in-place ``kv_a``, the index-K plane bytes of
``index_k_layernorm_fp8_scatter_and_quantize_query`` and the E4M3 query and FP32
weights ``quantize_index_query_weights`` gives for the front's own BF16 index
query. Its BF16 projection sums K in its own order: every value is within one
BF16 step of the FP64 product (plus the FP32 accumulation bound where the dot
product cancels), and on positive data, which cannot cancel, within one BF16
step of the GEMM it replaces.
"""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_cute_compile, is_nvidia, move_pages_past_int32

if not is_nvidia():
    pytest.skip("requires an NVIDIA GPU", allow_module_level=True)
pytest.importorskip("cutlass")
from tokenspeed_kernel import platform  # noqa: E402

if (
    not platform.ArchVersion(10, 0)
    <= platform.current_platform().arch_version
    <= platform.ArchVersion(10, 3)
):
    pytest.skip("requires SM100 or SM103", allow_module_level=True)

from tokenspeed_kernel import selection, signature  # noqa: E402
from tokenspeed_kernel.ops import gemm  # noqa: E402
from tokenspeed_kernel.ops.attention.dsa import cute_dsl as dsa_cute_dsl  # noqa: E402
from tokenspeed_kernel.ops.attention.dsa._triton import (  # noqa: E402
    topk as dsa_triton_topk,
)
from tokenspeed_kernel.ops.layernorm.cuda import rmsnorm_fused_parallel  # noqa: E402

if dsa_cute_dsl._decode_front is None:
    pytest.skip("the fused CuTe DSL decode front is not built", allow_module_level=True)

Q_LORA, LATENT, ROPE, GATE = 1536, 512, 64, 1536
HEADS, DIM, PAGE = 32, 128, 64
Q_B = 2304  # TP8: 12 local heads of 192
WIDTH = Q_B + HEADS * DIM
# The decode chain's packed projection row: [q_a | kv_a | rope | gate | wk | weights].
PACKED = Q_LORA + LATENT + ROPE + GATE + DIM + HEADS
SOFTMAX_SCALE = DIM**-0.5 * HEADS**-0.5
NORM_EPS, KEY_EPS = 1e-5, 1e-6
PAGES = 24


def _case(rows, *, seed, slot_dtype, norm_dtype, positive=False):
    """One decode round's inputs as the K3 decode chain hands them.

    They are column views of a packed projection, the layer's weights and a
    random-byte plane.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", generator=generator).to(dtype)

    packed = randn(rows, PACKED) * 2
    weight = randn(WIDTH, Q_LORA) * 0.03
    if positive:
        packed.abs_()
        weight.abs_()
    q_a, latent, _, key, head_w = packed.split(
        (Q_LORA, LATENT + ROPE, GATE, DIM, HEADS), dim=-1
    )
    if rows > 1 and not positive:
        q_a[1].zero_()  # an all-zero row: rsqrt(eps)
        weight[Q_B : Q_B + DIM].zero_()  # index head 0 takes the amax floor
        # BF16 subnormals: RMSNormFusedParallel (-use_fast_math) flushes them.
        q_a[0, :7] = torch.tensor(1e-39, dtype=torch.bfloat16)
        latent[0, 3] = torch.tensor(-5e-40, dtype=torch.bfloat16)
        key[0, :5] *= 300  # an outlier the key's FP8 scale follows
    plane = torch.randint(
        0,
        256,
        (PAGES * PAGE, DIM + 4),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    slots = torch.randperm(PAGES * PAGE, device="cuda", generator=generator)[:rows]
    return dict(
        packed=packed,
        q_a=q_a,
        kv_a=latent[:, :LATENT],
        key=key,
        head_w=head_w,
        q_gamma=1 + randn(Q_LORA) * 0.1,
        kv_gamma=1 + randn(LATENT) * 0.1,
        weight=weight.contiguous(),
        k_gamma=(1 + randn(DIM) * 0.1).to(norm_dtype),
        k_beta=(randn(DIM) * 0.1).to(norm_dtype),
        plane=plane,
        slots=slots.to(slot_dtype),
    )


def _front(case):
    return dsa_cute_dsl.cute_dsl_dsa_decode_front(
        case["q_a"],
        case["kv_a"],
        case["key"],
        case["head_w"],
        q_norm_weight=case["q_gamma"],
        kv_norm_weight=case["kv_gamma"],
        norm_eps=NORM_EPS,
        query_weight=case["weight"],
        q_b_rows=Q_B,
        key_norm_weight=case["k_gamma"],
        key_norm_bias=case["k_beta"],
        key_norm_eps=KEY_EPS,
        index_k_cache=case["plane"],
        slots=case["slots"],
        page_size=PAGE,
        softmax_scale=SOFTMAX_SCALE,
    )


def _bits(x: torch.Tensor) -> torch.Tensor:
    """The bit patterns of BF16 or FP32 ``x`` (``torch.equal`` takes -0 for +0)."""
    return x.view(torch.int16 if x.element_size() == 2 else torch.int32)


def _bf16_step(x: torch.Tensor) -> torch.Tensor:
    """The BF16 spacing at |x| (8 significant bits; the subnormal spacing below)."""
    exponent = torch.floor(torch.log2(x.abs().double().clamp_min(2.0**-126)))
    return torch.exp2(exponent - 7)


def _check(case, *, positive):
    """Run the front on ``case`` and compare it with the separate launches."""
    rows = case["q_a"].shape[0]
    q_a, kv_a = case["q_a"].clone(), case["kv_a"].clone()
    q_norm_ref, kv_ref = torch.empty_like(q_a), kv_a.clone()
    rmsnorm_fused_parallel(
        q_a, case["q_gamma"], q_norm_ref, kv_ref, case["kv_gamma"], kv_ref, NORM_EPS
    )
    plane_ref = case["plane"].clone()

    q_norm, projected, q_fp8, weights = _front(case)
    assert torch.equal(_bits(q_norm), _bits(q_norm_ref))
    assert torch.equal(_bits(case["kv_a"]), _bits(kv_ref))

    # The plane and the query quantization of the front's own BF16 index query.
    index_query = projected[:, Q_B:].view(rows, HEADS, DIM)
    q_ref, w_ref = dsa_triton_topk.index_k_layernorm_fp8_scatter_and_quantize_query(
        plane_ref,
        case["key"],
        case["k_gamma"],
        case["k_beta"],
        case["slots"],
        index_query,
        case["head_w"],
        eps=KEY_EPS,
        page_size=PAGE,
        group_size=128,
        write_mask=None,
        softmax_scale=SOFTMAX_SCALE,
    )
    assert torch.equal(case["plane"], plane_ref)
    assert q_fp8.shape == (rows, HEADS, DIM) and q_fp8.is_contiguous()
    assert torch.equal(q_fp8.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(_bits(weights), _bits(w_ref))

    # The projection: one BF16 step from FP64, plus the FP32 accumulation bound.
    exact = q_norm_ref.double() @ case["weight"].double().T
    bound = 2.0**-20 * (q_norm_ref.double().abs() @ case["weight"].double().abs().T)
    got = projected.double()
    step = _bf16_step(torch.maximum(got.abs(), exact.abs()))
    assert ((got - exact).abs() <= step + bound).all()
    library = gemm.mm(q_norm_ref, case["weight"]).double()
    step = _bf16_step(torch.maximum(got.abs(), library.abs()))
    slack = 0 if positive else 2 * bound
    assert ((got - library).abs() <= step + slack).all()


@pytest.mark.parametrize(
    "rows", [1, 3, 8, 16, 17, 24, 33, 48, 64], ids=lambda rows: f"rows{rows}"
)
@pytest.mark.parametrize("positive", [False, True], ids=["signed", "positive"])
def test_the_front_matches_the_launches_it_replaces(rows, positive) -> None:
    case = _case(
        rows,
        seed=rows,
        slot_dtype=torch.int64,
        norm_dtype=torch.float32,
        positive=positive,
    )
    _check(case, positive=positive)


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("norm_dtype", [torch.float32, torch.bfloat16])
def test_the_front_takes_either_slot_and_key_norm_dtype(slot_dtype, norm_dtype) -> None:
    case = _case(13, seed=7, slot_dtype=slot_dtype, norm_dtype=norm_dtype)
    _check(case, positive=False)


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_far_pages_past_int32_offsets(slot_dtype) -> None:
    """Slots in pages past 2**31 bytes of the plane write as near slots do."""
    near = _case(8, seed=19, slot_dtype=slot_dtype, norm_dtype=torch.float32)
    far = _case(8, seed=19, slot_dtype=slot_dtype, norm_dtype=torch.float32)
    page_bytes = PAGE * (DIM + 4)
    (pool,), first = move_pages_past_int32(
        (far["plane"].view(PAGES, page_bytes),), torch.zeros(1, dtype=torch.int64)
    )
    first = int(first)
    assert first * page_bytes > 2**31
    far["plane"] = pool.view(-1, DIM + 4)
    far["slots"] = far["slots"] + first * PAGE
    for got, want in zip(_front(far), _front(near)):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
    assert torch.equal(pool[first:].view(-1, DIM + 4), near["plane"])
    assert torch.equal(_bits(far["kv_a"]), _bits(near["kv_a"]))


def test_the_front_replays_in_a_cuda_graph_without_tracing() -> None:
    """Captured once per row tile, the front replays over refreshed inputs."""
    case = _case(8, seed=11, slot_dtype=torch.int64, norm_dtype=torch.float32)
    _front(case)  # compiles the tile's kernel before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with assert_no_cute_compile(), torch.cuda.graph(graph):
        captured = _front(case)
    fresh = _case(8, seed=12, slot_dtype=torch.int64, norm_dtype=torch.float32)
    for name in ("packed", "weight", "q_gamma", "kv_gamma", "k_gamma", "k_beta"):
        case[name].copy_(fresh[name])
    case["slots"].copy_(fresh["slots"])
    plane = case["plane"].clone()
    graph.replay()
    torch.cuda.synchronize()
    # The eager front over the same inputs from the same plane.
    replayed = [t.clone() for t in captured]
    replayed_plane = case["plane"].clone()
    kv_after = case["kv_a"].clone()
    case["packed"].copy_(fresh["packed"])
    case["plane"].copy_(plane)
    eager = _front(case)
    for got, want in zip(replayed, eager):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
    assert torch.equal(replayed_plane, case["plane"])
    assert torch.equal(_bits(kv_after), _bits(case["kv_a"]))


def test_the_k31_traits_select_the_front() -> None:
    kernel = selection.select_kernel(
        "attention",
        "dsa_decode_front",
        signature.format_signature(
            q_a=signature.dense_tensor_format(torch.bfloat16),
            weight=signature.dense_tensor_format(torch.bfloat16),
        ),
        traits={
            "q_lora_rank": Q_LORA,
            "kv_lora_rank": LATENT,
            "q_b_rows": Q_B,
            "index_heads": HEADS,
            "head_dim": DIM,
            "page_size": PAGE,
            "index_k_format": "fp8_scaled",
            "index_k_layout": "packed",
        },
        solution="cute_dsl",
    )
    assert kernel.name == "cute_dsl_dsa_decode_front"


@pytest.mark.parametrize("rows", [0, 65])
def test_rows_outside_one_to_64_are_refused(rows) -> None:
    case = _case(max(rows, 1), seed=3, slot_dtype=torch.int64, norm_dtype=torch.float32)
    packed = torch.zeros(rows, PACKED, dtype=torch.bfloat16, device="cuda")
    q_a, latent, _, key, head_w = packed.split(
        (Q_LORA, LATENT + ROPE, GATE, DIM, HEADS), dim=-1
    )
    kv_a = latent[:, :LATENT]
    assert not dsa_cute_dsl.decode_front_rows_supported(q_a, kv_a, key, head_w)
    case.update(q_a=q_a, kv_a=kv_a, key=key, head_w=head_w)
    with pytest.raises(ValueError, match="fused decode front"):
        _front(case)


def test_misaligned_rows_are_refused() -> None:
    case = _case(4, seed=5, slot_dtype=torch.int64, norm_dtype=torch.float32)
    shifted = torch.zeros(4, PACKED + 1, dtype=torch.bfloat16, device="cuda")[:, 1:]
    q_a = shifted[:, :Q_LORA]
    assert not dsa_cute_dsl.decode_front_rows_supported(
        q_a, case["kv_a"], case["key"], case["head_w"]
    )
