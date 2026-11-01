# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.numerics.torch.attention.gdn import (
    gdn_gate,
    gdn_l2norm,
    gdn_l2norm_clamped,
    torch_gdn_recurrence,
)
from tokenspeed_kernel.ops.attention.gdn import (
    GDN_TREE_VERIFY_CHUNKED_MAX_NODES,
    GDN_TREE_VERIFY_CHUNKED_MIN_NODES,
    GdnCheckpointLayout,
    GdnChunkPrefillResult,
    GdnStateCheckpoints,
    _qkv_rows_strided,
    gdn_chain_verify,
    gdn_chunk_prefill,
    gdn_decode_mtp,
    gdn_decode_step,
    gdn_tree_verify_needs_node_states,
)
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile


@pytest.mark.parametrize("batch", [1, 4])
def test_strided_qkv_trait_spares_packed_rows(batch: int):
    """Sections of one dense q|k|v row (the dense decode path's split) keep
    kernels without strided reads eligible; views into a wider projection
    row do not, from batch 2 on."""
    heads, value_heads, dim = 2, 4, 8
    widths = [heads * dim, heads * dim, value_heads * dim]

    def split(rows: torch.Tensor):
        q, k, v = rows[:, : sum(widths)].split(widths, dim=-1)
        return (
            q.view(batch, 1, heads, dim),
            k.view(batch, 1, heads, dim),
            v.view(batch, 1, value_heads, dim),
        )

    packed = split(torch.empty(batch, sum(widths)))
    projection = split(torch.empty(batch, sum(widths) + value_heads * dim))
    assert not _qkv_rows_strided(*packed)
    assert _qkv_rows_strided(*projection) == (batch > 1)
    assert not _qkv_rows_strided(*(t.contiguous() for t in projection))


def test_pack_gdn_projection_copies_the_row_strided_views(device: str):
    """Dense copies of a projection's [qkv | z | b | a] views, bit for bit,
    compiled once across row counts."""
    from tokenspeed_kernel.ops.attention.gdn._triton.qkv_split import (
        _pack_gdn_projection_kernel,
    )
    from tokenspeed_kernel.ops.attention.gdn.triton import pack_gdn_projection

    qkv_dim, z_dim, heads = 2 * 512 + 1536, 1536, 12

    def views(rows):
        projection = torch.randn(
            rows, qkv_dim + z_dim + 2 * heads, device=device, dtype=torch.bfloat16
        )
        b, a = projection[:, qkv_dim + z_dim :].split(heads, dim=-1)
        return projection[:, :qkv_dim], a, b

    for rows in (1, 16):
        sources = views(rows)
        for packed, source in zip(pack_gdn_projection(*sources), sources):
            assert packed.is_contiguous() and packed.data_ptr() % 16 == 0
            assert packed.data_ptr() != source.data_ptr()
            assert torch.equal(packed, source)
    with assert_no_triton_compile(_pack_gdn_projection_kernel):
        for rows in (3, 64):
            pack_gdn_projection(*views(rows))


def test_pack_gdn_projection_reads_rows_past_int32_offsets(device: str):
    """Row offsets times the projection's row stride pass int32."""
    from tokenspeed_kernel.ops.attention.gdn.triton import pack_gdn_projection

    qkv_dim, heads, row_stride = 64, 4, 1 << 20
    rows = (1 << 31) // row_stride + 2
    projection = torch.empty(rows, row_stride, device=device, dtype=torch.uint8)
    used = projection[:, : qkv_dim + 2 * heads]
    used.copy_(torch.randint(0, 256, used.shape, device=device, dtype=torch.uint8))
    b, a = used[:, qkv_dim:].split(heads, dim=-1)
    sources = (used[:, :qkv_dim], a, b)
    for packed, source in zip(pack_gdn_projection(*sources), sources):
        assert torch.equal(packed, source)


def _fla_chunk_gated_delta_rule():
    from tokenspeed_kernel.ops.attention.gdn._triton.chunk import (
        chunk_gated_delta_rule,
    )

    return chunk_gated_delta_rule


def _make_inputs(
    *,
    device: str,
    dtype: torch.dtype,
    seq_len: int,
    num_q_heads: int,
    num_v_heads: int,
    state_dtype: torch.dtype,
    nonzero_state: bool,
):
    torch.manual_seed(0)
    head_dim = 128
    q = torch.randn(1, seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, seq_len, num_v_heads, head_dim, device=device, dtype=dtype)
    beta = torch.rand(1, seq_len, num_v_heads, device=device, dtype=dtype).sigmoid()
    g = F.logsigmoid(
        torch.rand(1, seq_len, num_v_heads, device=device, dtype=torch.float32)
    )
    state_shape = (1, num_v_heads, head_dim, head_dim)
    initial_state = (
        torch.randn(state_shape, device=device, dtype=state_dtype) * 0.1
        if nonzero_state
        else torch.zeros(state_shape, device=device, dtype=state_dtype)
    )
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    return q, k, v, g, beta, initial_state, cu_seqlens


def _torch_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return gdn_l2norm_clamped(x, eps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GDN L2 norm requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gdn_l2norm_fwd_varying_prefill_rows(dtype: torch.dtype) -> None:
    from tokenspeed_kernel.ops.attention.gdn._triton.l2norm import (
        l2norm_fwd,
        l2norm_fwd_kernel,
    )

    l2norm_fwd(torch.randn((1476, 4, 128), device="cuda", dtype=dtype))
    with assert_no_triton_compile(l2norm_fwd_kernel):
        for rows in (1499, 1620, 2051, 2064):
            x = torch.randn((rows, 4, 128), device="cuda", dtype=dtype)
            actual = l2norm_fwd(x)
            reference = x.float() * torch.rsqrt(
                x.float().square().sum(dim=-1, keepdim=True) + 1e-6
            )
            tolerance = 5e-3 if dtype is torch.bfloat16 else 1e-5
            torch.testing.assert_close(
                actual.float(), reference, atol=tolerance, rtol=tolerance
            )


def _torch_gdn_chunk_prefill_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token recurrence of each ``cu_seqlens`` sequence (batch 1).

    ``initial_state``/the returned final state follow ``gdn_chunk_prefill``'s
    public K-last ``[N, Hv, V, K]`` contract.
    """
    assert q.shape[0] == 1
    out = torch.empty(*v.shape, device=q.device, dtype=torch.float32)
    final_states = []
    bounds = cu_seqlens.to(torch.int64).tolist()
    for seq_idx, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        tokens = slice(start, end)
        out[:, tokens], states = torch_gdn_recurrence(
            q[:, tokens],
            k[:, tokens],
            v[:, tokens],
            g[:, tokens],
            beta[:, tokens],
            initial_state[seq_idx, None],
            scale=scale,
            parents=None,
            compute_dtype=torch.float32,
        )
        final_states.append(
            states[0, -1] if end > start else initial_state[seq_idx].float()
        )
    return out.to(q.dtype), torch.stack(final_states).to(initial_state.dtype)


def test_gdn_chunk_prefill_triton_matches_torch_reference(device: str, require):
    # The Triton wrapper should match an independent token-by-token recurrence.
    require("attention", "gdn_chunk_prefill", "triton", torch.bfloat16, "q")

    torch.manual_seed(1234)
    seq_len = 130
    num_q_heads = 16
    num_v_heads = 32
    head_dim = 128
    dtype = torch.bfloat16
    q = torch.randn(1, seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, seq_len, num_v_heads, head_dim, device=device, dtype=dtype) * 0.5
    g = F.logsigmoid(
        torch.randn(1, seq_len, num_v_heads, device=device, dtype=torch.float32)
    )
    beta = torch.rand(1, seq_len, num_v_heads, device=device, dtype=dtype).sigmoid()
    initial_state = (
        torch.randn(1, num_v_heads, head_dim, head_dim, device=device, dtype=dtype)
        * 0.01
    )
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    scale = head_dim**-0.5

    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        qk_l2norm=True,
        output_final_state=True,
        solution="triton",
    )
    assert isinstance(result, GdnChunkPrefillResult)
    assert result.h_layout is GdnCheckpointLayout.NONE
    ref_out, ref_state = _torch_gdn_chunk_prefill_reference(
        _torch_l2norm(q),
        _torch_l2norm(k),
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(
        result.out.float(), ref_out.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        result.final_state.float(),
        ref_state.float(),
        rtol=2e-2,
        atol=3e-2,
    )


def test_gdn_chunk_prefill_triton_matches_torch_reference_varlen(device: str, require):
    # Varlen cu_seqlens should reset recurrent state independently per sequence.
    require("attention", "gdn_chunk_prefill", "triton", torch.bfloat16, "q")

    torch.manual_seed(4321)
    seq_lens = [63, 67]
    total_tokens = sum(seq_lens)
    num_q_heads = 4
    num_v_heads = 8
    head_dim = 128
    dtype = torch.bfloat16
    q = torch.randn(1, total_tokens, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, total_tokens, num_q_heads, head_dim, device=device, dtype=dtype)
    v = (
        torch.randn(1, total_tokens, num_v_heads, head_dim, device=device, dtype=dtype)
        * 0.5
    )
    g = F.logsigmoid(
        torch.randn(1, total_tokens, num_v_heads, device=device, dtype=torch.float32)
    )
    beta = torch.rand(
        1, total_tokens, num_v_heads, device=device, dtype=dtype
    ).sigmoid()
    initial_state = (
        torch.randn(
            len(seq_lens), num_v_heads, head_dim, head_dim, device=device, dtype=dtype
        )
        * 0.01
    )
    cu_seqlens = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0)], device=device)
    cu_seqlens = cu_seqlens.to(torch.int32)
    scale = head_dim**-0.5

    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        qk_l2norm=True,
        output_final_state=True,
        solution="triton",
    )
    assert isinstance(result, GdnChunkPrefillResult)
    assert result.h_layout is GdnCheckpointLayout.NONE
    ref_out, ref_state = _torch_gdn_chunk_prefill_reference(
        _torch_l2norm(q),
        _torch_l2norm(k),
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(
        result.out.float(), ref_out.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        result.final_state.float(),
        ref_state.float(),
        rtol=2e-2,
        atol=3e-2,
    )


def test_flashinfer_prefill_supported_shapes(device: str, require):
    require("attention", "gdn_chunk_prefill", "flashinfer", torch.bfloat16, "q")
    from tokenspeed_kernel.ops.attention.gdn.flashinfer import is_supported

    assert is_supported(128, torch.bfloat16, 16, 16)
    assert is_supported(128, torch.bfloat16, 16, 32)
    assert not is_supported(64, torch.bfloat16, 16, 16)
    assert not is_supported(128, torch.float16, 16, 16)
    assert not is_supported(128, torch.bfloat16, 32, 16)


@pytest.mark.parametrize("solution", ["triton", "flashinfer"])
@pytest.mark.parametrize(
    "num_q_heads,num_v_heads,seq_len,state_dtype,nonzero_state",
    [
        (16, 16, 130, torch.bfloat16, False),
        (16, 32, 130, torch.bfloat16, True),
        (16, 64, 512, torch.float32, True),
        (4, 12, 2048, torch.float32, False),
    ],
)
def test_gdn_chunk_prefill_matches_fla_reference(
    device: str,
    solution: str,
    num_q_heads: int,
    num_v_heads: int,
    seq_len: int,
    state_dtype: torch.dtype,
    nonzero_state: bool,
    require,
):
    # Each selectable backend should match the FLA reference output/state contract.
    require("attention", "gdn_chunk_prefill", solution, torch.bfloat16, "q")

    q, k, v, g, beta, initial_state, cu_seqlens = _make_inputs(
        device=device,
        dtype=torch.bfloat16,
        seq_len=seq_len,
        num_q_heads=num_q_heads,
        num_v_heads=num_v_heads,
        state_dtype=state_dtype,
        nonzero_state=nonzero_state,
    )
    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        qk_l2norm=True,
        output_final_state=True,
        solution=solution,
    )
    assert isinstance(result, GdnChunkPrefillResult)
    assert result.h_layout is GdnCheckpointLayout.NONE

    # FLA's own native state layout is [N, Hv, K, V] (V-last); gdn_chunk_prefill's
    # public contract is K-last [N, Hv, V, K] (matches the runtime pool +
    # flashinfer), so transpose at this reference's boundary (mirrors
    # triton_gdn_chunk_prefill's own in/out transpose).
    ref_out, ref_state_kv = _fla_chunk_gated_delta_rule()(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state.clone().transpose(-2, -1).contiguous(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    ref_state = ref_state_kv.transpose(-2, -1)

    assert result.out.shape == ref_out.shape
    assert result.final_state.shape == ref_state.shape
    # The FLA implementation is nondeterministic due to atomics; mean error is
    # the stable signal, while max error can be noisy on a few elements.
    assert (result.out.float() - ref_out.float()).abs().mean() < 1e-3
    assert (result.final_state.float() - ref_state.float()).abs().mean() < 1e-3
    torch.testing.assert_close(result.out, ref_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("solution", ["triton", "flashinfer"])
def test_gdn_chunk_prefill_output_h_contract(device: str, solution: str, require):
    # output_h exposes backend-native checkpoint layouts used by hybrid GDN caching.
    require("attention", "gdn_chunk_prefill", solution, torch.bfloat16, "q")

    q, k, v, g, beta, initial_state, cu_seqlens = _make_inputs(
        device=device,
        dtype=torch.bfloat16,
        seq_len=130,
        num_q_heads=16,
        num_v_heads=32,
        state_dtype=torch.bfloat16,
        nonzero_state=True,
    )
    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        qk_l2norm=True,
        output_final_state=True,
        output_h=True,
        solution=solution,
    )
    assert isinstance(result, GdnChunkPrefillResult)

    if solution == "triton":
        assert result.h_layout is GdnCheckpointLayout.FLA
        assert result.h is not None
        assert result.h_cu_starts is None
        assert result.h.shape == (1, 3, 32, 128, 128)
    else:
        assert result.h_layout is GdnCheckpointLayout.FLASHINFER
        assert result.h is not None
        assert result.h_cu_starts is not None
        assert result.h.shape == (2, 32, 128, 128)
        torch.testing.assert_close(
            result.h_cu_starts, torch.tensor([0, 2], device=device)
        )

    assert result.out.shape == v.shape
    assert result.final_state.shape == initial_state.shape


def test_gdn_chunk_prefill_state_checkpoints_are_the_truncated_final_states(
    device: str, require
):
    # A caller-owned checkpoint buffer at a 128-token interval: sequence i's
    # j-th state is the final state of the same scan cut after (j + 1) * 128
    # of its tokens, bit for bit.
    require("attention", "gdn_chunk_prefill", "flashinfer", torch.bfloat16, "q")
    lengths = (300, 520)
    interval = 128
    q, k, v, g, beta, initial_state, _ = _make_inputs(
        device=device,
        dtype=torch.bfloat16,
        seq_len=sum(lengths),
        num_q_heads=16,
        num_v_heads=32,
        state_dtype=torch.float32,
        nonzero_state=True,
    )
    initial_state = initial_state.repeat(len(lengths), 1, 1, 1)
    initial_state[1] *= -1
    starts = (0, lengths[0])
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    counts = [length // interval for length in lengths]
    cu_starts = torch.tensor([0, counts[0], counts[0] + counts[1]], dtype=torch.int32)
    states = torch.full(
        (sum(counts), *initial_state.shape[1:]), float("nan"), device=device
    )
    scale = q.shape[-1] ** -0.5
    gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens.to(device=device, dtype=torch.int32),
        qk_l2norm=True,
        output_final_state=True,
        state_checkpoints=GdnStateCheckpoints(
            every_n_tokens=interval, states=states, cu_starts=cu_starts.to(device)
        ),
    )
    for i, (start, count) in enumerate(zip(starts, counts)):
        for j in range(count):
            end = start + (j + 1) * interval
            truncated = gdn_chunk_prefill(
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                g[:, start:end],
                beta[:, start:end],
                scale=scale,
                initial_state=initial_state[i : i + 1],
                cu_seqlens=torch.tensor(
                    [0, end - start], device=device, dtype=torch.int32
                ),
                qk_l2norm=True,
                output_final_state=True,
            )
            assert torch.equal(
                states[int(cu_starts[i]) + j], truncated.final_state[0]
            ), (i, j)


def test_gdn_chunk_prefill_state_checkpoints_exclude_output_h(device: str, require):
    require("attention", "gdn_chunk_prefill", "flashinfer", torch.bfloat16, "q")
    q, k, v, g, beta, initial_state, cu_seqlens = _make_inputs(
        device=device,
        dtype=torch.bfloat16,
        seq_len=128,
        num_q_heads=16,
        num_v_heads=32,
        state_dtype=torch.float32,
        nonzero_state=False,
    )
    checkpoints = GdnStateCheckpoints(
        every_n_tokens=64,
        states=torch.empty((2, *initial_state.shape[1:]), device=device),
        cu_starts=torch.tensor([0, 2], device=device, dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="exclusive"):
        gdn_chunk_prefill(
            q,
            k,
            v,
            g,
            beta,
            scale=None,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            qk_l2norm=True,
            output_h=True,
            state_checkpoints=checkpoints,
        )


# ===-----------------------------------------------------------------------===#
# GDN decode / MTP (K-last, gdn_decode_step / gdn_decode_mtp)
# ===-----------------------------------------------------------------------===#


def _torch_gdn_decode_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_klast: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Token-by-token oracle for gdn_decode_step/gdn_decode_mtp.

    q, k: [B, T, H, K]; v, a, b: [B, T, HV, V] / [B, T, HV]; state_klast is
    K-last ``[B, HV, V, K]``. Returns ``(out [B, T, HV, V], per_step_states)``,
    the K-last ``[B, HV, V, K]`` state after each step.
    """
    out, states = torch_gdn_recurrence(
        gdn_l2norm(q.float(), 1e-6),
        gdn_l2norm(k.float(), 1e-6),
        v,
        gdn_gate(a.float(), dt_bias.float(), A_log.float()),
        torch.sigmoid(b.float()),
        state_klast,
        scale=scale,
        parents=None,
        compute_dtype=torch.float32,
    )
    return out, list(states.unbind(1))


def _make_decode_inputs(
    *,
    device: str,
    dtype: torch.dtype,
    T: int,
    batch: int = 4,
    num_q_heads: int = 4,
    num_v_heads: int = 8,
    head_dim: int = 128,
    pool_size: int = 16,
    state_dtype: torch.dtype | None = None,
    parameter_dtype: torch.dtype = torch.float32,
    parameter_requires_grad: bool = False,
):
    torch.manual_seed(7)
    q = torch.randn(batch, T, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, T, num_q_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, T, num_v_heads, head_dim, device=device, dtype=dtype) * 0.5
    a = torch.randn(batch, T, num_v_heads, device=device, dtype=torch.float32)
    b = torch.randn(batch, T, num_v_heads, device=device, dtype=torch.float32)
    A_log = torch.randn(num_v_heads, device=device, dtype=parameter_dtype)
    dt_bias = torch.randn(num_v_heads, device=device, dtype=parameter_dtype)
    if parameter_requires_grad:
        A_log = torch.nn.Parameter(A_log)
        dt_bias = torch.nn.Parameter(dt_bias)
    # K-last pool [pool_size, HV, V, K], small values to keep the recurrence stable.
    state_dtype = dtype if state_dtype is None else state_dtype
    pool = (
        torch.randn(
            pool_size,
            num_v_heads,
            head_dim,
            head_dim,
            device=device,
            dtype=state_dtype,
        )
        * 0.02
    )
    return q, k, v, a, b, A_log, dt_bias, pool


@pytest.mark.parametrize("solution", ["triton", "flashinfer", "gluon"])
@pytest.mark.parametrize(
    ("state_dtype", "parameter_dtype", "parameter_requires_grad"),
    [
        (torch.bfloat16, torch.float32, False),
        (torch.float32, torch.bfloat16, True),
        (torch.float32, torch.float32, True),
    ],
)
def test_gdn_decode_step_matches_torch_reference(
    device: str,
    solution: str,
    state_dtype: torch.dtype,
    parameter_dtype: torch.dtype,
    parameter_requires_grad: bool,
    require,
):
    # T=1 decode step, in-place state update (output_state_indices=None).
    require("attention", "gdn_decode_step", solution, torch.bfloat16, "q")

    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=torch.bfloat16,
        T=1,
        state_dtype=state_dtype,
        parameter_dtype=parameter_dtype,
        parameter_requires_grad=parameter_requires_grad,
    )
    read_idx = torch.tensor([1, 3, 5, 7], device=device, dtype=torch.int32)
    scale = q.shape[-1] ** -0.5

    pool_copy = pool.clone()
    out = gdn_decode_step(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool_copy,
        initial_state_indices=read_idx,
        scale=scale,
        use_qk_l2norm=True,
        solution=solution,
    )

    ref_out, ref_states = _torch_gdn_decode_reference(
        q, k, v, a, b, A_log, dt_bias, pool[read_idx], scale
    )

    assert out.shape == v.shape
    torch.testing.assert_close(
        out.float(), ref_out.to(out.dtype).float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        pool_copy[read_idx].float(),
        ref_states[0].to(pool.dtype).float(),
        rtol=2e-2,
        atol=3e-2,
    )
    # Rows never referenced by read_idx must stay untouched.
    untouched = torch.tensor(
        [i for i in range(pool.shape[0]) if i not in read_idx.tolist()],
        device=device,
    )
    torch.testing.assert_close(pool_copy[untouched], pool[untouched])


@pytest.mark.parametrize("solution", ["triton", "flashinfer", "gluon"])
def test_gdn_decode_step_output_state_indices_remap(
    device: str, solution: str, require
):
    # Dual-index paging: write lands on output_state_indices, not the read row.
    require("attention", "gdn_decode_step", solution, torch.bfloat16, "q")

    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device, dtype=torch.bfloat16, T=1
    )
    read_idx = torch.tensor([1, 3, 5, 7], device=device, dtype=torch.int32)
    out_idx = torch.tensor([8, 9, -1, 10], device=device, dtype=torch.int32)
    scale = q.shape[-1] ** -0.5

    pool_copy = pool.clone()
    out = gdn_decode_step(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool_copy,
        initial_state_indices=read_idx,
        output_state_indices=out_idx,
        scale=scale,
        use_qk_l2norm=True,
        solution=solution,
    )

    # The read rows must never be mutated when output_state_indices is given.
    torch.testing.assert_close(pool_copy[read_idx], pool[read_idx])

    ref_out, ref_states = _torch_gdn_decode_reference(
        q, k, v, a, b, A_log, dt_bias, pool[read_idx], scale
    )
    torch.testing.assert_close(
        out.float(), ref_out.to(out.dtype).float(), rtol=2e-2, atol=2e-2
    )
    for bi in range(q.shape[0]):
        oi = int(out_idx[bi])
        if oi < 0:
            continue
        torch.testing.assert_close(
            pool_copy[oi].float(),
            ref_states[0][bi].to(pool.dtype).float(),
            rtol=2e-2,
            atol=3e-2,
        )


@pytest.mark.parametrize("solution", ["triton", "flashinfer", "gluon"])
def test_gdn_decode_step_padding_index_is_isolated(device: str, solution: str, require):
    # -1 (CUDA-graph padding) rows must not corrupt any other batch entry's
    # read or write, regardless of how each backend treats the padding row
    # itself (skip vs sacrificial-row redirect are both valid per-kernel).
    require("attention", "gdn_decode_step", solution, torch.bfloat16, "q")

    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device, dtype=torch.bfloat16, T=1
    )
    read_idx = torch.tensor([1, -1, 5, 7], device=device, dtype=torch.int32)
    scale = q.shape[-1] ** -0.5

    pool_copy = pool.clone()
    out = gdn_decode_step(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool_copy,
        initial_state_indices=read_idx,
        scale=scale,
        use_qk_l2norm=True,
        solution=solution,
    )

    valid = [i for i, r in enumerate(read_idx.tolist()) if r >= 0]
    valid_rows = read_idx[valid]
    valid_idx = torch.tensor(valid, device=device)
    ref_out, ref_states = _torch_gdn_decode_reference(
        q[valid_idx],
        k[valid_idx],
        v[valid_idx],
        a[valid_idx],
        b[valid_idx],
        A_log,
        dt_bias,
        pool[valid_rows],
        scale,
    )
    torch.testing.assert_close(
        out[valid_idx].float(), ref_out.to(out.dtype).float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        pool_copy[valid_rows].float(),
        ref_states[0].to(pool.dtype).float(),
        rtol=2e-2,
        atol=3e-2,
    )
    # Rows unrelated to any (valid or padding) index stay untouched. Row 0 is
    # the reserved null page and is excluded because FlashInfer's bf16 path
    # redirects -1 padding there.
    referenced = {int(x) for x in valid_rows.tolist()} | {0}
    untouched = torch.tensor(
        [i for i in range(pool.shape[0]) if i not in referenced], device=device
    )
    torch.testing.assert_close(pool_copy[untouched], pool[untouched])


@pytest.mark.parametrize("solution", ["triton", "flashinfer", "gluon"])
@pytest.mark.parametrize(
    ("state_dtype", "parameter_dtype", "parameter_requires_grad"),
    [
        (torch.bfloat16, torch.float32, False),
        (torch.float32, torch.bfloat16, True),
        (torch.float32, torch.float32, True),
    ],
)
def test_gdn_decode_mtp_intermediate_states_buffer_matches_reference(
    device: str,
    solution: str,
    state_dtype: torch.dtype,
    parameter_dtype: torch.dtype,
    parameter_requires_grad: bool,
    require,
):
    # T>1 MTP verify: every step's post-update state must land in
    # intermediate_states_buffer[i_n, step], and the pool itself must stay
    # untouched (disable_state_update=True, the runtime's target_verify mode).
    require("attention", "gdn_decode_mtp", solution, torch.bfloat16, "q")

    T = 3
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=torch.bfloat16,
        T=T,
        state_dtype=state_dtype,
        parameter_dtype=parameter_dtype,
        parameter_requires_grad=parameter_requires_grad,
    )
    read_idx = torch.tensor([1, 3, 5, 7], device=device, dtype=torch.int32)
    scale = q.shape[-1] ** -0.5
    batch, _, num_v_heads, head_dim = v.shape

    scratch = torch.empty(
        batch, T, num_v_heads, head_dim, head_dim, device=device, dtype=pool.dtype
    )
    pool_copy = pool.clone()
    out = gdn_decode_mtp(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool_copy,
        initial_state_indices=read_idx,
        scale=scale,
        disable_state_update=True,
        use_qk_l2norm=True,
        intermediate_states_buffer=scratch,
        solution=solution,
        tree_ancestors=None,
    )

    # disable_state_update=True: the read rows must never be mutated.
    torch.testing.assert_close(pool_copy[read_idx], pool[read_idx])

    ref_out, ref_states = _torch_gdn_decode_reference(
        q, k, v, a, b, A_log, dt_bias, pool[read_idx], scale
    )
    assert out.shape == v.shape
    torch.testing.assert_close(
        out.float(), ref_out.to(out.dtype).float(), rtol=2e-2, atol=2e-2
    )
    for t in range(T):
        torch.testing.assert_close(
            scratch[:, t].float(),
            ref_states[t].to(pool.dtype).float(),
            rtol=2e-2,
            atol=3e-2,
        )


@pytest.mark.parametrize("solution", ["triton", "flashinfer", "gluon"])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_gdn_decode_mtp_output_state_indices_scatter_matches_reference(
    device: str,
    solution: str,
    state_dtype: torch.dtype,
    require,
):
    # FlashInfer 0.6.15 and the Triton fallback scatter every post-token state
    # directly into the scheduler-provided pool row, avoiding a dense B*T state
    # scratch buffer plus a follow-up PyTorch scatter.
    require("attention", "gdn_decode_mtp", solution, torch.bfloat16, "q")

    T = 3
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=torch.bfloat16,
        T=T,
        pool_size=24,
        state_dtype=state_dtype,
    )
    read_idx = torch.tensor([1, 3, 5, 7], device=device, dtype=torch.int32)
    output_idx = torch.arange(8, 20, device=device, dtype=torch.int32).view(4, T)
    scale = q.shape[-1] ** -0.5

    pool_copy = pool.clone()
    out = gdn_decode_mtp(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool_copy,
        initial_state_indices=read_idx,
        output_state_indices=output_idx,
        scale=scale,
        disable_state_update=False,
        use_qk_l2norm=True,
        solution=solution,
        tree_ancestors=None,
    )

    ref_out, ref_states = _torch_gdn_decode_reference(
        q, k, v, a, b, A_log, dt_bias, pool[read_idx], scale
    )
    torch.testing.assert_close(
        out.float(), ref_out.to(out.dtype).float(), rtol=2e-2, atol=2e-2
    )
    # Per-token scatter must not overwrite the read rows.
    torch.testing.assert_close(pool_copy[read_idx], pool[read_idx])
    for t in range(T):
        torch.testing.assert_close(
            pool_copy[output_idx[:, t].long()].float(),
            ref_states[t].to(pool.dtype).float(),
            rtol=2e-2,
            atol=3e-2,
        )


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_gdn_decode_mtp_tree_parents_match_per_path_reference(
    device: str, state_dtype: torch.dtype, require
):
    """Draft trees: each node continues from its parent's state, not the previous row."""
    require("attention", "gdn_decode_mtp", "triton", torch.bfloat16, "q")

    T = 6
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device, dtype=torch.bfloat16, T=T, pool_size=32, state_dtype=state_dtype
    )
    parents = [
        [-1, 0, 1, 1, 0, 4],  # branches under node 1 and at the root
        [-1, -1, 1, -1, 3, 3],  # several children of the root
        [-1, 0, 1, 2, 3, 4],  # a chain
        [-1, 0, 0, 0, 0, 0],  # one level, five siblings
    ]
    read_idx = torch.tensor([1, 3, 5, 7], device=device, dtype=torch.int32)
    output_idx = torch.arange(8, 8 + 4 * T, device=device, dtype=torch.int32).view(4, T)
    scale = q.shape[-1] ** -0.5

    pool_copy = pool.clone()
    out = gdn_decode_mtp(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool_copy,
        initial_state_indices=read_idx,
        output_state_indices=output_idx,
        tree_ancestors=_tree_ancestors(parents, device),
        scale=scale,
        disable_state_update=False,
        use_qk_l2norm=True,
    )

    for bi, par in enumerate(parents):
        for t in range(T):
            path, node = [], t
            while node >= 0:
                path.append(node)
                node = par[node]
            path = path[::-1]
            pick = lambda x: x[bi : bi + 1, path]
            ref_out, ref_states = _torch_gdn_decode_reference(
                pick(q),
                pick(k),
                pick(v),
                pick(a),
                pick(b),
                A_log,
                dt_bias,
                pool[read_idx[bi : bi + 1].long()],
                scale,
            )
            torch.testing.assert_close(
                out[bi, t].float(),
                ref_out[0, -1].to(out.dtype).float(),
                rtol=2e-2,
                atol=2e-2,
            )
            torch.testing.assert_close(
                pool_copy[int(output_idx[bi, t])].float(),
                ref_states[-1][0].to(pool.dtype).float(),
                rtol=2e-2,
                atol=3e-2,
            )

    # A chain given as parents never reloads: it matches the parent-less kernel to rounding.
    chain_pool = pool.clone()
    chain_out = gdn_decode_mtp(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=chain_pool,
        initial_state_indices=read_idx,
        output_state_indices=output_idx,
        scale=scale,
        disable_state_update=False,
        use_qk_l2norm=True,
        solution="triton",
        tree_ancestors=None,
    )
    torch.testing.assert_close(out[2], chain_out[2])
    rows = output_idx[2].long()
    torch.testing.assert_close(pool_copy[rows], chain_pool[rows])


@pytest.mark.parametrize(
    ("solution", "tree"),
    [("triton", False), ("triton", True), ("gluon", False)],
)
def test_gdn_decode_mtp_output_state_matches_in_place(
    device: str, solution: str, tree: bool, require
):
    """Per-token rows written to a separate ``output_state`` pool equal the
    in-place rows, and the read pool is left untouched (verify reads the
    committed state in place and stages positions in a scratch)."""
    require("attention", "gdn_decode_mtp", solution, torch.bfloat16, "q")

    T = 4
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=torch.bfloat16,
        T=T,
        pool_size=32,
        state_dtype=torch.float32,
    )
    read_idx = torch.tensor([1, -1, 5, 7], device=device, dtype=torch.int32)
    output_idx = torch.arange(8, 8 + 4 * T, device=device, dtype=torch.int32).view(4, T)
    ancestors = (
        _tree_ancestors(
            [[-1, 0, 0, 2], [-1, 0, 1, 2], [-1, -1, 1, 1], [-1, 0, 1, 2]], device
        )
        if tree
        else None
    )
    common = dict(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state_indices=read_idx,
        output_state_indices=output_idx,
        tree_ancestors=ancestors,
        disable_state_update=False,
        use_qk_l2norm=True,
        solution=solution,
    )
    in_place = pool.clone()
    expected = gdn_decode_mtp(q, k, v, initial_state=in_place, **common)
    read_pool = pool.clone()
    scratch = torch.zeros_like(pool)
    actual = gdn_decode_mtp(
        q, k, v, initial_state=read_pool, output_state=scratch, **common
    )

    live = read_idx >= 0
    torch.testing.assert_close(actual[live], expected[live], rtol=0, atol=0)
    rows = output_idx.flatten().long()
    torch.testing.assert_close(scratch[rows], in_place[rows], rtol=0, atol=0)
    assert torch.equal(read_pool, pool)


@pytest.mark.parametrize(
    ("solution", "state_dtype"),
    [
        ("triton", torch.bfloat16),
        ("triton", torch.float32),
        ("flashinfer", torch.float32),
        ("gluon", torch.bfloat16),
        ("gluon", torch.float32),
    ],
)
def test_gdn_decode_mtp_padding_indices_skip_state_writes(
    device: str, solution: str, state_dtype: torch.dtype, require
):
    require("attention", "gdn_decode_mtp", solution, torch.bfloat16, "q")

    T = 3
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=torch.bfloat16,
        T=T,
        pool_size=24,
        state_dtype=state_dtype,
    )
    read_idx = torch.tensor([1, -1, 5, 7], device=device, dtype=torch.int32)
    output_idx = torch.tensor(
        [[8, 9, 10], [-1, -1, -1], [11, 12, 13], [14, 15, 16]],
        device=device,
        dtype=torch.int32,
    )
    scale = q.shape[-1] ** -0.5

    pool_copy = pool.clone()
    out = gdn_decode_mtp(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool_copy,
        initial_state_indices=read_idx,
        output_state_indices=output_idx,
        scale=scale,
        disable_state_update=False,
        use_qk_l2norm=True,
        solution=solution,
        tree_ancestors=None,
    )

    valid_batch = torch.tensor([0, 2, 3], device=device)
    valid_read = read_idx[valid_batch]
    ref_out, ref_states = _torch_gdn_decode_reference(
        q[valid_batch],
        k[valid_batch],
        v[valid_batch],
        a[valid_batch],
        b[valid_batch],
        A_log,
        dt_bias,
        pool[valid_read],
        scale,
    )
    torch.testing.assert_close(
        out[valid_batch].float(),
        ref_out.to(out.dtype).float(),
        rtol=2e-2,
        atol=2e-2,
    )
    valid_output = output_idx[valid_batch]
    for t in range(T):
        torch.testing.assert_close(
            pool_copy[valid_output[:, t].long()].float(),
            ref_states[t].to(pool.dtype).float(),
            rtol=2e-2,
            atol=3e-2,
        )

    written = set(valid_output.flatten().tolist())
    untouched = torch.tensor(
        [i for i in range(pool.shape[0]) if i not in written], device=device
    )
    torch.testing.assert_close(pool_copy[untouched], pool[untouched])


@pytest.mark.parametrize("solution", ["triton", "flashinfer", "gluon"])
def test_gdn_decode_mtp_disable_state_update_false_writes_back(
    device: str, solution: str, require
):
    # disable_state_update=False: the final step's state writes back to the
    # SAME row (initial_state_indices), matching gdn_decode_mtp's contract.
    require("attention", "gdn_decode_mtp", solution, torch.bfloat16, "q")

    T = 2
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device, dtype=torch.bfloat16, T=T
    )
    read_idx = torch.tensor([1, 3, 5, 7], device=device, dtype=torch.int32)
    scale = q.shape[-1] ** -0.5

    pool_copy = pool.clone()
    gdn_decode_mtp(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool_copy,
        initial_state_indices=read_idx,
        scale=scale,
        disable_state_update=False,
        use_qk_l2norm=True,
        solution=solution,
        tree_ancestors=None,
    )

    _, ref_states = _torch_gdn_decode_reference(
        q, k, v, a, b, A_log, dt_bias, pool[read_idx], scale
    )
    torch.testing.assert_close(
        pool_copy[read_idx].float(),
        ref_states[-1].to(pool.dtype).float(),
        rtol=2e-2,
        atol=3e-2,
    )


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize(
    ("state_mode", "strided_pool"),
    [
        ("readonly", False),
        ("readonly", True),
        ("writeback", False),
        ("writeback", True),
        # Upstream scatter indexes a flat 3D pool, as used by verify scratch;
        # its native 4D strided-pool path supports the other state modes.
        ("scatter", False),
        ("cache", False),
        ("cache", True),
    ],
)
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_flashinfer_mtp_uninitialized_buffers(
    device: str,
    batch: int,
    strided_pool: bool,
    state_mode: str,
    enable_pdl: bool,
    require,
) -> None:
    """Live output/cache rows overwrite poison; skipped rows and slab gaps survive."""
    require("attention", "gdn_decode_mtp", "flashinfer", torch.bfloat16, "q")
    from flashinfer.gdn_decode import gated_delta_rule_mtp as reference_mtp
    from tokenspeed_kernel.ops.attention.gdn._flashinfer.adapter import (
        gated_delta_rule_mtp,
    )

    steps, heads, dim = 3, 32, 128
    pool_size = 1 + batch * (steps + 1)
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=torch.bfloat16,
        T=steps,
        batch=batch,
        num_q_heads=heads // 2,
        num_v_heads=heads,
        head_dim=dim,
        pool_size=pool_size,
        state_dtype=torch.float32,
        parameter_dtype=torch.float32,
        parameter_requires_grad=False,
    )
    # B*HV selects both FlashInfer's inline and warp-specialized kernels.
    read_rows = torch.arange(1, batch + 1, dtype=torch.int64, device=device)
    if batch > 1:
        read_rows[-1] = -1
    output_rows = None
    if state_mode == "scatter":
        output_rows = torch.arange(
            batch + 1, pool_size, dtype=torch.int32, device=device
        ).view(batch, steps)
        if batch > 1:
            output_rows[-1].fill_(-1)
    backing = torch.full(
        (pool_size * (2 if strided_pool else 1), heads, dim, dim),
        123.0,
        dtype=pool.dtype,
        device=device,
    )
    actual_pool = backing[::2] if strided_pool else backing
    actual_pool.copy_(pool)
    expected_pool = pool.clone()
    output = torch.full_like(v, float("nan"))
    expected_output = output.clone()
    cache = None
    expected_cache = None
    if state_mode == "cache":
        cache = torch.full(
            (batch, steps, heads, dim, dim),
            float("nan"),
            device=device,
            dtype=torch.float32,
        )
        expected_cache = cache.clone()
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        initial_state_indices=read_rows,
        A_log=A_log,
        a=a.to(q.dtype),
        dt_bias=dt_bias,
        b=b.to(q.dtype),
        scale=dim**-0.5,
        ssm_state_indices=output_rows,
        disable_state_update=state_mode in ("readonly", "cache"),
        use_qk_l2norm=True,
    )
    reference_mtp(
        **kwargs,
        initial_state=expected_pool,
        output=expected_output,
        intermediate_states_buffer=expected_cache,
        output_state_indices=None,
    )
    gated_delta_rule_mtp(
        **kwargs,
        enable_pdl=enable_pdl,
        initial_state=actual_pool,
        output=output,
        intermediate_states_buffer=cache,
    )

    live_rows = batch - 1 if batch > 1 else batch
    assert torch.isfinite(output[:live_rows]).all()
    torch.testing.assert_close(output, expected_output, equal_nan=True)
    torch.testing.assert_close(actual_pool, expected_pool)
    if cache is not None:
        assert torch.isfinite(cache[:live_rows]).all()
        torch.testing.assert_close(cache, expected_cache, equal_nan=True)
    if strided_pool:
        assert (backing[1::2] == 123.0).all()


def test_flashinfer_mtp_preserves_fp16_output_dtype(device: str, require) -> None:
    require("attention", "gdn_decode_mtp", "flashinfer", torch.float16, "q")
    from flashinfer.gdn_decode import gated_delta_rule_mtp as reference_mtp

    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=torch.float16,
        T=3,
        batch=1,
        num_q_heads=4,
        num_v_heads=8,
        head_dim=128,
        pool_size=4,
        state_dtype=torch.float32,
        parameter_dtype=torch.bfloat16,
        parameter_requires_grad=True,
    )
    read_rows = torch.tensor([1], dtype=torch.int32, device=device)
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        initial_state=pool,
        initial_state_indices=read_rows,
        a=a.to(q.dtype),
        b=b.to(q.dtype),
        scale=None,
        disable_state_update=True,
        use_qk_l2norm=True,
        intermediate_states_buffer=None,
        output_state_indices=None,
    )
    actual = gdn_decode_mtp(
        **kwargs,
        A_log=A_log,
        dt_bias=dt_bias,
        tree_ancestors=None,
        override=None,
        solution="flashinfer",
    )
    expected, _ = reference_mtp(
        **kwargs,
        A_log=A_log.detach().float(),
        dt_bias=dt_bias.detach().float(),
        output=None,
        ssm_state_indices=None,
    )
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("solution", ["triton", "gluon"])
@pytest.mark.parametrize(
    ("op", "steps"),
    [("gdn_decode_step", 1), ("gdn_decode_mtp", 1), ("gdn_decode_mtp", 3)],
)
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_gdn_packed_decode_inputs(
    device: str,
    solution: str,
    op: str,
    steps: int,
    state_dtype: torch.dtype,
    require,
):
    """Packed QKV and strided scalar gates must match the independent oracle."""
    require("attention", op, solution, torch.bfloat16, "q")
    batch, heads, value_heads, dim = 4, 4, 12, 128
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=torch.bfloat16,
        T=steps,
        batch=batch,
        num_q_heads=heads,
        num_v_heads=value_heads,
        head_dim=dim,
        pool_size=batch + 1,
        state_dtype=state_dtype,
        parameter_dtype=torch.float32,
        parameter_requires_grad=False,
    )
    reads = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    expected, states = _torch_gdn_decode_reference(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        pool[reads],
        dim**-0.5,
    )
    packed = torch.cat([x.flatten(2) for x in (q, k, v)], dim=-1)
    q, k, v = packed.split([heads * dim, heads * dim, value_heads * dim], dim=-1)
    gates = torch.stack((a, b), dim=-1)
    inputs = dict(
        q=q.view(batch, steps, heads, dim),
        k=k.view(batch, steps, heads, dim),
        v=v.view(batch, steps, value_heads, dim),
        a=gates[..., 0],
        b=gates[..., 1],
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=pool,
        initial_state_indices=reads,
        scale=dim**-0.5,
        use_qk_l2norm=True,
        output_state_indices=None,
        solution=solution,
        override=None,
    )
    if op == "gdn_decode_step":
        actual = gdn_decode_step(**inputs)
    else:
        actual = gdn_decode_mtp(
            **inputs,
            disable_state_update=False,
            intermediate_states_buffer=None,
            tree_ancestors=None,
        )
    torch.testing.assert_close(
        actual.float(), expected.to(actual.dtype).float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        pool[reads].float(), states[-1].to(state_dtype).float(), rtol=2e-2, atol=3e-2
    )


def test_gdn_decode_mtp_rejects_strided_tree_ancestors(device: str, require):
    """The kernels read ancestors as a contiguous [B, T] block; a strided view is refused."""
    require("attention", "gdn_decode_mtp", "triton", torch.bfloat16, "q")
    T = 4
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=torch.bfloat16,
        T=T,
        pool_size=32,
        state_dtype=torch.float32,
    )
    wide = torch.ones((q.shape[0], 2 * T), device=device, dtype=torch.int64)
    output_idx = torch.arange(8, 8 + q.shape[0] * T, device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="contiguous int64"):
        gdn_decode_mtp(
            q,
            k,
            v,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            initial_state=pool,
            initial_state_indices=torch.arange(
                q.shape[0], device=device, dtype=torch.int32
            ),
            output_state_indices=output_idx.view(q.shape[0], T),
            tree_ancestors=wide[:, ::2],
            scale=q.shape[-1] ** -0.5,
            disable_state_update=False,
            use_qk_l2norm=True,
        )


def _tree_ancestors(parents: list[list[int]], device: str) -> torch.Tensor:
    """Ancestor-or-self bitmask rows for parent lists: bit s of row t marks s on t's root path."""
    rows = []
    for par in parents:
        row = []
        for t in range(len(par)):
            bits, node = 0, t
            while node >= 0:
                bits |= 1 << node
                node = par[node]
            row.append(bits - (1 << 64) if bits >= 1 << 63 else bits)
        rows.append(row)
    return torch.tensor(rows, device=device, dtype=torch.int64)


def _branch_points(parents: list[int]) -> set[int]:
    """Steps with a child other than the next step: the states a tree reloads."""
    return {p for t, p in enumerate(parents) if p >= 0 and t != p + 1}


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_gdn_decode_mtp_replay_tree_keeps_node_states_out_of_the_pool(
    device: str, state_dtype: torch.dtype, require
):
    """ReplaySSM tree verify: branch-point states go to the intermediate buffer,
    children reload from it, the pool is untouched, and outputs match the
    pool-row tree."""
    require("attention", "gdn_decode_mtp", "triton", torch.bfloat16, "q")

    T = 6
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device, dtype=torch.bfloat16, T=T, pool_size=32, state_dtype=state_dtype
    )
    parents = [
        [-1, 0, 1, 1, 0, 4],
        [-1, -1, 1, -1, 3, 3],
        [-1, 0, 1, 2, 3, 4],
        [-1, 0, 0, 0, 0, 0],
    ]
    read_idx = torch.tensor([1, 3, 5, 7], device=device, dtype=torch.int32)
    output_idx = torch.arange(8, 8 + 4 * T, device=device, dtype=torch.int32).view(4, T)
    scale = q.shape[-1] ** -0.5
    common = dict(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state_indices=read_idx,
        tree_ancestors=_tree_ancestors(parents, device),
        scale=scale,
        use_qk_l2norm=True,
    )

    pool_rows = pool.clone()
    want = gdn_decode_mtp(
        q,
        k,
        v,
        initial_state=pool_rows,
        output_state_indices=output_idx,
        disable_state_update=False,
        **common,
    )
    replay_pool = pool.clone()
    states = torch.full(
        (4, T, *pool.shape[1:]), float("nan"), device=device, dtype=state_dtype
    )
    got = gdn_decode_mtp(
        q,
        k,
        v,
        initial_state=replay_pool,
        intermediate_states_buffer=states,
        disable_state_update=True,
        **common,
    )

    assert torch.equal(replay_pool, pool)
    assert torch.equal(got, want)
    for bi in range(4):
        branch_points = _branch_points(parents[bi])
        for t in range(T):
            if t in branch_points:
                assert torch.equal(states[bi, t], pool_rows[int(output_idx[bi, t])])
            else:
                assert states[bi, t].isnan().all(), (bi, t)


def test_gdn_decode_mtp_replay_tree_node_states_beyond_int32_offsets(
    device: str, require
):
    """A 64-node ReplaySSM tree batch at the Qwen3.8 TP1 GDN shape spans more than
    2**31 node-state elements; the last request still matches running it alone."""
    require("attention", "gdn_decode_mtp", "triton", torch.bfloat16, "q")
    B, T, H, HV, K, V = 44, 64, 16, 48, 128, 128
    assert B * T * HV * V * K > 2**31
    gen = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(B, T, H, K, generator=gen, device=device).bfloat16()
    k = torch.randn(B, T, H, K, generator=gen, device=device).bfloat16()
    v = torch.randn(B, T, HV, V, generator=gen, device=device).bfloat16()
    a = torch.randn(B, T, HV, generator=gen, device=device).bfloat16()
    b = torch.randn(B, T, HV, generator=gen, device=device).bfloat16()
    A_log = torch.rand(HV, generator=gen, device=device) * 2 - 1
    dt_bias = torch.randn(HV, generator=gen, device=device).bfloat16()
    pool = (torch.randn(B + 1, HV, V, K, generator=gen, device=device) * 0.1).bfloat16()
    read_idx = torch.arange(1, B + 1, dtype=torch.int32, device=device)
    parents = torch.tensor(
        [[-1] + [t // 2 for t in range(1, T)]] * B, dtype=torch.int32, device=device
    )
    ancestors = _tree_ancestors(parents.tolist(), device)

    def run(rows: slice) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(range(B)[rows])
        states = torch.empty((n, T, HV, V, K), device=device, dtype=torch.bfloat16)
        out = gdn_decode_mtp(
            q[rows],
            k[rows],
            v[rows],
            A_log=A_log,
            a=a[rows],
            dt_bias=dt_bias,
            b=b[rows],
            initial_state=pool,
            initial_state_indices=read_idx[rows],
            tree_ancestors=ancestors[rows],
            intermediate_states_buffer=states,
            use_qk_l2norm=True,
            disable_state_update=True,
        )
        return out, states

    out, states = run(slice(0, B))
    alone_out, alone_states = run(slice(B - 1, B))
    assert torch.equal(out[B - 1], alone_out[0])
    kept = sorted(_branch_points(parents[B - 1].tolist()))
    assert torch.equal(states[B - 1, kept], alone_states[0, kept])


def _random_tree(T: int, gen: torch.Generator) -> list[int]:
    """A tree in topological order: each node hangs off an earlier node or the initial state."""
    return [int(torch.randint(-1, t, (1,), generator=gen)) for t in range(T)]


def _torch_gdn_tree_reference(
    q, k, v, a, b, A_log, dt_bias, state_klast, scale, parents, *, use_qk_l2norm
):
    """Float64 per-node recurrence from each node's parent: outputs and K-last states."""
    q, k = q.double(), k.double()
    if use_qk_l2norm:
        q, k = gdn_l2norm(q, 1e-6), gdn_l2norm(k, 1e-6)
    return torch_gdn_recurrence(
        q,
        k,
        v,
        gdn_gate(a.double(), dt_bias.double(), A_log.double()),
        torch.sigmoid(b.double()),
        state_klast,
        scale=scale,
        parents=parents,
        compute_dtype=torch.float64,
    )


def _tree_verify(q, k, v, pool, parents, *, rows: bool, **common):
    """Both tree modes: pool rows (step by step), or ReplaySSM with a node-state buffer when the tree needs one."""
    B, T = q.shape[:2]
    common.update(tree_ancestors=_tree_ancestors(parents, q.device.type))
    if rows:
        out_idx = torch.arange(B, B + B * T, device=q.device, dtype=torch.int32).view(
            B, T
        )
        return (
            gdn_decode_mtp(
                q,
                k,
                v,
                initial_state=pool,
                output_state_indices=out_idx,
                disable_state_update=False,
                **common,
            ),
            out_idx,
        )
    buffer = None
    if gdn_tree_verify_needs_node_states(T):
        buffer = torch.empty(B, T, *pool.shape[1:], device=q.device, dtype=pool.dtype)
    return (
        gdn_decode_mtp(
            q,
            k,
            v,
            initial_state=pool,
            intermediate_states_buffer=buffer,
            disable_state_update=True,
            **common,
        ),
        None,
    )


@pytest.mark.parametrize("T", [7, 8, 14, 16, 17, 33, 64])
@pytest.mark.parametrize(
    ("dtype", "state_dtype"),
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
    ],
)
def test_gdn_decode_mtp_random_trees_match_float64_reference(
    device: str, T: int, dtype: torch.dtype, state_dtype: torch.dtype, require
):
    """Random multi-root trees and a chain on both sides of the chunked sizes, both state modes."""
    require("attention", "gdn_decode_mtp", "triton", dtype, "q")
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=dtype,
        T=T,
        pool_size=4 + 4 * T,
        state_dtype=state_dtype,
    )
    gen = torch.Generator().manual_seed(T)
    parents = [_random_tree(T, gen) for _ in range(3)] + [[t - 1 for t in range(T)]]
    read_idx = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
    scale = q.shape[-1] ** -0.5
    common = dict(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state_indices=read_idx,
        scale=scale,
        use_qk_l2norm=True,
    )
    want_out, want_states = _torch_gdn_tree_reference(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        pool[read_idx.long()],
        scale,
        parents,
        use_qk_l2norm=True,
    )

    rows_pool = pool.clone()
    out, out_idx = _tree_verify(q, k, v, rows_pool, parents, rows=True, **common)
    replay_pool = pool.clone()
    replay_out, _ = _tree_verify(q, k, v, replay_pool, parents, rows=False, **common)

    # The ReplaySSM verify writes nothing, within two bf16 ulps of the float64 recurrence.
    torch.testing.assert_close(replay_out.double(), want_out, rtol=2**-7, atol=1e-4)
    assert torch.equal(replay_pool, pool)
    torch.testing.assert_close(out.double(), want_out, rtol=2**-7, atol=1e-4)
    state_tol = dict(rtol=1e-4, atol=1e-4)
    if state_dtype == torch.bfloat16:
        # The pool-row tree stores every node's state rounded to the pool dtype.
        state_tol = dict(rtol=2**-7, atol=4e-3)
    torch.testing.assert_close(
        rows_pool[out_idx.long()].double(), want_states, **state_tol
    )


@pytest.mark.parametrize(
    "T,num_v_heads,head_dim",
    [
        (GDN_TREE_VERIFY_CHUNKED_MIN_NODES, 4, 128),
        (11, 8, 128),
        (GDN_TREE_VERIFY_CHUNKED_MAX_NODES, 8, 128),
        # Too narrow to halve: the V tile stays a full tl.dot operand.
        (GDN_TREE_VERIFY_CHUNKED_MIN_NODES, 4, 16),
    ],
)
def test_gdn_decode_mtp_chunked_tree_fills_the_replay_payload(
    device: str, require, T: int, num_v_heads: int, head_dim: int
):
    """A chunked ReplaySSM tree reading packed q/k/v and strided gate views
    writes each token's ``[k | v | a | b]`` as read into the payload, idle rows
    included, and the same output as from contiguous inputs without one."""
    require("attention", "gdn_decode_mtp", "triton", torch.bfloat16, "q")
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=torch.bfloat16,
        T=T,
        num_v_heads=num_v_heads,
        head_dim=head_dim,
        pool_size=4,
        state_dtype=torch.float32,
    )
    a, b = a.bfloat16(), b.bfloat16()
    batch, _, H, K = k.shape
    HV, V = v.shape[2:]
    rows = batch * T
    parents = [
        [t - 1 for t in range(T)],
        [-1] + [0] * (T - 1),
        [-1, 0, 1, 1, 0, 4] + [t - 1 for t in range(6, T)],
        [-1, -1, 1, -1, 3, 3] + [t - 1 for t in range(6, T)],
    ]
    common = dict(
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=pool,
        initial_state_indices=torch.tensor(
            [0, 1, -1, 3], device=device, dtype=torch.int32
        ),
        scale=K**-0.5,
        use_qk_l2norm=True,
        disable_state_update=True,
        tree_ancestors=_tree_ancestors(parents, device),
    )
    want = gdn_decode_mtp(q, k, v, a=a, b=b, **common)

    mixed = torch.cat([t.reshape(rows, -1) for t in (q, k, v)], dim=1)
    q_view, k_view, v_view = (
        part.view(batch, T, -1, last)
        for part, last in zip(mixed.split([H * K, H * K, HV * V], dim=1), (K, K, V))
    )
    gates = torch.cat([b.reshape(rows, -1), a.reshape(rows, -1)], dim=1)
    b_view, a_view = (gate.view(batch, T, HV) for gate in gates.split(HV, dim=1))
    payload = torch.full(
        (rows, H * K + HV * V + 2 * HV), float("nan"), device=device, dtype=q.dtype
    )
    got = gdn_decode_mtp(
        q_view, k_view, v_view, a=a_view, b=b_view, replay_payload=payload, **common
    )

    assert torch.equal(got, want)
    expected = torch.cat([t.reshape(rows, -1) for t in (k, v, a, b)], dim=1)
    assert torch.equal(payload, expected)


@pytest.mark.parametrize(
    ("A_log_value", "state_value"), [(88.0, None), (89.0, None), (10.0, 3e38)]
)
def test_gdn_decode_mtp_chunked_tree_keeps_extreme_gates_finite(
    device: str, A_log_value: float, state_value: float | None, require
):
    """Gates whose path sums overflow, or that are infinite themselves, decay to zero instead of NaN, even over a near-overflow state."""
    require("attention", "gdn_decode_mtp", "triton", torch.bfloat16, "q")
    T = GDN_TREE_VERIFY_CHUNKED_MAX_NODES
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device, dtype=torch.bfloat16, T=T, pool_size=4, state_dtype=torch.float32
    )
    A_log = torch.full_like(A_log, A_log_value)
    if state_value is not None:
        pool.fill_(state_value)
    parents = [[t - 1 for t in range(T)]] * 4
    read_idx = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
    scale = q.shape[-1] ** -0.5
    common = dict(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state_indices=read_idx,
        scale=scale,
        use_qk_l2norm=True,
    )

    out, _ = _tree_verify(q, k, v, pool, parents, rows=False, **common)

    want, _ = _torch_gdn_tree_reference(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        pool[read_idx.long()],
        scale,
        parents,
        use_qk_l2norm=True,
    )
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.double(), want, rtol=2**-7, atol=1e-4)


@pytest.mark.parametrize("regime", ["correlated_keys", "mixed_gates", "raw_qk"])
def test_gdn_decode_mtp_chunked_tree_matches_float64_within_one_ulp(
    device: str, regime: str, require
):
    """Hard regimes for the chunked form: near-parallel keys at full write strength
    and no decay (the solve), large root-path gates over O(1) segments (the decays),
    and q/k taken without the in-kernel L2 norm; Qwen3.8's 3 value heads per key head.
    """
    require("attention", "gdn_decode_mtp", "triton", torch.bfloat16, "q")
    T = GDN_TREE_VERIFY_CHUNKED_MAX_NODES
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=torch.bfloat16,
        T=T,
        num_v_heads=12,
        pool_size=4,
        state_dtype=torch.float32,
    )
    use_qk_l2norm = True
    if regime == "correlated_keys":
        k = (k[:, :1] + 0.01 * k).to(k.dtype)
        v = 4 * v
        b = torch.full_like(b, 9.0)
        A_log = torch.full_like(A_log, -14.0)
    elif regime == "mixed_gates":
        A_log = torch.full_like(A_log, 7.0)
        dt_bias = torch.full_like(dt_bias, -10.0)
        a = torch.full_like(a, 2.0)
        a[:, :4] = 40.0
    else:
        q, k = ((x / x.shape[-1] ** 0.5).to(x.dtype) for x in (q, k))
        use_qk_l2norm = False
    # Read row -1 is the zero state; the row before the pool holds NaN.
    backing = torch.full((5, *pool.shape[1:]), float("nan"), device=device)
    backing[1:] = pool
    pool = backing[1:]
    read_idx = torch.tensor([0, -1, 2, 3], device=device, dtype=torch.int32)
    start = torch.where(read_idx[:, None, None, None] >= 0, pool[read_idx.long()], 0.0)
    parents = [[t - 1 for t in range(T)], [-1, 0, 0] + list(range(2, T - 1))] * 2
    scale = 0.05
    common = dict(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state_indices=read_idx,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
    )

    out, _ = _tree_verify(q, k, v, pool, parents, rows=False, **common)

    want, _ = _torch_gdn_tree_reference(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        start,
        scale,
        parents,
        use_qk_l2norm=use_qk_l2norm,
    )
    # Within one bf16 ulp of the float64 recurrence.
    ulp = 2.0 ** (torch.floor(torch.log2(want.abs())) - 7)
    assert ((out.double() - want).abs() <= ulp + 1e-6).all()


def test_gdn_tree_verify_wide_tiles_match_narrow_bit_for_bit(device: str, require):
    """The chunked verify's wide V tiles, taken once (request, head) programs
    outnumber the SMs, give each request the same bits as the narrow tiles of a
    small batch."""
    require("attention", "gdn_decode_mtp", "triton", torch.bfloat16, "q")
    T, num_v_heads = 14, 8
    wide_batch = current_platform().sm_count // num_v_heads + 1
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=torch.bfloat16,
        T=T,
        batch=wide_batch,
        pool_size=wide_batch,
        num_v_heads=num_v_heads,
        state_dtype=torch.float32,
    )
    gen = torch.Generator().manual_seed(7)
    parents = [_random_tree(T, gen) for _ in range(wide_batch)]
    indices = torch.arange(wide_batch, device=device, dtype=torch.int32)
    common = dict(A_log=A_log, dt_bias=dt_bias, use_qk_l2norm=True)
    wide, _ = _tree_verify(
        q,
        k,
        v,
        pool,
        parents,
        rows=False,
        a=a,
        b=b,
        initial_state_indices=indices,
        **common,
    )
    narrow, _ = _tree_verify(
        q[:2],
        k[:2],
        v[:2],
        pool,
        parents[:2],
        rows=False,
        a=a[:2],
        b=b[:2],
        initial_state_indices=indices[:2],
        **common,
    )
    assert torch.equal(wide[:2], narrow)


def test_gdn_tree_verify_compiles_once_across_batch_sizes(device: str, require):
    """Both tree kernels compile once per tree size and V tile width, never per batch."""
    require("attention", "gdn_decode_mtp", "triton", torch.bfloat16, "q")
    from tokenspeed_kernel.ops.attention.gdn.triton import (
        _fused_gdn_decode_update_kernel,
        _gdn_tree_verify_chunked_kernel,
    )

    num_v_heads = 8

    def run(batch: int, T: int, rows: bool, payload: bool) -> None:
        q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
            device=device,
            dtype=torch.bfloat16,
            T=T,
            batch=batch,
            pool_size=batch + batch * T,
            num_v_heads=num_v_heads,
            state_dtype=torch.float32,
        )
        gen = torch.Generator().manual_seed(batch)
        parents = [_random_tree(T, gen) for _ in range(batch)]
        replay = {}
        if payload:
            a, b = a.bfloat16(), b.bfloat16()
            width = k[0, 0].numel() + v[0, 0].numel() + 2 * v.shape[2]
            replay["replay_payload"] = q.new_empty(batch * T, width)
        _tree_verify(
            q,
            k,
            v,
            pool,
            parents,
            rows=rows,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            initial_state_indices=torch.arange(batch, device=device, dtype=torch.int32),
            use_qk_l2norm=True,
            **replay,
        )

    # V tile tiers: half width at a few requests, full up to one program per SM, wider past it.
    full_batch = current_platform().sm_count // num_v_heads
    for T, rows, payload in (
        (14, False, False),
        (14, False, True),
        (24, False, False),
        (14, True, False),
    ):
        for batch in (4, full_batch, full_batch + 1):
            run(batch, T, rows, payload)
        with assert_no_triton_compile(
            _gdn_tree_verify_chunked_kernel, _fused_gdn_decode_update_kernel
        ):
            for batch in (1, 2, 3, 5, 8, 17, 33):
                run(batch, T, rows, payload)


def _chain_verify(q, k, v, pool, read_idx, **common):
    """The chunked chain verify, checking it leaves the pool untouched."""
    before = pool.clone()
    out = gdn_chain_verify(
        q, k, v, initial_state=pool, initial_state_indices=read_idx, **common
    )
    assert torch.equal(pool, before)
    return out


@pytest.mark.parametrize("T", [1, 3, 4, 5, 8, 9, 13, 16])
@pytest.mark.parametrize(
    ("dtype", "num_v_heads", "batch"),
    [
        (torch.bfloat16, 12, 2),
        (torch.bfloat16, 12, 8),
        (torch.bfloat16, 12, 20),
        (torch.bfloat16, 12, 40),
        (torch.float16, 8, 4),
    ],
)
def test_gdn_chain_verify_matches_float64_reference(
    device: str, T: int, dtype: torch.dtype, num_v_heads: int, batch: int, require
):
    """Every row bucket and launch tier against float64, with a zero-output padding request."""
    require("attention", "gdn_chain_verify", "triton", dtype, "q")
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=dtype,
        T=T,
        batch=batch,
        num_v_heads=num_v_heads,
        pool_size=batch + 1,
    )
    a, b = a.to(dtype), b.to(dtype)
    read_idx = torch.randperm(batch, device=device).to(torch.int32) + 1
    read_idx[1] = -1
    scale = q.shape[-1] ** -0.5
    common = dict(
        A_log=A_log, a=a, dt_bias=dt_bias, b=b, scale=scale, use_qk_l2norm=True
    )
    out = _chain_verify(q, k, v, pool, read_idx, **common)
    start = torch.where(
        read_idx[:, None, None, None] >= 0, pool[read_idx.clamp(min=0).long()], 0.0
    )
    want, _ = _torch_gdn_tree_reference(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        start,
        scale,
        [[t - 1 for t in range(T)]] * batch,
        use_qk_l2norm=True,
    )
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    live = read_idx >= 0
    # The products round q and the normalized key (with its residual) to the state dtype.
    torch.testing.assert_close(out[live].double(), want[live], rtol=2**-7, atol=2e-4)


@pytest.mark.parametrize("T", [4, 16])
@pytest.mark.parametrize(
    ("A_log_value", "state_value", "one_row"),
    [(88.0, None, False), (89.0, None, False), (10.0, 3e38, False), (None, None, True)],
)
def test_gdn_chain_verify_keeps_extreme_gates_finite(
    device: str,
    T: int,
    A_log_value: float | None,
    state_value: float | None,
    one_row: bool,
    require,
):
    """Overflowing or infinite gates decay to zero, not NaN; later decays stay exact."""
    require("attention", "gdn_chain_verify", "triton", torch.bfloat16, "q")
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device, dtype=torch.bfloat16, T=T, pool_size=4
    )
    a, b = a.bfloat16(), b.bfloat16()
    if A_log_value is not None:
        A_log = torch.full_like(A_log, A_log_value)
    if one_row:
        a[:, T // 2] = 3e38
    if state_value is not None:
        pool.fill_(state_value)
    read_idx = torch.arange(4, device=device, dtype=torch.int32)
    scale = q.shape[-1] ** -0.5
    common = dict(
        A_log=A_log, a=a, dt_bias=dt_bias, b=b, scale=scale, use_qk_l2norm=True
    )
    out = _chain_verify(q, k, v, pool, read_idx, **common)
    want, _ = _torch_gdn_tree_reference(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        pool[read_idx.long()],
        scale,
        [[t - 1 for t in range(T)]] * 4,
        use_qk_l2norm=True,
    )
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.double(), want, rtol=2**-7, atol=2e-4)


@pytest.mark.parametrize("T", [3, 13])
def test_gdn_chain_verify_reads_strided_rows_and_gates(device: str, T: int, require):
    """Strided q/k/v rows and unaligned gate views give bitwise the dense inputs' outputs."""
    require("attention", "gdn_chain_verify", "triton", torch.bfloat16, "q")
    batch, num_v_heads = 5, 12
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=torch.bfloat16,
        T=T,
        batch=batch,
        num_v_heads=num_v_heads,
        pool_size=batch,
    )
    a, b = a.bfloat16(), b.bfloat16()
    rows = batch * T
    common = dict(
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=pool,
        initial_state_indices=torch.arange(batch, device=device, dtype=torch.int32),
        scale=None,
        use_qk_l2norm=True,
    )
    dense = gdn_chain_verify(q, k, v, a=a, b=b, **common)
    # [q | k | v] rows padded to a wider stride, gates at an odd column offset.
    widths = [q[0, 0].numel(), k[0, 0].numel(), v[0, 0].numel()]
    packed = torch.zeros(rows, sum(widths) + 64, device=device, dtype=torch.bfloat16)
    projection = torch.zeros(
        rows, 3 + 2 * num_v_heads, device=device, dtype=torch.bfloat16
    )
    views = []
    start = 0
    for tensor, width in zip((q, k, v), widths):
        view = packed[:, start : start + width]
        view.copy_(tensor.reshape(rows, width))
        views.append(view.view(batch, T, -1, tensor.shape[-1]))
        start += width
    gates = []
    for offset, tensor in ((3, a), (3 + num_v_heads, b)):
        view = projection[:, offset : offset + num_v_heads]
        view.copy_(tensor.reshape(rows, num_v_heads))
        gates.append(view.view(batch, T, num_v_heads))
    strided = gdn_chain_verify(*views, a=gates[0], b=gates[1], **common)
    assert torch.equal(strided, dense)


def test_gdn_chain_verify_without_qk_l2norm(device: str, require):
    """Callers that normalize q and k themselves pass an explicit scale."""
    require("attention", "gdn_chain_verify", "triton", torch.bfloat16, "q")
    T, batch = 4, 3
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device, dtype=torch.bfloat16, T=T, batch=batch, pool_size=batch
    )
    q, k = (F.normalize(x.float(), dim=-1).bfloat16() for x in (q, k))
    a, b = a.bfloat16(), b.bfloat16()
    read_idx = torch.arange(batch, device=device, dtype=torch.int32)
    out = _chain_verify(
        q,
        k,
        v,
        pool,
        read_idx,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=0.5,
        use_qk_l2norm=False,
    )
    want, _ = _torch_gdn_tree_reference(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        pool[read_idx.long()],
        0.5,
        [[t - 1 for t in range(T)]] * batch,
        use_qk_l2norm=False,
    )
    torch.testing.assert_close(out.double(), want, rtol=2**-7, atol=2e-4)


def test_gdn_chain_verify_compiles_once_per_launch_tier(device: str, require):
    """Launches at ``gdn_chain_verify_warm_batch_sizes`` compile every batch and row count."""
    require("attention", "gdn_chain_verify", "triton", torch.bfloat16, "q")
    from tokenspeed_kernel.ops.attention.gdn._triton.chain_verify import (
        gdn_chain_verify_kernel,
    )
    from tokenspeed_kernel.ops.attention.gdn.triton import (
        gdn_chain_verify_warm_batch_sizes,
    )

    num_v_heads = 12

    def run(batch: int, T: int) -> None:
        q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
            device=device,
            dtype=torch.bfloat16,
            T=T,
            batch=batch,
            num_v_heads=num_v_heads,
            pool_size=batch,
        )
        gdn_chain_verify(
            q,
            k,
            v,
            A_log=A_log,
            a=a.bfloat16(),
            dt_bias=dt_bias,
            b=b.bfloat16(),
            initial_state=pool,
            initial_state_indices=torch.arange(batch, device=device, dtype=torch.int32),
            scale=None,
            use_qk_l2norm=True,
        )

    for T in (3, 8, 16):
        for batch in gdn_chain_verify_warm_batch_sizes(range(1, 129), num_v_heads, T):
            run(batch, T)
    with assert_no_triton_compile(gdn_chain_verify_kernel):
        for T in (3, 4, 5, 8, 13, 16):
            for batch in (1, 5, 9, 20, 32, 40, 65, 90, 128):
                run(batch, T)
