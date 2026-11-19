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

"""Retained AMD KDA against eager convolution and delta-rule references."""

from __future__ import annotations

import importlib

import pytest
import torch
from tokenspeed_kernel.ops.attention.kda import (
    KdaOutputNorm,
    KdaRetainedHistory,
    KdaRetainedRecords,
    kda_fused_paged_verify,
)
from utils import assert_no_triton_compile, is_cdna4, is_cdna5

if not (is_cdna4() or is_cdna5()):
    pytest.skip("requires gfx950 or gfx1250", allow_module_level=True)

from tokenspeed_kernel.ops.attention.kda import gluon as api

ARCH = "gfx950" if is_cdna4() else "gfx1250"
DECODE = importlib.import_module(
    f"tokenspeed_kernel_amd.ops.{ARCH}.attention.kda.decode"
)
from tokenspeed_kernel_amd.ops import _kda_records as records

KERNELS = (
    getattr(DECODE, f"gluon_kda_fused_paged_verify_nostore_vmajor_{ARCH}"),
    records.kda_retained_consolidate,
    records.kda_retained_reset,
    records.kda_retained_conv_commit,
    records.kda_retained_publish,
)
VERIFY = getattr(api, f"gluon_kda_retained_verify_{ARCH}")
COMMIT = getattr(api, f"gluon_kda_retained_commit_{ARCH}")


def _reconstruct(base, history, page, length):
    current = base.clone()
    for t in range(length):
        current = current * history.decay[page, :, t, None, :]
        current = (
            current
            + history.correction[page, :, t, :, None] * history.key[page, :, t, None, :]
        )
    return current


def _reference(qkv, weights, conv, gate, beta, a, bias, state, lower_bound):
    outputs, states, us, keys, decays = [], [], [], [], []
    taps = conv.float().clone()
    for t in range(qkv.shape[0]):
        current = qkv[t].float()
        value = (taps * weights[:, :3]).sum(-1) + current * weights[:, 3]
        value = torch.nn.functional.silu(value).to(qkv.dtype).float()
        q, k, v = value.reshape(3, -1, 128).unbind(0)
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * 128**-0.5
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        g = gate[t].float().reshape(-1, 128) + bias
        log_decay = (
            lower_bound * torch.sigmoid(a.exp()[:, None] * g)
            if lower_bound is not None
            else -a.exp()[:, None] * torch.nn.functional.softplus(g)
        )
        decay = log_decay.exp()
        state = state * decay[:, None, :]
        u = beta[t].float().sigmoid()[:, None] * (
            v - torch.einsum("hvk,hk->hv", state, k)
        )
        state = state + u[:, :, None] * k[:, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", state, q))
        states.append(state.clone())
        us.append(u)
        keys.append(k)
        decays.append(decay)
        taps = torch.cat((taps[:, 1:], current[:, None]), -1)
    return (
        torch.stack(outputs),
        states,
        torch.stack(us),
        torch.stack(keys),
        torch.stack(decays),
    )


def _arena(value, step):
    # Keep channels strided too: retained storage is an arbitrary arena view.
    backing = torch.empty(
        tuple(step * n for n in value.shape), device=value.device, dtype=value.dtype
    )
    view = backing[tuple(slice(None, None, step) for _ in value.shape)]
    view.copy_(value)
    return view


def _case(
    capacity,
    width,
    initial_length,
    accepted_rounds,
    *,
    cross_page,
    strided,
    projected,
    lower_bound,
    lengths=None,
    stamped=True,
    corruption=None,
    graph=False,
    null_row=False,
    guard=False,
    retain_input=True,
    stamp_capacity=None,
    arena_step=2,
    gate_error=None,
    state_dtype=torch.float32,
    output_norm=False,
):
    torch.manual_seed(611)
    heads, dim = 1, 128
    lengths = [width] if lengths is None else lengths
    batch = len(lengths)
    stamp_capacity = capacity if stamp_capacity is None else stamp_capacity
    pages = batch + 2
    source = torch.ones(batch, dtype=torch.int32, device="cuda")
    if not cross_page:
        source = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    live = torch.arange(
        2 if cross_page else 1,
        batch + (2 if cross_page else 1),
        dtype=torch.int32,
        device="cuda",
    )
    state = (torch.randn(pages, heads, dim, dim, device="cuda") * 0.02).to(state_dtype)
    conv = torch.randn(pages, 3 * heads * dim, 3, device="cuda").bfloat16()
    u = torch.randn(pages, heads, capacity, dim, device="cuda") * 0.02
    key = torch.nn.functional.normalize(torch.randn_like(u), dim=-1)
    decay = torch.rand_like(key) * 0.2 + 0.7
    length = torch.full((pages, 1), initial_length, device="cuda", dtype=torch.int32)
    if strided:
        u, key, decay, length, live = [
            _arena(x, arena_step) for x in (u, key, decay, length, live)
        ]
        source = _arena(source, arena_step)
    history = KdaRetainedHistory(u, key, decay, length, live)
    # Deliberately wrong live convolution must never affect a cross-page verify.
    if cross_page:
        conv[live.long()] = 11
    if null_row:
        source[-1] = 0
    expected = [
        _reconstruct(state[int(s)], history, int(s), initial_length) for s in source
    ]
    expected_conv = [conv[int(s)].clone() for s in source]
    weights = torch.randint(-3, 4, (3 * heads * dim, 4), device="cuda").float() * 0.125
    a = torch.randn(heads, device="cuda") * 0.1 - 2
    bias = torch.randn(heads, dim, device="cuda") * 0.1
    starts_list = [0]
    for n in lengths:
        starts_list.append(starts_list[-1] + n)
    starts = torch.tensor(starts_list, device="cuda", dtype=torch.int32)
    if strided:
        starts = _arena(starts, arena_step)
    metadata = (
        torch.zeros(batch, 3, device="cuda", dtype=torch.int64) if stamped else None
    )
    stamps = (
        torch.full((batch, stamp_capacity, 2), -1, device="cuda", dtype=torch.int64)
        if stamped
        else None
    )
    source_state, source_conv = state[1].clone(), conv[1].clone()
    source_records = [x[1].clone() for x in (u, key, decay, length)]
    for round_idx, accepted in enumerate(accepted_rounds):
        tokens = starts_list[-1]
        mixed = (
            torch.randint(-8, 9, (tokens, 3 * heads * dim), device="cuda").bfloat16()
            * 0.125
        )
        original = torch.full_like(mixed, 17) if retain_input else mixed
        beta = torch.randn(tokens, heads, device="cuda").bfloat16()
        f_a = torch.randint(-2, 3, (tokens, dim), device="cuda").bfloat16() * 0.125
        f_b = torch.randint(-2, 3, (heads * dim, dim), device="cuda").bfloat16() * 0.125
        gate = (
            torch.nn.functional.linear(f_a, f_b)
            if projected
            else torch.randn(tokens, heads * dim, device="cuda").bfloat16()
        )
        if stamped and batch:
            metadata.copy_(
                torch.tensor(
                    [
                        [r, 7 + round_idx, capacity - 1 + round_idx * width]
                        for r in range(batch)
                    ],
                    device="cuda",
                )
            )
        before = state.clone()
        conv_before = conv.clone()
        old_lengths = [int(length[int(s), 0]) for s in source]
        references = []
        for r, n in enumerate(lengths):
            if n and int(source[r]) > 0:
                begin, end = starts_list[r : r + 2]
                if int(source[r]) != int(live[r]) or old_lengths[r] + n > capacity:
                    expected[r] = expected[r].to(state_dtype).float()
                references.append(
                    _reference(
                        mixed[begin:end],
                        weights,
                        expected_conv[r],
                        gate[begin:end],
                        beta[begin:end],
                        a,
                        bias,
                        expected[r],
                        lower_bound,
                    )
                )
            else:
                references.append(None)
        kwargs = dict(
            mixed_qkv=mixed,
            original_qkv=original if retain_input else None,
            conv_weights=weights,
            conv_states=conv,
            raw_g=None if projected else gate,
            f_a_out=f_a if projected else None,
            f_b_weight=f_b if projected else None,
            beta_logits=beta,
            A_log=a,
            dt_bias=bias,
            state_pool=state,
            read_indices=source,
            query_start_loc=starts,
            retained_history=history,
            record_metadata=metadata,
            record_stamps=stamps,
            num_heads=heads,
            head_dim=dim,
            draft_token_num=width,
            lower_bound=lower_bound,
        )
        norm = None
        if output_norm:
            norm = KdaOutputNorm(
                gate=torch.randn(tokens, heads * dim, device="cuda").bfloat16(),
                weight=torch.randn(dim, device="cuda").bfloat16(),
                eps=1e-5,
            )

            def verify_with_norm(**inputs):
                retained = KdaRetainedRecords(
                    original_qkv=inputs.pop("original_qkv"),
                    history=inputs.pop("retained_history"),
                    metadata=inputs.pop("record_metadata"),
                    stamps=inputs.pop("record_stamps"),
                    query_start_loc=inputs.pop("query_start_loc"),
                )
                inputs["g_raw"] = inputs.pop("raw_g")
                return kda_fused_paged_verify(
                    **inputs,
                    retained_records=retained,
                    output_norm=norm,
                    recurrent_layout="v_major",
                    override=None,
                    solution=None,
                )

            verify = verify_with_norm
        else:
            verify = VERIFY
        if gate_error:
            if gate_error == "both":
                kwargs.update(raw_g=gate, f_a_out=f_a, f_b_weight=f_b)
            else:
                kwargs.update(raw_g=None, f_a_out=f_a, f_b_weight=None)
            with pytest.raises(ValueError, match="gate"):
                verify(**kwargs)
            return history
        if guard or graph:
            verify(**kwargs)
        if guard:
            with assert_no_triton_compile(*KERNELS[:3]):
                out = verify(**kwargs)
        elif graph:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                out = verify(**kwargs)
            # Captured kernels must consume refreshed metadata in stable storage.
            if stamped:
                metadata[:, 1].add_(1)
                metadata[:, 2].add_(width)
            capture.replay()
        else:
            out = verify(**kwargs)
        torch.testing.assert_close(conv, conv_before, atol=0, rtol=0)
        assert out.shape == (
            (tokens, heads * dim) if output_norm else (1, tokens, heads, dim)
        )
        out_heads = out.reshape(1, tokens, heads, dim)
        for r, ref in enumerate(references):
            dst, src = int(live[r]), int(source[r])
            if ref is None:
                torch.testing.assert_close(state[dst], before[dst], atol=0, rtol=0)
                if n := lengths[r]:
                    begin, end = starts_list[r : r + 2]
                    torch.testing.assert_close(
                        out_heads[0, begin:end],
                        torch.zeros_like(out_heads[0, begin:end]),
                        atol=0,
                        rtol=0,
                    )
                continue
            consolidated = dst != src or old_lengths[r] + lengths[r] > capacity
            old = 0 if consolidated else old_lengths[r]
            assert int(length[dst, 0]) == old
            torch.testing.assert_close(
                state[dst].float(),
                expected[r] if consolidated else before[dst].float(),
                atol=2e-6,
                rtol=2e-5,
            )
            begin, end = starts_list[r : r + 2]
            target_output = ref[0]
            if norm is not None:
                core = target_output.to(mixed.dtype).float()
                target_output = (
                    (
                        core
                        * torch.rsqrt(core.square().mean(-1, keepdim=True) + norm.eps)
                        * norm.weight.float()
                        * norm.gate[begin:end].reshape(-1, heads, dim).float().sigmoid()
                    )
                    .to(mixed.dtype)
                    .float()
                )
            torch.testing.assert_close(
                out_heads[0, begin:end].float(),
                target_output,
                atol=3e-3 if norm is not None else 2e-3,
                rtol=1e-2 if norm is not None else 5e-3,
            )
            for actual, target in zip((u, key, decay), ref[2:]):
                torch.testing.assert_close(
                    actual[dst, :, old : old + lengths[r]],
                    target.transpose(0, 1),
                    atol=3e-6,
                    rtol=3e-5,
                )
        for r, ref in enumerate(references):
            begin, end = starts_list[r : r + 2]
            target = (
                mixed[begin:end]
                if ref is not None or not retain_input
                else torch.full_like(mixed[begin:end], 17)
            )
            torch.testing.assert_close(original[begin:end], target, atol=0, rtol=0)
        records_before = [x.clone() for x in (u, key, decay)]
        dense_before = state.clone()
        counts = torch.full((batch,), accepted, device="cuda", dtype=torch.int32)
        if corruption:
            if corruption in ("position", "generation", "rejected"):
                position = int(metadata[0, 2]) + (
                    lengths[0] - 1 if corruption == "rejected" else 0
                )
                stamps[
                    0, position % stamp_capacity, 1 if corruption == "generation" else 0
                ] += 1
            elif corruption == "slot":
                metadata[0, 0] = stamps.shape[0]
            elif corruption == "start":
                metadata[0, 2] = -1
        length_before = length.clone()
        conv_before = conv.clone()
        commit_kwargs = dict(
            original_qkv=original,
            conv_states=conv,
            read_indices=source,
            query_start_loc=starts,
            accepted_length=counts,
            retained_history=history,
            record_metadata=metadata,
            record_stamps=stamps,
            num_heads=heads,
            head_dim=dim,
            draft_token_num=width,
        )
        if guard:
            # A0 warm-up exercises the same launch geometry without advancing length.
            counts.zero_()
            COMMIT(**commit_kwargs)
            counts.fill_(accepted)
            with assert_no_triton_compile(*KERNELS[3:]):
                COMMIT(**commit_kwargs)
        elif graph:
            counts.zero_()
            COMMIT(**commit_kwargs)
            capture_commit = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture_commit):
                COMMIT(**commit_kwargs)
            counts.fill_(accepted)
            capture_commit.replay()
        else:
            COMMIT(**commit_kwargs)
        if corruption and corruption != "rejected":
            torch.testing.assert_close(conv, conv_before, atol=0, rtol=0)
            torch.testing.assert_close(length, length_before, atol=0, rtol=0)
            return history
        torch.testing.assert_close(state, dense_before, atol=0, rtol=0)
        for actual, before_record in zip((u, key, decay), records_before):
            torch.testing.assert_close(actual, before_record, atol=0, rtol=0)
        for r, n in enumerate(lengths):
            if not n or references[r] is None:
                dst = int(live[r])
                torch.testing.assert_close(conv[dst], conv_before[dst], atol=0, rtol=0)
                torch.testing.assert_close(
                    length[dst], length_before[dst], atol=0, rtol=0
                )
                continue
            take = max(0, min(accepted, n))
            dst = int(live[r])
            old = (
                0
                if int(source[r]) != dst or old_lengths[r] + n > capacity
                else old_lengths[r]
            )
            assert int(length[dst, 0]) == old + take
            if take:
                expected[r] = references[r][1][take - 1]
            if take:
                begin = starts_list[r]
                expected_conv[r] = torch.cat(
                    (expected_conv[r], mixed[begin : begin + take].T), -1
                )[:, -3:]
            torch.testing.assert_close(conv[dst], expected_conv[r], atol=0, rtol=0)
            torch.testing.assert_close(
                _reconstruct(state[dst], history, dst, old + take),
                expected[r],
                atol=4e-6,
                rtol=4e-5,
            )
        if cross_page and round_idx == 0:
            torch.testing.assert_close(state[1], source_state, atol=0, rtol=0)
            torch.testing.assert_close(conv[1], source_conv, atol=0, rtol=0)
            for actual, before_record in zip((u, key, decay, length), source_records):
                torch.testing.assert_close(actual[1], before_record, atol=0, rtol=0)
        source = live.clone()
    return history


def test_retained_smoke_gfx1250():
    """One head, one row, T1 and two rounds; bounded simulator coverage."""
    _case(
        8,
        1,
        2,
        (0, 1),
        cross_page=False,
        strided=False,
        projected=False,
        lower_bound=-5.0,
    )


@pytest.mark.parametrize(
    "capacity,initial_length,cross_page",
    [(8, 6, False), (8, 7, False), (16, 3, True)],
    ids=["equal", "overflow", "source"],
)
def test_retained_capacity_and_source_gfx1250(capacity, initial_length, cross_page):
    _case(
        capacity,
        4,
        initial_length,
        (2,),
        cross_page=cross_page,
        strided=False,
        projected=False,
        lower_bound=None,
        lengths=[2],
    )


@pytest.mark.parametrize(
    "capacity,width,initial_length,accepted_rounds",
    [(8, 4, 2, (2, 4)), (16, 1, 5, (0, 1)), (32, 4, 28, (4, 0)), (8, 4, 6, (2, 4))],
)
@pytest.mark.parametrize(
    "strided,projected,lower_bound",
    [(False, False, -5.0), (True, False, None), (True, True, -5.0)],
)
def test_retained_rounds(
    capacity, width, initial_length, accepted_rounds, strided, projected, lower_bound
):
    _case(
        capacity,
        width,
        initial_length,
        accepted_rounds,
        cross_page=False,
        strided=strided,
        projected=projected,
        lower_bound=lower_bound,
    )


@pytest.mark.parametrize("accepted", [0, 2, 4, 99, -1])
def test_retained_shared_source(accepted):
    _case(
        8,
        4,
        3,
        (accepted,),
        cross_page=True,
        strided=True,
        projected=True,
        lower_bound=-5.0,
        lengths=[4, 2, 0],
    )


@pytest.mark.skipif(not is_cdna4(), reason="gfx950 projection GEMM crossover")
def test_retained_gfx950_large_projected_gate():
    _case(
        8,
        1,
        1,
        (1,),
        cross_page=False,
        strided=False,
        projected=True,
        lower_bound=None,
        lengths=[1] * 65,
    )


@pytest.mark.parametrize("stamped", [False, True])
def test_retained_unstamped_diagnostic(stamped):
    _case(
        8,
        1,
        1,
        (1,),
        cross_page=True,
        strided=False,
        projected=False,
        lower_bound=None,
        stamped=stamped,
    )


@pytest.mark.parametrize(
    "corruption", ["position", "generation", "slot", "start", "rejected"]
)
def test_retained_commit_guards_stale_accepted_stamps(corruption):
    _case(
        8,
        4,
        2,
        (2,),
        cross_page=True,
        strided=True,
        projected=False,
        lower_bound=-5.0,
        corruption=corruption,
    )


@pytest.mark.parametrize("graph", [False, True])
def test_retained_metadata_refresh_and_graph(graph):
    _case(
        8,
        1,
        2,
        (1, 1),
        cross_page=False,
        strided=True,
        projected=False,
        lower_bound=-5.0,
        graph=graph,
        guard=not graph,
    )


def test_retained_empty_and_null_rows():
    _case(
        8,
        4,
        2,
        (2,),
        cross_page=True,
        strided=True,
        projected=False,
        lower_bound=None,
        lengths=[2, 0, 1],
        null_row=True,
    )


def test_retained_input_owned_by_caller():
    _case(
        8,
        1,
        2,
        (1,),
        cross_page=False,
        strided=False,
        projected=False,
        lower_bound=None,
        retain_input=False,
    )


def test_retained_stamp_ring_independent_of_history_capacity():
    # Absolute windows wrap the C4 stamp ring while C16 history grows in place.
    _case(
        16,
        4,
        2,
        (2, 4),
        cross_page=False,
        strided=True,
        projected=False,
        lower_bound=None,
        stamp_capacity=4,
    )


def test_retained_empty_batch():
    _case(
        8,
        1,
        2,
        (0,),
        cross_page=False,
        strided=False,
        projected=False,
        lower_bound=None,
        lengths=[],
    )


@pytest.mark.parametrize("width", [1, 4])
def test_retained_runtime_scalar_geometry_does_not_recompile(width):
    # B4 and B3 share the existing value-split bucket. Page counts, request
    # strides and every retained arena stride change; C/T/gate stay fixed.
    _case(
        8,
        width,
        2,
        (1,),
        cross_page=True,
        strided=True,
        projected=False,
        lower_bound=None,
        lengths=[width] * 4,
        arena_step=2,
    )
    with assert_no_triton_compile(*KERNELS):
        _case(
            8,
            width,
            2,
            (1,),
            cross_page=True,
            strided=True,
            projected=False,
            lower_bound=None,
            lengths=[width] * 3,
            arena_step=3,
        )


@pytest.mark.parametrize("gate_error", ["both", "missing"])
def test_retained_gate_selection_is_explicit(gate_error):
    _case(
        8,
        1,
        2,
        (1,),
        cross_page=False,
        strided=False,
        projected=False,
        lower_bound=None,
        gate_error=gate_error,
    )


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "cross_page,initial_length", [(False, 2), (False, 6), (True, 2)]
)
def test_retained_dense_base_rounds_only_when_materialized(
    state_dtype, cross_page, initial_length
):
    _case(
        8,
        4,
        initial_length,
        (2, 4),
        cross_page=cross_page,
        strided=True,
        projected=False,
        lower_bound=-5.0,
        state_dtype=state_dtype,
    )


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_public_retained_verify_preserves_output_norm(width, state_dtype):
    _case(
        8,
        width,
        2,
        (1,),
        cross_page=True,
        strided=False,
        projected=False,
        lower_bound=-5.0,
        state_dtype=state_dtype,
        output_norm=True,
    )


def test_retained_bf16_materialization_smoke_gfx1250():
    _case(
        8,
        1,
        8,
        (1,),
        cross_page=False,
        strided=False,
        projected=False,
        lower_bound=-5.0,
        state_dtype=torch.bfloat16,
    )


def test_retained_output_norm_smoke_gfx1250():
    _case(
        8,
        1,
        2,
        (1,),
        cross_page=True,
        strided=False,
        projected=False,
        lower_bound=-5.0,
        state_dtype=torch.bfloat16,
        output_norm=True,
    )
