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

import json
from contextlib import nullcontext
from dataclasses import asdict, replace

import pytest
import torch
from tokenspeed_kernel.op_profile import (
    CostRegistry,
    OpProfile,
    ProfileSession,
    profiled,
    wrap_op_for_profiling,
)
from tokenspeed_kernel.profile_costs import (
    AttentionMatmulCost,
    CollectiveCost,
    CollectiveTraffic,
    DenseMatmulCost,
    register_compute_costs,
)
from tokenspeed_kernel.profile_utilization import (
    HardwareProfile,
    ScopeTiming,
    analyze_utilization,
)
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, register_kernel


@pytest.mark.parametrize("batched", [False, True])
def test_registered_compute_costs(batched):
    from tokenspeed_kernel.numerics.torch.gemm import torch_bmm, torch_mm

    batch = (2,) if batched else ()
    a, b = torch.randn(*batch, 3, 5), torch.randn(*batch, 4, 5)
    fn = torch_bmm if batched else torch_mm
    session = ProfileSession(run_id="compute", model_id="test", rank=0, backend="none")
    with session.iteration(iter_id=0):
        result = fn(a, b, None, None, torch.float64)
    torch.testing.assert_close(result, (a @ b.transpose(-1, -2)).double())
    costs = CostRegistry()
    register_compute_costs(costs, hbm_policy="unavailable")
    record = session.ops[0]
    assert record.implementation == ("torch_bmm" if batched else "torch_mm")
    report = session.report(costs=costs)
    cost = report["analysis"]["ops"][0]["cost"]
    assert cost["useful_flops"] == 2 * (2 if batched else 1) * 3 * 4 * 5
    assert cost["logical_read_bytes"] == (a.numel() + b.numel()) * 4
    assert cost["logical_write_bytes"] == result.numel() * 8
    assert cost["compute_type"] is None
    assert cost["estimated_hbm_bytes"] is None


@pytest.mark.parametrize("hbm_policy", ["unavailable", "logical_io_once"])
def test_fp8_block_matmul_cost_counts_scales_and_uses_operand_precision(hbm_policy):
    def reference(A, B, A_scales, B_scales, *, out):
        a = A.float() * A_scales.repeat_interleave(128, 1)
        b = (
            B.float()
            * B_scales.repeat_interleave(128, 0).repeat_interleave(128, 1)[: B.shape[0]]
        )
        out.copy_(a @ b.T)
        return out

    registry = KernelRegistry()
    registry.register(
        KernelSpec(name="profile_fp8_reference", family="gemm", mode="mm"),
        reference,
        profile=OpProfile(cost_rule="fp8_block_mm.v1"),
    )
    a = torch.full((3, 256), 0.5).to(torch.float8_e4m3fn)
    b = torch.full((192, 256), 0.25).to(torch.float8_e4m3fn)
    a_scale, b_scale = torch.full((3, 2), 0.5), torch.full((2, 2), 2.0)
    out = torch.empty(3, 192, dtype=torch.bfloat16)
    session = ProfileSession(run_id="fp8", model_id="test", rank=0, backend="none")
    with session.iteration(iter_id=0):
        result = registry.get_impl("profile_fp8_reference")(
            a, b, a_scale, b_scale, out=out
        )
    assert result is out
    torch.testing.assert_close(result, torch.full_like(result, 32))
    costs = CostRegistry()
    register_compute_costs(costs, hbm_policy=hbm_policy)
    cost = session.report(costs=costs)["analysis"]["ops"][0]["cost"]
    assert cost["useful_flops"] == 294912
    assert cost["logical_read_bytes"] == 49960
    assert cost["logical_write_bytes"] == 1152
    assert cost["estimated_hbm_bytes"] == (
        51112 if hbm_policy == "logical_io_once" else None
    )
    assert cost["compute_type"] is None

    # Only descriptors move to CUDA; this precision-policy check uses no device.
    record = session.ops[0]
    session.ops[0] = replace(
        record,
        inputs={key: replace(t, device="cuda:0") for key, t in record.inputs.items()},
        outputs={key: replace(t, device="cuda:0") for key, t in record.outputs.items()},
    )
    assert (
        session.report(costs=costs)["analysis"]["ops"][0]["cost"]["compute_type"]
        == "fp8_e4m3_tensor"
    )
    session.ops[0] = replace(session.ops[0], autocast_enabled=True)
    assert (
        session.report(costs=costs)["analysis"]["ops"][0]["cost"]["compute_type"]
        is None
    )
    session.ops[0] = replace(
        record,
        inputs={
            **record.inputs,
            "B_scales": replace(record.inputs["B_scales"], shape=(1, 2)),
        },
    )
    invalid = session.report(costs=costs)["analysis"]["ops"][0]
    assert invalid["cost"] is None
    assert "scales disagree" in invalid["missing_reason"]


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("hbm_policy", ["unavailable", "logical_io_once"])
def test_rmsnorm_cost_counts_explicit_outputs(parallel, hbm_policy):
    def reference(input, weight, eps=1e-6, out=None):
        value = input.float() * torch.rsqrt(
            input.float().square().mean(-1, keepdim=True) + eps
        )
        value = (value * weight.float()).to(input.dtype)
        return value if out is None else out.copy_(value)

    def reference_parallel(
        input1, weight1, output1, input2, weight2, output2, eps=1e-6
    ):
        reference(input1, weight1, eps, out=output1)
        reference(input2, weight2, eps, out=output2)

    rule = "rmsnorm_fused_parallel.v1" if parallel else "rmsnorm.v1"
    fn = wrap_op_for_profiling(
        reference_parallel if parallel else reference,
        spec=KernelSpec(
            name="profile_norm_reference", family="layernorm", mode="rmsnorm"
        ),
        profile=OpProfile(cost_rule=rule),
    )
    x = (
        torch.arange(8 if parallel else 24, dtype=torch.bfloat16).reshape(
            (2, 4) if parallel else (2, 3, 4)
        )
        / 8
    )
    weight = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=x.dtype)
    out = torch.empty_like(x)
    x2, weight2 = torch.ones(2, 8, dtype=x.dtype), torch.full((8,), 0.5, dtype=x.dtype)
    session = ProfileSession(run_id="norm", model_id="test", rank=0, backend="none")
    with session.iteration(iter_id=0):
        result = (
            fn(x, weight, out, x2, weight2, x2) if parallel else fn(x, weight, out=out)
        )
    if parallel:
        assert result is None and not session.ops[0].outputs
        torch.testing.assert_close(x2, torch.full_like(x2, 0.5))
    else:
        assert result is out
    torch.testing.assert_close(
        out,
        torch.nn.functional.rms_norm(x.float(), (4,), weight.float(), eps=1e-6).to(
            x.dtype
        ),
    )
    costs = CostRegistry()
    register_compute_costs(costs, hbm_policy=hbm_policy)
    cost = session.report(costs=costs)["analysis"]["ops"][0]["cost"]
    # Both cases normalize 24 elements. Weight reads differ; output1/out is
    # write-only, while the aliased input2/output2 has one read and one write.
    reads = 72 if parallel else 56
    assert cost["useful_flops"] == 96
    assert cost["logical_read_bytes"] == reads
    assert cost["logical_write_bytes"] == 48
    assert cost["estimated_hbm_bytes"] == (
        reads + 48 if hbm_policy == "logical_io_once" else None
    )
    assert cost["compute_type"] is None
    record = session.ops[0]
    session.ops[0] = replace(
        record,
        inputs={key: replace(t, device="cuda:0") for key, t in record.inputs.items()},
        outputs={key: replace(t, device="cuda:0") for key, t in record.outputs.items()},
    )
    assert (
        session.report(costs=costs)["analysis"]["ops"][0]["cost"]["compute_type"]
        == "fp32_simt"
    )
    inputs = dict(record.inputs)
    name = "output2" if parallel else "weight"
    inputs[name] = replace(inputs[name], shape=(1,))
    session.ops[0] = replace(record, inputs=inputs)
    invalid = session.report(costs=costs)["analysis"]["ops"][0]
    assert invalid["cost"] is None and "RMSNorm" in invalid["missing_reason"]


@pytest.mark.parametrize("hbm_policy", ["unavailable", "logical_io_once"])
@pytest.mark.parametrize("fixed_format", [False, True])
def test_fp8_quantization_cost_counts_values_and_scales(hbm_policy, fixed_format):
    from tokenspeed_kernel.numerics.torch.quantization import (
        torch_quantize_fp8_with_scale,
    )

    registry = KernelRegistry()
    registry.register(
        KernelSpec(
            name="profile_quant_reference",
            family="quantization",
            mode="fp8_with_scale",
            traits={
                "granularity": frozenset(
                    {"token_group_128"}
                    if fixed_format
                    else {"token", "token_group_128"}
                ),
                "scale_encoding": frozenset(
                    {"float32"} if fixed_format else {"float32", "ue8m0"}
                ),
                "scale_layout": frozenset({"mn_major"}),
            },
        ),
        torch_quantize_fp8_with_scale,
        profile=OpProfile(cost_rule="fp8_token_group_128.v1"),
    )
    x = torch.full((3, 256), 0.5, dtype=torch.bfloat16)
    session = ProfileSession(run_id="quant", model_id="test", rank=0, backend="none")
    with session.iteration(iter_id=0):
        values, scales = registry.get_impl("profile_quant_reference")(
            x,
            granularity="token_group",
            group_size=128,
            scale_encoding="float32",
            enable_pdl=False,
        )
    torch.testing.assert_close(
        values.float() * scales.repeat_interleave(128, 1), x.float()
    )
    costs = CostRegistry()
    register_compute_costs(costs, hbm_policy=hbm_policy)
    cost = session.report(costs=costs)["analysis"]["ops"][0]["cost"]
    assert cost["useful_flops"] == 768
    assert cost["logical_read_bytes"] == 1536
    assert cost["logical_write_bytes"] == 792
    assert cost["estimated_hbm_bytes"] == (
        2328 if hbm_policy == "logical_io_once" else None
    )
    assert cost["compute_type"] is None
    record = session.ops[0]
    assert "group_size" not in record.scalar_args
    for scales in (
        replace(record.outputs["return[1]"], shape=(3, 4)),
        replace(record.outputs["return[1]"], dtype="torch.uint8"),
    ):
        session.ops[0] = replace(
            record, outputs={**record.outputs, "return[1]": scales}
        )
        invalid = session.report(costs=costs)["analysis"]["ops"][0]
        assert invalid["cost"] is None and "FP32 scales" in invalid["missing_reason"]
    session.ops[0] = replace(record, scalar_args={})
    without_args = session.report(costs=costs)["analysis"]["ops"][0]
    if fixed_format:
        assert record.scalar_args == {}
        assert without_args["cost"] == cost
        assert record.static_traits == {
            "granularity": "token_group_128",
            "scale_encoding": "float32",
            "scale_layout": "mn_major",
        }
    else:
        assert without_args["cost"] is None
        assert "128-wide groups" in without_args["missing_reason"]
    session.ops[0] = replace(
        record,
        scalar_args={
            "granularity": "token",
            "scale_encoding": "float32",
        },
    )
    invalid = session.report(costs=costs)["analysis"]["ops"][0]
    assert invalid["cost"] is None and "128-wide groups" in invalid["missing_reason"]


@pytest.mark.parametrize("hbm_policy", ["unavailable", "logical_io_once"])
def test_mla_nope_cost_describes_holders_and_counts_touched_cache_rows(hbm_policy):
    from tokenspeed_kernel.ops.attention.prologue.types import (
        LatentKVCache,
        MLAPrologueOutput,
    )

    def reference(*, query, q_pe, latent_cache, cache, rotary=None, expanded=None):
        query[..., -q_pe.shape[-1] :].copy_(q_pe)
        cache.kv_cache[cache.slots, 0] = latent_cache
        return MLAPrologueOutput(query=query, key=None, value=None)

    registry = KernelRegistry()
    registry.register(
        KernelSpec(
            name="profile_nope_reference", family="attention", mode="mla_prologue"
        ),
        reference,
        profile=OpProfile(cost_rule="mla_nope_prologue.v1"),
    )
    query = torch.full((2, 3, 8), 42, dtype=torch.bfloat16)
    q_pe = torch.full((2, 3, 2), 0.5, dtype=query.dtype)
    latent = torch.arange(16, dtype=query.dtype).reshape(2, 8)
    cache = LatentKVCache(
        kv_cache=torch.zeros(128, 1, 8, dtype=query.dtype),
        sanitize=False,
        slots=torch.tensor([3, 17]),
        write_mask=None,
    )
    session = ProfileSession(run_id="cache", model_id="test", rank=0, backend="none")
    with session.iteration(iter_id=0):
        output = registry.get_impl("profile_nope_reference")(
            query=query, q_pe=q_pe, latent_cache=latent, cache=cache
        )
    assert output.query is query
    torch.testing.assert_close(query[..., :6], torch.full_like(query[..., :6], 42))
    torch.testing.assert_close(query[..., 6:], q_pe)
    torch.testing.assert_close(cache.kv_cache[cache.slots, 0], latent)
    costs = CostRegistry()
    register_compute_costs(costs, hbm_policy=hbm_policy)
    cost = session.report(costs=costs)["analysis"]["ops"][0]["cost"]
    assert cost["useful_flops"] == 0 and cost["compute_type"] is None
    # Twelve query-tail elements + sixteen latent elements, plus two int64 slots.
    # The other 126 cache rows and 36 query-prefix elements are untouched.
    assert cost["logical_read_bytes"] == 72
    assert cost["logical_write_bytes"] == 56
    assert cost["estimated_hbm_bytes"] == (
        128 if hbm_policy == "logical_io_once" else None
    )
    record = session.ops[0]
    assert record.inputs["cache.kv_cache"].shape == (128, 1, 8)
    assert record.outputs["return.query"].shape == (2, 3, 8)
    # Unsupported mask/rotary variants must not silently reuse the NoPE rule.
    for field, message in (
        ("cache.write_mask", "masked write count"),
        ("rotary.cos_sin_cache", "excludes RoPE"),
    ):
        session.ops[0] = replace(
            record, inputs={**record.inputs, field: record.inputs["cache.slots"]}
        )
        invalid = session.report(costs=costs)["analysis"]["ops"][0]
        assert invalid["cost"] is None and message in invalid["missing_reason"]


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("hbm_policy", ["unavailable", "logical_io_once"])
def test_mla_cache_copy_cost_counts_only_accessed_rows(paged, hbm_policy):
    def scatter(kv_buffer, loc, cache_k_nope, cache_k_rope, *, write_mask):
        values = torch.cat((cache_k_nope, cache_k_rope), dim=-1)
        if write_mask is not None:
            loc, values = loc[write_mask], values[write_mask]
        indices = (loc // 16, loc % 16) if paged else loc
        kv_buffer[indices] = values.to(kv_buffer.dtype)

    def gather(kv_buffer, loc, cache_k_nope, cache_k_rope):
        indices = (loc // 16, loc % 16) if paged else loc
        values = kv_buffer[indices]
        cache_k_nope.copy_(values[..., :6])
        cache_k_rope.copy_(values[..., 6:])

    scatter, gather = (
        wrap_op_for_profiling(
            fn,
            spec=KernelSpec(
                name=f"reference_mla_{mode}", family="kvcache", mode=f"mla_{mode}"
            ),
            profile=OpProfile(cost_rule=f"mla_cache_{mode}.v1"),
        )
        for fn, mode in ((scatter, "scatter"), (gather, "gather"))
    )
    # Paged storage deliberately has a gap between pages. Neither those gaps
    # nor the 61 untouched cache rows belong in the transfer byte count.
    storage = torch.full(
        (4, 19, 1, 8) if paged else (64, 1, 8), -3, dtype=torch.bfloat16
    )
    cache = storage[:, :16] if paged else storage
    source = torch.arange(24, dtype=torch.float32).reshape(3, 1, 8) / 4
    output = torch.empty_like(source)
    slots = torch.tensor([1, 31, 52])
    indices = (slots // 16, slots % 16) if paged else slots
    expected = storage.clone()
    expected[indices] = source.to(storage.dtype)
    session = ProfileSession(
        run_id="cache-copy", model_id="test", rank=0, backend="none"
    )
    with session.iteration(iter_id=0):
        assert (
            scatter(cache, slots, source[..., :6], source[..., 6:], write_mask=None)
            is None
        )
        assert gather(cache, slots, output[..., :6], output[..., 6:]) is None
    torch.testing.assert_close(output, source, rtol=0, atol=0)
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)
    assert all(not record.outputs for record in session.ops)
    costs = CostRegistry()
    register_compute_costs(costs, hbm_policy=hbm_policy)
    analysis = session.report(costs=costs)["analysis"]["ops"]
    for entry, reads, writes in zip(analysis, (120, 72), (48, 96)):
        cost = entry["cost"]
        assert cost is not None
        assert cost["useful_flops"] == 0 and cost["compute_type"] is None
        assert cost["logical_read_bytes"] == reads
        assert cost["logical_write_bytes"] == writes
        assert cost["estimated_hbm_bytes"] == (
            168 if hbm_policy == "logical_io_once" else None
        )

    # A real partial write still executes, but its cost must not charge all
    # rows without an explicit snapshot of the mask's values.
    mask = torch.tensor([True, False, True])
    with session.iteration(iter_id=1):
        scatter(
            cache, slots, source[..., :6] + 100, source[..., 6:] + 100, write_mask=mask
        )
    torch.testing.assert_close(
        cache[indices][1], source[1].to(cache.dtype), rtol=0, atol=0
    )
    torch.testing.assert_close(
        cache[indices][mask], (source[mask] + 100).to(cache.dtype), rtol=0, atol=0
    )
    masked = session.report(costs=costs)["analysis"]["ops"][-1]
    assert masked["cost"] is None and "masked write count" in masked["missing_reason"]


@pytest.mark.parametrize(
    "query_length,causal", [(1, True), (3, True), (8, True), (3, False)]
)
def test_paged_mla_uses_visible_length_snapshots(query_length, causal):
    def reference(
        q, k_cache, block_table, cache_seqlens, head_dim_v, causal, is_fp8_kvcache=False
    ):
        outputs, normalizers = [], []
        for i, length in enumerate(cache_seqlens.tolist()):
            kv = k_cache[block_table[i]].flatten(0, 2)[:length].float()
            scores = torch.einsum("qhd,kd->hqk", q[i].float(), kv)
            if causal:
                visible = (
                    torch.arange(length)[None, :]
                    <= length - q.shape[1] + torch.arange(q.shape[1])[:, None]
                )
                scores.masked_fill_(~visible, -torch.inf)
            probabilities = scores.softmax(-1).nan_to_num()
            outputs.append(
                torch.einsum("hqk,kv->qhv", probabilities, kv[:, :head_dim_v])
            )
            normalizers.append(scores.logsumexp(-1))
        return torch.stack(outputs).to(q.dtype), torch.stack(normalizers)

    registry = KernelRegistry()
    registry.register(
        KernelSpec(
            name="profile_paged_mla_reference", family="attention", mode="mla_decode"
        ),
        reference,
        profile=OpProfile(
            cost_rule="flashmla_decode.v1",
            scalar_args=("head_dim_v", "causal", "is_fp8_kvcache"),
            integer_tensor_args=("cache_seqlens",),
        ),
    )
    q = torch.zeros(2, query_length, 2, 4, dtype=torch.bfloat16)
    cache = torch.ones(32, 4, 1, 4, dtype=q.dtype)
    table = torch.tensor([[3, 8], [5, 11]], dtype=torch.int32)
    lengths = torch.tensor([2, 5], dtype=torch.int32)
    session = ProfileSession(run_id="paged", model_id="test", rank=0, backend="none")
    with session.iteration(iter_id=0):
        output, lse = registry.get_impl("profile_paged_mla_reference")(
            q, cache, table, lengths, head_dim_v=2, causal=causal
        )
        lengths.fill_(99)  # A later step reuses the same metadata buffer.
    assert session.ops[0].tensor_values["cache_seqlens"] == (2, 5)
    pair_count = 0
    for b, length in enumerate((2, 5)):
        for i in range(query_length):
            visible = sum(
                not causal or j <= length - query_length + i for j in range(length)
            )
            pair_count += visible
            torch.testing.assert_close(
                output[b, i], torch.full_like(output[b, i], 1 if visible else 0)
            )
            expected_lse = torch.tensor(float(visible)).log().expand(2)
            torch.testing.assert_close(lse[b, :, i], expected_lse)
    costs = CostRegistry()
    register_compute_costs(costs, hbm_policy="logical_io_once")
    cost = session.report(costs=costs)["analysis"]["ops"][0]["cost"]
    assert cost["useful_flops"] == 2 * 2 * pair_count * (4 + 2)
    # Seven logical latent rows, not all 128 allocated rows; V is a key prefix.
    assert cost["logical_read_bytes"] == q.numel() * 2 + 7 * 4 * 2 + 3 * 4 + 2 * 4
    assert cost["logical_write_bytes"] == output.numel() * 2 + lse.numel() * 4
    assert (
        cost["estimated_hbm_bytes"]
        == cost["logical_read_bytes"] + cost["logical_write_bytes"]
    )
    assert cost["compute_type"] is None
    record = session.ops[0]
    for values, message in (
        (None, "snapshot unavailable"),
        ((2, 9), "visible lengths disagree"),
    ):
        session.ops[0] = replace(record, tensor_values={"cache_seqlens": values})
        invalid = session.report(costs=costs)["analysis"]["ops"][0]
        assert invalid["cost"] is None and message in invalid["missing_reason"]


@pytest.mark.parametrize("causal", [False, True])
def test_varlen_attention_cost_uses_actual_causal_pairs(causal):
    def reference(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        *,
        causal,
        window_size=(-1, -1),
        attention_chunk=0,
    ):
        outputs, normalizers = [], []
        for qs, qe, ks, ke in zip(
            cu_seqlens_q, cu_seqlens_q[1:], cu_seqlens_k, cu_seqlens_k[1:]
        ):
            key = k[ks:ke].repeat_interleave(2, dim=1).float()
            value = v[ks:ke].repeat_interleave(2, dim=1).float()
            scores = torch.einsum("qhd,khd->hqk", q[qs:qe].float(), key)
            if causal:
                mask = (
                    torch.arange(ke - ks)[None, :]
                    > ke - ks - (qe - qs) + torch.arange(qe - qs)[:, None]
                )
                scores.masked_fill_(mask[None], -torch.inf)
            outputs.append(
                torch.einsum("hqk,khd->qhd", scores.softmax(-1).nan_to_num(), value)
            )
            normalizers.append(scores.logsumexp(-1))
        return torch.cat(outputs).to(q.dtype), torch.cat(normalizers, dim=1)

    registry = KernelRegistry()
    registry.register(
        KernelSpec(name="varlen_reference", family="attention", mode="mha_varlen"),
        reference,
        profile=OpProfile(
            cost_rule="attention_varlen.v1",
            scalar_args=("causal", "window_size", "attention_chunk"),
            integer_tensor_args=("cu_seqlens_q", "cu_seqlens_k"),
        ),
    )
    q = torch.zeros(4, 4, 4, dtype=torch.bfloat16)
    k = torch.ones(7, 2, 4, dtype=q.dtype)
    v = torch.ones(7, 2, 3, dtype=q.dtype)
    cu_q = torch.tensor([0, 3, 4], dtype=torch.int32)
    cu_k = torch.tensor([0, 2, 7], dtype=torch.int32)
    session = ProfileSession(run_id="varlen", model_id="test", rank=0, backend="none")
    with session.iteration(iter_id=0):
        out, lse = registry.get_impl("varlen_reference")(
            q, k, v, cu_q, cu_k, causal=causal
        )
        cu_q.fill_(99)
        cu_k.fill_(99)
    pairs = 0
    row = 0
    for qlen, klen in ((3, 2), (1, 5)):
        for i in range(qlen):
            visible = sum(not causal or j <= klen - qlen + i for j in range(klen))
            pairs += visible
            torch.testing.assert_close(
                out[row], torch.full_like(out[row], 1 if visible else 0)
            )
            torch.testing.assert_close(
                lse[:, row], torch.tensor(float(visible)).log().expand(4)
            )
            row += 1
    costs = CostRegistry()
    register_compute_costs(costs, hbm_policy="logical_io_once")
    cost = session.report(costs=costs)["analysis"]["ops"][0]["cost"]
    assert cost["useful_flops"] == 2 * 4 * pairs * (4 + 3)
    assert cost["logical_read_bytes"] == (q.numel() + k.numel() + v.numel()) * 2 + 24
    assert cost["logical_write_bytes"] == out.numel() * 2 + lse.numel() * 4
    assert (
        cost["estimated_hbm_bytes"]
        == cost["logical_read_bytes"] + cost["logical_write_bytes"]
    )
    record = session.ops[0]
    used = torch.tensor([1, 3], dtype=torch.int32)
    used_desc = replace(record.inputs["cu_seqlens_k"], shape=(2,), logical_bytes=8)
    session.ops[0] = replace(
        record,
        inputs={**record.inputs, "seqused_k": used_desc},
        tensor_values={**record.tensor_values, "seqused_k": tuple(used.tolist())},
    )
    used_cost = session.report(costs=costs)["analysis"]["ops"][0]["cost"]
    assert used_cost["useful_flops"] == 2 * 4 * (4 if causal else 6) * (4 + 3)
    assert used_cost["logical_read_bytes"] == cost["logical_read_bytes"] + 8
    with_lengths = session.ops[0]
    for values in (None, (3, 3)):
        session.ops[0] = replace(
            with_lengths,
            tensor_values={**record.tensor_values, "seqused_k": values},
        )
        invalid = session.report(costs=costs)["analysis"]["ops"][0]
        assert (
            invalid["cost"] is None
            and "seqused_k live lengths" in invalid["missing_reason"]
        )
    for lengths, reason in ((None, "snapshot unavailable"), ((0, 5, 4), "partition")):
        session.ops[0] = replace(
            record, tensor_values={**record.tensor_values, "cu_seqlens_q": lengths}
        )
        invalid = session.report(costs=costs)["analysis"]["ops"][0]
        assert invalid["cost"] is None and reason in invalid["missing_reason"]
    session.ops[0] = replace(
        record, scalar_args={**record.scalar_args, "window_size": (1, 0)}
    )
    assert (
        "native attention only"
        in session.report(costs=costs)["analysis"]["ops"][0]["missing_reason"]
    )


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("hbm_policy", ["unavailable", "logical_io_once"])
def test_attention_merge_cost_counts_boundary_io(inplace, hbm_policy):
    def merge(out_a, lse_a, out_b, lse_b, *, inplace):
        lse = torch.logaddexp(lse_a, lse_b)
        output = (
            out_a.float() * (lse_a - lse).exp()[..., None]
            + out_b.float() * (lse_b - lse).exp()[..., None]
        ).to(out_a.dtype)
        if inplace:
            return out_a.copy_(output), lse_a.copy_(lse)
        return output, lse

    registry = KernelRegistry()
    registry.register(
        KernelSpec(name="merge_reference", family="attention", mode="attn_merge_state"),
        merge,
        profile=OpProfile(cost_rule="attention_merge_state.v1"),
    )
    out_a = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
    out_b = out_a.flip(-1)
    lse_a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    lse_b = -lse_a
    logits = torch.stack((lse_a, lse_b))
    weights = logits.softmax(0)
    expected = (
        out_a.float() * weights[0, ..., None] + out_b.float() * weights[1, ..., None]
    ).to(out_a.dtype)
    expected_lse = logits.logsumexp(0)
    session = ProfileSession(run_id="merge", model_id="test", rank=0, backend="none")
    with session.iteration(iter_id=0):
        output, lse = registry.get_impl("merge_reference")(
            out_a, lse_a, out_b, lse_b, inplace=inplace
        )
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(lse, expected_lse)
    assert (output is out_a) is inplace and (lse is lse_a) is inplace
    costs = CostRegistry()
    register_compute_costs(costs, hbm_policy=hbm_policy)
    cost = session.report(costs=costs)["analysis"]["ops"][0]["cost"]
    assert cost is not None
    assert cost["useful_flops"] == 72
    assert cost["executed_flops"] is None
    assert cost["logical_read_bytes"] == 144  # Both BF16 values and FP32 LSEs.
    assert cost["logical_write_bytes"] == 72
    assert cost["estimated_hbm_bytes"] == (
        216 if hbm_policy == "logical_io_once" else None
    )
    assert cost["compute_type"] is None
    record = session.ops[0]
    session.ops[0] = replace(
        record,
        inputs={name: replace(t, device="cuda:0") for name, t in record.inputs.items()},
        outputs={
            name: replace(t, device="cuda:0") for name, t in record.outputs.items()
        },
    )
    assert (
        session.report(costs=costs)["analysis"]["ops"][0]["cost"]["compute_type"]
        == "fp32_simt"
    )
    session.ops[0] = replace(
        record,
        inputs={
            **record.inputs,
            "lse_b": replace(record.inputs["lse_b"], dtype="torch.bfloat16"),
        },
    )
    invalid = session.report(costs=costs)["analysis"]["ops"][0]
    assert invalid["cost"] is None and "FP32" in invalid["missing_reason"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hbm_policy", ["unavailable", "logical_io_once"])
def test_registered_causal_conv_cost_counts_packed_channels_and_used_state(
    hbm_policy, weight_dtype
):
    from tokenspeed_kernel.ops.attention.gdn._triton.causal_conv1d_metadata import (
        build_causal_conv1d_prefill_metadata,
    )
    from tokenspeed_kernel.ops.conv.triton import triton_causal_conv1d_prefill

    channels, tokens, width = 96, 12, 4
    x = torch.full((tokens, channels), 2.0, device="cuda", dtype=torch.bfloat16).T
    weight = torch.ones((channels, width), device="cuda", dtype=weight_dtype)
    state = torch.ones((8, channels, width - 1), device="cuda", dtype=torch.bfloat16)
    boundaries = torch.tensor([0, 5, 12], device="cuda", dtype=torch.int32)
    metadata = build_causal_conv1d_prefill_metadata(boundaries, torch.tensor([5, 7]), 8)
    session = ProfileSession(run_id="conv", model_id="test", rank=0, backend="none")
    with session.iteration(iter_id=0):
        output = triton_causal_conv1d_prefill(
            x,
            weight,
            None,
            state,
            boundaries,
            cache_indices=torch.tensor([2, 5], device="cuda", dtype=torch.int32),
            has_initial_state=torch.ones(2, device="cuda", dtype=torch.bool),
            activation=None,
            prefill_metadata=metadata,
        )
    expected = torch.tensor([5, 6, 7, 8, 8, 5, 6, 7, 8, 8, 8, 8], dtype=output.dtype)
    torch.testing.assert_close(output.cpu(), expected.expand(channels, -1))
    torch.testing.assert_close(
        state[[2, 5]].cpu(), torch.full((2, channels, 3), 2.0, dtype=state.dtype)
    )
    costs = CostRegistry()
    register_compute_costs(costs, hbm_policy=hbm_policy)
    report = session.report(costs=costs)
    cost = report["analysis"]["ops"][0]["cost"]
    assert cost["useful_flops"] == 2 * tokens * channels * width
    # Only two state rows are accessed, although eight rows are allocated.
    state_bytes = 2 * channels * (width - 1) * 2
    reads = x.numel() * 2 + weight.numel() * weight.element_size() + state_bytes
    writes = output.numel() * 2 + state_bytes
    assert cost["logical_read_bytes"] == reads
    assert cost["logical_write_bytes"] == writes
    assert cost["estimated_hbm_bytes"] == (
        reads + writes if hbm_policy == "logical_io_once" else None
    )
    assert cost["compute_type"] == "fp32_simt"
    timing = ScopeTiming(
        trace_label=session.ops[0].trace_label,
        start_timestamp=0,
        end_timestamp=1000,
        gpu_duration_ns=1000,
    )
    hardware = HardwareProfile(
        "test",
        {"fp32_simt": 1e12, "bf16_tensor": 10e12},
        1e12,
        "synthetic peaks for normalization validation",
    )
    row = analyze_utilization(report, [timing], hardware)[0]
    assert row["mfu"] == pytest.approx(cost["useful_flops"] / 1e6)
    assert "MFU=N/A" not in row["nvtx_label"]
    if hbm_policy == "logical_io_once":
        assert row["estimated_mbu"] == pytest.approx((reads + writes) / 1e6)
    missing = analyze_utilization(
        report,
        [timing],
        replace(hardware, compute_flops_per_second={"bf16_tensor": 10e12}),
    )[0]
    assert missing["mfu"] is None
    assert missing["mfu_missing_reason"] == "peak unavailable for fp32_simt"


def test_gated_delta_cost_counts_recurrence_and_returned_state():
    def scan(q, k, v, g, initial_state):
        state = initial_state[0].clone()
        outputs = []
        for qt, kt, vt, gt in zip(q, k, v, g, strict=True):
            state = state * gt[:, None, :]
            correction = vt - torch.einsum("hvk,hk->hv", state, kt)
            state = state + correction[:, :, None] * kt[:, None, :]
            outputs.append(torch.einsum("hvk,hk->hv", state, qt))
        return torch.stack(outputs), state.unsqueeze(0)

    registry = KernelRegistry()
    registry.register(
        KernelSpec(name="profile_delta", family="attention", mode="delta"),
        scan,
        profile=OpProfile(cost_rule="gated_delta_prefill.v1"),
    )
    q = torch.full((4, 2, 3), 0.1)
    k, g = torch.full_like(q, 0.2), torch.full_like(q, 0.9)
    v = torch.ones(4, 2, 5)
    initial_state = torch.zeros(1, 2, 5, 3)
    session = ProfileSession(run_id="delta", model_id="test", rank=0, backend="none")
    with session.iteration(iter_id=0):
        output, final_state = registry.get_impl("profile_delta")(
            q, k, v, g, initial_state
        )
    first, state = scan(q[:2], k[:2], v[:2], g[:2], initial_state)
    second, split_final = scan(q[2:], k[2:], v[2:], g[2:], state)
    torch.testing.assert_close(output, torch.cat((first, second)))
    torch.testing.assert_close(final_state, split_final)
    costs = CostRegistry()
    register_compute_costs(costs, hbm_policy="logical_io_once")
    cost = session.report(costs=costs)["analysis"]["ops"][0]["cost"]
    assert cost["useful_flops"] == 880
    assert cost["logical_read_bytes"] == 568
    assert cost["logical_write_bytes"] == 280
    assert cost["estimated_hbm_bytes"] == 848
    assert cost["executed_flops"] is None and cost["compute_type"] is None


@pytest.fixture
def fixture(fresh_registry):
    @register_kernel(
        "gemm",
        "mm",
        name="utilization_mm",
        solution="reference",
        signatures=frozenset(),
        profile=OpProfile(cost_rule="mm"),
    )
    def mm(a, b):
        return a @ b

    class Layer:
        @profiled(level="layer")
        def forward(self):
            return mm(torch.ones(2, 3), torch.ones(3, 4))

    layer = Layer()
    session = ProfileSession(run_id="test", model_id="test", rank=0, backend="none")
    session.bind_layer(layer, layer_id=7)
    with session.iteration(iter_id=0):
        layer.forward()
        layer.forward()
    costs = CostRegistry()
    dense = DenseMatmulCost(lhs="a", rhs="b", output="return")
    costs.register(
        "mm",
        lambda op: replace(
            dense(op), compute_type="synthetic", estimated_hbm_bytes=104
        ),
    )
    timings = [
        ScopeTiming(
            r.trace_label, r.cpu_start_ns, r.cpu_start_ns + r.cpu_duration_ns, duration
        )
        for records, duration in (
            (session.ops, 100_000_000),
            (session.layers, 200_000_000),
        )
        for r in records
    ]
    hardware = HardwareProfile("test", {"synthetic": 1000}, 1000, "analytical fixture")
    return session, costs, timings, hardware, layer


def test_metrics_match_analytical_reference_and_repeated_calls(fixture):
    session, costs, timings, hardware, _ = fixture
    rows = analyze_utilization(session.report(costs=costs), timings, hardware)
    assert len(rows) == 4
    ops = [r for r in rows if r["kind"] == "op"]
    layers = [r for r in rows if r["kind"] == "layer"]
    for row in ops:
        assert row["useful_flops_per_second"] == pytest.approx(480)
        assert row["estimated_hbm_bytes_per_second"] == pytest.approx(1040)
        assert row["mfu"] == pytest.approx(0.48)
        assert row["estimated_mbu"] == pytest.approx(1.04)  # Never clamp to 100%.
        assert "MFU=48.00% Est.MBU(HBM)=104.00%" in row["nvtx_label"]
    for row in layers:
        assert row["op_count"] == 1
        assert row["useful_flops_per_second"] == pytest.approx(240)
        assert row["estimated_hbm_bytes_per_second"] == pytest.approx(520)
        assert row["mfu"] == pytest.approx(0.24)
        assert row["estimated_mbu"] == pytest.approx(0.52)
    assert [r["occurrence"] for r in layers] == [0, 1]


def test_nested_layer_counts_each_descendant_op_once(fixture):
    session, costs, _, hardware, child = fixture

    class Parent:
        @profiled(level="layer")
        def forward(self):
            child.forward()
            return child.forward()

    parent = Parent()
    session.bind_layer(parent, layer_id=0)
    with session.iteration(iter_id=1):
        parent.forward()
    timings = [
        ScopeTiming(
            r.trace_label,
            r.cpu_start_ns,
            r.cpu_start_ns + r.cpu_duration_ns,
            400_000_000,
        )
        for r in [*session.ops, *session.layers]
    ]
    rows = analyze_utilization(session.report(costs=costs), timings, hardware)
    row = next(r for r in rows if r["kind"] == "layer" and r["layer_id"] == 0)
    assert row["op_count"] == row["covered_op_count"] == 2
    assert row["useful_flops"] == 96
    assert row["mfu"] == pytest.approx(0.24)


def test_missing_cost_is_unavailable_and_has_reason(fixture):
    session, _, timings, hardware, _ = fixture
    rows = analyze_utilization(session.report(costs=CostRegistry()), timings, hardware)
    for row in rows:
        assert row["mfu"] is None and row["estimated_mbu"] is None
        assert row["useful_flops_per_second"] is None
        assert row["estimated_hbm_bytes_per_second"] is None
        assert row["covered_op_count"] == 0
        assert "cost rule unavailable" in row["mfu_missing_reason"]
        if row["kind"] == "op":
            assert "MFU=N/A Est.MBU(HBM)=N/A" in row["nvtx_label"]
        else:
            assert row["nvtx_label"] == f"L{row['layer_id']}"


def test_missing_peaks_and_hbm_estimates_are_independent(fixture):
    session, costs, timings, hardware, _ = fixture
    report = session.report(costs=costs)
    rows = analyze_utilization(
        report, timings, replace(hardware, compute_flops_per_second={})
    )
    assert rows[0]["mfu"] is None and rows[0]["estimated_mbu"] is not None
    rows = analyze_utilization(
        report,
        timings,
        replace(hardware, compute_flops_per_second={}, hbm_bytes_per_second=None),
    )
    assert rows[0]["mfu"] is None and rows[0]["estimated_mbu"] is None
    assert rows[0]["useful_flops_per_second"] == pytest.approx(480)
    assert rows[0]["estimated_hbm_bytes_per_second"] == pytest.approx(1040)
    report["analysis"]["ops"][0]["cost"]["estimated_hbm_bytes"] = None
    rows = analyze_utilization(report, timings, hardware)
    assert rows[0]["mfu"] is not None and rows[0]["estimated_mbu"] is None
    assert rows[0]["useful_flops_per_second"] == pytest.approx(480)
    assert rows[0]["estimated_hbm_bytes_per_second"] is None
    assert rows[0]["mbu_missing_reason"] == "HBM traffic estimate unavailable"


def test_unqualified_device_time_invalidates_parent_utilization(fixture):
    session, costs, timings, hardware, _ = fixture
    timings[0] = replace(timings[0], unavailable_reason="different CUDA device")
    rows = analyze_utilization(session.report(costs=costs), timings, hardware)
    for row in (rows[0], rows[2]):
        assert row["mfu"] is None and row["estimated_mbu"] is None
        assert row["useful_flops_per_second"] is None
        assert row["estimated_hbm_bytes_per_second"] is None
        assert "different CUDA device" in row["mfu_missing_reason"]
    assert rows[1]["mfu"] is not None


@pytest.mark.parametrize("other_flops", [48, 0])
def test_layer_normalizes_only_nonzero_compute_types(fixture, other_flops):
    session, costs, timings, hardware, _ = fixture
    report = session.report(costs=costs)
    other_cost = report["analysis"]["ops"][1]["cost"]
    other_cost["useful_flops"] = other_flops
    other_cost["compute_type"] = "other" if other_flops else None
    # A parent layer encloses both ops; its own event window is the denominator.
    parent = report["layers"][0]
    label = parent["trace_label"]
    timings = [
        (
            replace(t, end_timestamp=max(x.end_timestamp for x in timings))
            if t.trace_label == label
            else t
        )
        for t in timings
    ]
    hardware = replace(
        hardware, compute_flops_per_second={"synthetic": 1000, "other": 2000}
    )
    rows = analyze_utilization(report, timings, hardware)
    row = next(r for r in rows if r["trace_label"] == label)
    assert row["estimated_mbu"] == pytest.approx(1.04)
    if other_flops:
        assert row["mfu"] == pytest.approx(0.36)
        assert row["mfu_by_compute_type"] == pytest.approx(
            {"synthetic": 0.24, "other": 0.12}
        )
        assert row["mfu_missing_reason"] is None
        assert row["mfu_total"] == pytest.approx(0.36)
        assert row["mfu_total_missing_reason"] is None
    else:
        assert row["mfu"] == pytest.approx(0.24)
        assert row["mfu_by_compute_type"] == pytest.approx({"synthetic": 0.24})
        assert row["mfu_missing_reason"] is None


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_device_time_never_uses_cpu_time(fixture, duration):
    session, costs, timings, hardware, _ = fixture
    timings[0] = replace(timings[0], gpu_duration_ns=duration)
    row = analyze_utilization(session.report(costs=costs), timings, hardware)[0]
    assert row["mfu"] is None and row["estimated_mbu"] is None
    assert row["useful_flops_per_second"] is None
    assert row["estimated_hbm_bytes_per_second"] is None
    assert "GPU duration unavailable" in row["mfu_missing_reason"]


def test_attribution_gaps_and_duplicate_timings_are_rejected(fixture):
    session, costs, timings, hardware, _ = fixture
    report = session.report(costs=costs)
    with pytest.raises(ValueError, match="exactly one timing"):
        analyze_utilization(report, timings[1:], hardware)
    with pytest.raises(ValueError, match="Duplicate"):
        analyze_utilization(report, timings + timings[:1], hardware)


@pytest.mark.parametrize("failed", [True, False])
def test_invalid_layer_does_not_report_throughput_or_utilization(fixture, failed):
    session, costs, timings, hardware, _ = fixture
    report = session.report(costs=costs)
    if failed:
        report["layers"][0]["error"] = "RuntimeError"
        reason = "layer failed: RuntimeError"
    else:
        reason = "unqualified layer timing"
        timings[2] = replace(timings[2], unavailable_reason=reason)
    rows = analyze_utilization(report, timings, hardware)
    row = next(r for r in rows if r["kind"] == "layer")
    assert row["mfu"] is None and row["estimated_mbu"] is None
    assert row["useful_flops_per_second"] is None
    assert row["estimated_hbm_bytes_per_second"] is None
    assert row["mfu_missing_reason"] == reason


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_hardware_file_formats_preserve_metrics(fixture, tmp_path, suffix):
    if suffix != ".json":
        pytest.importorskip("yaml", reason="YAML profiles require the profiling extra")
    session, costs, timings, hardware, _ = fixture
    path = tmp_path / f"hardware{suffix}"
    path.write_text(
        json.dumps(asdict(hardware))
        if suffix == ".json"
        else """# Peaks use FLOP/s and byte/s, not TFLOP/s or GB/s.
name: test
compute_flops_per_second:
  synthetic: 1.0e+3
hbm_bytes_per_second: 1000
source: analytical fixture
"""
    )
    loaded = HardwareProfile.from_file(path)
    assert loaded == hardware
    report = session.report(costs=costs)
    assert analyze_utilization(report, timings, loaded) == analyze_utilization(
        report, timings, hardware
    )
    if suffix == ".json":
        assert HardwareProfile.from_json(path) == loaded
    unknown = replace(hardware, compute_flops_per_second={}, hbm_bytes_per_second=None)
    path.write_text(json.dumps(asdict(unknown)))  # JSON is also valid YAML.
    loaded = HardwareProfile.from_file(path)
    assert loaded == unknown
    rows = analyze_utilization(report, timings, loaded)
    assert rows[0]["mfu"] is None and rows[0]["estimated_mbu"] is None
    assert rows[0]["useful_flops_per_second"] == pytest.approx(480)
    path.write_text("[]")
    with pytest.raises(ValueError, match="must be a mapping"):
        HardwareProfile.from_file(path)


@pytest.mark.parametrize("peak", [0, -1, float("nan"), float("inf"), True, "1e12"])
def test_invalid_hardware_peaks_rejected(peak):
    with pytest.raises(ValueError, match="finite and positive"):
        HardwareProfile("test", {"type": peak}, 1000, "test")
    with pytest.raises(ValueError, match="finite and positive"):
        HardwareProfile("test", {}, peak, "test")


@pytest.mark.parametrize(
    "kind,source,role,total,bus",
    [
        ("all_reduce_sum", "input", "a", 24, 24),
        ("reduce_scatter_sum", "input", "a", 24, 12),
        ("all_gather", "output", "return", 32, 16),
    ],
)
@pytest.mark.parametrize("fused_compute", [False, True])
def test_collective_bytes_and_busbw(
    fixture, kind, source, role, total, bus, fused_compute
):
    session, _, timings, hardware, _ = fixture
    # Reuse real tensor metadata to check the offline full-array role and units.
    rule = CollectiveCost(kind, role, source, 2, "tp")
    cost = rule(session.ops[0])
    assert cost.communication.total_bytes == total  # dtype size is already included.
    assert cost.communication.bus_bytes == bus
    expected = 448 if kind == "all_reduce_sum" else 224
    assert CollectiveTraffic(kind, 256, 8, "tp").bus_bytes == expected
    costs = CostRegistry()
    # Fused ops carry both communication traffic and useful compute FLOPs.
    costs.register(
        "mm",
        lambda op: replace(
            rule(op),
            useful_flops=48 if fused_compute else 0,
            compute_type="synthetic" if fused_compute else None,
        ),
    )
    row = analyze_utilization(session.report(costs=costs), timings, hardware)[0]
    assert row["busbw_bytes_per_second"] == pytest.approx(bus / 0.1)
    assert row["algbw_bytes_per_second"] == pytest.approx(total / 0.1)
    assert row["estimated_mbu"] is None
    assert "Est.MBU(HBM)=N/A" in row["nvtx_label"]
    if fused_compute:
        assert row["mfu"] == pytest.approx(48 / (1000 * 0.1))
        assert "MFU=48.00%" in row["nvtx_label"]
    else:
        assert row["mfu"] == row["mfu_total"] == 0
        assert row["mfu_missing_reason"] is None
        assert "MFU=" not in row["nvtx_label"]
    assert "commAlgBW=" in row["nvtx_label"]
    assert "commBusBW=" in row["nvtx_label"]
    heading = row["nvtx_label"].split(" | ")
    assert heading[:2] == [session.ops[0].op_name, "Est.CommBWUtil=N/A"]
    assert heading[2].startswith("GPU=") and heading[2].endswith("us")
    assert float(heading[2][4:-2]) == 100000
    assert (
        row["nvtx_label"].index("Est.CommBWUtil=")
        < row["nvtx_label"].index("Est.MBU(HBM)=")
        < row["nvtx_label"].index("commAlgBW=")
    )
    assert "comm util N/A:" in row["nvtx_label"]


def test_layer_busbw_window_and_unavailable_results(fixture):
    session, costs, timings, hardware, _ = fixture
    report = session.report(costs=costs)
    report["analysis"]["ops"][0]["cost"] = asdict(
        CollectiveCost("all_reduce_sum", "a", "input", 2, "tp")(session.ops[0])
    )
    layer_label = session.layers[0].trace_label
    timings = [
        (
            replace(t, end_timestamp=max(x.end_timestamp for x in timings))
            if t.trace_label == layer_label
            else t
        )
        for t in timings
    ]
    rows = analyze_utilization(report, timings, hardware)
    assert rows[0]["busbw_bytes_per_second"] == pytest.approx(24 / 0.1)
    assert rows[2]["busbw_bytes_per_second"] == pytest.approx(24 / 0.2)
    assert rows[2]["mfu"] == pytest.approx(48 / (1000 * 0.2))
    invalid = [replace(timings[0], gpu_duration_ns=0), *timings[1:]]
    assert (
        analyze_utilization(report, invalid, hardware)[0]["busbw_bytes_per_second"]
        is None
    )
    report["layers"][0]["error"] = "failed"
    assert (
        analyze_utilization(report, timings, hardware)[2]["busbw_bytes_per_second"]
        is None
    )
    report["layers"][0]["error"] = None
    saved = report["analysis"]["ops"][1]["cost"]
    report["analysis"]["ops"][1].update(cost=None, missing_reason="unknown")
    row = analyze_utilization(report, timings, hardware)[2]
    assert row["busbw_bytes_per_second"] is None
    assert "incomplete cost coverage" in row["busbw_missing_reason"]
    report["analysis"]["ops"][1]["cost"] = saved
    saved["communication"] = asdict(CollectiveTraffic("all_reduce_sum", 32, 2, "dp"))
    row = analyze_utilization(report, timings, hardware)[2]
    assert row["busbw_bytes_per_second"] is None
    assert "mixed communication groups" in row["busbw_missing_reason"]
    assert CollectiveTraffic("all_reduce_sum", 64, 1, "one").bus_bytes == 0
    with pytest.raises(ValueError, match="positive integer"):
        CollectiveTraffic("all_reduce_sum", 64, 0, "empty")


def test_attention_cost_uses_query_heads_and_causal_pairs():
    def attention(q, k, v, *, attn_mask):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, enable_gqa=True
        )

    registry = KernelRegistry()
    registry.register(
        KernelSpec(name="test", family="attention", mode="gqa"),
        attention,
        profile=OpProfile(cost_rule="attention"),
    )
    wrapped = registry.get_impl("test")
    session = ProfileSession(
        run_id="attention", model_id="test", rank=0, backend="none"
    )
    cases = (
        (4, 4, "full", 16),
        (4, 4, "causal_lower_right", 10),
        (1, 8, "causal_lower_right", 8),
        (2, 5, "causal_lower_right", 9),
        (5, 2, "causal_lower_right", 3),
        (2, 5, "causal_upper_left", 3),
        (5, 2, "causal_upper_left", 9),
    )
    for iter_id, (query_length, kv_length, mask, pairs) in enumerate(cases):
        attn_mask = None
        if mask != "full":
            offset = kv_length - query_length if mask == "causal_lower_right" else 0
            attn_mask = (
                torch.arange(kv_length)[None, :]
                <= torch.arange(query_length)[:, None] + offset
            )
            assert attn_mask.sum().item() == pairs
        with session.iteration(iter_id=iter_id):
            wrapped(
                torch.randn(2, 4, query_length, 3),
                torch.randn(2, 2, kv_length, 3),
                torch.randn(2, 2, kv_length, 5),
                attn_mask=attn_mask,
            )
        rule = AttentionMatmulCost(q="q", k="k", v="v", output="return", mask=mask)
        cost = rule(session.ops[-1])
        assert cost.useful_flops == 2 * 2 * 4 * pairs * (3 + 5)
        assert cost.executed_flops is None and cost.estimated_hbm_bytes is None


@pytest.mark.parametrize(
    "missing_gpu,wrong_capture,python_payload,signed_failure",
    [
        (False, False, False, None),
        (False, False, True, None),
        (True, False, True, None),
        (False, True, True, None),
        (False, False, True, "foreign_only"),
        (False, False, True, "wrong_event"),
        (False, False, True, "wrong_prefix"),
        (False, False, True, "cross_thread"),
        (False, False, True, "open_range"),
        (False, False, True, "no_domain_schema"),
    ],
)
def test_nsight_timestamps_are_the_only_gpu_denominator(
    fixture, tmp_path, missing_gpu, wrong_capture, python_payload, signed_failure
):
    import csv
    import sqlite3

    from tokenspeed_kernel.profile_nsys import analyze_nsight

    session, costs, timings, hardware, _ = fixture
    report = session.report(costs=costs)
    report["hardware"] = asdict(hardware)
    report["device_analysis"] = {"process_id": 17, "thread_id": 23, "scopes": []}
    payload_column = "int64Value" if python_payload else "uint64Value"
    id_base = 1 << (53 if python_payload else 63)
    if python_payload:
        report["device_analysis"]["nvtx_payload_type"] = "int64"
    database, projection = tmp_path / "trace.sqlite", tmp_path / "projection.csv"
    rows = []
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE StringIds(id INTEGER,value TEXT)")
        db.execute(
            "CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,"
            f"endGlobalTid INTEGER,text TEXT,textId INTEGER,{payload_column} INTEGER)"
        )
        for index, timing in enumerate(timings, 1):
            label = "gemm.mm" if index <= 2 else "L7"
            identity = id_base + index
            stored = identity if python_payload else identity - (1 << 64)
            if python_payload:
                db.execute("INSERT INTO StringIds VALUES(?,?)", (index, label))
            report["device_analysis"]["scopes"].append(
                {
                    "trace_label": timing.trace_label,
                    "nvtx_label": label,
                    "nvtx_id": identity,
                    "gpu_duration_ns": 1,
                }
            )
            db.execute(
                "INSERT INTO NVTX_EVENTS VALUES(?,?,?,NULL,?,?,?)",
                (
                    timing.start_timestamp,
                    timing.end_timestamp,
                    (17 << 24) | 23,
                    None if python_payload else label,
                    index if python_payload else None,
                    stored,
                ),
            )
            if index != 1 or not missing_gpu:
                rows.append(
                    {
                        "RangeId": index,
                        "Name": f"TokenSpeed:{label}",
                        "Orig Start (ns)": timing.start_timestamp + int(wrong_capture),
                        "Orig Duration (ns)": timing.end_timestamp
                        - timing.start_timestamp,
                        "PID": 17,
                        "TID": 23,
                        "Projected Start (ns)": index * 300_000_000,
                        "Projected Duration (ns)": int(timing.gpu_duration_ns / 2),
                        "NumGPUOps": 2,
                    }
                )
        stored_first = id_base + 1 if python_payload else id_base + 1 - (1 << 64)
        # Same TID, payload and label in a different process must not match.
        db.execute(
            "INSERT INTO NVTX_EVENTS VALUES(1,2,?,NULL,'gemm.mm',NULL,?)",
            ((18 << 24) | 23, stored_first),
        )
        # Other annotations may reuse the numeric payload on our own thread.
        db.execute(
            "INSERT INTO NVTX_EVENTS VALUES(1,2,?,NULL,'unrelated',NULL,?)",
            ((17 << 24) | 23, stored_first),
        )
        if python_payload and signed_failure != "no_domain_schema":
            db.executescript("""
                ALTER TABLE NVTX_EVENTS ADD COLUMN domainId INTEGER;
                ALTER TABLE NVTX_EVENTS ADD COLUMN eventType INTEGER;
                UPDATE NVTX_EVENTS SET domainId=7, eventType=59;
            """)
            db.execute("INSERT INTO StringIds VALUES(99,'TokenSpeed')")
            db.executemany(
                "INSERT INTO NVTX_EVENTS(start,globalTid,text,textId,domainId,eventType) VALUES(0,?,?,?,?,75)",
                [((17 << 24) | 9, None, 99, 7), ((17 << 24) | 9, "Foreign", None, 8)],
            )
            # Deliberately reuse the exact signed ID, registered label, and
            # thread. Only the owning TokenSpeed range may take GPU timing.
            foreign = db.execute("""
                INSERT INTO NVTX_EVENTS(start,end,globalTid,endGlobalTid,text,textId,int64Value,domainId,eventType)
                SELECT start,end,globalTid,endGlobalTid,text,textId,int64Value,8,59
                FROM NVTX_EVENTS WHERE rowid=1
                """).lastrowid
            if not missing_gpu:
                rows.append(
                    {
                        **rows[0],
                        "RangeId": foreign,
                        "Name": "Foreign:gemm.mm",
                        "Projected Duration (ns)": 17,
                    }
                )
            if signed_failure == "foreign_only":
                db.execute("UPDATE NVTX_EVENTS SET domainId=8 WHERE rowid=1")
            elif signed_failure == "wrong_event":
                db.execute("UPDATE NVTX_EVENTS SET eventType=60 WHERE rowid=1")
            elif signed_failure == "cross_thread":
                db.execute(
                    "UPDATE NVTX_EVENTS SET endGlobalTid=? WHERE rowid=1",
                    ((17 << 24) | 31,),
                )
            elif signed_failure == "open_range":
                db.execute("UPDATE NVTX_EVENTS SET end=NULL WHERE rowid=1")
            elif signed_failure == "wrong_prefix":
                rows[0]["Name"] = "Foreign:gemm.mm"
    with projection.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if signed_failure:
        reason = {
            "no_domain_schema": "lacks TokenSpeed domain or push/pop evidence",
            "wrong_prefix": "projection and NVTX capture do not match",
        }.get(signed_failure, "closed same-thread TokenSpeed push/pop range")
        with pytest.raises(ValueError, match=reason):
            analyze_nsight(report, database=database, projection=projection)
        return
    if wrong_capture:
        with pytest.raises(ValueError, match="do not match"):
            analyze_nsight(report, database=database, projection=projection)
        return
    result = analyze_nsight(report, database=database, projection=projection)
    assert result["device_analysis"]["timing_kind"] == "nsight_gpu_projection"
    analyzed = result["device_analysis"]["scopes"]
    for index, row in enumerate(analyzed):
        assert row["nsight_range_id"] == index + 1
        if missing_gpu and index in (0, 2):
            assert row["mfu"] is None and row["estimated_mbu"] is None
            assert "no GPU activity" in row["mfu_missing_reason"]
        else:
            assert row["gpu_duration_ns"] == timings[index].gpu_duration_ns / 2
            assert row["mfu"] == pytest.approx(0.96 if index < 2 else 0.48)
            assert row["op_count"] == 1
    assert report["device_analysis"]["scopes"][0]["gpu_duration_ns"] == 1
    if python_payload and not missing_gpu:
        report["device_analysis"]["scopes"][0]["nvtx_id"] = float(id_base + 1)
        with pytest.raises(ValueError, match="exact int64 NVTX scope IDs"):
            analyze_nsight(report, database=database, projection=projection)


def test_nsight_legacy_label_projection_without_payload_or_domain_columns(tmp_path):
    import sqlite3

    from tokenspeed_kernel.profile_nsys import _join_nsight_projection

    database = tmp_path / "old.sqlite"
    projection = tmp_path / "old.csv"
    with sqlite3.connect(database) as db:
        db.executescript("""
            CREATE TABLE StringIds(id INTEGER,value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,
                                     endGlobalTid INTEGER,text TEXT,textId INTEGER);
        """)
        db.execute(
            "INSERT INTO NVTX_EVENTS VALUES(10,20,?,NULL,'legacy scope',NULL)",
            ((17 << 24) | 23,),
        )
    projection.write_text(
        "RangeId,Name,Orig Start (ns),Orig Duration (ns),PID,TID,"
        "Projected Start (ns),Projected Duration (ns),NumGPUOps\n"
        "1,Legacy:legacy scope,10,10,17,23,100,20,1\n"
    )
    report = {
        "ops": [],
        "device_analysis": {
            "thread_id": 23,
            "scopes": [{"trace_label": "original", "nvtx_label": "legacy scope"}],
        },
    }
    timings, evidence = _join_nsight_projection(
        report, database=database, projection=projection
    )
    assert timings[0].gpu_duration_ns == 20
    assert evidence["original"]["nsight_range_id"] == 1


@pytest.mark.parametrize("serving", [False, True])
def test_native_collection_records_identity_without_events_or_timing_sync(
    fixture, monkeypatch, tmp_path, serving
):
    from itertools import count

    import tokenspeed_kernel.profile_nvtx as native
    from tokenspeed_kernel.profile_eager import (
        merge_forward_binding,
        read_eager_sidecar,
    )
    from tokenspeed_kernel.profile_graph import GraphReplayCollector, ProfileStep

    _, costs, _, hardware, layer = fixture
    emitted = []

    class Stream:
        device = torch.device("cuda:0")
        cuda_stream = 7

        def synchronize(self):
            pytest.fail("Collection without metadata snapshots must not synchronize")

    class Domain:
        def __init__(self):
            self.stack = []

        def push_range(self, *, message, payload):
            self.stack.append(payload)
            emitted.append((message, payload))

        def pop_range(self):
            self.stack.pop()

    stream = Stream()
    domain = Domain()
    empty = torch.empty
    monkeypatch.setattr(native, "_nvtx_domain", lambda: domain)
    monkeypatch.setattr(native, "_scope_ids", count((1 << 63) - 4))
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: empty(
            *args, **{k: v for k, v in kwargs.items() if k != "pin_memory"}
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda *args, **kwargs: pytest.fail("CUDA Event timing is forbidden"),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "CPU fixture")
    session = ProfileSession(
        run_id="native:eager:0" if serving else "native",
        model_id="test",
        rank=0,
        backend="nvtx",
    )
    collector = step = eager_step = None
    index = tmp_path / "replays.jsonl"
    if serving:
        collector = GraphReplayCollector(
            output=index,
            metadata_slots=1,
            max_metadata_bytes=64,
            max_steps=1,
            source_context={"profile_config_sha256": "a" * 64},
        )
        step = ProfileStep(
            owner=collector,
            run_id="native",
            rank=0,
            serving_step=0,
            request_ids=("test",),
            pool_indices=(0,),
        )
        eager_step = {**step.identity(), "profile_config_sha256": "a" * 64}
    session.bind_layer(layer, layer_id=7)
    output = tmp_path / "metadata.json"
    reservation = (
        collector.reserve_eager_metadata(step) if serving else nullcontext(True)
    )
    with reservation as reserved:
        assert reserved
        with native.NvtxTimeline(
            session,
            costs=costs,
            hardware=hardware,
            stream=stream,
            max_scopes=5,
            output=output,
            eager_step=eager_step,
        ) as timeline:
            with session.iteration(iter_id=0):
                layer.forward()
                layer.forward()
                with pytest.raises(OverflowError, match="scope_id"):
                    with timeline.record_func("overflow", display_name="overflow"):
                        pytest.fail("Overflowing scope was entered")
            if not serving:
                timeline.finish()
                before = output.read_bytes()
                timeline.finish()
                assert output.read_bytes() == before
        if serving:
            step.bind_eager_timeline(timeline)
    if serving:
        assert not output.exists()
        event = object()  # Already synchronized by the existing serving result owner.
        step.bind_result_event(event)
        assert not output.exists()
        step.device_results_ready(event)
        step.bind_forward_id(9)
        assert not output.exists()
        assert [
            json.loads(line)["event"] for line in index.read_text().splitlines()
        ] == ["forward_identity_bound"]
        window = collector.flush_ready()
        collector.close()
        assert window["persisted_eager"] == 1 and window["dropped_metadata"] == 0
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        ready = next(row for row in rows if row["event"] == "device_results_ready")
        binding = next(row for row in rows if row["event"] == "forward_identity_bound")
        assert ready["device_ready_sequence"] == 1 and ready["forward_id"] is None
        merge_forward_binding(ready, binding)
        assert ready["forward_id"] == 9
        report = read_eager_sidecar(
            ready, config_sha256=window["profile_config_sha256"]
        )
    else:
        report = json.loads(output.read_text())
    assert report["device_analysis"]["timing_kind"] == "pending_nsight"
    assert [label for label, _ in emitted] == ["L7", "gemm.mm", "L7", "gemm.mm"]
    assert not domain.stack
    assert report["device_analysis"]["nvtx_payload_type"] == "int64"
    assert [scope["nvtx_id"] for scope in report["device_analysis"]["scopes"]] == [
        (1 << 63) - offset for offset in (4, 3, 2, 1)
    ]
    assert all(
        "gpu_duration_ns" not in row for row in report["device_analysis"]["scopes"]
    )
