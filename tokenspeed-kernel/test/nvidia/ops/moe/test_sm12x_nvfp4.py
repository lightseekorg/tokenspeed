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


"""SM120 NVFP4 expert and packed Engram numerical contracts."""

from dataclasses import dataclass
from datetime import timedelta

import pytest
import torch
from tokenspeed_kernel.thirdparty.flashinfer.sm12x_nvfp4_moe import (
    apply_sm12x_nvfp4,
    prepare_sm12x_nvfp4,
)


@dataclass
class SwigluArg:
    alpha: float
    limit: float


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Requires an available SM120 GPU",
)


def unpack(codes, sf, global_scale, group):
    bits = torch.stack((codes & 15, codes >> 4), -1).flatten(-2).long()
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=codes.device)
    value = levels[bits & 7] * torch.where(bits & 8 != 0, -1, 1)
    return (
        value.unflatten(-1, (-1, group)) * sf.float().unsqueeze(-1) * global_scale
    ).flatten(-2)


def quant(x, scale):
    from flashinfer import fp4_quantize

    q, sf = fp4_quantize(
        x.contiguous(), scale.reciprocal(), sf_vec_size=16, is_sf_swizzled_layout=False
    )
    return q, sf.view(torch.float8_e4m3fn).reshape(x.shape[0], x.shape[1] // 16)


class Experts(torch.nn.Module):
    def __init__(self, e=2, h=512, i=256):
        super().__init__()
        self.w13_input_layout = "concatenated"
        self.activation = "swiglu"
        self.swiglu_arg = SwigluArg(alpha=1.0, limit=10.0)
        self.swiglu_beta = 0.0
        self.w13_input_scale = torch.tensor([0.003, 0.011], device="cuda")
        self.w2_input_scale = torch.tensor([0.007, 0.023], device="cuda")
        ws1, ws2, q1, q2, sf1, sf2 = [], [], [], [], [], []
        for expert in range(e):
            a = (torch.randn(2 * i, h, device="cuda") * 0.08).bfloat16()
            b = (torch.randn(h, i, device="cuda") * 0.08).bfloat16()
            s1, s2 = a.float().abs().max() / (6 * 448), b.float().abs().max() / (
                6 * 448
            )
            qa, sa = quant(a, s1)
            qb, sb = quant(b, s2)
            ws1.append(torch.stack((s1, s1)))
            ws2.append(s2)
            q1.append(qa)
            sf1.append(sa)
            q2.append(qb)
            sf2.append(sb)
        self.w13_weight = torch.stack(q1)
        self.w13_weight_scale = torch.stack(sf1)
        self.w13_weight_scale_2 = torch.stack(ws1)
        self.w2_weight = torch.stack(q2)
        self.w2_weight_scale = torch.stack(sf2)
        self.w2_weight_scale_2 = torch.stack(ws2)


def reference(x, w, ids, weights):
    out = torch.zeros_like(x, dtype=torch.float32)
    for e in range(w.w2_weight.shape[0]):
        w13 = unpack(
            w.w13_weight[e], w.w13_weight_scale[e], w.w13_weight_scale_2[e, 0], 16
        )
        w2 = unpack(w.w2_weight[e], w.w2_weight_scale[e], w.w2_weight_scale_2[e], 16)
        q, sf = quant(x, w.w13_input_scale[e])
        qx = unpack(q, sf, w.w13_input_scale[e], 16)
        gate, up = torch.nn.functional.linear(qx, w13).chunk(2, -1)
        act = torch.nn.functional.silu(gate.clamp_max(10.0)) * up.clamp(-10, 10)
        q, sf = quant(act.bfloat16(), w.w2_input_scale[e])
        qa = unpack(q, sf, w.w2_input_scale[e], 16)
        value = torch.nn.functional.linear(qa, w2)
        routing = torch.where(ids == e, weights, 0).sum(-1)
        out += value * routing[:, None]
    return out.bfloat16()


@pytest.mark.parametrize("rows", [1, 4, 8, 17])
def test_nvfp4_per_expert_scales_graph_and_clamp(rows):
    torch.manual_seed(17)
    w = Experts()
    # Large activations exercise both branches of the checkpoint clamp.
    x = (torch.randn(rows, 512, device="cuda") * 6).bfloat16()
    ids = (
        torch.tensor([[0, 1]], device="cuda", dtype=torch.int32).expand(rows, 2).clone()
    )
    weights = torch.tensor([[0.3, 0.7]], device="cuda").expand(rows, 2).contiguous()
    ref = reference(x, w, ids, weights)
    state = prepare_sm12x_nvfp4(w)
    result = apply_sm12x_nvfp4(x, state, ids, weights, False)
    error = ((result.float() - ref.float()).norm() / ref.float().norm()).item()
    assert error < 0.025, error
    for _ in range(3):
        apply_sm12x_nvfp4(x, state, ids, weights, False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = apply_sm12x_nvfp4(x, state, ids, weights, False)
    for _ in range(3):
        graph.replay()
        torch.testing.assert_close(captured, result, atol=0, rtol=0)
    ids.fill_(-1)
    graph.replay()
    assert torch.count_nonzero(captured).item() == 0


def _local_nvfp4_tuning(rank, world_size, rendezvous):
    import torch.distributed as dist
    from tokenspeed_kernel.ops.tuning import autotune, set_autotune_process_group

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(17 + rank)
        weights = Experts()
        rows = 1 if rank == 0 else 17
        x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
        ids = torch.zeros((rows, 1), dtype=torch.int32, device="cuda")
        routing = torch.ones((rows, 1), dtype=torch.float32, device="cuda")
        state = prepare_sm12x_nvfp4(weights)
        baseline = apply_sm12x_nvfp4(x, state, ids, routing, False)
        set_autotune_process_group(dist.group.WORLD)
        with autotune(tune_mode=True, tuning_buckets=(1, 16, 32), round_up=True):
            for _ in range(rank + 1):
                result = apply_sm12x_nvfp4(x, state, ids, routing, False)
        torch.testing.assert_close(result, baseline, atol=0, rtol=0)
        dist.barrier()
    finally:
        set_autotune_process_group(None)
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two SM120 GPUs")
def test_nvfp4_local_tuning_allows_different_work_per_rank(tmp_path):
    torch.multiprocessing.spawn(
        _local_nvfp4_tuning,
        args=(2, str(tmp_path / "rendezvous")),
        nprocs=2,
        join=True,
    )


def test_engram_packed_lookup_decode_boundaries():
    from tokenspeed_kernel.ops.embedding import mxfp4_embedding, mxfp4_embedding_decode

    codes = torch.arange(256, device="cuda", dtype=torch.uint8).reshape(4, 64)
    exponents = torch.tensor(
        [[0, 126, 127, 255], [120, 128, 129, 132], [127] * 4, [124] * 4],
        device="cuda",
        dtype=torch.uint8,
    )
    scale = exponents.view(torch.float8_e8m0fnu).float()
    ref = unpack(codes, scale, 1.0, 32).bfloat16()
    result = mxfp4_embedding_decode(codes, exponents)
    torch.testing.assert_close(result, ref, atol=0, rtol=0, equal_nan=True)
    ids = torch.tensor([[2, 2, 0, -1, 4]], device="cuda")
    got = mxfp4_embedding(codes, exponents, ids, 0, 4)
    torch.testing.assert_close(
        got[:, :3], ref[ids[:, :3]], atol=0, rtol=0, equal_nan=True
    )
    assert torch.count_nonzero(got[:, 3:]).item() == 0


def test_wire1_compact_decode_restore_and_empty_graph():
    from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.low_latency_wire1 import (
        restore_expert_rows,
    )
    from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.nvfp4_receive import (
        decode_wire_nvfp4,
    )

    payload = torch.randint(0, 256, (2, 4, 256), dtype=torch.uint8, device="cuda")
    scales = torch.ones((2, 4, 32), device="cuda").to(torch.float8_e4m3fn)
    globals_ = torch.arange(1, 9, device="cuda", dtype=torch.float32).reshape(2, 4) / 16
    counts = torch.tensor([2, 1], device="cuda", dtype=torch.int32)
    ref = unpack(payload, scales, globals_[..., None, None], 16)
    x, ids, weights = decode_wire_nvfp4(payload, scales, globals_, counts, 8)
    torch.testing.assert_close(
        x[:3], torch.cat((ref[0, :2], ref[1, :1])).bfloat16(), atol=0, rtol=0
    )
    assert ids.flatten().tolist() == [0, 0, 1, -1, -1, -1, -1, -1]
    assert weights.flatten().tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    restored = restore_expert_rows(x, counts, 4, zero_padding=True)
    torch.testing.assert_close(restored[0, :2], x[:2], atol=0, rtol=0)
    torch.testing.assert_close(restored[1, :1], x[2:3], atol=0, rtol=0)
    assert torch.count_nonzero(restored[0, 2:]).item() == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gx, gi, gw = decode_wire_nvfp4(payload, scales, globals_, counts, 8)
    counts.zero_()
    graph.replay()
    assert torch.count_nonzero(gx).item() == torch.count_nonzero(gw).item() == 0
    assert (gi == -1).all().item()
    counts.copy_(torch.tensor([1, 3], device="cuda"))
    graph.replay()
    torch.testing.assert_close(
        gx[:4], torch.cat((ref[0, :1], ref[1, :3])).bfloat16(), atol=0, rtol=0
    )


@pytest.mark.parametrize("hidden", [512, 5120])
@pytest.mark.parametrize("experts", [4, 12, 48])
@pytest.mark.parametrize("zero_padding", [False, True])
def test_wire1_restore_dynamic_counts_and_values(hidden, experts, zero_padding):
    from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.low_latency_wire1 import (
        _restore_expert_rows,
        restore_expert_rows,
    )

    capacity = 128
    values = torch.randn((256, hidden), dtype=torch.bfloat16, device="cuda")
    counts = torch.zeros(experts, dtype=torch.int32, device="cuda")
    counts[1], counts[-1] = 2, 3
    graph = torch.cuda.CUDAGraph()
    # Warm the same kernel specialization before capture.
    restore_expert_rows(values, counts, capacity, zero_padding=zero_padding)
    with torch.cuda.graph(graph):
        out = restore_expert_rows(values, counts, capacity, zero_padding=zero_padding)
    for distribution in (
        [0] * experts,
        [128] + [0] * (experts - 1),
        [0, 1] + [0] * (experts - 3) + [127],
    ):
        counts.copy_(torch.tensor(distribution, dtype=torch.int32, device="cuda"))
        values.add_(0.125)
        graph.replay()
        torch.cuda.synchronize()
        start = 0
        for expert, count in enumerate(distribution):
            torch.testing.assert_close(
                out[expert, :count], values[start : start + count], atol=0, rtol=0
            )
            if zero_padding:
                assert torch.count_nonzero(out[expert, count:]).item() == 0
            start += count
        # Poisoned padding independently checks the scatter's valid write set.
        poisoned = torch.full_like(out, float("nan"))
        _restore_expert_rows[(values.shape[0], (hidden + 1023) // 1024)](
            values, counts, poisoned, hidden, experts, capacity, 1024, num_warps=4
        )
        start = 0
        for expert, count in enumerate(distribution):
            torch.testing.assert_close(
                poisoned[expert, :count], values[start : start + count], atol=0, rtol=0
            )
            assert torch.isnan(poisoned[expert, count:]).all().item()
            start += count


@pytest.mark.parametrize("hidden", [512, 5120])
def test_elastic_receive_fp8_packed_scales_and_tail(hidden):
    from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.nvfp4_receive import (
        decode_elastic_fp8,
    )
    from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.q0_pack import pack_q0

    x = torch.randn(7, hidden, device="cuda", dtype=torch.bfloat16)
    q, s = pack_q0(x)
    counts = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
    got = decode_elastic_fp8(q, s, counts)
    shifts = torch.tensor([0, 8, 16, 24], device="cuda")
    exp = ((s[..., None] >> shifts) & 255).reshape(7, hidden // 128).to(torch.uint8)
    sf = exp.view(torch.float8_e8m0fnu).float().repeat_interleave(128, -1)
    ref = (q.float() * sf).bfloat16()
    torch.testing.assert_close(got[:4], ref[:4], atol=0, rtol=0)
    assert torch.count_nonzero(got[4:]).item() == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = decode_elastic_fp8(q, s, counts)
    counts.zero_()
    graph.replay()
    assert torch.count_nonzero(captured).item() == 0


def test_wire1_sparse_experts_with_compact_capacity():
    from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.nvfp4_receive import (
        decode_wire_nvfp4,
    )

    payload = torch.randint(0, 256, (4, 8, 256), dtype=torch.uint8, device="cuda")
    scales = torch.ones((4, 8, 32), device="cuda").to(torch.float8_e4m3fn)
    globals_ = torch.ones((4, 8), device="cuda", dtype=torch.float32)
    counts = torch.tensor([0, 2, 0, 1], device="cuda", dtype=torch.int32)
    ref = unpack(payload, scales, globals_[..., None, None], 16)
    x, ids, weights = decode_wire_nvfp4(payload, scales, globals_, counts, 8)
    assert x.shape[0] == 8
    assert ids.flatten().tolist() == [1, 1, 3, -1, -1, -1, -1, -1]
    torch.testing.assert_close(
        x[:3], torch.cat((ref[1, :2], ref[3, :1])).bfloat16(), atol=0, rtol=0
    )
    assert torch.count_nonzero(x[3:]).item() == 0
