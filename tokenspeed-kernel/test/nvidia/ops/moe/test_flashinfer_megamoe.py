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

import copy
import math
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.numerics.torch.quantization import quantize_nvfp4_blocks
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_process_weights
from tokenspeed_kernel.ops.tuning import (
    get_autotune_max_num_tokens,
    kernel_warmup,
    set_autotune_max_num_tokens,
)
from tokenspeed_kernel.thirdparty.flashinfer.megamoe import (
    megamoe_unavailable_reason,
    prepare_megamoe,
)


class ExpertWeights(torch.nn.Module):
    def __init__(self, experts: int, hidden: int, intermediate: int, top_k: int):
        super().__init__()
        self.tp_size = self.ep_size = 1
        self.top_k = top_k
        self._moe_backend_state: object | None = None
        gate_up = torch.randn(experts, 2 * intermediate, hidden, device="cuda") * 0.02
        down = torch.randn(experts, hidden, intermediate, device="cuda") * 0.02
        first_scales = torch.linspace(0.125, 0.25, experts, device="cuda")
        second_scales = torch.linspace(0.25, 0.5, experts, device="cuda")
        q1, s1 = quantize_nvfp4_blocks(gate_up / first_scales[:, None, None])
        q2, s2 = quantize_nvfp4_blocks(down / second_scales[:, None, None])
        self.w13_weight = torch.nn.Parameter(q1, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(q2, requires_grad=False)
        self.w13_weight_scale = torch.nn.Parameter(s1, requires_grad=False)
        self.w2_weight_scale = torch.nn.Parameter(s2, requires_grad=False)
        self.w13_weight_scale_2 = torch.nn.Parameter(
            first_scales[:, None].repeat(1, 2), requires_grad=False
        )
        self.w2_weight_scale_2 = torch.nn.Parameter(second_scales, requires_grad=False)
        self.w13_input_scale = torch.nn.Parameter(
            torch.tensor([0.25], device="cuda"), requires_grad=False
        )
        self.w2_input_scale = torch.nn.Parameter(
            torch.full((experts,), 0.125, device="cuda"), requires_grad=False
        )


def plan_for(
    experts: int, hidden: int, intermediate: int, top_k: int, *, solution: str
):
    return moe_plan(
        weight_dtype="nvfp4",
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ep_size=1,
        ispp=intermediate,
        hidden=hidden,
        num_experts=experts,
        top_k=top_k,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
        solution=solution,
    )


def dequant(q, sf):
    q = q.view(torch.uint8)
    codes = torch.stack((q & 15, q >> 4), dim=-1).flatten(-2).long()
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=q.device)
    return (
        table[codes & 7]
        * torch.where(codes < 8, 1.0, -1.0)
        * sf.float().repeat_interleave(16, -1)
    )


def quant(v):
    """Independent FP32 -> E4M3 scale and E2M1 RNE; no BF16 intermediate."""
    sf = (
        (v.reshape(*v.shape[:-1], -1, 16).abs().amax(-1) * (1.0 / 6.0))
        .clamp(max=448.0)
        .to(torch.float8_e4m3fn)
    )
    inv = torch.where(sf.float() == 0, 0.0, sf.float().reciprocal())
    scaled = v * inv.repeat_interleave(16, -1)
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=v.device)
    odd = torch.tensor([False, True, False, True, False, True, False], device=v.device)
    absolute = scaled.abs()[..., None]
    code = ((absolute > bounds) | ((absolute == bounds) & odd)).sum(-1).to(torch.uint8)
    code |= torch.signbit(scaled).to(torch.uint8) << 3
    return code[..., 0::2] | (code[..., 1::2] << 4), sf


def oracle(q, sf, raw, ids, scores, dtype):
    x = dequant(q, sf)
    terms = torch.empty(
        ids.shape[0],
        ids.shape[1],
        raw.w2_weight.shape[1],
        device="cuda",
        dtype=torch.bfloat16,
    )
    for e in ids.unique().tolist():
        row, slot = torch.where(ids == e)
        w1 = dequant(raw.w13_weight[e], raw.w13_weight_scale[e])
        w2 = dequant(raw.w2_weight[e], raw.w2_weight_scale[e])
        acc = F.linear(x[row].to(dtype), w1.to(dtype)).float()
        alpha1 = (raw.w13_input_scale * raw.w13_weight_scale_2[e, 0]).reshape(())
        gate, up = (acc * alpha1).chunk(2, -1)
        middle = (up * gate) * (
            1.0 + torch.exp2(gate * -math.log2(math.e))
        ).reciprocal()
        qmid, smid = quant(middle * raw.w2_input_scale[e].reciprocal())
        alpha2 = raw.w2_input_scale[e] * raw.w2_weight_scale_2[e]
        terms[row, slot] = (
            F.linear(dequant(qmid, smid).to(dtype), w2.to(dtype)).float() * alpha2
        ).bfloat16()
    return (terms.float() * scores[..., None]).sum(1).bfloat16()


@triton.jit
def reciprocal_approx_kernel(source, destination, count):
    idx = tl.program_id(0) * 256 + tl.arange(0, 256)
    value = tl.load(source + idx, idx < count, other=1.0)
    reciprocal = tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(destination + idx, reciprocal, idx < count)


def contract_quant_input(x, norm):
    # Match the documented scalar order, not the upstream quantizer implementation.
    multiplier = torch.tensor(norm / 6.0, device=x.device, dtype=torch.float32)
    block_amax = x.float().reshape(x.shape[0], -1, 16).abs().amax(-1)
    sf = (block_amax * multiplier).clamp(max=448.0).to(torch.float8_e4m3fn)
    sf32 = sf.float()
    reciprocal = torch.empty_like(sf32)
    reciprocal_approx_kernel[(triton.cdiv(sf32.numel(), 256),)](
        sf32, reciprocal, sf32.numel()
    )
    encode = (norm * reciprocal).clamp(max=torch.finfo(torch.float32).max)
    encode = encode * (sf32 * 1.0e30).clamp(max=1.0)
    scaled = x.float() * encode.repeat_interleave(16, -1)
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    odd = torch.tensor([False, True, False, True, False, True, False], device=x.device)
    absolute = scaled.abs()[..., None]
    codes = ((absolute > bounds) | ((absolute == bounds) & odd)).sum(-1).to(torch.uint8)
    codes |= torch.signbit(scaled).to(torch.uint8) << 3
    return codes[..., 0::2] | (codes[..., 1::2] << 4), sf


def reference(x, raw, ids, scores):
    q, sf = contract_quant_input(x, raw.w13_input_scale.float().reciprocal().item())
    # -1 routes are padding; skip them and give them zero contribution.
    valid = ids >= 0
    return oracle(q, sf, raw, ids.clamp_min(0), scores * valid, torch.float64)


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("FlashInfer MegaMoE NVFP4 requires SM107")
    reason = megamoe_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    previous = get_autotune_max_num_tokens()
    set_autotune_max_num_tokens(128)
    yield
    set_autotune_max_num_tokens(previous)


def apply(w, x, ids, scores):
    return w._moe_backend_state.apply(x, ids, scores, capacity=max(1, x.shape[0]))


def inputs(tokens):
    x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
    ids = torch.rand(tokens, 8, device="cuda").topk(3).indices.int()
    scores = torch.rand(tokens, 3, device="cuda").softmax(-1)
    return x, ids, scores


@pytest.mark.parametrize(
    "tokens,amplitude", [(1, 1), (1, 4), (8, 1), (8, 4), (17, 1), (128, 1)]
)
def test_reference_and_changing_graph(tokens, amplitude):
    torch.manual_seed(37)
    w = ExpertWeights(8, 256, 128, 3)
    # Per-expert FC2 global scales must survive weight preparation.
    w.w2_input_scale.data.copy_(torch.linspace(0.125, 0.25, 8, device="cuda"))
    raw = copy.deepcopy(w)
    w._moe_backend_state = prepare_megamoe(w, fast_math=True)
    x, ids, scores = inputs(tokens)
    x.mul_(amplitude)
    actual = apply(w, x, ids, scores)
    torch.testing.assert_close(
        actual, reference(x, raw, ids, scores), atol=0.002, rtol=0.02
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = apply(w, x, ids, scores)
    x.mul_(0.75)
    ids.copy_((ids + 1) % 8)
    scores.copy_(scores.roll(1, -1))
    graph.replay()
    torch.testing.assert_close(captured, apply(w, x, ids, scores), atol=0, rtol=0)
    torch.testing.assert_close(
        captured, reference(x, raw, ids, scores), atol=0.002, rtol=0.02
    )


def test_workspace_growth_preserves_old_graph_and_owned_outputs():
    torch.manual_seed(41)
    w = ExpertWeights(8, 256, 128, 3)
    raw = copy.deepcopy(w)
    w._moe_backend_state = prepare_megamoe(w, fast_math=True)
    x, ids, scores = inputs(8)
    with kernel_warmup(enabled=True):
        saved = apply(w, x, ids, scores)
    saved_copy = saved.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = apply(w, x, ids, scores)
    # A second layer shares scratch, but has different weights and scales.
    other = ExpertWeights(8, 256, 128, 3)
    other.w13_input_scale.data.mul_(2)
    other_raw = copy.deepcopy(other)
    other._moe_backend_state = prepare_megamoe(other, fast_math=True)
    y = apply(other, x, ids, scores)
    torch.testing.assert_close(
        y, reference(x, other_raw, ids, scores), atol=0.002, rtol=0.02
    )
    torch.testing.assert_close(saved, saved_copy, atol=0, rtol=0)
    large = inputs(129)
    apply(w, *large)
    # Replay a graph retaining the original capacity after a larger eager batch.
    ids.fill_(-1)
    ids[:, 0] = 2
    scores[:, 1:].zero_()
    graph.replay()
    torch.testing.assert_close(
        captured, reference(x, raw, ids, scores), atol=0.002, rtol=0.02
    )
    torch.testing.assert_close(captured, apply(w, x, ids, scores), atol=0, rtol=0)
    empty = inputs(0)
    assert apply(w, *empty).shape == (0, 256)


def test_rejects_missing_prewarm_and_unsupported_contract(monkeypatch):
    w = ExpertWeights(8, 256, 128, 3)
    monkeypatch.delenv("MEGA_NO_DIST")
    with pytest.raises(ValueError, match="MEGA_NO_DIST"):
        w._moe_backend_state = prepare_megamoe(w, fast_math=True)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    with pytest.raises(ValueError, match="fast_math"):
        prepare_megamoe(w, fast_math=False)
    w.w13_weight_scale_2.data[0, 1] *= 2
    with pytest.raises(ValueError, match="equal gate/up"):
        w._moe_backend_state = prepare_megamoe(w, fast_math=True)
    w.w13_weight_scale_2.data[0, 1] /= 2
    w._moe_backend_state = prepare_megamoe(w, fast_math=True)
    x, ids, scores = inputs(1)
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            apply(w, x, ids, scores)
    with pytest.raises(ValueError, match="FP32"):
        apply(w, x, ids, scores.bfloat16())


@pytest.mark.parametrize(
    "missing",
    ["flashinfer.moe_ep", "flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe"],
)
def test_missing_optional_dependency_keeps_default_dispatch(missing):
    import subprocess
    import sys
    import textwrap

    # A fresh interpreter models a deployment without this optional package;
    # blocking only that import leaves the real registry and TRT-LLM intact.
    code = textwrap.dedent("""
        import importlib.abc
        import sys
        class MissingDependency(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == sys.argv[1] or fullname.startswith(sys.argv[1] + "."):
                    raise ModuleNotFoundError(f"Unavailable optional dependency: {fullname}")
                return None
        sys.meta_path.insert(0, MissingDependency())
        import torch
        from tokenspeed_kernel.ops.moe import moe_plan
        from tokenspeed_kernel.selection import NoKernelFoundError
        from tokenspeed_kernel.registry import load_builtin_kernels
        load_builtin_kernels()
        options = dict(
            weight_dtype="nvfp4", input_dtype=torch.bfloat16, activation="silu",
            routing_mode="precomputed_topk", ep_size=1, ispp=640, hidden=2560,
            num_experts=512, top_k=10, swiglu_form=None, activation_clamped=False,
            expert_id_repeats=False, fast_math=True, combine_order="rank",
            features=frozenset(),
        )
        fallback = moe_plan(**options, solution=None)
        expected = moe_plan(**options, solution="flashinfer_trtllm")
        assert fallback["apply_kernel"] == expected["apply_kernel"]
        try:
            moe_plan(**options, solution="flashinfer_megamoe")
        except NoKernelFoundError:
            pass
        else:
            raise AssertionError("Missing MegaMoE must not be selectable")
    """)
    subprocess.run([sys.executable, "-c", code, missing], check=True, timeout=120)


def test_maximum_warmup_prepares_unseen_decode_shapes():
    torch.manual_seed(43)
    w = ExpertWeights(8, 256, 128, 3)
    raw = copy.deepcopy(w)
    w._moe_backend_state = prepare_megamoe(w, fast_math=True)
    x, ids, scores = inputs(128)
    with kernel_warmup(enabled=True):
        apply(w, x, ids, scores)
    # No eager call at these live row counts before capture. Upstream's
    # capture guards reject any missing compilation or lazy initialization.
    for count in (1, 8, 17):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = apply(w, x[:count], ids[:count], scores[:count])
        graph.replay()
        torch.testing.assert_close(
            output,
            reference(x[:count], raw, ids[:count], scores[:count]),
            atol=0.002,
            rtol=0.02,
        )
        live_ids = ids[:count].clone()
        ids[:count].fill_(-1)
        graph.replay()
        torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
        ids[:count].copy_(live_ids)


@dataclass(frozen=True)
class ExpertSpec:
    num_experts: int
    num_local_experts: int
    top_k: int
    ep_rank: int


class QwenExpertWeights(ExpertWeights):
    def __init__(self):
        # Eight different expert weights repeated across the full geometry keep
        # fixture construction bounded while changing routes changes the output.
        super().__init__(8, 2560, 640, 10)
        self._spec = ExpertSpec(512, 512, 10, 0)
        self.intermediate_size_per_partition = 640
        self._correction_bias = None
        self._routing_logits_dtype = torch.bfloat16
        self._routing_method_type = 4
        self._normalize_topk_weights = True
        self._n_group = self._topk_group = 0
        self._routed_scaling_factor = 1.0
        self.swiglu_arg = None
        for parameter in (
            self.w13_weight,
            self.w2_weight,
            self.w13_weight_scale,
            self.w2_weight_scale,
            self.w13_weight_scale_2,
            self.w2_weight_scale_2,
            self.w2_input_scale,
        ):
            parameter.data = parameter.repeat(64, *([1] * (parameter.ndim - 1)))


def test_single_layout_all_sizes_and_startup_capture():
    torch.manual_seed(47)
    w = QwenExpertWeights()
    raw = copy.deepcopy(w)
    plan = plan_for(512, 2560, 640, 10, solution="flashinfer_megamoe")
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    moe_process_weights(plan, w)
    torch.cuda.synchronize()
    # Preparation replaces ~1.3 GiB of weights, not a second resident layout.
    # Allow allocator rounding and small scalar/bootstrap buffers.
    assert torch.cuda.memory_allocated() - before < 16 * 1024**2
    state = w._moe_backend_state
    count = 8193
    x = torch.randn(count, 2560, device="cuda", dtype=torch.bfloat16) * 0.2
    ids = torch.rand(count, 512, device="cuda").topk(10).indices.int()
    scores = torch.randn(count, 10, device="cuda").softmax(-1)

    def run(rows, *, capacity=None):
        return moe_apply(
            plan,
            x[:rows],
            w,
            None,
            topk_ids=ids[:rows],
            topk_weights=scores[:rows],
            max_num_tokens_per_gpu=capacity,
        )

    # Empty inputs do not allocate scratch. Decode uses MegaMoE immediately.
    assert run(0).shape == (0, 2560)
    assert state.workspace is None
    small = run(1)
    torch.testing.assert_close(
        small, reference(x[:1], raw, ids[:1], scores[:1]), atol=0.002, rtol=0.02
    )
    assert state.workspace.max_tokens_per_rank == 1
    with kernel_warmup(enabled=True):
        run(count, capacity=count)
    workspace = state.workspace
    assert workspace.max_tokens_per_rank == 16384
    for rows in (0, 1, 8, 6655, 6656, 8192, count):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(rows)
        graph.replay()
        expected = state.apply(x[:rows], ids[:rows], scores[:rows], capacity=count)
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
        saved = captured.clone()
        x[:rows].mul_(0.75)
        ids[:rows].copy_((ids[:rows] + 1) % 512)
        scores[:rows].copy_(scores[:rows].roll(1, -1))
        graph.replay()
        torch.testing.assert_close(captured, run(rows), atol=0, rtol=0)
        if rows in (1, 8, count):
            torch.testing.assert_close(
                captured,
                reference(x[:rows], raw, ids[:rows], scores[:rows]),
                atol=0.002,
                rtol=0.02,
            )
        later = run(count)
        torch.testing.assert_close(captured, run(rows), atol=0, rtol=0)
        if rows:
            assert captured.data_ptr() != later.data_ptr()
            assert not torch.equal(captured, saved)
        assert state.workspace is workspace
    with pytest.raises(ValueError, match="finalized"):
        moe_apply(
            plan,
            x[:1],
            w,
            None,
            topk_ids=ids[:1],
            topk_weights=scores[:1],
            do_finalize=False,
        )
    with pytest.raises(ValueError, match="smaller"):
        run(8, capacity=1)
    # Explicit capacity is honored even when the current batch is small.
    run(1, capacity=32768)
    assert state.workspace.max_tokens_per_rank == 32768


@pytest.mark.parametrize(
    "shape", [(8, 256, 128, 3), (512, 2560, 768, 10), (512, 2560, 640, 8)]
)
def test_unmeasured_shapes_are_not_selected(shape):
    from tokenspeed_kernel.selection import NoKernelFoundError

    with pytest.raises(NoKernelFoundError):
        plan_for(*shape, solution="flashinfer_megamoe")


def test_logits_and_explicit_routes_use_single_layout():
    torch.manual_seed(53)
    w = QwenExpertWeights()
    raw = copy.deepcopy(w)
    plan = plan_for(512, 2560, 640, 10, solution="flashinfer_megamoe")
    moe_process_weights(plan, w)
    count = 8193
    x = torch.randn(count, 2560, device="cuda", dtype=torch.bfloat16) * 0.2
    logits = torch.full((count, 512), -4.0, device="cuda", dtype=torch.bfloat16)
    chosen = torch.rand(count, 512, device="cuda").topk(10).indices
    # Distinct selected logits avoid ambiguous tied routes in the FP64 oracle.
    values = torch.arange(10, device="cuda", dtype=torch.bfloat16) * 0.5
    logits.scatter_(1, chosen, values.expand(count, -1))

    def run(rows):
        return moe_apply(plan, x[:rows], w, logits[:rows])

    with kernel_warmup(enabled=True):
        run(count)
    for rows in (0, 1, 8, 8192, count):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(rows)
        logits.copy_(logits.roll(1, -1))
        x[:rows].mul_(0.75)
        graph.replay()
        torch.testing.assert_close(captured, run(rows), atol=0, rtol=0)
        if rows:
            probabilities = logits[:rows].float().softmax(-1)
            scores, ids = probabilities.topk(10, dim=-1)
            scores = scores / scores.sum(-1, keepdim=True)
            expected = reference(x[:rows], raw, ids.int(), scores)
            torch.testing.assert_close(captured, expected, atol=0.002, rtol=0.02)
    # Explicit routes remain authoritative even when unrelated logits are supplied.
    ids = chosen[:8].int()
    scores = torch.full((8, 10), 0.1, device="cuda", dtype=torch.float32)
    explicit = moe_apply(plan, x[:8], w, logits[:8], topk_ids=ids, topk_weights=scores)
    torch.testing.assert_close(
        explicit, reference(x[:8], raw, ids, scores), atol=0.002, rtol=0.02
    )
    with pytest.raises(ValueError, match="together"):
        moe_apply(plan, x[:1], w, logits[:1], topk_ids=chosen[:1].int())
    w._normalize_topk_weights = False
    with pytest.raises(ValueError, match="RenormalizeNaive"):
        run(1)
