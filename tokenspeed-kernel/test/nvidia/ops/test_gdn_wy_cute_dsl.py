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

"""Output-only WY verify against the token recurrence, including graph replay."""

import pytest
import torch
from tokenspeed_kernel.platform import current_platform, pdl_enabled

if not current_platform().is_hopper_plus:
    pytest.skip("CuTe GDN WY requires NVIDIA SM90+", allow_module_level=True)
pytest.importorskip("cutlass")

from tokenspeed_kernel.ops.attention.gdn.cute_dsl import (
    cute_dsl_gdn_decode_mtp as run_wy,
)


def make_case(*, batch, tokens, dtype, state_dtype, tree, strided, heads, vheads):
    generator = torch.Generator(device="cuda").manual_seed(8273)

    def rand(*shape, dtype=dtype):
        return torch.randn(shape, device="cuda", dtype=dtype, generator=generator)

    q, k = (rand(batch, tokens, heads, 256 if strided else 128) for _ in range(2))
    v = rand(batch, tokens, vheads, 256 if strided else 128) * 0.5
    a, b = (rand(batch, tokens, vheads * 2 if strided else vheads) for _ in range(2))
    if strided:
        q, k, v, a, b = (x[..., ::2] for x in (q, k, v, a, b))
    # Retain the padded physical row stride in the state pool.
    backing = rand(batch + 3, vheads + 1, 128, 128, dtype=state_dtype) * 0.02
    state = backing[:, :vheads]
    indices = torch.arange(batch, 0, -1, device="cuda", dtype=torch.int32)
    ancestors = None
    if tree:
        masks = []
        for request in range(batch):
            row = []
            for node in range(tokens):
                parent = (node - 1) // 2 if node and node != 7 else -1
                row.append((1 << node) | (row[parent] if parent >= 0 else 0))
            masks.append(row)
        ancestors = torch.tensor(masks, device="cuda", dtype=torch.int64)
    return dict(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=rand(vheads, dtype=torch.float32),
        dt_bias=rand(vheads, dtype=torch.float32),
        initial_state=state,
        initial_state_indices=indices,
        scale=128**-0.5,
        disable_state_update=True,
        use_qk_l2norm=True,
        intermediate_states_buffer=None,
        output_state_indices=None,
        output_state=None,
        tree_ancestors=ancestors,
        replay_payload=None,
    )


def reference(case):
    q, k, v = (case[name].double() for name in ("q", "k", "v"))
    if case["use_qk_l2norm"]:
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    group = v.shape[2] // q.shape[2]
    q, k = (x.repeat_interleave(group, 2) for x in (q, k))
    gate = -case["A_log"].double().exp() * torch.nn.functional.softplus(
        case["a"].double() + case["dt_bias"].double()
    )
    decay = gate.exp()
    beta = case["b"].double().sigmoid()
    indices = case["initial_state_indices"].long()
    initial = case["initial_state"][indices.clamp_min(0)].double()
    initial = torch.where((indices >= 0)[:, None, None, None], initial, 0)
    tokens = q.shape[1]
    if case["tree_ancestors"] is None:
        masks = [
            [(1 << (node + 1)) - 1 for node in range(tokens)] for _ in range(q.shape[0])
        ]
    else:
        masks = case["tree_ancestors"].cpu().tolist()
    outputs = []
    for request in range(q.shape[0]):
        states = []
        result = []
        for node in range(tokens):
            parent = (masks[request][node] ^ (1 << node)).bit_length() - 1
            h = initial[request] if parent < 0 else states[parent]
            h = h * decay[request, node, :, None, None]
            delta = (
                v[request, node] - (h * k[request, node, :, None, :]).sum(-1)
            ) * beta[request, node, :, None]
            h = h + delta[..., None] * k[request, node, :, None, :]
            scale = 128**-0.5 if case["scale"] is None else case["scale"]
            result.append((h * q[request, node, :, None, :]).sum(-1) * scale)
            states.append(h)
        outputs.append(torch.stack(result))
    return torch.stack(outputs)


@pytest.mark.parametrize(
    "dtype,state_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float16),
        (torch.float16, torch.float32),
    ],
)
@pytest.mark.parametrize(
    "tokens,tree,strided",
    [(3, False, False), (8, False, True), (13, True, True), (16, True, False)],
)
def test_wy_recurrence(dtype, state_dtype, tokens, tree, strided):
    case = make_case(
        batch=2,
        tokens=tokens,
        dtype=dtype,
        state_dtype=state_dtype,
        tree=tree,
        strided=strided,
        heads=2,
        vheads=6,
    )
    before = case["initial_state"].clone()
    actual = run_wy(**case)
    torch.testing.assert_close(actual.double(), reference(case), atol=1e-4, rtol=2**-7)
    assert torch.equal(case["initial_state"], before)


@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("tokens", [4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_wy_wide_tile_and_payload(tree, tokens, dtype):
    case = make_case(
        batch=12,
        tokens=tokens,
        dtype=dtype,
        state_dtype=dtype,
        tree=tree,
        strided=True,
        heads=8,
        vheads=24,
    )
    width = 8 * 128 + 24 * 128 + 2 * 24
    backing = torch.full(
        (12 * tokens + 2, width), float("nan"), device="cuda", dtype=dtype
    )
    case["replay_payload"] = backing[1:-1]
    state_before = case["initial_state"].clone()
    actual = run_wy(**case)
    torch.testing.assert_close(actual.double(), reference(case), atol=1e-4, rtol=2**-7)
    expected = torch.cat(
        (case["k"].flatten(2), case["v"].flatten(2), case["a"], case["b"]), -1
    ).flatten(0, 1)
    assert torch.equal(case["replay_payload"], expected)
    assert backing[0].isnan().all() and backing[-1].isnan().all()
    assert torch.equal(case["initial_state"], state_before)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    "tokens,gate_node", [(4, 0), (4, 1), (16, 0), (16, 1), (16, 7)]
)
def test_wy_unit_state_negative_row_and_strong_gate(normalize, tokens, gate_node):
    case = make_case(
        batch=2,
        tokens=tokens,
        dtype=torch.bfloat16,
        state_dtype=torch.float32,
        tree=True,
        strided=False,
        heads=2,
        vheads=6,
    )
    case["initial_state"].mul_(50)
    case["initial_state_indices"][-1] = -1
    case["use_qk_l2norm"] = normalize
    if not normalize:
        for name in ("q", "k"):
            tensor = case[name].float()
            case[name] = (
                tensor / (tensor.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            ).to(case[name].dtype)
    # A huge decay at any ancestor must not erase the segment decays
    # between its descendants through cancellation. Node7 is a separate root.
    case["A_log"].zero_()
    case["dt_bias"].zero_()
    case["a"][:, gate_node] = 1e20
    actual = run_wy(**case)
    torch.testing.assert_close(actual.double(), reference(case), atol=1e-4, rtol=2**-7)
    assert actual.isfinite().all()


@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("tokens,warm_tokens", [(3, 4), (5, 8), (13, 16)])
def test_wy_graph_refreshes_inputs_and_runtime_geometry(tree, tokens, warm_tokens):
    warm = make_case(
        batch=1,
        tokens=warm_tokens,
        dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        tree=tree,
        strided=False,
        heads=2,
        vheads=6,
    )
    run_wy(**warm)
    case = make_case(
        batch=5,
        tokens=tokens,
        dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        tree=tree,
        strided=True,
        heads=2,
        vheads=6,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run_wy(**case)
    graph.replay()
    torch.testing.assert_close(output.double(), reference(case), atol=1e-4, rtol=2**-7)
    before = output.clone()
    for name in ("q", "k", "v", "a", "b", "A_log", "dt_bias", "initial_state"):
        case[name].mul_(-0.75)
    case["initial_state_indices"].copy_(case["initial_state_indices"].flip(0))
    if tree:
        steps = torch.arange(tokens, device="cuda", dtype=torch.int64)
        case["tree_ancestors"].copy_(((1 << (steps + 1)) - 1).expand(5, tokens))
    pool = case["initial_state"].clone()
    graph.replay()
    torch.testing.assert_close(output.double(), reference(case), atol=1e-4, rtol=2**-7)
    assert not torch.equal(output, before)
    assert torch.equal(case["initial_state"], pool)
    torch.testing.assert_close(output, run_wy(**case), rtol=0, atol=0)


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("batch,heads,vheads,offset", [(9, 2, 6, 1), (12, 8, 24, 0)])
def test_wy_graph_with_unaligned_state_rows(state_dtype, batch, heads, vheads, offset):
    dtype = torch.float16 if state_dtype == torch.float16 else torch.bfloat16
    case = make_case(
        batch=batch,
        tokens=8,
        dtype=dtype,
        state_dtype=state_dtype,
        tree=False,
        strided=True,
        heads=heads,
        vheads=vheads,
    )
    row_stride = vheads * 128 * 128 + 1
    generator = torch.Generator(device="cuda").manual_seed(2147)
    backing = (
        torch.randn(
            (batch + 3) * row_stride + offset,
            device="cuda",
            dtype=state_dtype,
            generator=generator,
        )
        * 0.02
    )
    case["initial_state"] = torch.as_strided(
        backing,
        (batch + 3, vheads, 128, 128),
        (row_stride, 128 * 128, 128, 1),
        storage_offset=offset,
    )
    # Odd row strides mix aligned and unaligned rows in the same invocation.
    # Graph replay must reevaluate alignment after the physical indices change.
    case["initial_state_indices"][-1] = -1
    before = backing.clone()
    run_wy(**case)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run_wy(**case)
    graph.replay()
    torch.testing.assert_close(output.double(), reference(case), atol=1e-4, rtol=2**-7)
    initial_output = output.clone()
    case["initial_state_indices"].copy_(case["initial_state_indices"].roll(1))
    graph.replay()
    torch.testing.assert_close(output.double(), reference(case), atol=1e-4, rtol=2**-7)
    assert not torch.equal(output, initial_output)
    assert torch.equal(backing, before)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [4, 8, 16])
def test_wy_graph_with_mixed_qkv_row_alignment(dtype, tokens):
    case = make_case(
        batch=2,
        tokens=tokens,
        dtype=dtype,
        state_dtype=torch.float32,
        tree=True,
        strided=False,
        heads=2,
        vheads=6,
    )
    backing = []
    for name in ("q", "k", "v"):
        source = case[name]
        batch, count, heads, width = source.shape
        storage = torch.full(
            (batch * count * heads * (width + 1) + 1,),
            float("nan"),
            device="cuda",
            dtype=dtype,
        )
        view = torch.as_strided(
            storage,
            source.shape,
            (count * heads * (width + 1), heads * (width + 1), width + 1, 1),
            storage_offset=1,
        )
        view.copy_(source)
        case[name] = view
        backing.append(storage)
    # Unit inner strides can still alternate aligned and unaligned physical rows.
    width = 2 * 128 + 6 * 128 + 2 * 6
    case["replay_payload"] = torch.empty(2 * tokens, width, device="cuda", dtype=dtype)
    state = case["initial_state"].clone()
    run_wy(**case)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run_wy(**case)
    graph.replay()
    torch.testing.assert_close(output.double(), reference(case), atol=1e-4, rtol=2**-7)
    case["q"].neg_()
    case["k"].neg_()
    graph.replay()
    torch.testing.assert_close(output.double(), reference(case), atol=1e-4, rtol=2**-7)
    expected = torch.cat(
        (case["k"].flatten(2), case["v"].flatten(2), case["a"], case["b"]), -1
    ).flatten(0, 1)
    assert torch.equal(case["replay_payload"], expected)
    assert torch.equal(case["initial_state"], state)
    for storage in backing:
        assert storage[::129].isnan().all()


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_wy_two_value_heads_per_key_with_pdl(state_dtype):
    case = make_case(
        batch=1,
        tokens=4,
        dtype=torch.bfloat16,
        state_dtype=state_dtype,
        tree=False,
        strided=False,
        heads=16,
        vheads=32,
    )
    case["replay_payload"] = torch.empty(
        4,
        16 * 128 + 32 * 128 + 2 * 32,
        device="cuda",
        dtype=torch.bfloat16,
    )
    before = case["initial_state"].clone()
    previous = pdl_enabled()
    try:
        pdl_enabled(True)
        run_wy(**case)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run_wy(**case)
        graph.replay()
        torch.testing.assert_close(
            output.double(), reference(case), atol=1e-4, rtol=2**-7
        )
        case["v"].mul_(0.5)
        case["a"].add_(0.25)
        case["b"].sub_(0.5)
        case["A_log"].add_(0.1)
        case["dt_bias"].sub_(0.25)
        graph.replay()
        torch.testing.assert_close(
            output.double(), reference(case), atol=1e-4, rtol=2**-7
        )
    finally:
        pdl_enabled(previous)
    expected = torch.cat(
        (case["k"].flatten(2), case["v"].flatten(2), case["a"], case["b"]),
        -1,
    ).flatten(0, 1)
    assert torch.equal(case["replay_payload"], expected)
    assert torch.equal(case["initial_state"], before)


def test_wy_pdl_equivalence():
    case = make_case(
        batch=2,
        tokens=8,
        dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        tree=True,
        strided=True,
        heads=2,
        vheads=6,
    )
    previous = pdl_enabled()
    try:
        pdl_enabled(False)
        plain = run_wy(**case)
        pdl_enabled(True)
        pdl = run_wy(**case)
        torch.testing.assert_close(plain, pdl, rtol=0, atol=0)
    finally:
        pdl_enabled(previous)


@pytest.mark.parametrize(
    "argument",
    [
        "disable_state_update",
        "intermediate_states_buffer",
        "output_state_indices",
        "output_state",
    ],
)
def test_wy_rejects_state_writes(argument):
    case = make_case(
        batch=1,
        tokens=4,
        dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        tree=False,
        strided=False,
        heads=2,
        vheads=6,
    )
    case[argument] = (
        False if argument == "disable_state_update" else case["initial_state"]
    )
    with pytest.raises(ValueError, match="output-only"):
        run_wy(**case)


@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("solution", [None, "cute_dsl"])
def test_wy_registry_selection(monkeypatch, tree, solution):
    from tokenspeed_kernel.ops.attention import gdn

    selected = []
    select = gdn.select_kernel

    def record_selection(*args, **kwargs):
        kernel = select(*args, **kwargs)
        selected.append(kernel.name)
        return kernel

    monkeypatch.setattr(gdn, "select_kernel", record_selection)

    case = make_case(
        batch=2,
        tokens=8,
        dtype=torch.bfloat16,
        state_dtype=torch.float32,
        tree=tree,
        strided=True,
        heads=2,
        vheads=6,
    )
    pool = case["initial_state"].clone()
    output = gdn.gdn_decode_mtp(**case, solution=solution)
    assert selected == ["cute_dsl_gdn_decode_mtp"]
    torch.testing.assert_close(output.double(), reference(case), atol=1e-4, rtol=2**-7)
    assert torch.equal(case["initial_state"], pool)


@pytest.mark.parametrize("tokens,state_update", [(17, False), (8, True)])
def test_wy_selection_refuses_unsupported_execution(tokens, state_update):
    from tokenspeed_kernel.ops.attention.gdn import gdn_decode_mtp
    from tokenspeed_kernel.selection import NoKernelFoundError

    case = make_case(
        batch=1,
        tokens=tokens,
        dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        tree=False,
        strided=False,
        heads=2,
        vheads=6,
    )
    case["disable_state_update"] = not state_update
    with pytest.raises(NoKernelFoundError):
        gdn_decode_mtp(**case, solution="cute_dsl")


@pytest.mark.parametrize("tokens,scale", [(1, None), (4, 0.25)])
def test_wy_scale_and_int64_indices(tokens, scale):
    case = make_case(
        batch=2,
        tokens=tokens,
        dtype=torch.float16,
        state_dtype=torch.float16,
        tree=False,
        strided=True,
        heads=2,
        vheads=6,
    )
    case["scale"] = scale
    case["initial_state_indices"] = case["initial_state_indices"].long()
    torch.testing.assert_close(
        run_wy(**case).double(), reference(case), atol=1e-4, rtol=2**-7
    )


@pytest.mark.parametrize("parameter_dtype", [torch.float16, torch.bfloat16])
def test_wy_float32_gates_and_half_parameters(parameter_dtype):
    case = make_case(
        batch=1,
        tokens=8,
        dtype=torch.bfloat16,
        state_dtype=torch.float32,
        tree=True,
        strided=False,
        heads=2,
        vheads=6,
    )
    for name in ("a", "b"):
        case[name] = case[name].float()
    for name in ("A_log", "dt_bias"):
        case[name] = case[name].to(parameter_dtype)
    torch.testing.assert_close(
        run_wy(**case).double(), reference(case), atol=1e-4, rtol=2**-7
    )


def test_wy_empty_batch():
    case = make_case(
        batch=0,
        tokens=4,
        dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        tree=False,
        strided=False,
        heads=2,
        vheads=6,
    )
    result = run_wy(**case)
    assert result.shape == (0, 4, 6, 128) and result.numel() == 0


@pytest.mark.parametrize(
    "head_dim,value_head_dim,tokens,output_only,tree,expected",
    [
        (128, 128, 1, True, False, "cute_dsl_gdn_decode_mtp"),
        (128, 128, 16, True, True, "cute_dsl_gdn_decode_mtp"),
        (128, 128, 17, True, False, "flashinfer_gdn_decode_mtp"),
        (64, 128, 4, True, False, "triton_gdn_decode_mtp"),
        (128, 64, 8, True, True, "triton_gdn_decode_mtp"),
        (128, 64, 4, True, False, "triton_gdn_decode_mtp"),
        (128, 128, 8, False, True, "triton_gdn_decode_mtp"),
        (128, 128, 8, False, False, "flashinfer_gdn_decode_mtp"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_wy_selection_traits(
    dtype, head_dim, value_head_dim, tokens, output_only, tree, expected
):
    from tokenspeed_kernel.ops.attention.gdn import _attention_format_signature
    from tokenspeed_kernel.registry import KernelRegistry
    from tokenspeed_kernel.selection import select_kernel

    if (
        expected == "flashinfer_gdn_decode_mtp"
        and KernelRegistry.get().get_by_name(expected) is None
    ):
        expected = "triton_gdn_decode_mtp"
    probe = torch.empty(0, dtype=dtype, device="meta")
    assert (
        select_kernel(
            "attention",
            "gdn_decode_mtp",
            _attention_format_signature(q=probe, k=probe, v=probe),
            traits={
                "head_dim": head_dim,
                "value_head_dim": value_head_dim,
                "verify_tokens": tokens,
                "output_only": output_only,
                "tree": tree,
            },
        ).name
        == expected
    )


def test_wy_selection_on_other_vendor(mi350_platform):
    from tokenspeed_kernel.ops.attention.gdn import _attention_format_signature
    from tokenspeed_kernel.selection import select_kernel

    probe = torch.empty(0, dtype=torch.bfloat16, device="meta")
    assert (
        select_kernel(
            "attention",
            "gdn_decode_mtp",
            _attention_format_signature(q=probe, k=probe, v=probe),
            platform=mi350_platform,
            traits={
                "head_dim": 128,
                "value_head_dim": 128,
                "verify_tokens": 8,
                "output_only": True,
                "tree": True,
            },
        ).name
        == "triton_gdn_decode_mtp"
    )


def test_wy_explicit_solution_is_preserved():
    from tokenspeed_kernel.ops.attention.gdn import gdn_decode_mtp
    from tokenspeed_kernel.selection import NoKernelFoundError

    case = make_case(
        batch=1,
        tokens=8,
        dtype=torch.bfloat16,
        state_dtype=torch.float32,
        tree=True,
        strided=False,
        heads=2,
        vheads=6,
    )
    # The portable kernel's state pool is packed rather than row-padded.
    case["initial_state"] = case["initial_state"].contiguous()
    actual = gdn_decode_mtp(**case, solution="triton")
    torch.testing.assert_close(actual.double(), reference(case), atol=1e-4, rtol=2**-7)
    with pytest.raises(NoKernelFoundError):
        gdn_decode_mtp(**case, solution="flashinfer")


def test_flashinfer_bf16_state_rejects_unsupported_value_dim(require):
    require("attention", "gdn_decode_mtp", "flashinfer", torch.bfloat16, "q")
    from tokenspeed_kernel.ops.attention.gdn.flashinfer import flashinfer_gdn_decode_mtp

    case = make_case(
        batch=1,
        tokens=4,
        dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        tree=False,
        strided=False,
        heads=2,
        vheads=6,
    )
    case["v"] = case["v"][..., :64].contiguous()
    case["initial_state"] = case["initial_state"][..., :64, :].contiguous()
    with pytest.raises(NotImplementedError, match="BF16.*K=V=128"):
        flashinfer_gdn_decode_mtp(**case)


@pytest.mark.parametrize(
    "tokens,state_writes", [(8, "none"), (17, "none"), (8, "per_token")]
)
def test_wy_benchmark_preparation(tokens, state_writes):
    from tokenspeed_kernel.benchmark.generators.gdn import prepare_gdn_decode_mtp
    from tokenspeed_kernel.benchmark.harness import (
        BenchmarkCaseError,
        BenchmarkRequest,
        BenchmarkStatus,
    )

    request = BenchmarkRequest(
        family="attention",
        mode="gdn_decode_mtp",
        parameters={
            "model_profile": "qwen38_flash_tp2",
            "dtype": "bfloat16",
            "batch": 1,
            "tokens": tokens,
            "state_writes": state_writes,
        },
        solution="cute_dsl",
        registration=None,
        cold_cache=False,
        seed=8273,
    )
    if tokens > 16 or state_writes != "none":
        with pytest.raises(BenchmarkCaseError) as error:
            prepare_gdn_decode_mtp(request, current_platform())
        assert error.value.status == BenchmarkStatus.NOT_APPLICABLE
    else:
        prepared = prepare_gdn_decode_mtp(request, current_platform())
        result = prepared.invocation.invoke()
        assert result.shape == (1, tokens, 24, 128) and result.isfinite().all()
