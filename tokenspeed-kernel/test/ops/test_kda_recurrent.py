"""KDA chunk-prefill and dispatch coverage."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from kimi3_reference import kda_gate
from kimi3_reference import kda_recurrent as reference_kda_recurrent
from tokenspeed_kernel.ops import attention as attention_ops
from tokenspeed_kernel.ops.attention import (
    KdaPrefillResult,
    _attention_format_signature,
    kda_paged_decode,
    kda_paged_prefill,
    kda_recurrent_layout,
    kda_replay_commit_supported,
    try_kda_fused_paged_decode,
    try_kda_fused_paged_verify,
    try_kda_replay_commit,
)
from tokenspeed_kernel.platform import Platform, current_platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    SelectedKernel,
    select_kernel,
)

# K3 serving values: gate lower bound from the config, RMSNorm eps from the model.
LOWER_BOUND = -5.0
NORM_EPS = 1e-6


@pytest.mark.parametrize(
    ("supported_layouts", "expect_relayout"),
    [
        (None, False),
        (frozenset({"k_major", "v_major"}), False),
        (frozenset({"k_major"}), True),
    ],
)
def test_kda_prefill_relayouts_only_for_declaring_kernels(
    monkeypatch, supported_layouts, expect_relayout
) -> None:
    captured = {}

    def fake_kernel(**kwargs):
        captured["initial_state"] = kwargs["initial_state"]
        return KdaPrefillResult(torch.empty(0), kwargs["initial_state"])

    selected = SelectedKernel("fake_kda_prefill", fake_kernel)
    monkeypatch.setattr(
        attention_ops, "select_kernel", lambda *_args, **_kwargs: selected
    )
    traits = (
        {} if supported_layouts is None else {"recurrent_layout": supported_layouts}
    )
    registry = KernelRegistry.get()
    original_get_by_name = registry.get_by_name
    monkeypatch.setattr(
        registry,
        "get_by_name",
        lambda name: (
            SimpleNamespace(traits=traits)
            if name == selected.name
            else original_get_by_name(name)
        ),
    )
    q = torch.empty(1, 1, 1, 2)
    initial_state = torch.arange(4.0).view(1, 1, 2, 2)
    result = kda_paged_prefill(
        q,
        q,
        q,
        q,
        torch.empty(1, 1, 1),
        torch.empty(1),
        torch.empty(1, 2),
        initial_state=initial_state,
        cu_seqlens=torch.tensor([0, 1]),
        cu_seqlens_cpu=torch.tensor([0, 1], dtype=torch.int64),
        recurrent_layout="v_major",
    )

    if expect_relayout:
        torch.testing.assert_close(
            captured["initial_state"], initial_state.transpose(-1, -2).contiguous()
        )
        torch.testing.assert_close(result.final_state, initial_state)
    else:
        assert captured["initial_state"] is initial_state
        assert result.final_state is initial_state


@pytest.mark.parametrize(
    "batch_size,value_dim,store_states,expected",
    [
        (3, 128, False, 32),
        (4, 128, False, 64),
        (7, 128, False, 64),
        (8, 128, False, 128),
        (7, 128, True, 32),
        (8, 128, True, 32),
        (16, 128, True, 32),
        (8, 64, False, 32),
        (8, 256, False, 32),
    ],
)
def test_kda_verify_bv_routing(batch_size, value_dim, store_states, expected) -> None:
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import _kda_verify_bv

    assert _kda_verify_bv(batch_size, value_dim, store_states) == expected


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [(1, (8, 4, 3)), (3, (8, 4, 3)), (4, (8, 1, 3)), (15, (8, 1, 3)), (16, (16, 1, 3))],
)
def test_kda_verify_split_launch_routing(batch_size, expected) -> None:
    """Both routed knobs, including where the wide-block window closes."""
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        _kda_verify_launch_config,
    )

    assert (
        _kda_verify_launch_config(
            batch_size,
            128,
            False,
            split_producers=True,
            bv=None,
            num_warps=None,
            num_stages=None,
        )
        == expected
    )


def test_kda_verify_split_launch_requires_both_hoists_and_honors_overrides() -> None:
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        _kda_verify_launch_config,
    )

    assert _kda_verify_launch_config(
        4,
        128,
        False,
        split_producers=False,
        bv=None,
        num_warps=None,
        num_stages=None,
    ) == (64, 4, 2)
    assert _kda_verify_launch_config(
        8,
        128,
        True,
        split_producers=True,
        bv=None,
        num_warps=None,
        num_stages=None,
    ) == (32, 4, 2)
    assert _kda_verify_launch_config(
        4,
        128,
        False,
        split_producers=True,
        bv=32,
        num_warps=8,
        num_stages=5,
    ) == (32, 8, 5)


@pytest.mark.parametrize(
    "recurrent_layout,store_states",
    [
        ("k_major", False),
        ("k_major", True),
        ("v_major", False),
        ("v_major", True),
    ],
)
def test_kda_fused_verify_selects_all_layout_traits(
    monkeypatch, recurrent_layout, store_states
) -> None:
    selected = {}

    def fake_select_kernel(*args, **kwargs):
        selected.update(kwargs["traits"])
        return lambda **kernel_kwargs: kernel_kwargs["mixed_qkv"]

    monkeypatch.setattr(attention_ops, "select_kernel", fake_select_kernel)
    tensor = torch.empty(1, dtype=torch.bfloat16)
    result = try_kda_fused_paged_verify(
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        state_pool=tensor,
        state_scratch=tensor,
        read_indices=tensor,
        write_indices=tensor,
        num_heads=1,
        head_dim=1,
        draft_token_num=1,
        recurrent_layout=recurrent_layout,
        store_states=store_states,
    )
    assert result is tensor
    expected = {
        "paged_state": True,
        "store_states": store_states,
        "recurrent_layout": recurrent_layout,
    }
    assert selected == expected


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (torch.bfloat16, "triton_nvidia_kda_fused_paged_verify_split"),
        (torch.float16, "triton_nvidia_kda_fused_paged_verify_no_store"),
    ],
)
def test_no_store_verify_resolves_by_priority_not_by_the_caller(
    b300_platform, dtype, expected
) -> None:
    """Callers ask for store_states only; the registry picks the producers.

    The split variant outranks its inline-producer twin, so bf16 gets it
    without naming it, and fp16 -- which the split producer cannot read --
    falls back instead of losing the fused verify altogether.
    """
    registry = KernelRegistry.get()
    real_platform = Platform.get()
    probe = torch.empty(0, dtype=dtype, device="meta")
    try:
        Platform.override(b300_platform)
        registry.clear_cache()
        kernel = select_kernel(
            "attention",
            "kda_fused_paged_verify",
            _attention_format_signature(q=probe, k=probe, v=probe),
            traits={
                "paged_state": True,
                "store_states": False,
                "recurrent_layout": "v_major",
            },
        )
        assert kernel.name == expected
    finally:
        Platform.override(real_platform)
        registry.clear_cache()


def test_kda_replay_is_registered_for_the_platform_layout() -> None:
    """The workspace planner and the backend both probe this layout.

    A layout with no replay kernel makes the planner reserve the non-replay
    workspace while the backend replays, and startup rejects the mismatch.
    """
    if not current_platform().is_nvidia:
        pytest.skip("NVIDIA replay registration")
    assert kda_recurrent_layout() == "v_major"
    assert kda_replay_commit_supported(
        torch.bfloat16, recurrent_layout=kda_recurrent_layout()
    )


@pytest.mark.parametrize(
    "kernel_name,expected_traits",
    [
        (
            "triton_nvidia_kda_fused_paged_verify",
            {
                "paged_state": frozenset({True}),
                "store_states": frozenset({True}),
                "recurrent_layout": frozenset({"v_major"}),
            },
        ),
        (
            "triton_nvidia_kda_fused_paged_verify_no_store",
            {
                "paged_state": frozenset({True}),
                "store_states": frozenset({False}),
                "recurrent_layout": frozenset({"v_major"}),
            },
        ),
        (
            "triton_nvidia_kda_fused_paged_decode",
            {
                "paged_state": frozenset({True}),
                "fused_output_norm": frozenset({False, True}),
                "recurrent_layout": frozenset({"v_major"}),
            },
        ),
    ],
)
def test_nvidia_kda_verify_and_decode_registration_traits(
    kernel_name, expected_traits
) -> None:
    """A vanished or renamed registration must fail here, never skip.

    Selection ignores traits a spec does not declare, so a renamed trait
    key silently loosens matching; only exact equality catches it.
    """
    spec = KernelRegistry.get().get_by_name(kernel_name)
    assert spec is not None, kernel_name
    assert spec.traits == expected_traits, kernel_name


@pytest.mark.parametrize(
    "kernel_name",
    [
        "triton_nvidia_kda_paged_prefill",
        "flashkda_nvidia_kda_paged_prefill",
        "cutedsl_kda_nvidia_paged_prefill",
    ],
)
def test_nvidia_kda_prefill_kernels_declare_k_major(kernel_name) -> None:
    """kda_paged_prefill relayouts only for kernels that declare a layout;
    a dropped declaration silently hands them the V-major state as-is."""
    spec = KernelRegistry.get().get_by_name(kernel_name)
    assert spec is not None, kernel_name
    assert spec.traits.get("recurrent_layout") == frozenset({"k_major"}), kernel_name


def test_kda_replay_supported_on_the_nvidia_serving_platform(b300_platform) -> None:
    """The eager-replay probe must answer True for the GB300 serving config.

    Every runtime replay test skips itself when this probe flips False, so
    this is the one that fails when a replay registration vanishes, a trait
    is renamed, or the probe drifts from what the backend selects. fp16 has
    no split producer (it dereferences bf16), so it resolves to the
    inline-producer twin instead.
    """
    registry = KernelRegistry.get()
    real_platform = Platform.get()
    try:
        Platform.override(b300_platform)
        registry.clear_cache()
        assert kda_recurrent_layout() == "v_major"
        assert kda_replay_commit_supported(torch.bfloat16)
        assert kda_replay_commit_supported(torch.float16)
        assert not kda_replay_commit_supported(torch.float32)
    finally:
        Platform.override(real_platform)
        registry.clear_cache()


def test_kda_split_verify_registration_traits() -> None:
    spec = KernelRegistry.get().get_by_name(
        "triton_nvidia_kda_fused_paged_verify_split"
    )
    assert spec is not None
    assert spec.traits == {
        "paged_state": frozenset({True}),
        "store_states": frozenset({False}),
        "recurrent_layout": frozenset({"v_major"}),
    }


@pytest.mark.parametrize(
    "kernel_name,extra_traits",
    [
        ("triton_nvidia_kda_paged_decode", {"indexed_state": True}),
        ("triton_nvidia_kda_replay_commit", {"flat_state": True}),
        (
            "triton_nvidia_kda_batched_replay_commit",
            {"flat_state": True, "batched_layers": True},
        ),
    ],
)
def test_nvidia_kda_state_consumers_are_v_major(kernel_name, extra_traits) -> None:
    spec = KernelRegistry.get().get_by_name(kernel_name)
    assert spec is not None
    expected = {key: frozenset({value}) for key, value in extra_traits.items()}
    expected["recurrent_layout"] = frozenset({"v_major"})
    assert spec.traits == expected


@pytest.mark.parametrize("recurrent_layout", ["k_major", "v_major"])
def test_kda_decode_and_replay_select_layout_trait(
    monkeypatch, recurrent_layout
) -> None:
    selected = []

    def fake_select_kernel(*args, **kwargs):
        selected.append(kwargs["traits"])
        return lambda **_kwargs: torch.empty(0)

    monkeypatch.setattr(attention_ops, "select_kernel", fake_select_kernel)
    tensor = torch.empty(1, dtype=torch.bfloat16)
    kda_paged_decode(
        tensor.view(1, 1, 1, 1),
        tensor.view(1, 1, 1, 1),
        tensor.view(1, 1, 1, 1),
        tensor.view(1, 1, 1, 1),
        tensor.view(1, 1, 1),
        tensor,
        tensor,
        state_pool=tensor,
        read_indices=tensor,
        write_indices=tensor,
        cu_seqlens=torch.empty(2),
        recurrent_layout=recurrent_layout,
    )
    try_kda_replay_commit(
        *([tensor] * 9),
        state_pool=tensor,
        state_out=tensor,
        read_indices=tensor,
        write_indices=tensor,
        accepted_length=tensor,
        num_heads=1,
        head_dim=1,
        draft_token_num=1,
        recurrent_layout=recurrent_layout,
    )
    assert selected[0]["recurrent_layout"] == recurrent_layout
    assert selected[1]["recurrent_layout"] == recurrent_layout


def test_nvidia_kda_pool_matches_the_reference_recurrence() -> None:
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        fused_recurrent_kda_pool,
    )

    if not current_platform().is_nvidia:
        pytest.skip("NVIDIA KDA pool layout test")
    torch.manual_seed(29)
    batch, heads, dim, pages = 3, 2, 32, 8
    q, k, v, raw_g = (
        torch.randn(1, batch, heads, dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    )
    beta = torch.randn(1, batch, heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(heads, dim, device="cuda", dtype=torch.float32)
    # The oracle's state is [H, V, K], which is exactly the V-major slab page.
    pool = torch.randn(pages, heads, dim, dim, device="cuda")
    reads = torch.arange(batch, device="cuda", dtype=torch.int32)
    writes = reads + batch
    cu_seqlens = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    expected = [
        reference_kda_recurrent(
            q[0, row : row + 1],
            k[0, row : row + 1],
            v[0, row : row + 1],
            raw_g[0, row : row + 1],
            beta[0, row : row + 1],
            pool[row],
            a_log,
            dt_bias,
            lower_bound=LOWER_BOUND,
        )
        for row in range(batch)
    ]

    out = fused_recurrent_kda_pool(
        q,
        k,
        v,
        raw_g,
        beta,
        a_log,
        dt_bias,
        pool,
        reads,
        writes,
        cu_seqlens=cu_seqlens,
        lower_bound=LOWER_BOUND,
    )

    for row, (expected_out, expected_state) in enumerate(expected):
        torch.testing.assert_close(out[0, row : row + 1], expected_out)
        torch.testing.assert_close(pool[writes[row]], expected_state)


def test_k3_safe_gate_reference_matches_sigmoid_contract() -> None:
    """Distinguish K3's safe sigmoid gate from a clamped softplus gate."""
    raw_g = torch.tensor([[[-2.0, 0.0, 2.0]]])
    a_log = torch.tensor([0.25])
    dt_bias = torch.tensor([[0.5, -0.5, 1.0]])
    lower_bound = -5.0

    gate_input = raw_g + dt_bias
    expected = lower_bound * torch.sigmoid(torch.exp(a_log)[None, :, None] * gate_input)
    actual = kda_gate(raw_g, a_log, dt_bias, lower_bound=lower_bound)
    torch.testing.assert_close(actual, expected)

    legacy = torch.clamp_min(
        -torch.exp(a_log)[None, :, None] * torch.nn.functional.softplus(gate_input),
        lower_bound,
    )
    assert not torch.allclose(actual, legacy)


def test_kda_paged_prefill_preserves_native_state_layout() -> None:
    """Native prefill preserves each backend's physical state layout."""
    platform = current_platform()
    if not (platform.is_cdna4 or platform.is_cdna5):
        pytest.skip("gfx950/gfx1250 KDA dispatch test")

    device = "cuda"
    torch.manual_seed(3)
    tokens, heads, key_dim, value_dim = 65, 2, 128, 128
    q = torch.randn(tokens, heads, key_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(
        tokens,
        heads,
        value_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    raw_g = torch.randn_like(q)
    beta = torch.randn(tokens, heads, device=device, dtype=torch.bfloat16)
    state = torch.randn(
        1,
        heads,
        value_dim,
        key_dim,
        device=device,
        dtype=torch.float32,
    )
    a_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, key_dim, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, tokens], device=device, dtype=torch.int32)

    expected_out, expected_state = reference_kda_recurrent(
        q,
        k,
        v,
        raw_g,
        beta,
        state[0],
        a_log,
        dt_bias,
    )
    result = kda_paged_prefill(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        raw_g.unsqueeze(0),
        beta.unsqueeze(0),
        a_log,
        dt_bias,
        initial_state=state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens.to("cpu", torch.int64),
    )

    torch.testing.assert_close(
        result.out[0].float(), expected_out.float(), atol=6e-2, rtol=6e-2
    )
    expected_final_state = expected_state.unsqueeze(0)
    torch.testing.assert_close(
        result.final_state,
        expected_final_state,
        atol=6e-2,
        rtol=6e-2,
    )


@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("strided_inputs", [False, True])
@pytest.mark.parametrize(
    ("heads", "key_dim", "value_dim"),
    [(2, 8, 8), (2, 8, 5), (12, 128, 128)],
)
def test_kda_paged_decode_defaults_to_specialized_kernel_on_amd(
    lower_bound: float | None,
    strided_inputs: bool,
    heads: int,
    key_dim: int,
    value_dim: int,
) -> None:
    """AMD single-token dispatch must select Gluon and preserve native math."""
    platform = current_platform()
    if not (platform.is_cdna4 or platform.is_cdna5):
        pytest.skip("gfx950/gfx1250 KDA dispatch test")

    device = "cuda"
    torch.manual_seed(13)
    tokens = 3
    if strided_inputs:
        qkv = torch.randn(
            1,
            tokens,
            heads * (2 * key_dim + value_dim) + 7,
            device=device,
            dtype=torch.bfloat16,
        )
        q_end = heads * key_dim
        k_end = 2 * q_end
        v_end = k_end + heads * value_dim
        q = qkv[..., :q_end].view(1, tokens, heads, key_dim)
        k = qkv[..., q_end:k_end].view(1, tokens, heads, key_dim)
        v = qkv[..., k_end:v_end].view(1, tokens, heads, value_dim)
        raw_g = torch.randn(
            1,
            tokens,
            heads * key_dim + 5,
            device=device,
            dtype=torch.bfloat16,
        )[..., :q_end].view(1, tokens, heads, key_dim)
        beta = torch.randn(
            1,
            tokens,
            heads + 3,
            device=device,
            dtype=torch.bfloat16,
        )[..., :heads]
    else:
        q = torch.randn(1, tokens, heads, key_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(
            1, tokens, heads, value_dim, device=device, dtype=torch.bfloat16
        )
        raw_g = torch.randn_like(q)
        beta = torch.randn(1, tokens, heads, device=device, dtype=torch.bfloat16)
    state_pool = torch.randn(
        6,
        heads,
        value_dim,
        key_dim,
        device=device,
        dtype=torch.float32,
    )
    initial_pool = state_pool.clone()
    expected_pool = state_pool.clone()
    a_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, key_dim, device=device, dtype=torch.float32)
    cu_seqlens = torch.arange(tokens + 1, device=device, dtype=torch.int32)
    read_indices = torch.tensor([0, 1, 1], device=device, dtype=torch.int32)
    write_indices = torch.tensor([2, 3, 4], device=device, dtype=torch.int32)

    selected = select_kernel(
        "attention",
        "kda_paged_decode",
        _attention_format_signature(q=q, k=k, v=v),
        traits={"indexed_state": True, "single_token": True},
    )
    expected_kernel = (
        "gluon_kda_paged_decode_gfx1250"
        if platform.is_cdna5
        else "gluon_kda_paged_decode_gfx950"
    )
    assert selected.name == expected_kernel

    expected_out = []
    for row in range(tokens):
        out, final_state = reference_kda_recurrent(
            q[0, row : row + 1],
            k[0, row : row + 1],
            v[0, row : row + 1],
            raw_g[0, row : row + 1],
            beta[0, row : row + 1],
            initial_pool[read_indices[row].long()],
            a_log,
            dt_bias,
            lower_bound=lower_bound,
        )
        expected_out.append(out[0])
        expected_pool[write_indices[row].long()] = final_state
    expected_out = torch.stack(expected_out).unsqueeze(0)
    actual_out = kda_paged_decode(
        q,
        k,
        v,
        raw_g,
        beta,
        a_log,
        dt_bias,
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
    )

    torch.testing.assert_close(
        actual_out.float(), expected_out.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(state_pool, expected_pool, atol=2e-5, rtol=2e-5)


def test_kda_paged_decode_graph_padding_and_page_stride() -> None:
    """Gluon decode supports padded graph batches and strided state pages."""
    if not (current_platform().is_cdna4 or current_platform().is_cdna5):
        pytest.skip("gfx950/gfx1250 KDA dispatch test")

    torch.manual_seed(23)
    batch, active, heads, key_dim, value_dim = 4, 2, 2, 8, 5
    state_elements = heads * key_dim * value_dim
    raw_pool = torch.randn(7, state_elements + 11, device="cuda", dtype=torch.float32)
    state_pool = raw_pool[:, :state_elements].view(
        7,
        heads,
        value_dim,
        key_dim,
    )
    q = torch.randn(1, batch, heads, key_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, batch, heads, value_dim, device="cuda", dtype=torch.bfloat16)
    raw_g = torch.randn_like(q)
    beta = torch.randn(1, batch, heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(heads, key_dim, device="cuda", dtype=torch.float32)
    read_indices = torch.tensor([1, 2, -1, -1], device="cuda", dtype=torch.int32)
    write_indices = torch.tensor([3, 4, -1, -1], device="cuda", dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 1, 2, 2, 2], device="cuda", dtype=torch.int32)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kda_paged_decode(
            q,
            k,
            v,
            raw_g,
            beta,
            a_log,
            dt_bias,
            state_pool=state_pool,
            read_indices=read_indices,
            write_indices=write_indices,
            cu_seqlens=cu_seqlens,
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        captured[:, active:],
        torch.zeros_like(captured[:, active:]),
    )


@pytest.mark.parametrize(
    ("batch", "active"),
    [(1, 1), (2, 2), (4, 2), (8, 8), (16, 16), (32, 32)],
)
def test_kda_fused_paged_decode_matches_reference(batch: int, active: int) -> None:
    """The K3 megafusion preserves state paging and its fused norm epilogue."""
    # NVIDIA is covered by test_kda_megafuse_fused_norm_matches_separate_norm: the
    # 2e-4 state tolerance below is tuned to the gfx950 kernel's accumulation.
    if not (current_platform().is_cdna4 or current_platform().is_cdna5):
        pytest.skip("gfx950/gfx1250 KDA fusion test")

    torch.manual_seed(31)
    heads, head_dim, pages = 12, 128, 2 * active
    projection_width = heads * head_dim
    used_width = 4 * projection_width + head_dim + heads
    packed_width = (used_width + 15) // 16 * 16
    packed_projection = torch.randn(
        batch,
        packed_width,
        device="cuda",
        dtype=torch.bfloat16,
    )
    mixed_qkv = packed_projection[:, : 3 * projection_width]
    assert mixed_qkv.stride() == (packed_width, 1)
    conv_weights = 0.1 * torch.randn(
        3 * projection_width,
        4,
        device="cuda",
        dtype=torch.bfloat16,
    )
    conv_states = 0.1 * torch.randn(
        pages,
        3 * projection_width,
        3,
        device="cuda",
        dtype=torch.bfloat16,
    )
    initial_conv_states = conv_states.clone()
    expected_conv_states = conv_states.clone()
    f_a_start = 4 * projection_width
    f_a_out = packed_projection[:, f_a_start : f_a_start + head_dim]
    f_b_weight = 0.1 * torch.randn(
        projection_width,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    beta_start = f_a_start + head_dim
    beta_logits = packed_projection[:, beta_start : beta_start + heads]
    a_log = torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(projection_width, device="cuda", dtype=torch.float32)
    output_gate = packed_projection[:, 3 * projection_width : f_a_start]
    norm_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    norm_eps = 1e-6
    state_pool = 0.01 * torch.randn(
        pages,
        heads,
        head_dim,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    initial_state_pool = state_pool.clone()
    expected_state_pool = state_pool.clone()
    read_indices = torch.full((batch,), -1, device="cuda", dtype=torch.int32)
    write_indices = torch.full((batch,), -1, device="cuda", dtype=torch.int32)
    read_indices[:active] = torch.arange(active, device="cuda", dtype=torch.int32)
    write_indices[:active] = torch.arange(
        active, 2 * active, device="cuda", dtype=torch.int32
    )
    cu_seqlens = torch.cat(
        (
            torch.arange(active + 1, device="cuda", dtype=torch.int32),
            torch.full((batch - active,), active, device="cuda", dtype=torch.int32),
        )
    )

    raw_g = torch.nn.functional.linear(f_a_out, f_b_weight)
    expected_out = torch.zeros(
        batch,
        heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    for row in range(active):
        read_idx = read_indices[row].item()
        write_idx = write_indices[row].item()
        current = mixed_qkv[row].view(3, heads, head_dim).float()
        history = initial_conv_states[read_idx].view(3, heads, head_dim, 3).float()
        weights = conv_weights.view(3, heads, head_dim, 4).float()
        convolved = torch.nn.functional.silu(
            (history * weights[..., :3]).sum(dim=-1) + current * weights[..., 3]
        )
        q, k, v = convolved.unbind(dim=0)
        core_out, final_state = reference_kda_recurrent(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            raw_g[row].view(1, heads, head_dim),
            beta_logits[row].unsqueeze(0),
            initial_state_pool[read_idx],
            a_log,
            dt_bias.view(heads, head_dim),
        )
        core_out = core_out[0].float()
        inverse_rms = torch.rsqrt(core_out.square().mean(dim=-1) + norm_eps)
        expected_out[row] = (
            core_out
            * inverse_rms[:, None]
            * norm_weight.float()[None, :]
            * torch.sigmoid(output_gate[row].view(heads, head_dim).float())
        ).to(torch.bfloat16)
        expected_state_pool[write_idx] = final_state
        expected_conv_states[write_idx] = torch.stack(
            (history[..., 1], history[..., 2], current), dim=-1
        ).reshape(3 * projection_width, 3)

    result = try_kda_fused_paged_decode(
        mixed_qkv,
        conv_weights,
        conv_states,
        f_a_out,
        f_b_weight,
        beta_logits,
        a_log,
        dt_bias,
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        num_heads=heads,
        head_dim=head_dim,
        cu_seqlens=cu_seqlens,
        output_gate=output_gate,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        recurrent_layout="v_major",
    )

    assert result is not None
    assert result.output_norm_applied
    torch.testing.assert_close(
        result.out[0].float(), expected_out.float(), atol=5e-2, rtol=5e-2
    )
    torch.testing.assert_close(state_pool, expected_state_pool, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(conv_states, expected_conv_states)


def test_kda_fused_decode_override_preserves_external_output_norm(monkeypatch) -> None:
    """A core-only override must not claim that it applied output normalization."""
    assert KernelRegistry.get().get_by_name(
        "triton_nvidia_kda_fused_paged_decode"
    ).traits["fused_output_norm"] == frozenset({False, True})

    # The verify kernel is registered without the trait at all.
    kernel_name = "triton_nvidia_kda_fused_paged_verify_no_store"
    spec = KernelRegistry.get().get_by_name(kernel_name)
    assert spec is not None
    assert "fused_output_norm" not in spec.traits

    captured_kwargs = {}

    class CoreOnlyKernel:
        name = kernel_name

        def __call__(self, **kwargs):
            captured_kwargs.update(kwargs)
            return kwargs["mixed_qkv"]

    monkeypatch.setattr(
        attention_ops,
        "select_kernel",
        lambda *_args, **_kwargs: CoreOnlyKernel(),
    )
    tensor = torch.empty(1, dtype=torch.bfloat16)
    result = try_kda_fused_paged_decode(
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        state_pool=tensor,
        read_indices=tensor,
        write_indices=tensor,
        num_heads=12,
        head_dim=128,
        cu_seqlens=tensor,
        output_gate=tensor,
        norm_weight=tensor,
        norm_eps=1e-6,
        override=kernel_name,
    )

    assert result is not None
    assert not result.output_norm_applied
    assert captured_kwargs["output_gate"] is None
    assert captured_kwargs["norm_weight"] is None
    assert captured_kwargs["norm_eps"] is None


def test_kda_fused_decode_rejects_unsupported_conv_width() -> None:
    """Unsupported convolution widths must fall back before kernel execution."""
    if not (current_platform().is_cdna4 or current_platform().is_cdna5):
        pytest.skip("gfx950/gfx1250 KDA fusion dispatch test")

    tensor = torch.empty(1, dtype=torch.bfloat16)
    conv_weights = torch.empty(1, 5, dtype=torch.bfloat16)
    result = try_kda_fused_paged_decode(
        tensor,
        conv_weights,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        state_pool=tensor,
        read_indices=tensor,
        write_indices=tensor,
        num_heads=12,
        head_dim=128,
        cu_seqlens=tensor,
        output_gate=tensor,
        norm_weight=tensor,
        norm_eps=1e-6,
    )

    assert result is None


def test_kda_paged_decode_does_not_select_nvidia_kernel_on_amd() -> None:
    """The NVIDIA portable adapter remains outside the AMD dispatch surface."""
    if not current_platform().is_amd:
        pytest.skip("AMD KDA dispatch test")

    q = torch.randn(1, 3, 2, 8, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with pytest.raises(NoKernelFoundError):
        select_kernel(
            "attention",
            "kda_paged_decode",
            _attention_format_signature(q=q, k=k, v=v),
            traits={"indexed_state": True, "single_token": False},
        )


def _megafuse_inputs(batch: int, seed: int = 17):
    """Build K3-shaped megafuse decode inputs (TP8 rank: 12 heads, K=V=128)."""
    torch.manual_seed(seed)
    heads, head_dim = 12, 128
    pages = 2 * batch
    width = heads * head_dim
    used = 4 * width + head_dim + heads
    packed = torch.randn(
        batch, (used + 15) // 16 * 16, device="cuda", dtype=torch.bfloat16
    )
    return {
        "heads": heads,
        "head_dim": head_dim,
        "mixed_qkv": packed[:, : 3 * width],
        "output_gate": packed[:, 3 * width : 4 * width],
        "f_a_out": packed[:, 4 * width : 4 * width + head_dim],
        "beta_logits": packed[:, 4 * width + head_dim : used],
        "conv_weights": 0.1
        * torch.randn(3 * width, 4, device="cuda", dtype=torch.bfloat16),
        "conv_states": 0.1
        * torch.randn(pages, 3 * width, 3, device="cuda", dtype=torch.bfloat16),
        "f_b_weight": 0.1
        * torch.randn(width, head_dim, device="cuda", dtype=torch.bfloat16),
        "a_log": torch.randn(heads, device="cuda", dtype=torch.float32),
        "dt_bias": torch.randn(width, device="cuda", dtype=torch.float32),
        "norm_weight": torch.randn(head_dim, device="cuda", dtype=torch.bfloat16),
        "state_pool": 0.01
        * torch.randn(
            pages, heads, head_dim, head_dim, device="cuda", dtype=torch.float32
        ),
        "read_indices": torch.arange(batch, device="cuda", dtype=torch.int32),
        "write_indices": torch.arange(
            batch, 2 * batch, device="cuda", dtype=torch.int32
        ),
        "cu_seqlens": torch.arange(batch + 1, device="cuda", dtype=torch.int32),
    }


def _run_megafuse(inp, *, fused: bool):
    """One megafuse decode, with the norm epilogue fused in or applied after."""
    from tokenspeed_kernel.ops.activation.triton import rmsnorm_gated_sigmoid
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        fused_recurrent_kda_megafuse,
    )

    heads, head_dim = inp["heads"], inp["head_dim"]
    # Serving passes a row-strided slice of the packed projection, not a copy.
    gate = inp["output_gate"]
    out = fused_recurrent_kda_megafuse(
        inp["mixed_qkv"],
        inp["conv_weights"],
        inp["conv_states"],
        inp["f_a_out"],
        inp["f_b_weight"],
        inp["beta_logits"],
        inp["a_log"],
        inp["dt_bias"],
        h_pool=inp["state_pool"],
        read_indices=inp["read_indices"],
        write_indices=inp["write_indices"],
        num_heads=heads,
        head_dim=head_dim,
        cu_seqlens=inp["cu_seqlens"],
        lower_bound=LOWER_BOUND,
        output_gate=gate if fused else None,
        norm_weight=inp["norm_weight"] if fused else None,
        norm_eps=NORM_EPS if fused else None,
    )
    if fused:
        return out
    return rmsnorm_gated_sigmoid(
        out.reshape(-1, heads * head_dim).contiguous(),
        gate.contiguous(),
        inp["norm_weight"],
        NORM_EPS,
        heads,
        head_dim,
    ).view_as(out)


@pytest.mark.parametrize("batch", [1, 4])
def test_kda_megafuse_fused_norm_matches_separate_norm(batch: int) -> None:
    """The fused epilogue reproduces megafuse followed by rmsnorm_gated_sigmoid."""
    if not current_platform().is_nvidia:
        pytest.skip("NVIDIA triton KDA megafusion test")

    inp = _megafuse_inputs(batch)
    conv0, state0 = inp["conv_states"].clone(), inp["state_pool"].clone()

    expected = _run_megafuse(inp, fused=False)
    conv_ref, state_ref = inp["conv_states"].clone(), inp["state_pool"].clone()

    inp["conv_states"].copy_(conv0)
    inp["state_pool"].copy_(state0)
    fused = _run_megafuse(inp, fused=True)

    torch.testing.assert_close(fused.float(), expected.float(), atol=0.5, rtol=2e-2)
    # The epilogue must not disturb the state it writes back.
    torch.testing.assert_close(inp["conv_states"], conv_ref)
    torch.testing.assert_close(inp["state_pool"], state_ref)


def test_kda_megafuse_fused_norm_is_cuda_graph_safe() -> None:
    """The fused epilogue captures and replays inside a CUDA graph."""
    if not current_platform().is_nvidia:
        pytest.skip("NVIDIA triton KDA megafusion test")

    inp = _megafuse_inputs(1)
    conv0, state0 = inp["conv_states"].clone(), inp["state_pool"].clone()

    # Warm eagerly: the JIT must not compile inside the capture.
    eager = _run_megafuse(inp, fused=True)
    inp["conv_states"].copy_(conv0)
    inp["state_pool"].copy_(state0)

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run_megafuse(inp, fused=True)
    torch.cuda.current_stream().wait_stream(stream)
    inp["conv_states"].copy_(conv0)
    inp["state_pool"].copy_(state0)

    with torch.cuda.graph(graph):
        captured = _run_megafuse(inp, fused=True)

    inp["conv_states"].copy_(conv0)
    inp["state_pool"].copy_(state0)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured.float(), eager.float(), atol=0.5, rtol=2e-2)
