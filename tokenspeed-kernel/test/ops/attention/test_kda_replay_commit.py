"""Correctness of the KDA speculative replay-commit.

Speculative verification runs the whole draft window, but only an unknown
prefix of it is accepted. Rather than storing a recurrent state per draft
position and committing the accepted one, verification saves U/K/raw-g records and the
accepted prefix is recovered from the still-intact committed page.

What has to hold for that to be lossless:

1. replaying ``a`` positions reproduces what ``a`` ordinary decode steps would
   have produced (the pure-torch reference below, and the production
   non-speculative decode kernel);
2. the rejected suffix cannot influence the committed state at all;
3. ``a = 0`` commits the pre-draft state unchanged;
4. requests in one batch may accept different lengths;
5. committing into the source page (the usual case -- the new position rarely
   crosses a flat page boundary) matches committing into a fresh page. The
   conv history must have one writer per head. NVIDIA uses a separate
   history launch; AMD's unsplit recovery program owns that head's history.
"""

import pytest
import torch
from kda_replay_reference import batched_recurrent_kda_replay_commit
from utils import is_cdna4, is_cdna5

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.kda import (  # noqa: E402
    kda_replay_commit_supported,
    resolve_kda_batched_replay_commit,
)
from tokenspeed_kernel.platform import current_platform  # noqa: E402

#: The registry entries behind the probe and the no-store fused verify are
#: vendor-gated; the shared records tests exercise the selected implementation.
requires_registered_replay = pytest.mark.skipif(
    not kda_replay_commit_supported(torch.bfloat16),
    reason="KDA replay ops are not registered on this platform",
)

from kda_replay_reference import _gate_tiling
from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (  # noqa: E402
    batched_kda_commit_conv_window_kernel,
    fused_recurrent_kda_verify_megafuse,
)

# K3 TP8 rank geometry, trimmed to 4 heads to keep the reference loop quick.
HV, K, V, D_FA = 4, 128, 128, 128
P = HV * K
LOWER_BOUND = -5.0
DEV = "cuda"


def _window(n, t, pages=32, seed=0, num_heads=HV):
    """A draft window of ``t`` positions for ``n`` requests, plus the pools."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    projection_width = num_heads * K

    def rnd(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(
            device=DEV, dtype=dtype
        )

    return dict(
        qkv_raw=rnd(n * t, 3 * projection_width),
        conv_w=rnd(3 * projection_width, 4, scale=0.3).contiguous(),
        conv_pool=rnd(pages, 3 * projection_width, 3),
        f_a=rnd(n * t, D_FA),
        w_fb=rnd(projection_width, D_FA, scale=0.05).contiguous(),
        beta=rnd(n * t, num_heads),
        A_log=rnd(num_heads, dtype=torch.float32, scale=0.5),
        dt_bias=rnd(projection_width, dtype=torch.float32),
        h_pool=rnd(pages, num_heads, K, V, dtype=torch.float32),
        gate_scratch=torch.empty(
            n * t, projection_width, device=DEV, dtype=torch.float32
        ),
        read_indices=torch.arange(1, n + 1, device=DEV, dtype=torch.int32),
    )


def _prepare_records(x, t, *, raw_gate, original_qkv=None):
    """Produce BF16 K/raw-gate and FP32 U through the registered serving path."""
    from tokenspeed_kernel.ops.attention.kda import (
        KdaReplayRecords,
        kda_fused_paged_verify_uses_split_producers,
        kda_verify_conv_update,
        try_kda_fused_paged_verify,
    )

    n, heads = x["read_indices"].numel(), x["beta"].shape[-1]
    producers = {}
    if kda_fused_paged_verify_uses_split_producers(
        x["qkv_raw"].dtype,
        store_states=False,
        recurrent_layout="v_major",
        num_heads=heads,
        head_dim=K,
    ):
        producers["conv_qkv"] = kda_verify_conv_update(
            x["qkv_raw"],
            x["conv_w"],
            x["conv_pool"],
            x["read_indices"],
            num_heads=heads,
            head_dim=K,
            draft_token_num=t,
            bias=None,
            recurrent_layout="v_major",
            original_qkv=original_qkv,
        )
        original_qkv = None
    if raw_gate is None and "conv_qkv" in producers:
        raw_gate = torch.mm(x["f_a"], x["w_fb"].t())
    x["correction"] = torch.empty(n, t, heads, V, device=DEV, dtype=torch.float32)
    x["key_gate"] = torch.empty(n, t, heads, 2 * K, device=DEV, dtype=torch.bfloat16)
    x["metadata"] = torch.zeros(n, 3, device=DEV, dtype=torch.int64)
    x["metadata"][:, 0] = torch.arange(n, device=DEV)
    # Exercise ring wrap and 64-bit absolute positions on every backend.
    x["metadata"][:, 1] = 7
    x["metadata"][:, 2] = 2**33 + 1
    x["stamps"] = torch.full((n, t, 2), -1, device=DEV, dtype=torch.int64)
    records = KdaReplayRecords(
        original_qkv=original_qkv,
        correction=x["correction"].transpose(1, 2),
        key_gate=x["key_gate"].transpose(1, 2),
        metadata=x["metadata"],
        stamps=x["stamps"],
        query_start_loc=torch.arange(n + 1, device=DEV, dtype=torch.int32) * t,
    )

    def verify():
        x["verify_output"] = try_kda_fused_paged_verify(
            x["qkv_raw"],
            x["conv_w"],
            x["conv_pool"],
            x["conv_pool"],
            x["f_a"],
            x["w_fb"],
            x["beta"],
            x["A_log"],
            x["dt_bias"],
            state_pool=x["h_pool"],
            state_scratch=None,
            read_indices=x["read_indices"],
            write_indices=x["read_indices"],
            num_heads=heads,
            head_dim=K,
            draft_token_num=t,
            lower_bound=LOWER_BOUND,
            recurrent_layout="v_major",
            store_states=False,
            replay_records=records,
            g_raw=raw_gate,
            **producers,
        )
        assert x["verify_output"] is not None

    verify()
    return verify


def _records_commit(xs, groups, reads, writes, accepted, t, *, corrupt=False):
    kernel = resolve_kda_batched_replay_commit(
        num_heads=xs[0]["beta"].shape[-1], head_dim=K
    )
    assert kernel is not None
    verifies = []
    for x, group in zip(xs, groups, strict=True):
        x["read_indices"] = reads[group]
        x["retained_qkv"] = torch.full_like(x["qkv_raw"], float("nan"))
        verifies.append(
            _prepare_records(
                x,
                t,
                raw_gate=torch.mm(x["f_a"], x["w_fb"].t()),
                original_qkv=x["retained_qkv"],
            )
        )
        torch.testing.assert_close(x["retained_qkv"], x["qkv_raw"], atol=0, rtol=0)
    addresses = _batched_descriptor(xs)
    for layer, x in enumerate(xs):
        addresses[layer, 0] = x["retained_qkv"].data_ptr()
        addresses[layer, 3] = x["correction"].data_ptr()
        addresses[layer, 5] = x["key_gate"].data_ptr()
    if corrupt:
        addresses[2, 5] = xs[1]["key_gate"].data_ptr()
    x = xs[0]
    group_tensor = torch.tensor(groups, device=DEV, dtype=torch.int32)
    accepted_counts = accepted.clamp(0, t)
    stamps = torch.stack([layer["stamps"] for layer in xs])

    def commit():
        for verify in verifies:
            verify()
        kernel(
            descriptors=addresses,
            group_indices=group_tensor,
            read_indices=reads,
            write_indices=writes,
            accepted_length=accepted_counts,
            activation_dtype=x["qkv_raw"].dtype,
            conv_state_dtype=x["conv_pool"].dtype,
            draft_token_num=t,
            num_heads=x["beta"].shape[-1],
            head_dim=K,
            qkv_stride=x["qkv_raw"].stride(0),
            conv_stride=x["conv_pool"].stride(0),
            correction_stride=x["correction"].stride(1),
            key_gate_stride=x["key_gate"].stride(1),
            state_stride=x["h_pool"].stride(0),
            conv_width=4,
            lower_bound=LOWER_BOUND,
            record_metadata=x["metadata"],
            record_stamps=stamps,
        )

    commit()
    return commit


def _replay(x, write_indices, accepted, t):
    """Verify once, then recover the accepted prefix into the destination pool."""
    x = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
    _records_commit(
        [x],
        [0],
        x["read_indices"][None],
        write_indices[None],
        accepted,
        t,
    )
    return x


def _batched_descriptor(xs):
    addresses = []
    for x in xs:
        addresses.append(
            [
                x["qkv_raw"].data_ptr(),
                x["conv_w"].data_ptr(),
                x["conv_pool"].data_ptr(),
                x["f_a"].data_ptr(),
                x["w_fb"].data_ptr(),
                x["beta"].data_ptr(),
                x["A_log"].data_ptr(),
                x["dt_bias"].data_ptr(),
                x["h_pool"].data_ptr(),
                x["gate_scratch"].data_ptr(),
            ]
        )
    return torch.tensor(addresses, dtype=torch.uint64, device=DEV)


def _batched_static_args(x):
    return dict(
        qkv_stride=x["qkv_raw"].stride(0),
        conv_stride=x["conv_pool"].stride(0),
        f_a_stride=x["f_a"].stride(0),
        beta_stride=x["beta"].stride(0),
        state_stride=x["h_pool"].stride(0),
        gate_stride=x["gate_scratch"].stride(0),
        conv_width=x["conv_w"].shape[1],
        lower_bound=LOWER_BOUND,
    )


def test_gate_tiling_counts_every_layer_in_the_grid():
    """The batched launch tiles for its own grid, which spans all layers."""
    dev = torch.device(DEV)
    rows = 8
    one = _gate_tiling(rows, HV, K, dev)
    many = _gate_tiling(rows, HV, K, dev, layers=69)
    # More programs can only let an earlier, wider rung clear the threshold.
    assert many[0] >= one[0] and many[1] >= one[1]
    assert _gate_tiling(rows, HV, K, dev, layers=10**6) == (8, 32)
    # A rung wider than the row count spends most of its lanes on rows that
    # do not exist, so the ladder skips it however wide the grid looks.
    for narrow in (1, 2, 4):
        assert _gate_tiling(narrow, HV, K, dev, layers=10**6)[0] <= narrow


@pytest.mark.parametrize("block", [64, 128, 256, 512, 2048])
def test_batched_conv_window_is_independent_of_its_column_block(block):
    """Splitting the window publish across columns must not move a byte."""
    layers, n, t = 3, 4, 4
    source = [_window(n, t, seed=700 + layer) for layer in range(layers)]
    reads = torch.stack([source[0]["read_indices"]] * layers).to(torch.int32)
    writes = torch.stack(
        [torch.arange(9, 9 + n, device=DEV, dtype=torch.int32)] * layers
    )
    accepted = torch.tensor([0, t, 1, 3], device=DEV, dtype=torch.int32)
    conv_dim = 3 * HV * K

    def publish(block_size):
        xs = [
            {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
            for x in source
        ]
        batched_kda_commit_conv_window_kernel[
            (layers, n, (conv_dim + block_size - 1) // block_size)
        ](
            _batched_descriptor(xs),
            xs[0]["qkv_raw"],
            xs[0]["conv_pool"],
            torch.arange(layers, device=DEV, dtype=torch.int32),
            reads,
            writes,
            accepted,
            n,
            T=t,
            STRIDE_QKV=xs[0]["qkv_raw"].stride(0),
            STRIDE_CONV=xs[0]["conv_pool"].stride(0),
            CONV_DIM=conv_dim,
            BLOCK=block_size,
            num_warps=min(8, max(1, block_size // 128)),
        )
        return [x["conv_pool"] for x in xs]

    for narrow, wide in zip(publish(block), publish(4096), strict=True):
        torch.testing.assert_close(narrow, wide, atol=0, rtol=0)


def test_batched_replay_is_bit_identical_and_descriptor_sensitive():
    """One launch matches the layer loop; a wrong descriptor must be detected."""
    layers, n, t = 5, 5, 4
    source = [_window(n, t, seed=100 + layer) for layer in range(layers)]
    loop = []
    writes = torch.stack(
        [
            torch.arange(17, 17 + n, device=DEV, dtype=torch.int32),
            torch.arange(24, 24 + n, device=DEV, dtype=torch.int32),
        ]
    )
    reads = torch.stack([source[0]["read_indices"], source[0]["read_indices"] + 5]).to(
        torch.int32
    )
    accepted = torch.tensor([0, t, 1, 3, 2], device=DEV, dtype=torch.int32)
    groups = [0, 0, 0, 1, 1]
    group_indices = torch.tensor(groups, device=DEV, dtype=torch.int32)
    for x, group in zip(source, groups, strict=True):
        local = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
        local["read_indices"] = reads[group]
        loop.append(_replay(local, writes[group], accepted, t))

    batched = [
        {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
        for x in source
    ]
    for x, group in zip(batched, groups, strict=True):
        x["read_indices"] = reads[group]
    _records_commit(batched, groups, reads, writes, accepted, t)
    for expected, actual in zip(loop, batched, strict=True):
        torch.testing.assert_close(
            actual["correction"], expected["correction"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            actual["key_gate"], expected["key_gate"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            actual["conv_pool"], expected["conv_pool"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            actual["h_pool"], expected["h_pool"], atol=1e-6, rtol=0
        )

    negative = [
        {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
        for x in source
    ]
    _records_commit(negative, groups, reads, writes, accepted, t, corrupt=True)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            negative[2]["h_pool"], loop[2]["h_pool"], atol=0, rtol=0
        )


@pytest.mark.skipif(
    not (is_cdna4() or is_cdna5()),
    reason="AMD CDNA4 or CDNA5 is required for Gluon KDA replay execution",
)
@pytest.mark.parametrize("t", [1, 4, 8])
@pytest.mark.parametrize("graph", [False, True])
def test_gluon_batched_replay_uneven_groups_match_per_layer_launches(t, graph):
    """Records verify and recovery honor uneven groups in decode and speculation."""
    layers, n, pages = 5, 2, 10
    groups = [0, 0, 0, 1, 1]
    source = [_window(n, t, pages=pages, seed=500 + layer) for layer in range(layers)]
    batched = [
        {name: value.clone() for name, value in layer.items()} for layer in source
    ]
    per_layer = [
        {name: value.clone() for name, value in layer.items()} for layer in source
    ]
    reads = torch.tensor([[1, 2], [3, 4]], device=DEV, dtype=torch.int32)
    writes = torch.tensor([[5, 6], [7, 8]], device=DEV, dtype=torch.int32)
    accepted = torch.tensor([1, t], device=DEV, dtype=torch.int32)
    run = _records_commit(batched, groups, reads, writes, accepted, t)
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            run()
        capture.replay()
    for layer, group in zip(per_layer, groups, strict=True):
        _records_commit(
            [layer],
            [0],
            reads[group : group + 1],
            writes[group : group + 1],
            accepted,
            t,
        )
    torch.cuda.synchronize()
    for original, actual, expected, group in zip(
        source, batched, per_layer, groups, strict=True
    ):
        original["read_indices"] = reads[group]
        _check_against_reference(original, actual, writes[group], n, t, accepted)
        _, _, outputs, corrections = _reference(original, n, t, [t] * n)
        expected_outputs = torch.stack([torch.stack(row) for row in outputs])
        expected_corrections = torch.stack([torch.stack(row) for row in corrections])
        torch.testing.assert_close(
            actual["verify_output"].view_as(expected_outputs),
            expected_outputs,
            atol=1.6e-2,
            rtol=1e-2,
        )
        positions = (2**33 + 1 + torch.arange(t, device=DEV)) % t
        torch.testing.assert_close(
            actual["correction"].index_select(1, positions),
            expected_corrections,
            atol=2e-6,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            actual["h_pool"][reads[group].long()],
            original["h_pool"][reads[group].long()],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            actual["conv_pool"][reads[group].long()],
            original["conv_pool"][reads[group].long()],
            atol=0,
            rtol=0,
        )

        torch.testing.assert_close(
            actual["conv_pool"], expected["conv_pool"], atol=0, rtol=0
        )
        torch.testing.assert_close(actual["h_pool"], expected["h_pool"], atol=0, rtol=0)
        torch.testing.assert_close(
            actual["verify_output"], expected["verify_output"], atol=0, rtol=0
        )

    # Batch/descriptor strides vary during serving without changing the model.
    from utils import assert_no_triton_compile

    if is_cdna4():
        from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
            gluon_kda_fused_paged_verify_nostore_vmajor_gfx950 as verify_kernel,
        )
        from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
            gluon_kda_fused_replay_gfx950 as commit_kernel,
        )
    else:
        from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.decode import (
            gluon_kda_fused_paged_verify_nostore_vmajor_gfx1250 as verify_kernel,
        )
        from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.decode import (
            gluon_kda_fused_replay_gfx1250 as commit_kernel,
        )
    wider = _window(3, t, pages=10, seed=730)
    original = {name: value.clone() for name, value in wider.items()}
    next_reads = torch.tensor([[1, 2, 3]], device=DEV, dtype=torch.int32)
    next_writes = torch.tensor([[4, 5, 6]], device=DEV, dtype=torch.int32)
    next_accepted = torch.tensor([0, 1, t], device=DEV, dtype=torch.int32)
    # Verify keeps the original power-of-two value-split buckets.
    # Warm B=4, then B=3 must reuse that bucket despite its different strides.
    warm = _window(4, t, pages=10, seed=731)
    _records_commit(
        [warm],
        [0],
        torch.tensor([[1, 2, 3, 4]], device=DEV, dtype=torch.int32),
        torch.tensor([[5, 6, 7, 8]], device=DEV, dtype=torch.int32),
        torch.full((4,), t, device=DEV, dtype=torch.int32),
        t,
    )
    with assert_no_triton_compile(verify_kernel, commit_kernel):
        _records_commit([wider], [0], next_reads, next_writes, next_accepted, t)
    _check_against_reference(original, wider, next_writes[0], 3, t, next_accepted)

    # Check automatic gate production below and above the gfx950 GEMM cutoff.
    # Dyadic inputs make the FP32 reduction exact, isolating projection indexing
    # from BF16 midpoint differences between otherwise valid reduction orders.
    for batch in (3, 17):
        projected = _window(batch, t, pages=2 * batch + 2, seed=740 + batch)
        projected["f_a"].round_()
        projected["w_fb"].mul_(32).round_().div_(8)
        raw_gate = torch.mm(projected["f_a"], projected["w_fb"].t())
        activation = projected["f_a"].clone()
        rows, width = projected["qkv_raw"].shape
        for padding in (7, 16):
            projected["f_a"] = torch.empty_strided(
                activation.shape,
                (D_FA + padding, 1),
                device=DEV,
                dtype=torch.bfloat16,
            )
            projected["f_a"].copy_(activation)
            destination = torch.full(
                (rows, width + padding), float("nan"), device=DEV, dtype=torch.bfloat16
            )
            retained = destination[:, :width]
            if padding == 7:
                _prepare_records(projected, t, raw_gate=None, original_qkv=retained)
            with assert_no_triton_compile(verify_kernel):
                verify = _prepare_records(
                    projected, t, raw_gate=None, original_qkv=retained
                )
                if graph:
                    capture = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(capture):
                        verify()
                    retained.fill_(float("nan"))
                    capture.replay()
            torch.testing.assert_close(retained, projected["qkv_raw"], atol=0, rtol=0)
            assert torch.isnan(destination[:, width:]).all()
            actual = {
                name: projected[name].clone()
                for name in ("verify_output", "correction", "key_gate")
            }
            _prepare_records(projected, t, raw_gate=raw_gate)
            torch.testing.assert_close(
                actual["verify_output"],
                projected["verify_output"],
                atol=1.6e-2,
                rtol=1e-2,
            )
            torch.testing.assert_close(
                actual["correction"], projected["correction"], atol=2e-6, rtol=2e-5
            )
            torch.testing.assert_close(
                actual["key_gate"], projected["key_gate"], atol=0, rtol=0
            )


@pytest.mark.parametrize(
    ("accepted", "crossing", "t"),
    [
        ([1, 1, 1, 1], [False, False, False, False], 3),
        ([2, 3, 2, 3], [True, True, True, True], 3),
        ([1, 1, 2, 3], [False, False, True, True], 3),
        # DSpark's draft width: longer replays cross the boundary more often.
        ([1, 8, 4, 1], [False, True, True, False], 8),
    ],
    ids=[
        "in-place-only",
        "cross-only",
        "mixed-checkpoint-boundary",
        "mixed-boundary-t8",
    ],
)
def test_batched_replay_checkpoint_boundary_matches_layer_loop(accepted, crossing, t):
    """TP1 checkpoint-boundary schedule agrees with the layer-loop oracle."""
    layers, n = 69, 4
    source = [_window(n, t, pages=24, seed=2701 + layer) for layer in range(layers)]
    for x in source:
        qkv = torch.empty(n * t, 9 * P, device=DEV, dtype=torch.bfloat16)
        qkv[:, : 3 * P].copy_(x["qkv_raw"])
        x["qkv_raw"] = qkv[:, : 3 * P]
        beta = torch.empty(n * t, 3 * HV, device=DEV, dtype=torch.bfloat16)
        beta[:, :HV].copy_(x["beta"])
        x["beta"] = beta[:, :HV]
        x["gate_scratch"] = torch.empty(n * t, 3 * P, device=DEV, dtype=torch.float32)[
            :, :P
        ]
        pages = 24
        page_bytes = 96 * 3 * P * 3 * torch.bfloat16.itemsize
        arena = torch.empty(pages * page_bytes, device=DEV, dtype=torch.uint8)
        conv_storage = arena.view(torch.bfloat16)
        conv = torch.as_strided(
            conv_storage, (pages, 9 * P, 3), (page_bytes // 2, 3, 1)
        )
        conv.zero_()
        conv[:, : 3 * P].copy_(x["conv_pool"])
        x["conv_pool"] = conv
        state_storage = arena[3 * P * 3 * 2 :].view(torch.float32)
        state = torch.as_strided(
            state_storage,
            (pages, 3 * HV, K, V),
            (page_bytes // 4, K * V, V, 1),
        )
        state.zero_()
        state[:, :HV].copy_(x["h_pool"])
        x["h_pool"] = state
    reads = torch.stack([torch.arange(2, 2 + n, device=DEV, dtype=torch.int32)] * 3)
    crossed = torch.arange(10, 10 + n, device=DEV, dtype=torch.int32)
    writes = torch.stack(
        [torch.where(torch.tensor(crossing, device=DEV), crossed, reads[0])] * 3
    ).to(torch.int32)
    accepted_tensor = torch.tensor(accepted, device=DEV, dtype=torch.int32)
    groups = [group for group in range(3) for _ in range(23)]

    loop = []
    for x, group in zip(source, groups, strict=True):
        local = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
        local["conv_pool"].zero_()
        local["h_pool"].zero_()
        local["read_indices"] = reads[group]
        loop.append(_replay(local, writes[group], accepted_tensor, t))

    batched = [
        {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
        for x in source
    ]
    for x, group in zip(batched, groups, strict=True):
        x["conv_pool"].zero_()
        x["h_pool"].zero_()
        x["read_indices"] = reads[group]
    _records_commit(batched, groups, reads, writes, accepted_tensor, t)
    for expected, actual in zip(loop, batched, strict=True):
        torch.testing.assert_close(
            actual["conv_pool"], expected["conv_pool"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            actual["h_pool"], expected["h_pool"], atol=1e-6, rtol=0
        )


def _canonical(page):
    """View a V-major state page as the ``[HV, K, V]`` the reference builds."""
    return page.transpose(-1, -2)


def _reference(x, n, t, accepted):
    """Independent recurrence from the rounded serving producer inputs.

    Producer accuracy is checked by test_verify_derived_records_with_serving_producers.
    Reusing its inputs here isolates accepted-state recovery from producer FMA
    differences that can round to adjacent BF16 values.
    """
    from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
        fused_kda_verify_conv_update,
    )

    windows, states, outputs, corrections = [], [], [], []
    conv_qkv = fused_kda_verify_conv_update(
        x["qkv_raw"],
        x["conv_w"],
        x["conv_pool"],
        x["read_indices"],
        num_heads=HV,
        head_dim=K,
        draft_token_num=t,
    ).float()
    raw_gate = torch.mm(x["f_a"], x["w_fb"].t()).float()
    for request in range(n):
        page = int(x["read_indices"][request])
        window = (
            x["conv_pool"][page, : 3 * P].float().clone()
            if page >= 0
            else torch.zeros(3 * P, 3, device=DEV)
        )
        state = (
            x["h_pool"][page, :HV].float().clone()
            if page >= 0
            else torch.zeros(HV, V, K, device=DEV)
        )
        request_outputs, request_corrections = [], []
        for step in range(int(accepted[request])):
            token = request * t + step
            raw = x["qkv_raw"][token, : 3 * P].float()
            convolved = conv_qkv[token]
            query = convolved[:P].view(HV, K)
            query *= torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6) * K**-0.5
            key = convolved[P : 2 * P].view(HV, K)
            key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
            value = convolved[2 * P :].view(HV, V)
            gate_input = (raw_gate[token] + x["dt_bias"]).view(HV, K)
            decay = (
                LOWER_BOUND * torch.sigmoid(x["A_log"].exp()[:, None] * gate_input)
            ).exp()
            state *= decay[:, None, :]
            correction = value - torch.einsum("hvk,hk->hv", state, key)
            correction *= x["beta"][token].float().sigmoid()[:, None]
            state += correction[:, :, None] * key[:, None, :]
            request_outputs.append(
                torch.einsum("hvk,hk->hv", state, query).to(torch.bfloat16)
            )
            request_corrections.append(correction.clone())
            window = torch.cat((window[:, 1:], raw[:, None]), -1)
        windows.append(window)
        states.append(state.transpose(-1, -2))
        outputs.append(request_outputs)
        corrections.append(request_corrections)
    return windows, states, outputs, corrections


def _check_against_reference(x, out, write_indices, n, t, accepted):
    windows, states, _, _ = _reference(x, n, t, accepted)
    for i in range(n):
        w = int(write_indices[i])
        torch.testing.assert_close(
            out["conv_pool"][w].float(), windows[i], atol=0.0, rtol=0.0
        )
        # Producers round to BF16; recurrence and saved correction stay FP32.
        torch.testing.assert_close(
            _canonical(out["h_pool"][w]), states[i], atol=2e-6, rtol=2e-5
        )


@pytest.mark.parametrize("t", [1, 2, 3, 5])
def test_replay_matches_sequential_decode_at_every_accepted_length(t):
    """Invariant 1, swept over every acceptable length including 0 and T."""
    n = 6
    x = _window(n, t, seed=t)
    fresh = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 16
    for a in range(t + 1):
        accepted = torch.full((n,), a, device=DEV, dtype=torch.int32)
        out = _replay(x, fresh, accepted, t)
        _check_against_reference(x, out, fresh, n, t, accepted)


def test_rejected_suffix_cannot_reach_the_committed_state():
    """Invariant 2: perturbing the rejected tail changes nothing, bitwise."""
    n, t = 5, 4
    x = _window(n, t, seed=7)
    fresh = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 16
    accepted = torch.tensor([0, 1, 2, 3, 4], device=DEV, dtype=torch.int32)
    base = _replay(x, fresh, accepted, t)

    perturbed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
    for i in range(n):
        for step in range(int(accepted[i]), t):
            tok = i * t + step
            perturbed["qkv_raw"][tok].normal_()
            perturbed["f_a"][tok].normal_()
            perturbed["beta"][tok].normal_()
    other = _replay(perturbed, fresh, accepted, t)

    for i in range(n):
        w = int(fresh[i])
        torch.testing.assert_close(
            other["conv_pool"][w], base["conv_pool"][w], atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            other["h_pool"][w], base["h_pool"][w], atol=0.0, rtol=0.0
        )


def test_zero_accepted_commits_the_pre_draft_state_unchanged():
    """Invariant 3: an all-rejected window still has to move the state.

    The destination page can differ from the source, so ``a = 0`` is a real
    commit of the unchanged state, not a no-op the caller may skip.
    """
    n, t = 4, 3
    x = _window(n, t, seed=11)
    fresh = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 16
    out = _replay(x, fresh, torch.zeros(n, device=DEV, dtype=torch.int32), t)
    for i in range(n):
        r, w = int(x["read_indices"][i]), int(fresh[i])
        torch.testing.assert_close(
            out["conv_pool"][w], x["conv_pool"][r], atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(out["h_pool"][w], x["h_pool"][r], atol=0.0, rtol=0.0)


def test_mixed_accepted_lengths_in_one_batch():
    """Spec 8.3: per-request lengths, including 0 and T, and a skipped row."""
    n, t = 6, 4
    x = _window(n, t, seed=13)
    fresh = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 16
    accepted = torch.tensor([0, 4, 2, 1, 3, 4], device=DEV, dtype=torch.int32)
    out = _replay(x, fresh, accepted, t)
    _check_against_reference(x, out, fresh, n, t, accepted)

    # A negative destination (CUDA-graph padding) must leave the pools alone.
    padded = fresh.clone()
    padded[2] = -1
    out_padded = _replay(x, padded, accepted, t)
    w = int(fresh[2])
    torch.testing.assert_close(
        out_padded["conv_pool"][w], x["conv_pool"][w], atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        out_padded["h_pool"][w], x["h_pool"][w], atol=0.0, rtol=0.0
    )


def test_accepted_length_clamps_at_and_past_the_window_boundary():
    """Boundary pins: a = T is exact, a > T and a < 0 clamp bitwise to T / 0.

    The host entry clamps ``accepted_length`` to ``[0, T]`` before both the
    recurrence and the conv-window launches; over- and under-range values
    must therefore be indistinguishable from the boundary itself, bit for
    bit, in both pools.
    """
    n, t = 4, 3
    x = _window(n, t, seed=29)
    fresh = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 16
    at = _replay(x, fresh, torch.full((n,), t, device=DEV, dtype=torch.int32), t)
    past = _replay(x, fresh, torch.full((n,), t + 5, device=DEV, dtype=torch.int32), t)
    torch.testing.assert_close(past["conv_pool"], at["conv_pool"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(past["h_pool"], at["h_pool"], atol=0.0, rtol=0.0)
    zero = _replay(x, fresh, torch.zeros(n, device=DEV, dtype=torch.int32), t)
    neg = _replay(x, fresh, torch.full((n,), -2, device=DEV, dtype=torch.int32), t)
    torch.testing.assert_close(neg["conv_pool"], zero["conv_pool"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(neg["h_pool"], zero["h_pool"], atol=0.0, rtol=0.0)


def test_fresh_request_zero_accept_commits_zero_state():
    """A fresh request's zero checkpoint is copied even with zero acceptance.

    The scheduler allocates and zeros a real source page for a fresh request.
    Negative source pages identify padding and do not request state recovery.
    """
    n, t = 3, 2
    x = _window(n, t, seed=31)
    x["read_indices"] = torch.ones(n, device=DEV, dtype=torch.int32)
    x["conv_pool"][1].zero_()
    x["h_pool"][1].zero_()
    fresh = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 16
    out = _replay(x, fresh, torch.zeros(n, device=DEV, dtype=torch.int32), t)
    for i in range(n):
        w = int(fresh[i])
        assert (out["conv_pool"][w].float() == 0).all()
        assert (out["h_pool"][w] == 0).all()


def test_in_place_commit_full_pool_accounting():
    """Torn or stray writes: account for every page in the pool, bit for bit.

    First, a = 0 in place: every program must store exactly the bits it read,
    so the whole pool is a fixed point -- any program writing another
    program's [BK, BV] slice (or any out-of-bounds store) moves random
    sentinel bits somewhere detectable. Then a mixed-length in-place commit:
    only the n anchor pages may change, and each must equal the out-of-place
    result; all other pages (including the out-of-place run's targets) must
    survive untouched.
    """
    n, t, pages = 16, 4, 40
    x = _window(n, t, pages=pages, seed=37)
    zero = torch.zeros(n, device=DEV, dtype=torch.int32)
    for _ in range(8):  # a race or torn write would be intermittent
        out = _replay(x, x["read_indices"], zero, t)
        torch.testing.assert_close(out["conv_pool"], x["conv_pool"], atol=0.0, rtol=0.0)
        torch.testing.assert_close(out["h_pool"], x["h_pool"], atol=0.0, rtol=0.0)

    accepted = torch.arange(n, device=DEV, dtype=torch.int32) % (t + 1)
    fresh = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 20
    expected = _replay(x, fresh, accepted, t)
    for _ in range(8):
        got = _replay(x, x["read_indices"], accepted, t)
        for p in range(pages):
            if 1 <= p <= n:
                # Anchor page of request p-1: must hold the committed state.
                ref_c = expected["conv_pool"][p + 20]
                ref_h = expected["h_pool"][p + 20]
            else:
                # Every other page must be untouched, bit for bit.
                ref_c = x["conv_pool"][p]
                ref_h = x["h_pool"][p]
            torch.testing.assert_close(got["conv_pool"][p], ref_c, atol=0.0, rtol=0.0)
            torch.testing.assert_close(got["h_pool"][p], ref_h, atol=0.0, rtol=0.0)


@requires_registered_replay
def test_replay_commit_probe_tracks_dtype():
    """The capability probe must use the actual activation dtype."""
    from tokenspeed_kernel.ops.attention.kda import kda_replay_commit_supported

    assert not kda_replay_commit_supported(torch.float32)
    assert kda_replay_commit_supported(torch.bfloat16)


@requires_registered_replay
def test_replay_probe_requires_both_commit_and_fused_verify_kernels():
    """Eager replay has no decomposed fallback: either kernel missing means
    unsupported."""
    from unittest import mock

    import tokenspeed_kernel.ops.attention.kda as attention_ops
    from tokenspeed_kernel.selection import NoKernelFoundError

    real = attention_ops.select_kernel

    def missing(mode_to_drop):
        def probe(family, mode, *args, **kwargs):
            if mode == mode_to_drop:
                raise NoKernelFoundError(f"no {mode} on this platform")
            return real(family, mode, *args, **kwargs)

        return probe

    for mode in ("kda_fused_paged_verify", "kda_replay_commit"):
        with mock.patch.object(attention_ops, "select_kernel", missing(mode)):
            assert not attention_ops.kda_replay_commit_supported(torch.bfloat16)


@requires_registered_replay
def test_fused_verify_no_store_matches_store_and_leaves_tape_untouched():
    """The inline-producer no-store fusion returns the store output, sans tape.

    Selection prefers the split-producer variant for this trait set, so the
    twin is named directly: the point here is that dropping the tape does not
    disturb the recurrence, not which producers ran.
    """
    from tokenspeed_kernel.ops.attention.kda.triton import (
        triton_nvidia_kda_fused_paged_verify_no_store as kda_fused_paged_verify,
    )

    n, t, rows = 2, 3, 12
    x = _window(n, t, seed=47)
    writes = torch.arange(n * t, device=DEV, dtype=torch.int32).view(n, t)
    conv_tape = torch.randn(rows, 3 * P, 3, device=DEV, dtype=torch.bfloat16)
    state_tape = torch.randn(rows, HV, K, V, device=DEV, dtype=torch.float32)
    conv_before, state_before = conv_tape.clone(), state_tape.clone()

    no_store = kda_fused_paged_verify(
        x["qkv_raw"],
        x["conv_w"],
        x["conv_pool"],
        conv_tape,
        x["f_a"],
        x["w_fb"],
        x["beta"],
        x["A_log"],
        x["dt_bias"],
        state_pool=x["h_pool"],
        state_scratch=state_tape,
        read_indices=x["read_indices"],
        write_indices=writes,
        num_heads=HV,
        head_dim=V,
        draft_token_num=t,
        lower_bound=LOWER_BOUND,
    )
    torch.testing.assert_close(conv_tape, conv_before, atol=0.0, rtol=0.0)
    torch.testing.assert_close(state_tape, state_before, atol=0.0, rtol=0.0)

    stored = fused_recurrent_kda_verify_megafuse(
        x["qkv_raw"],
        x["conv_w"],
        x["conv_pool"],
        conv_tape,
        x["f_a"],
        x["w_fb"],
        x["beta"],
        x["A_log"],
        x["dt_bias"],
        x["h_pool"],
        state_tape,
        x["read_indices"],
        writes,
        num_heads=HV,
        head_dim=V,
        draft_token_num=t,
        lower_bound=LOWER_BOUND,
        store_states=True,
        enable_pdl=False,
    ).view(1, -1, HV, V)
    torch.testing.assert_close(no_store, stored, atol=0.0, rtol=0.0)
    assert not torch.equal(conv_tape, conv_before)
    assert not torch.equal(state_tape, state_before)

    # Negative control: the output oracle must notice a wrong committed base.
    corrupted = {**x, "h_pool": x["h_pool"].clone()}
    corrupted["h_pool"][x["read_indices"].long()] += 1
    wrong = kda_fused_paged_verify(
        corrupted["qkv_raw"],
        corrupted["conv_w"],
        corrupted["conv_pool"],
        conv_before,
        corrupted["f_a"],
        corrupted["w_fb"],
        corrupted["beta"],
        corrupted["A_log"],
        corrupted["dt_bias"],
        state_pool=corrupted["h_pool"],
        state_scratch=state_before,
        read_indices=corrupted["read_indices"],
        write_indices=writes,
        num_heads=HV,
        head_dim=V,
        draft_token_num=t,
        lower_bound=LOWER_BOUND,
    )
    assert not torch.equal(wrong, no_store)


@requires_registered_replay
@pytest.mark.parametrize("n", [1, 4])
def test_split_verify_wrapper_matches_fused_wrapper(n):
    from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
        fused_kda_verify_conv_update,
    )
    from tokenspeed_kernel.ops.attention.kda.triton import (
        triton_nvidia_kda_fused_paged_verify_no_store,
        triton_nvidia_kda_fused_paged_verify_split,
    )

    t = 3
    x = _window(n, t, seed=49 + n)
    writes = torch.arange(n * t, device=DEV, dtype=torch.int32).view(n, t)
    conv_scratch = torch.empty(n * t, 3 * P, 3, device=DEV, dtype=torch.bfloat16)
    state_scratch = torch.empty(n * t, HV, K, V, device=DEV, dtype=torch.float32)
    kwargs = dict(
        state_pool=x["h_pool"],
        state_scratch=state_scratch,
        read_indices=x["read_indices"],
        write_indices=writes,
        num_heads=HV,
        head_dim=V,
        draft_token_num=t,
        lower_bound=LOWER_BOUND,
    )
    args = (
        x["qkv_raw"],
        x["conv_w"],
        x["conv_pool"],
        conv_scratch,
        x["f_a"],
        x["w_fb"],
        x["beta"],
        x["A_log"],
        x["dt_bias"],
    )
    fused = triton_nvidia_kda_fused_paged_verify_no_store(*args, **kwargs)
    split = triton_nvidia_kda_fused_paged_verify_split(
        *args, **kwargs, replay_records=None
    )
    conv_qkv = fused_kda_verify_conv_update(
        x["qkv_raw"],
        x["conv_w"],
        x["conv_pool"],
        x["read_indices"],
        num_heads=HV,
        head_dim=V,
        draft_token_num=t,
    )
    g_raw = torch.mm(x["f_a"], x["w_fb"].t())
    precomputed = triton_nvidia_kda_fused_paged_verify_split(
        *args, **kwargs, g_raw=g_raw, conv_qkv=conv_qkv, replay_records=None
    )
    torch.testing.assert_close(split, fused, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(precomputed, split, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not current_platform().is_hopper_plus, reason="PDL requires SM90+")
def test_verify_to_gated_norm_pdl_chain_matches_serial():
    """The early verify trigger remains fenced before gated-norm reads."""
    from tokenspeed_kernel.ops.activation.triton import rmsnorm_gated_sigmoid

    n, t = 4, 4
    x = _window(n, t, seed=57)
    writes = torch.arange(n * t, device=DEV, dtype=torch.int32).view(n, t)
    gate = torch.randn(n * t, P, device=DEV, dtype=torch.bfloat16)
    weight = torch.randn(V, device=DEV, dtype=torch.bfloat16)

    def run(enable_pdl: bool) -> torch.Tensor:
        out = fused_recurrent_kda_verify_megafuse(
            x["qkv_raw"],
            x["conv_w"],
            x["conv_pool"],
            x["conv_pool"],
            x["f_a"],
            x["w_fb"],
            x["beta"],
            x["A_log"],
            x["dt_bias"],
            x["h_pool"],
            x["h_pool"],
            x["read_indices"],
            writes,
            num_heads=HV,
            head_dim=V,
            draft_token_num=t,
            lower_bound=LOWER_BOUND,
            store_states=False,
            enable_pdl=enable_pdl,
        )
        return rmsnorm_gated_sigmoid(
            out.reshape(-1, P),
            gate,
            weight,
            1e-6,
            HV,
            V,
            enable_pdl=enable_pdl,
        )

    serial = run(False)
    pdl = run(True)
    torch.testing.assert_close(pdl, serial, atol=0.0, rtol=0.0)


def test_fused_verify_default_store_is_bit_identical_to_explicit_trait():
    """The existing tape-writing API stays bitwise identical by default."""
    x = _window(2, 3, seed=53)
    writes = torch.arange(6, device=DEV, dtype=torch.int32).view(2, 3)
    tapes = []
    outputs = []
    for kwargs in ({}, {"store_states": True}):
        conv = torch.zeros(8, 3 * P, 3, device=DEV, dtype=torch.bfloat16)
        state = torch.zeros(8, HV, K, V, device=DEV, dtype=torch.float32)
        outputs.append(
            fused_recurrent_kda_verify_megafuse(
                x["qkv_raw"],
                x["conv_w"],
                x["conv_pool"],
                conv,
                x["f_a"],
                x["w_fb"],
                x["beta"],
                x["A_log"],
                x["dt_bias"],
                x["h_pool"],
                state,
                x["read_indices"],
                writes,
                num_heads=HV,
                head_dim=V,
                draft_token_num=3,
                lower_bound=LOWER_BOUND,
                enable_pdl=False,
                **kwargs,
            )
        )
        tapes.append((conv, state))
    torch.testing.assert_close(outputs[0], outputs[1], atol=0.0, rtol=0.0)
    torch.testing.assert_close(tapes[0][0], tapes[1][0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(tapes[0][1], tapes[1][1], atol=0.0, rtol=0.0)


@pytest.mark.parametrize("n,t", [(4, 3), (8, 4), (16, 4)])
def test_fused_verify_routed_bv_matches_bv32(n, t):
    """The routed wide tiles (64/128) must agree with BV=32 to one bf16 ulp.

    Each program owns its V rows outright, so tile width can only reorder
    the fp32 K reduction — a one-ulp wobble after the bf16 store. A masking
    or boundary bug shifts whole rows, far beyond that.
    """
    x = _window(n, t, seed=61)
    writes = torch.arange(n * t, device=DEV, dtype=torch.int32).view(n, t)
    results = []
    for bv in (32, None):
        out = fused_recurrent_kda_verify_megafuse(
            x["qkv_raw"],
            x["conv_w"],
            x["conv_pool"],
            x["conv_pool"],
            x["f_a"],
            x["w_fb"],
            x["beta"],
            x["A_log"],
            x["dt_bias"],
            x["h_pool"],
            x["h_pool"],
            x["read_indices"],
            writes,
            num_heads=HV,
            head_dim=V,
            draft_token_num=t,
            lower_bound=LOWER_BOUND,
            store_states=False,
            bv=bv,
            enable_pdl=False,
        )
        results.append(out)
    torch.testing.assert_close(results[0], results[1], atol=1e-5, rtol=8e-3)


def test_fused_verify_store_follows_permuted_write_rows():
    """Stored tapes land at write_indices[tok], not at the token index."""
    n, t = 2, 3
    x = _window(n, t, seed=63)
    rows = n * t
    perm = torch.randperm(rows, generator=torch.Generator().manual_seed(5))
    writes = perm.to(device=DEV, dtype=torch.int32).view(n, t)
    identity = torch.arange(rows, device=DEV, dtype=torch.int32).view(n, t)
    outs, convs, states = [], [], []
    for w in (identity, writes):
        conv = torch.zeros(rows, 3 * P, 3, device=DEV, dtype=torch.bfloat16)
        state = torch.zeros(rows, HV, K, V, device=DEV, dtype=torch.float32)
        outs.append(
            fused_recurrent_kda_verify_megafuse(
                x["qkv_raw"],
                x["conv_w"],
                x["conv_pool"],
                conv,
                x["f_a"],
                x["w_fb"],
                x["beta"],
                x["A_log"],
                x["dt_bias"],
                x["h_pool"],
                state,
                x["read_indices"],
                w,
                num_heads=HV,
                head_dim=V,
                draft_token_num=t,
                lower_bound=LOWER_BOUND,
                store_states=True,
                enable_pdl=False,
            )
        )
        convs.append(conv)
        states.append(state)
    torch.testing.assert_close(outs[0], outs[1], atol=0.0, rtol=0.0)
    torch.testing.assert_close(convs[1][perm], convs[0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(states[1][perm], states[0], atol=0.0, rtol=0.0)


def test_fused_verify_rejects_non_power_of_two_bv():
    x = _window(1, 3, seed=62)
    writes = torch.arange(3, device=DEV, dtype=torch.int32).view(1, 3)
    conv = torch.zeros(3, 3 * P, 3, device=DEV, dtype=torch.bfloat16)
    state = torch.zeros(3, HV, K, V, device=DEV, dtype=torch.float32)
    with pytest.raises(ValueError, match="positive power of two"):
        fused_recurrent_kda_verify_megafuse(
            x["qkv_raw"],
            x["conv_w"],
            x["conv_pool"],
            conv,
            x["f_a"],
            x["w_fb"],
            x["beta"],
            x["A_log"],
            x["dt_bias"],
            x["h_pool"],
            state,
            x["read_indices"],
            writes,
            num_heads=HV,
            head_dim=V,
            draft_token_num=3,
            lower_bound=LOWER_BOUND,
            bv=48,
            enable_pdl=False,
        )
