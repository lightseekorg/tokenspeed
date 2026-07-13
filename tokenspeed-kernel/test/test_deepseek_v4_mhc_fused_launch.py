# Copyright (c) 2026 LightSeek Foundation

"""Launch-table and GPU parity tests for the fused mHC backend selection.

The launch table was measured on B200 (768-combination sweep, graph-replay
timed, parity-checked against the composed two-stage path). These tests pin
its central constraint: backend=1 (fma_ksplit + big_fuse) writes
``num_k_splits * M`` partial rows into the y_acc/r_acc accumulator
workspaces, so they must be sized ``FUSED_HC_MAX_K_SPLITS x max_tokens``.
With M-row accumulators the spill corrupts small-M outputs and crashed a
production p8 run with ``cudaErrorIllegalAddress``.
"""

import pytest
import torch
from tokenspeed_kernel.ops.mhc.trtllm import (
    FUSED_HC_MAX_K_SPLITS,
    _select_fused_hc_launch,
)

HC, HID = 4, 7168
MIX = (2 + HC) * HC
RMS_EPS, HC_EPS, SINK = 1e-6, 1e-2, 4
REL_TOL = 0.03


def _is_sm100() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    return torch.cuda.get_device_capability(0) == (10, 0)


class TestSelectFusedHcLaunch:
    def test_small_and_medium_m_use_ksplit_tile2(self):
        # The two-stage path wins at every measured M >= 1 with
        # contract-sized workspaces (the earlier small-M gate was an
        # artifact of the workspace undersizing).
        for m in (1, 2, 4, 8, 12, 13, 16, 24, 32):
            assert _select_fused_hc_launch(m) == (1, 2, 2, 512)

    def test_large_m_uses_ksplit_tile4(self):
        for m in (33, 36, 48, 64, 128):
            assert _select_fused_hc_launch(m) == (1, 4, 2, 512)

    def test_cliff_backend_never_selected(self):
        # backend=2 (allinone tf32 mma) measured 206-241us at every M on
        # B200 -- the launch table must never dispatch to it.
        for m in range(1, 129):
            assert _select_fused_hc_launch(m)[0] != 2

    def test_k_splits_bounded_by_workspace_contract(self):
        for m in range(1, 129):
            assert _select_fused_hc_launch(m)[2] <= FUSED_HC_MAX_K_SPLITS


def _make_inputs(m, dev):
    torch.manual_seed(0)
    x = torch.randn(m, HID, device=dev, dtype=torch.bfloat16)
    res = torch.randn(m, HC, HID, device=dev, dtype=torch.bfloat16) * 0.5
    post = torch.rand(m, HC, device=dev, dtype=torch.float32)
    comb = torch.rand(m, HC, HC, device=dev, dtype=torch.float32) / HC
    fn = torch.randn(MIX, HC * HID, device=dev, dtype=torch.float32) * 0.02
    scale = torch.ones(3, device=dev, dtype=torch.float32)
    base = torch.zeros(MIX, device=dev, dtype=torch.float32)
    return x, res, post, comb, fn, scale, base


def _raw_fused_hc(inputs, m, backend, tile_n, ksp, bfbs=0, acc_rows=None):
    """Call the raw op with explicit launch params; return outputs and acc."""

    x, res, post, comb, fn, scale, base = inputs
    dev = x.device
    rows = acc_rows if acc_rows is not None else FUSED_HC_MAX_K_SPLITS * m
    shape_n = HC * (2 + HC)
    residual_cur = torch.empty(m, HC, HID, dtype=torch.bfloat16, device=dev)
    post_cur = torch.empty(m, HC, dtype=torch.float32, device=dev)
    comb_cur = torch.empty(m, HC * HC, dtype=torch.float32, device=dev)
    layer_input = torch.empty(m, HID, dtype=torch.bfloat16, device=dev)
    y_acc = torch.full((rows, shape_n), 12345.678, dtype=torch.float32, device=dev)
    r_acc = torch.full((rows,), 12345.678, dtype=torch.float32, device=dev)
    dc = torch.zeros(rows, dtype=torch.int32, device=dev)
    torch.ops.trtllm.mhc_fused_hc(
        x,
        res,
        post,
        comb,
        fn,
        scale,
        base,
        residual_cur,
        post_cur,
        comb_cur,
        layer_input,
        y_acc,
        r_acc,
        dc,
        m,
        HID,
        HC,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        2.0,
        SINK,
        backend,
        tile_n,
        ksp,
        bfbs,
        1,
        None,
        0.0,
    )
    torch.cuda.synchronize()
    return (residual_cur, layer_input, post_cur, comb_cur), (y_acc, r_acc)


def _assert_close(got, ref):
    for g, r in zip(got, ref):
        g = g.float()
        r = r.float()
        denom = r.abs().max().clamp_min(1e-6)
        rel = ((g - r).abs().max() / denom).item()
        assert rel <= REL_TOL, f"rel err {rel} > {REL_TOL}"


@pytest.mark.skipif(not _is_sm100(), reason="TRT-LLM mHC kernels need sm100")
class TestFusedHcLaunchTableParityGpu:
    @pytest.fixture(autouse=True)
    def _load_kernels(self):
        pytest.importorskip("trtllm_kernel")
        from tokenspeed_kernel.ops.mhc.trtllm import has_trtllm_mhc

        if not has_trtllm_mhc():
            pytest.skip("trtllm mHC kernels unavailable")

    @pytest.mark.parametrize("m", [1, 4, 8, 13, 16, 24, 32, 36, 48, 64])
    def test_selected_launch_matches_allinone_reference(self, m):
        # The allinone-fma backend (3) is the numerically trusted incumbent
        # at every M; the sweep-selected launch must agree with it.
        inputs = _make_inputs(m, "cuda:0")
        ref, _ = _raw_fused_hc(inputs, m, 3, 1, 1)
        backend, tile_n, ksp, bfbs = _select_fused_hc_launch(m)
        got, _ = _raw_fused_hc(inputs, m, backend, tile_n, ksp, bfbs=bfbs)
        _assert_close(got, ref)

    @pytest.mark.parametrize("m", [16, 32])
    def test_ksplit_writes_beyond_m_rows_but_within_contract(self, m):
        # Documents the workspace contract: the k-split backend spills
        # num_k_splits x M partial rows -- more than M (so M-row buffers
        # overflow) but never more than FUSED_HC_MAX_K_SPLITS x M.
        inputs = _make_inputs(m, "cuda:0")
        backend, tile_n, ksp, bfbs = _select_fused_hc_launch(m)
        assert backend == 1
        _, (y_acc, r_acc) = _raw_fused_hc(
            inputs,
            m,
            backend,
            tile_n,
            ksp,
            bfbs=bfbs,
            acc_rows=FUSED_HC_MAX_K_SPLITS * m,
        )
        y_rows = int((y_acc != 12345.678).any(dim=1).sum().item())
        assert y_rows > m
        assert y_rows <= FUSED_HC_MAX_K_SPLITS * m

    @pytest.mark.parametrize("m", list(range(1, 9)))
    def test_small_m_ksplit_parity_with_contract_sized_workspace(self, m):
        # With accumulators sized to the FUSED_HC_MAX_K_SPLITS x M contract,
        # backend=1 matches the default at small M too. (With M-row
        # accumulators its spill corrupts these outputs -- the same
        # undersizing that crashed production at scale.) The launch table
        # still keeps small M on the default backend because it is at least
        # as fast there.
        inputs = _make_inputs(m, "cuda:0")
        ref, _ = _raw_fused_hc(inputs, m, 3, 1, 1)
        got, _ = _raw_fused_hc(inputs, m, 1, 2, 2)
        _assert_close(got, ref)
