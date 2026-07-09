# Copyright (c) 2026 LightSeek Foundation

"""Routing tests for the fused-mHC allinone/composed token threshold."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

# CI registration (AST-parsed, runtime no-op). Routing logic is mocked and
# GPU-free, but the tokenspeed import graph requires a CUDA environment.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import torch

from tokenspeed.runtime.layers import deepseek_v4_mhc as mhc

HC, HID = 4, 64


def _inputs(m):
    x = torch.randn(m, HID, dtype=torch.bfloat16)
    res = torch.randn(m, HC, HID, dtype=torch.bfloat16)
    post = torch.rand(m, HC, 1)
    comb = torch.rand(m, HC, HC)
    fn = torch.randn((2 + HC) * HC, HC * HID)
    return x, res, post, comb, fn, torch.ones(3), torch.zeros((2 + HC) * HC)


def _call(m):
    x, res, post, comb, fn, scale, base = _inputs(m)
    return mhc._trtllm_mhc_fused_hc(
        x, res, post, comb, fn, scale, base, 1e-6, 1e-2, 4, mhc.MhcFusedWorkspace()
    )


def test_fused_hc_routes_large_tokens_to_composed():
    mhc._MHC_FUSED_ROUTING_LOGGED.clear()
    m = mhc._MHC_FUSED_ALLINONE_MAX_TOKENS + 1
    sentinel_res = torch.zeros(m, HC, HID, dtype=torch.bfloat16)
    with (
        patch.object(mhc, "_trtllm_mhc_post", return_value=sentinel_res) as post_fn,
        patch.object(
            mhc,
            "_trtllm_mhc_pre",
            return_value=(
                torch.zeros(m, HID, dtype=torch.bfloat16),
                torch.zeros(m, HC, 1),
                torch.zeros(m, HC, HC),
            ),
        ) as pre_fn,
        patch.object(mhc, "trtllm_mhc_fused_hc") as allinone,
    ):
        out = _call(m)
    post_fn.assert_called_once()
    pre_fn.assert_called_once()
    allinone.assert_not_called()
    assert out[0] is sentinel_res


def test_fused_hc_keeps_allinone_at_threshold_and_below():
    mhc._MHC_FUSED_ROUTING_LOGGED.clear()
    for m in (1, mhc._MHC_FUSED_ALLINONE_MAX_TOKENS):
        with (
            patch.object(mhc, "trtllm_mhc_fused_hc") as allinone,
            patch.object(mhc, "_trtllm_mhc_post") as post_fn,
            patch.object(mhc.MhcFusedWorkspace, "get") as ws_get,
        ):
            buf = lambda *s, dt=torch.bfloat16: torch.zeros(*s, dtype=dt)  # noqa: E731
            ws_get.return_value.get.return_value = (
                buf(m, HC, HID),
                buf(m, HC, 1, dt=torch.float32),
                buf(m, HC, HC, dt=torch.float32),
                buf(m, HID),
                buf(1, m, (2 + HC) * HC, dt=torch.float32),
                buf(1, m, dt=torch.float32),
                buf(1, dt=torch.int32),
            )
            _call(m)
        allinone.assert_called_once()
        post_fn.assert_not_called()


def test_fused_hc_empty_batch_short_circuits():
    out = _call(0)
    assert out[0].shape == (0, HC, HID)


def test_fused_hc_routing_logs_once_per_shape(caplog):
    mhc._MHC_FUSED_ROUTING_LOGGED.clear()
    m = mhc._MHC_FUSED_ALLINONE_MAX_TOKENS + 8
    with (
        patch.object(mhc, "_trtllm_mhc_post", return_value=torch.zeros(m, HC, HID)),
        patch.object(
            mhc,
            "_trtllm_mhc_pre",
            return_value=(
                torch.zeros(m, HID),
                torch.zeros(m, HC, 1),
                torch.zeros(m, HC, HC),
            ),
        ),
        caplog.at_level("INFO", logger=mhc.logger.name),
    ):
        _call(m)
        _call(m)
    routing_logs = [r for r in caplog.records if "fused_hc routing" in r.getMessage()]
    assert len(routing_logs) == 1
