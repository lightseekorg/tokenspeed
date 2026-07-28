# Copyright (c) 2026 LightSeek Foundation
#
# Regression test for the fused-output-norm DSM defects in the CuTe DSL KDA
# decode kernel (dev/decode-k3 dark corner, serve_onorm.log IMA):
#
#   1. Illegal distributed-shared-memory write: the epilogue's
#      ``st.shared::cluster`` of the o^2 partial into the sibling CTA ran
#      before any completed cluster-barrier phase, so it could target a CTA
#      that had not begun execution (compute-sanitizer: "Potential invalid
#      __shared__ write ... block that might not have entered yet") —
#      a timing-dependent cudaErrorIllegalAddress inside every captured
#      decode graph.
#   2. Cross-CTA WAR race: the sibling's partial landed in ``s_norm[CPL]``,
#      the same word phase 2 reads for the q L2 normalizer (racecheck WAR).
#
# The test runs the onorm kernel (nv=2 band: bs 1 and 3) under
# ``compute-sanitizer --tool racecheck`` in a subprocess and requires zero
# hazards AND zero invalid-shared-write reports. Skips when the sanitizer
# binary or a GPU is unavailable.
#
# Intended location:
#   tokenspeed-kernel/test/ops/attention/test_kda_onorm_dsm_regression.py

import shutil
import subprocess
import sys

import pytest
import torch

_SANITIZER_CANDIDATES = (
    shutil.which("compute-sanitizer"),
    "/usr/local/cuda/bin/compute-sanitizer",
)

_TARGET = r"""
import sys
import torch
from tokenspeed_kernel.ops.attention.cute_dsl.kda_fused_decode import (
    cutedsl_fused_recurrent_kda_megafuse,
)
from tokenspeed_kernel.ops.attention.kda_utils import KdaGatedNormRequest

torch.manual_seed(0)
dev = "cuda"
HV, K, V = 12, 128, 128
P = HV * K
D_FA = 128
PAGES = 8
MERGED = 3 * P + HV + D_FA + P

for bs in (1, 3):  # nv=2 band (cluster of 2 with the DSM exchange)
    merged = torch.randn(bs, MERGED, dtype=torch.bfloat16, device=dev) * 0.3
    qkv = merged[:, : 3 * P]
    beta = merged[:, 3 * P : 3 * P + HV]
    fa = merged[:, 3 * P + HV : 3 * P + HV + D_FA]
    gate = merged[:, 3 * P + HV + D_FA :]
    conv_pool = torch.randn(PAGES, 3 * P, 3, dtype=torch.bfloat16, device=dev) * 0.05
    h_pool = torch.randn(PAGES, HV, K, V, dtype=torch.float32, device=dev) * 0.05
    conv_w = torch.randn(3 * P, 4, dtype=torch.bfloat16, device=dev) * 0.1
    w_fb = torch.randn(P, D_FA, dtype=torch.bfloat16, device=dev) * 0.05
    A_log = torch.randn(HV, dtype=torch.float32, device=dev) * 0.1
    dt_bias = torch.randn(P, dtype=torch.float32, device=dev) * 0.1
    norm_w = torch.rand(V, dtype=torch.bfloat16, device=dev) + 0.5
    ri = torch.arange(bs, dtype=torch.int32, device=dev)
    wi = ri.clone()
    wi[-1] = bs % PAGES  # boundary crossing on the last sequence
    cu = torch.arange(bs + 1, dtype=torch.int32, device=dev)
    for _ in range(3):
        req = KdaGatedNormRequest(weight=norm_w, gate=gate, eps=1e-6)
        cutedsl_fused_recurrent_kda_megafuse(
            qkv, conv_w, conv_pool, fa, w_fb, beta, A_log, dt_bias, h_pool,
            ri, wi, num_heads=HV, head_dim=K, cu_seqlens=cu,
            lower_bound=-5.0, onorm=req,
        )
        assert req.consumed
    torch.cuda.synchronize()
print("SANIT TARGET DONE")
"""


def _sanitizer():
    for c in _SANITIZER_CANDIDATES:
        if c and shutil.which(c):
            return c
    return None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.skipif(_sanitizer() is None, reason="compute-sanitizer unavailable")
def test_onorm_dsm_exchange_is_race_and_hazard_free():
    out = subprocess.run(
        [_sanitizer(), "--tool", "racecheck", sys.executable, "-c", _TARGET],
        capture_output=True,
        text=True,
        timeout=900,
    )
    text = out.stdout + out.stderr
    assert "SANIT TARGET DONE" in text, f"target did not complete:\n{text[-4000:]}"
    # The pre-fix kernel reported ~300 WAR hazards on s_norm[CPL] plus
    # "invalid __shared__ write ... might not have entered yet" for the DSM
    # store; both must stay at zero.
    assert "might not have entered" not in text, text[-4000:]
    assert "RACECHECK SUMMARY: 0 hazards" in text, text[-4000:]
