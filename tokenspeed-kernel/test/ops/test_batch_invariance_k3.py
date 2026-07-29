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

"""Cross-batch correctness for every batch-size-switched Kimi-K3 decode path.

The K3 stack changes *implementation* at several batch/token-count
thresholds (see ``/scratch/jue/correctness/batch_paths.md`` for the full
inventory). Every threshold is a place where two kernels must agree but
nothing has been asserting that they do. This module asserts three
properties at each threshold:

1. **Reference agreement** -- every batch size matches an fp32 torch
   reference within a per-kernel tolerance documented at the assertion.
2. **Row invariance** -- one logical row produces the same output when it
   runs alone (``B = 1``) and when it runs inside a larger batch at
   several positions (first / middle / last / next to a padded row).
   Where the two sides run the *same* reduction tree we require **bitwise**
   equality; where the tiling legitimately re-associates the reduction we
   require a tight bound and say why in a comment.
3. **Padding isolation** -- CUDA-graph padding rows carry poison (NaN/Inf
   activations, ``-1`` page ids). Real rows must be bit-identical to a run
   without the poison, and no state page may be touched.

Tolerance policy
----------------
``atol``/``rtol`` pairs below are *not* arbitrary. Two regimes are used:

* ``EXACT`` (``atol=rtol=0``) wherever the two code paths evaluate the same
  expression in the same order. The KDA decode kernel's ``nv`` bands are the
  main example: ``nv`` only changes *how many blocks* split the V columns,
  and warp ``w`` always owns K-rows ``[16w, 16w+16)`` in both bands, so the
  per-column reduction tree is identical. A bitwise difference there is a
  bug, not noise.
* A documented bound wherever the reduction is re-associated. The fused
  output RMSNorm is the canonical case: at ``nv=1`` the ``o^2`` sum is
  ``((p0+p1)+p2)+p3`` inside one block, at ``nv=2`` it is
  ``(p0+p1) + (p0'+p1')`` across the DSM exchange. The two differ in the
  last fp32 ulps of the normalizer; after the bf16 output rounding this is
  almost always invisible, so we allow a small ``atol`` *and* additionally
  require the overwhelming majority of elements to be bit-identical.

Runtime
-------
Every GPU test is single-device and sized to stay well under 30 s *after*
the CuTe DSL kernels are compiled. The first test that touches a given
``(nv, onorm)`` combination pays a one-off JIT compile; the
``_warm_kda_configs`` session fixture front-loads those four compiles so no
individual test absorbs them.
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if find_spec("cutlass") is None or find_spec("quack") is None:
    pytest.skip("cutlass DSL + quack required", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("K3 decode stack is validated on SM100 only", allow_module_level=True)

from tokenspeed_kernel.ops.attention.cute_dsl.kda_fused_decode import (  # noqa: E402
    _pick_nv,
    cutedsl_fused_recurrent_kda_megafuse,
)
from tokenspeed_kernel.ops.attention.kda_utils import KdaGatedNormRequest  # noqa: E402

# K3 TP8-rank KDA decode geometry.
HV, K, V = 12, 128, 128
P = HV * K
D_FA = 128
LOWER_BOUND = -5.0

# Batch sizes straddling every KDA threshold:
#   batch * HV >= 96  ->  nv = 1 (bulk-TMA staging, smem-resident pass 2)
#   batch * HV <  96  ->  nv = 2 (cp.async staging, register-resident pass 2)
# At HV = 12 the crossover is B = 8. The remaining sizes bracket the default
# CUDA-graph capture ladder [1, 2, 4, 8, 16, 24, 32, ...] so that every
# "captured vs padded-up" pair is represented.
KDA_BATCHES = [1, 2, 4, 7, 8, 9, 15, 16, 31, 32, 63, 64, 65]

# Positions probed inside a large batch for the row-invariance tests: first,
# an early interior row, a mid row, the row immediately after a padded slot,
# and the last row.
PROBE_POSITIONS = (0, 1, 17, 32, 63)


# ---------------------------------------------------------------------------
# KDA fused decode helpers
# ---------------------------------------------------------------------------


def _kda_inputs(bs: int, *, pages: int, seed: int, device: str = "cuda") -> dict:
    """Independent per-row inputs plus a state pool with one page per row.

    ``ri[n] = n`` and ``wi[n] = pages // 2 + n`` keep read and write pages
    disjoint, so a batched run and a single-row rerun observe exactly the
    same input page and land in exactly the same output page.
    """
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape, dtype=torch.bfloat16, scale=1.0):
        return (
            torch.randn(*shape, generator=g, dtype=torch.float32, device=device) * scale
        ).to(dtype)

    assert pages >= 2 * bs
    return dict(
        qkv=rnd(bs, 3 * P),
        conv_w=rnd(3 * P, 4, scale=0.3),
        conv_pool=rnd(pages, 3 * P, 3, scale=0.5),
        f_a=rnd(bs, D_FA),
        w_fb=rnd(P, D_FA, scale=0.05),
        beta=rnd(bs, HV),
        A_log=rnd(HV, dtype=torch.float32, scale=0.5),
        dt_bias=rnd(P, dtype=torch.float32),
        h_pool=rnd(pages, HV, K, V, dtype=torch.float32, scale=0.5),
        gate=rnd(bs, P),
        norm_w=(torch.rand(V, generator=g, device=device) + 0.5).to(torch.bfloat16),
        ri=torch.arange(bs, dtype=torch.int32, device=device),
        wi=torch.arange(bs, dtype=torch.int32, device=device) + pages // 2,
        cu=torch.arange(bs + 1, dtype=torch.int32, device=device),
    )


def _clone(x: dict) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}


def _run_kda(
    x: dict, *, lower_bound=LOWER_BOUND, cu=True, onorm=None, enable_pdl=False
):
    return cutedsl_fused_recurrent_kda_megafuse(
        x["qkv"],
        x["conv_w"],
        x["conv_pool"],
        x["f_a"],
        x["w_fb"],
        x["beta"],
        x["A_log"],
        x["dt_bias"],
        x["h_pool"],
        x["ri"],
        x["wi"],
        num_heads=HV,
        head_dim=K,
        cu_seqlens=x["cu"] if cu else None,
        lower_bound=lower_bound,
        onorm=onorm,
        enable_pdl=enable_pdl,
    )


def _onorm_request(x: dict, eps: float = 1e-6) -> KdaGatedNormRequest:
    return KdaGatedNormRequest(weight=x["norm_w"], gate=x["gate"], eps=eps)


def _kda_reference(x: dict, *, lower_bound=LOWER_BOUND, apply_onorm=False):
    """fp32 torch reference for one fused decode step; mutates the pools.

    Mirrors ``fused_recurrent_kda_megafuse``: 4-tap causal conv + SiLU, the
    low-rank f_b decay-gate GEMV, QK L2 normalization, and the delta-rule
    rank-1 state update. Negative page ids read zeros / skip stores, as the
    kernel contracts.
    """
    qkv = x["qkv"].float()
    conv_w = x["conv_w"].float()
    f_a = x["f_a"].float()
    w_fb = x["w_fb"].float()
    beta = x["beta"].float()
    dev = qkv.device
    bs = qkv.shape[0]
    out = torch.zeros(bs, HV, V, dtype=torch.float32, device=dev)
    for n in range(bs):
        r = int(x["ri"][n])
        w = int(x["wi"][n])
        window = (
            x["conv_pool"][r].float() if r >= 0 else torch.zeros(3 * P, 3, device=dev)
        )
        xt = qkv[n]
        acc = (window * conv_w[:, :3]).sum(-1) + xt * conv_w[:, 3]
        y = acc * torch.sigmoid(acc)
        if w >= 0:
            new_window = torch.cat([window[:, 1:], xt[:, None]], dim=1)
            x["conv_pool"][w] = new_window.to(x["conv_pool"].dtype)
        q = y[:P].view(HV, K)
        k = y[P : 2 * P].view(HV, K)
        v = y[2 * P :].view(HV, V)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * K**-0.5
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        g = (w_fb @ f_a[n]) + x["dt_bias"]
        g = g.view(HV, K)
        exp_a = torch.exp(x["A_log"])[:, None]
        if lower_bound is not None:
            gk = lower_bound * torch.sigmoid(exp_a * g)
        else:
            gk = -exp_a * torch.where(g < 20.0, torch.log1p(torch.exp(g)), g)
        h = x["h_pool"][r].clone() if r >= 0 else torch.zeros(HV, K, V, device=dev)
        h = h * torch.exp(gk)[:, :, None]
        t = torch.einsum("hkv,hk->hv", h, k)
        v_new = (v - t) * torch.sigmoid(beta[n])[:, None]
        h = h + torch.einsum("hk,hv->hkv", k, v_new)
        out[n] = torch.einsum("hkv,hk->hv", h, q)
        if w >= 0:
            x["h_pool"][w] = h
    if apply_onorm:
        var = (out * out).mean(-1, keepdim=True)
        out = (
            out
            * torch.rsqrt(var + 1e-6)
            * x["norm_w"].float()
            * torch.sigmoid(x["gate"].float()).view(bs, HV, V)
        )
    return out.to(torch.bfloat16)


@pytest.fixture(scope="module", autouse=True)
def _warm_kda_configs():
    """Front-load the CuTe JIT so no individual test absorbs a compile.

    Four CUBINs cover the whole decode surface: ``nv`` in {1, 2} crossed with
    the fused output norm on/off. ``bs = 1`` picks ``nv = 2`` and ``bs = 8``
    picks ``nv = 1`` at ``HV = 12``.
    """
    for bs in (1, 8):
        x = _kda_inputs(bs, pages=2 * bs + 2, seed=1)
        _run_kda(_clone(x))
        y = _clone(x)
        _run_kda(y, onorm=_onorm_request(y))
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# 1. KDA fused decode: reference agreement across the nv threshold
# ---------------------------------------------------------------------------


class TestKdaDecodeAcrossBatch:
    @pytest.mark.parametrize("bs", KDA_BATCHES)
    @pytest.mark.parametrize("onorm", [False, True])
    def test_matches_fp32_reference(self, bs, onorm):
        """Every batch size, both nv bands, against the fp32 reference.

        Tolerances: the kernel accumulates in fp32 and rounds once to bf16,
        so the only slack is bf16 output rounding (~4e-3 relative at unit
        scale) plus fp32 reassociation inside the 128-wide dot products.
        """
        x = _kda_inputs(bs, pages=2 * bs + 2, seed=100 + bs)
        ref = _clone(x)
        req = _onorm_request(x) if onorm else None
        o = _run_kda(x, onorm=req)
        ref_o = _kda_reference(ref, apply_onorm=onorm)
        torch.cuda.synchronize()
        if onorm:
            assert req.consumed, "onorm must fuse for the K3 decode operand shapes"
        torch.testing.assert_close(o.float(), ref_o.float(), atol=6e-3, rtol=1e-2)
        # The state pool is fp32 end to end; only fp32 dot reassociation.
        torch.testing.assert_close(x["h_pool"], ref["h_pool"], atol=4e-3, rtol=1e-3)
        # The conv window shift is a pure bf16 copy: it must be exact.
        torch.testing.assert_close(
            x["conv_pool"].float(), ref["conv_pool"].float(), atol=0.0, rtol=0.0
        )

    def test_pick_nv_threshold_is_where_we_think_it_is(self):
        """Guard the documented crossover so a silent retune is caught here.

        If this fails, the batch sizes in ``KDA_BATCHES`` no longer straddle
        the band boundary and the invariance tests below stop testing the
        thing they claim to test.
        """
        assert [_pick_nv(b, HV) for b in (1, 2, 4, 7)] == [2, 2, 2, 2]
        assert [_pick_nv(b, HV) for b in (8, 9, 16, 64)] == [1, 1, 1, 1]
        # The rule is on batch * heads, not batch alone.
        assert _pick_nv(96, 1) == 1 and _pick_nv(95, 1) == 2


class TestKdaRowInvariance:
    """One row must not care how many neighbours it has.

    ``nv`` only splits the V columns across blocks; warp ``w`` owns K-rows
    ``[16w, 16w+16)`` and the per-column combine runs over warps 0..7 in
    order in *both* bands, so the reduction tree per output element is
    identical. Without the fused norm the two bands must therefore agree
    **bitwise** -- any difference is a real cross-batch bug (a stale smem
    slot, a mis-sized cp.async, a TMA overrun), not float noise.
    """

    @staticmethod
    def _single_row_view(x: dict, j: int) -> dict:
        """A B=1 input that replays row ``j`` against the same pages."""
        one = _clone(x)
        for key in ("qkv", "f_a", "beta", "gate"):
            one[key] = x[key][j : j + 1].clone()
        one["ri"] = x["ri"][j : j + 1].clone()
        one["wi"] = x["wi"][j : j + 1].clone()
        one["cu"] = torch.tensor([0, 1], dtype=torch.int32, device=x["ri"].device)
        return one

    @pytest.mark.parametrize("big_bs", [16, 64])
    def test_row_alone_matches_row_in_batch_bitwise(self, big_bs):
        batched = _kda_inputs(big_bs, pages=2 * big_bs + 2, seed=200 + big_bs)
        pristine = _clone(batched)
        o_batched = _run_kda(batched)
        torch.cuda.synchronize()

        for j in PROBE_POSITIONS:
            if j >= big_bs:
                continue
            one = self._single_row_view(pristine, j)
            o_one = _run_kda(one)
            torch.cuda.synchronize()
            w = int(pristine["wi"][j])
            assert torch.equal(o_one[0], o_batched[j]), (
                f"row {j} output differs between B=1 (nv="
                f"{_pick_nv(1, HV)}) and B={big_bs} (nv={_pick_nv(big_bs, HV)})"
            )
            assert torch.equal(
                one["h_pool"][w], batched["h_pool"][w]
            ), f"row {j} recurrent state page {w} differs across batch size"
            assert torch.equal(
                one["conv_pool"][w], batched["conv_pool"][w]
            ), f"row {j} conv window page {w} differs across batch size"

    @pytest.mark.parametrize("big_bs", [16, 64])
    def test_row_alone_matches_row_in_batch_with_fused_norm(self, big_bs):
        """Fused-norm epilogue: tight bound, not bitwise, and here is why.

        At ``nv = 1`` the ``sum(o^2)`` normalizer is reduced inside one block
        as ``((p0+p1)+p2)+p3`` over four 32-column warp partials. At
        ``nv = 2`` each block reduces ``p0+p1`` over its own 64 columns and
        then adds the sibling's partial that arrived over distributed shared
        memory, i.e. ``(p0+p1)+(p2+p3)``. Same terms, different association,
        so the fp32 normalizer can differ in its last ulps. That perturbation
        is ~1e-7 relative and only ever changes the bf16 output when a value
        sits exactly on a rounding boundary -- hence the tight atol plus the
        "almost everything is bit-identical" check.
        """
        batched = _kda_inputs(big_bs, pages=2 * big_bs + 2, seed=300 + big_bs)
        pristine = _clone(batched)
        req = _onorm_request(batched)
        o_batched = _run_kda(batched, onorm=req)
        torch.cuda.synchronize()
        assert req.consumed

        for j in PROBE_POSITIONS:
            if j >= big_bs:
                continue
            one = self._single_row_view(pristine, j)
            req_one = _onorm_request(one)
            o_one = _run_kda(one, onorm=req_one)
            torch.cuda.synchronize()
            assert req_one.consumed
            a, b = o_one[0].float(), o_batched[j].float()
            # One bf16 ulp at the observed magnitude, no more.
            torch.testing.assert_close(a, b, atol=8e-3, rtol=8e-3)
            identical = (o_one[0] == o_batched[j]).float().mean().item()
            assert identical > 0.98, (
                f"row {j}: only {identical:.3%} of the fused-norm output is "
                "bit-identical across nv bands; expected >98% (reassociation "
                "of the o^2 reduction should move at most a handful of "
                "bf16 roundings)"
            )
            # The state pool never depends on the epilogue.
            w = int(pristine["wi"][j])
            assert torch.equal(one["h_pool"][w], batched["h_pool"][w])

    def test_row_position_within_one_batch_is_irrelevant(self):
        """The same logical row placed at different offsets of one batch.

        Catches indexing bugs that only show up away from row 0 (a
        ``bos``/``i_n`` mixup, a page-id gather off by the block index).
        """
        bs = 64
        x = _kda_inputs(bs, pages=2 * bs + 2, seed=404)
        # Broadcast row 0's activations to the probe positions and point them
        # at page 0's state, so those rows are logically identical.
        for j in PROBE_POSITIONS[1:]:
            for key in ("qkv", "f_a", "beta", "gate"):
                x[key][j] = x[key][0]
            x["ri"][j] = x["ri"][0]
        # Distinct write pages so the duplicated rows do not race each other.
        o = _run_kda(x)
        torch.cuda.synchronize()
        for j in PROBE_POSITIONS[1:]:
            assert torch.equal(
                o[j], o[0]
            ), f"identical logical row gives a different result at position {j}"
            wj, w0 = int(x["wi"][j]), int(x["wi"][0])
            assert torch.equal(x["h_pool"][wj], x["h_pool"][w0])


class TestKdaGraphPaddingIsolation:
    """CUDA-graph padding rows must be inert.

    Contract (``CudaGraphWrapper`` + ``MambaAttnBackend``): padded rows get
    page id ``-1`` (``pad_slot_id``) in both the read and the write index
    tensor, and their activation slots hold whatever the previous replay
    left behind. The kernel's ``read_ok`` / ``write_ok`` gates must make
    those rows read zeros and skip every store.
    """

    @pytest.mark.parametrize(
        "real_bs,padded_bs",
        [
            (1, 2),  # inside the nv=2 band
            (3, 4),  # inside the nv=2 band
            (5, 8),  # nv=2 alone -> nv=1 once padded (capture ladder step)
            (9, 16),  # inside the nv=1 band
            (17, 24),  # nv=1, past the latent-tail token gate
        ],
    )
    @pytest.mark.parametrize("onorm", [False, True])
    def test_poisoned_padding_rows_do_not_perturb_real_rows(
        self, real_bs, padded_bs, onorm
    ):
        pages = 2 * padded_bs + 2
        clean = _kda_inputs(padded_bs, pages=pages, seed=500 + padded_bs)
        # Both runs use the SAME padded batch shape (so the same kernel and
        # the same nv band); only the contents of the padding rows differ.
        clean["ri"][real_bs:] = -1
        clean["wi"][real_bs:] = -1
        clean["qkv"][real_bs:] = 0
        clean["f_a"][real_bs:] = 0
        clean["beta"][real_bs:] = 0
        clean["gate"][real_bs:] = 0

        poisoned = _clone(clean)
        poison = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 1e30], device="cuda"
        )
        for key in ("qkv", "f_a", "beta", "gate"):
            rows = poisoned[key][real_bs:]
            rows.copy_(
                poison.to(rows.dtype)
                .repeat(rows.numel() // 4 + 1)[: rows.numel()]
                .view(rows.shape)
            )

        pool_before = clean["h_pool"].clone()
        conv_before = clean["conv_pool"].clone()

        req_c = _onorm_request(clean) if onorm else None
        o_clean = _run_kda(clean, onorm=req_c)
        req_p = _onorm_request(poisoned) if onorm else None
        o_poison = _run_kda(poisoned, onorm=req_p)
        torch.cuda.synchronize()

        # 1. Real rows are bit-identical: padding cannot influence them.
        assert torch.equal(
            o_clean[:real_bs], o_poison[:real_bs]
        ), "poison in the padded activation slots reached the real rows"

        # 2. No real row's output is NaN/Inf (poison did not leak numerically).
        assert torch.isfinite(o_poison[:real_bs].float()).all()

        # 3. Every page the real rows did not write is untouched, and no page
        #    ever received poison -- padded rows must skip their stores.
        written = {int(clean["wi"][i]) for i in range(real_bs)}
        untouched = [p for p in range(pages) if p not in written]
        assert torch.equal(
            poisoned["h_pool"][untouched], pool_before[untouched]
        ), "a padded row wrote into the recurrent state pool"
        assert torch.equal(
            poisoned["conv_pool"][untouched], conv_before[untouched]
        ), "a padded row wrote into the conv window pool"
        assert torch.isfinite(poisoned["h_pool"]).all()

        # 4. The real rows' pages agree between the two runs.
        for p in written:
            assert torch.equal(clean["h_pool"][p], poisoned["h_pool"][p])
            assert torch.equal(clean["conv_pool"][p], poisoned["conv_pool"][p])

    def test_real_rows_match_an_unpadded_run(self):
        """Padding a batch up to the next captured size is a no-op for real rows.

        ``real_bs = 5`` alone runs ``nv = 2``; padded to the captured size 8
        it runs ``nv = 1``. Both bands share the reduction tree (see
        ``TestKdaRowInvariance``), so the real rows must be bitwise equal.
        """
        real_bs, padded_bs = 5, 8
        pages = 2 * padded_bs + 2
        padded = _kda_inputs(padded_bs, pages=pages, seed=606)
        padded["ri"][real_bs:] = -1
        padded["wi"][real_bs:] = -1

        exact = _clone(padded)
        for key in ("qkv", "f_a", "beta", "gate", "ri", "wi"):
            exact[key] = padded[key][:real_bs].clone()
        exact["cu"] = torch.arange(real_bs + 1, dtype=torch.int32, device="cuda")

        o_padded = _run_kda(padded)
        o_exact = _run_kda(exact)
        torch.cuda.synchronize()

        assert _pick_nv(real_bs, HV) != _pick_nv(
            padded_bs, HV
        ), "this test only means something if padding crosses the nv band"
        assert torch.equal(o_padded[:real_bs], o_exact)
        for i in range(real_bs):
            w = int(exact["wi"][i])
            assert torch.equal(padded["h_pool"][w], exact["h_pool"][w])
            assert torch.equal(padded["conv_pool"][w], exact["conv_pool"][w])


class TestKdaPoolLayoutFallback:
    """The misaligned-pitch fallback must not change the answer.

    ``nv = 1`` stages the ``[K, V]`` slab with one 64 KB bulk TMA copy, which
    needs a 16 B-aligned base and pitch. When the pool is only 8 B aligned
    the launcher silently falls back to the ``nv = 2`` cp.async path. Both
    must produce the same bits (same reduction tree, different staging).
    """

    @pytest.mark.parametrize("pad_elems", [640, 642])
    def test_strided_pool_matches_dense_pool(self, pad_elems):
        bs = 16  # comfortably inside the nv=1 band on a dense pool
        pages = 2 * bs + 2
        dense = _kda_inputs(bs, pages=pages, seed=700 + pad_elems)
        strided = _clone(dense)

        row = HV * K * V
        storage = torch.zeros(
            pages * (row + pad_elems) + row, dtype=torch.float32, device="cuda"
        )
        view = storage.as_strided((pages, HV, K, V), (row + pad_elems, K * V, V, 1))
        view.copy_(dense["h_pool"])
        strided["h_pool"] = view

        o_dense = _run_kda(dense)
        o_strided = _run_kda(strided)
        torch.cuda.synchronize()

        # pad 640 keeps the pitch 16B-aligned (still nv=1 bulk TMA);
        # pad 642 is only 8B-aligned and must fall back to nv=2 cp.async.
        assert torch.equal(
            o_dense, o_strided
        ), f"pool pitch padding {pad_elems} changed the decode output"
        assert torch.equal(dense["h_pool"], strided["h_pool"])


class TestKdaGraphCapture:
    """Capture/replay at each rung of the default capture ladder.

    ``get_batch_sizes_to_capture`` defaults to ``[1, 2, 4] + [8, 16, ...]``.
    The nv band, the onorm fusion decision, and the TMA-vs-cp.async choice
    are all made on the host at capture time, so a replay must reproduce the
    eager result exactly for the size it was captured at.
    """

    @pytest.mark.parametrize("bs", [1, 2, 4, 8, 16])
    def test_replay_matches_eager(self, bs):
        pages = 2 * bs + 2
        x = _kda_inputs(bs, pages=pages, seed=800 + bs)
        eager = _clone(x)
        req_e = _onorm_request(eager)
        o_eager = _run_kda(eager, onorm=req_e)
        torch.cuda.synchronize()
        assert req_e.consumed

        stream = torch.cuda.Stream()
        # Side-stream launches must order after the default-stream producers,
        # or the kernel can read garbage page indices and fault OOB.
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                warm = _clone(x)
                _run_kda(warm, onorm=_onorm_request(warm))
        torch.cuda.synchronize()

        graphed = _clone(x)
        req_g = _onorm_request(graphed)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            o_graph = _run_kda(graphed, onorm=req_g)
        assert req_g.consumed, "fusion decision must be stable at capture"
        # Restore the pools the capture pass dirtied, then replay.
        graphed["h_pool"].copy_(x["h_pool"])
        graphed["conv_pool"].copy_(x["conv_pool"])
        g.replay()
        torch.cuda.synchronize()

        assert torch.equal(o_graph, o_eager)
        assert torch.equal(graphed["h_pool"], eager["h_pool"])
        assert torch.equal(graphed["conv_pool"], eager["conv_pool"])


class TestKdaVarlenGate:
    """``cu_seqlens`` vs dense indexing must agree at every batch size.

    ``has_cu_seqlens`` is a compile-time switch producing a different CUBIN.
    The dense path derives ``bos = i_n``; the varlen path reads
    ``cu_seqlens[i_n]`` and skips zero-length sequences. For a pure decode
    step (one token per sequence) the two are the same computation.
    """

    @pytest.mark.parametrize("bs", [1, 7, 8, 32])
    def test_dense_and_varlen_agree(self, bs):
        a = _kda_inputs(bs, pages=2 * bs + 2, seed=900 + bs)
        b = _clone(a)
        o_varlen = _run_kda(a, cu=True)
        o_dense = _run_kda(b, cu=False)
        torch.cuda.synchronize()
        assert torch.equal(o_varlen, o_dense)
        assert torch.equal(a["h_pool"], b["h_pool"])
        assert torch.equal(a["conv_pool"], b["conv_pool"])

    def test_zero_length_sequences_are_skipped(self):
        """A varlen batch with empty sequences interleaved among real ones.

        Zero-length rows must not write state. The kernel skips them
        block-uniformly (the whole ``t_len > 0`` body, including the cluster
        barriers), so this also exercises the nv=2 cluster path with a
        partially idle grid.
        """
        real = 4
        x = _kda_inputs(real, pages=2 * real + 2, seed=911)
        # Sequences 1 and 3 carry no token: cu_seqlens repeats its offset.
        counts = torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.int32, device="cuda")
        cu = torch.zeros(counts.numel() + 1, dtype=torch.int32, device="cuda")
        cu[1:] = torch.cumsum(counts, 0)
        assert int(cu[-1]) == real
        x["cu"] = cu
        # One page id per *sequence* now, not per token.
        x["ri"] = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device="cuda")
        x["wi"] = x["ri"] + 6

        pool_before = x["h_pool"].clone()
        conv_before = x["conv_pool"].clone()
        _run_kda(x)
        torch.cuda.synchronize()

        # Write pages of the empty sequences (7 and 9) must be untouched.
        for empty_seq in (1, 3):
            w = int(x["wi"][empty_seq])
            assert torch.equal(x["h_pool"][w], pool_before[w])
            assert torch.equal(x["conv_pool"][w], conv_before[w])


# ---------------------------------------------------------------------------
# 2. AttnRes residual mixing: the token-count dispatch table
# ---------------------------------------------------------------------------

try:
    from tokenspeed_kernel.ops.attn_res import attn_res_fwd
    from tokenspeed_kernel.ops.attn_res.cuda import _HAS_CUDA_KERNEL
    from tokenspeed_kernel.ops.attn_res.torch import torch_attn_res_fwd

    _ATTN_RES_AVAILABLE = _HAS_CUDA_KERNEL
except Exception:  # pragma: no cover - build without the compiled kernel
    _ATTN_RES_AVAILABLE = False

H_K3 = 7168


def _attn_res_inputs(t: int, n: int, seed: int):
    """``N = n`` candidates: ``n - 1`` block snapshots plus the layer stream."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape, scale=1.0):
        return (
            torch.randn(*shape, generator=g, dtype=torch.float32, device="cuda") * scale
        ).to(torch.bfloat16)

    return dict(
        layer_residual=rnd(t, H_K3),
        block_residual=rnd(max(n - 1, 0), t, H_K3),
        res_weight=rnd(H_K3, scale=0.05),
        rms_weight=rnd(H_K3, scale=0.5),
        out_norm_weight=rnd(H_K3, scale=0.5),
    )


def _attn_res_reference(x: dict, *, out_norm: bool, chunk: int = 256):
    """Chunked fp32 reference (bounds peak memory at T = 1024, N = 12)."""
    t = x["layer_residual"].shape[0]
    parts = []
    for lo in range(0, t, chunk):
        hi = min(lo + chunk, t)
        parts.append(
            torch_attn_res_fwd(
                layer_residual=x["layer_residual"][lo:hi],
                block_residual=x["block_residual"][:, lo:hi],
                res_weight=x["res_weight"],
                rms_weight=x["rms_weight"],
                eps=1e-6,
                out_norm_weight=x["out_norm_weight"] if out_norm else None,
            )
        )
    return torch.cat(parts, dim=0)


def _call_attn_res(x: dict, *, out_norm: bool, lo=None, hi=None):
    sl = slice(lo, hi)
    return attn_res_fwd(
        x["layer_residual"][sl],
        x["block_residual"][:, sl],
        x["res_weight"],
        x["rms_weight"],
        1e-6,
        out_norm_weight=x["out_norm_weight"] if out_norm else None,
    )


@pytest.mark.skipif(
    not _ATTN_RES_AVAILABLE, reason="compiled attn_res kernel unavailable"
)
class TestAttnResAcrossTokenCount:
    """``run_attn_res_fwd_tma`` dispatches on ``(H, N, T, out_norm)``.

    At the K3 hidden size 7168 the table (attn_res_fwd_tma.cu:1367-1404) is:

    ======================================  ==============================
    predicate                               kernel
    ======================================  ==============================
    ``T==1 && N in {1,2,4} && !out_norm``   ``s1_single_cta``
    ``T==1 && N in {8,12} && !out_norm``    ``s1_splitk`` (8-CTA cluster,
                                            DSM stat exchange)
    ``N==12 && T==1024``                    ``online_v2<FULL_N12>`` on
                                            ``num_sm - 1`` CTAs
    otherwise                               ``online_v2`` on ``num_sm`` CTAs
    ======================================  ==============================

    Three of those four rows are only reachable at *specific* token counts,
    which is exactly the shape of bug this module exists to find. Note the
    ``T == 1024`` row is an exact-equality special case: 1023 and 1025 take
    a different kernel *and* a different grid size.
    """

    # bf16 output of an fp32-internal mix; one bf16 ulp at unit scale.
    ATOL, RTOL = 8e-3, 8e-3

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 12])
    @pytest.mark.parametrize("t", [1, 2, 3, 4, 8, 16, 32, 33, 64, 65])
    @pytest.mark.parametrize("out_norm", [False, True])
    def test_matches_torch_reference(self, n, t, out_norm):
        x = _attn_res_inputs(t, n, seed=1000 + 17 * n + t)
        got = _call_attn_res(x, out_norm=out_norm)
        ref = _attn_res_reference(x, out_norm=out_norm)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            got.float(), ref.float(), atol=self.ATOL, rtol=self.RTOL
        )

    @pytest.mark.parametrize("t", [1023, 1024, 1025])
    @pytest.mark.parametrize("out_norm", [False, True])
    def test_n12_prefill_tile_special_case(self, t, out_norm):
        """``N == 12 && T == 1024`` gets its own kernel and one fewer CTA.

        A chunked prefill of exactly 1024 tokens is a completely ordinary
        production shape, so this branch is hot; its neighbours 1023 and
        1025 run different code. All three must agree with the reference.
        """
        x = _attn_res_inputs(t, 12, seed=2000 + t)
        got = _call_attn_res(x, out_norm=out_norm)
        ref = _attn_res_reference(x, out_norm=out_norm)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            got.float(), ref.float(), atol=self.ATOL, rtol=self.RTOL
        )

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 12])
    @pytest.mark.parametrize("out_norm", [False, True])
    def test_row_alone_matches_row_in_batch(self, n, out_norm):
        """Token 0 alone (``T=1``) vs the same token inside ``T=64``.

        Every candidate's statistics are per-token, so no legitimate
        reduction-order difference exists *between* tokens -- but the two
        sides may run entirely different kernels (``s1_*`` vs
        ``online_v2``), which reduce the per-token 7168-wide sums with
        different tile widths. Hence a tolerance rather than bitwise
        equality; the bound is one bf16 ulp.
        """
        big = 64
        x = _attn_res_inputs(big, n, seed=3000 + n)
        got_batched = _call_attn_res(x, out_norm=out_norm)
        for j in (0, 1, 31, 63):
            got_alone = _call_attn_res(x, out_norm=out_norm, lo=j, hi=j + 1)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                got_alone[0].float(),
                got_batched[j].float(),
                atol=self.ATOL,
                rtol=self.RTOL,
                msg=lambda m, j=j: f"attn_res token {j}: T=1 vs T=64 disagree\n{m}",
            )

    @pytest.mark.parametrize("n", [8, 12])
    def test_single_token_splitk_is_deterministic(self, n):
        """``T==1, N in {8,12}, no out_norm`` -> the 8-CTA cluster split-K kernel.

        That kernel gathers per-CTA ``(sum v^2, dot)`` statistics over
        distributed shared memory. Repeated launches on identical inputs must
        give identical bits; a cross-CTA read/write hazard on the statistics
        slots shows up here as run-to-run variation.

        This path is not currently reached from ``_apply_attn_res`` (the K3
        model always passes ``out_norm``), but it is live in the op API.
        """
        x = _attn_res_inputs(1, n, seed=4000 + n)
        first = _call_attn_res(x, out_norm=False).clone()
        ref = _attn_res_reference(x, out_norm=False)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            first.float(), ref.float(), atol=self.ATOL, rtol=self.RTOL
        )
        for i in range(64):
            again = _call_attn_res(x, out_norm=False)
            torch.cuda.synchronize()
            assert torch.equal(first, again), (
                f"split-K attn_res is not deterministic (iteration {i}); "
                "suspect the DSM statistics exchange in "
                "attn_res_fwd_s1_splitk_kernel"
            )


# ---------------------------------------------------------------------------
# 3. Kimi-3 projections: the m == 1 / m > 32 kernel swaps
# ---------------------------------------------------------------------------

from tokenspeed_kernel.ops.gemm.kimi3 import (  # noqa: E402
    KIMI3_HIDDEN_SIZE,
    KIMI3_LATENT_SIZE,
    KIMI3_QKVFAB_SIZE,
    KIMI3_ROUTER_SIZE,
    kimi3_latent_projection_add3,
    kimi3_mla_qkv_gate_projection,
    kimi3_qkvfab_projection,
    kimi3_router_projection,
)

PROJ_BATCHES = [1, 2, 4, 8, 16, 31, 32, 33, 64, 65]


def _bf16(*shape, seed, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (
        torch.randn(*shape, generator=g, dtype=torch.float32, device="cuda") * scale
    ).to(torch.bfloat16)


class TestKimi3ProjectionsAcrossBatch:
    """Every K3 projection swaps kernel at ``m == 1`` (some also at ``m == 32``).

    These are bf16 GEMMs, so the two sides legitimately differ: a row-per-CTA
    streaming GEMV reduces K=7168 in a single fp32 accumulator chain, while
    cuBLAS/cuBLASLt splits K across tiles and re-associates. The bound below
    is derived from that: fp32 accumulation of 7168 bf16 products has a
    relative error around ``sqrt(7168) * 2^-24 ~= 5e-6``; after rounding the
    result to bf16 (``2^-8`` relative) the dominant term is the output
    rounding, so we allow ``rtol = 1e-2`` on values whose magnitude is
    ``O(sqrt(K))``. Bitwise equality is NOT expected and is not asserted.
    """

    ATOL, RTOL = 6e-2, 1e-2

    @staticmethod
    def _ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x.float(), w.float())

    @pytest.mark.parametrize("m", PROJ_BATCHES)
    def test_router_projection_matches_fp32(self, m):
        """``m == 1`` -> rowcta GEMV; ``m > 1`` -> the dense CUDA router GEMM.

        Both claim fp32 accumulation into an fp32 output, so the reference
        agreement here is much tighter than the bf16-output projections.
        """
        w = _bf16(KIMI3_ROUTER_SIZE, KIMI3_HIDDEN_SIZE, seed=11, scale=0.02)
        x = _bf16(m, KIMI3_HIDDEN_SIZE, seed=12 + m, scale=0.5)
        got = kimi3_router_projection(x, w)
        ref = self._ref(x, w)
        torch.cuda.synchronize()
        assert got.dtype == torch.float32
        torch.testing.assert_close(got, ref, atol=2e-3, rtol=2e-3)

    def test_router_row_invariance_and_topk_stability(self):
        """The router decides which experts a token visits.

        A logit perturbation at the ``m == 1`` boundary that flips a top-k
        selection changes the token's experts entirely -- a categorical, not
        numerical, difference. We therefore assert (a) the logits agree
        within the GEMM tolerance and (b) any top-k disagreement is confined
        to genuinely tied candidates (gap below the observed logit noise).
        """
        topk = 8
        big = 64
        w = _bf16(KIMI3_ROUTER_SIZE, KIMI3_HIDDEN_SIZE, seed=21, scale=0.02)
        x = _bf16(big, KIMI3_HIDDEN_SIZE, seed=22, scale=0.5)
        batched = kimi3_router_projection(x, w)
        torch.cuda.synchronize()

        worst = 0.0
        for j in (0, 1, 31, 63):
            alone = kimi3_router_projection(x[j : j + 1].contiguous(), w)
            torch.cuda.synchronize()
            delta = (alone[0] - batched[j]).abs().max().item()
            worst = max(worst, delta)
            torch.testing.assert_close(alone[0], batched[j], atol=2e-3, rtol=2e-3)

            sel_a = torch.topk(alone[0], topk).indices.sort().values
            sel_b = torch.topk(batched[j], topk).indices.sort().values
            if torch.equal(sel_a, sel_b):
                continue
            # A flip is only acceptable if the k-th and (k+1)-th logits are
            # closer together than the measured cross-batch logit noise.
            ordered = torch.sort(batched[j], descending=True).values
            margin = (ordered[topk - 1] - ordered[topk]).item()
            assert margin <= 4 * max(delta, 1e-6), (
                f"router top-{topk} for token {j} changes between m=1 and "
                f"m={big} with a decision margin of {margin:.3e}, far above "
                f"the {delta:.3e} logit difference -- the two router kernels "
                "disagree structurally, not just in rounding"
            )

    @pytest.mark.parametrize("m", PROJ_BATCHES)
    def test_qkvfab_projection_matches_fp32(self, m):
        """``m == 1`` -> decode_gemv (rowcta); ``m > 1`` -> torch/cuBLAS.

        This feeds the KDA conv/gate front, so a divergence here propagates
        straight into the recurrent state.
        """
        w = _bf16(KIMI3_QKVFAB_SIZE, KIMI3_HIDDEN_SIZE, seed=31, scale=0.02)
        x = _bf16(m, KIMI3_HIDDEN_SIZE, seed=32 + m, scale=0.5)
        got = kimi3_qkvfab_projection(x, w)
        ref = self._ref(x, w).to(torch.bfloat16)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            got.float(), ref.float(), atol=self.ATOL, rtol=self.RTOL
        )

    def test_qkvfab_row_invariance(self):
        big = 64
        w = _bf16(KIMI3_QKVFAB_SIZE, KIMI3_HIDDEN_SIZE, seed=41, scale=0.02)
        x = _bf16(big, KIMI3_HIDDEN_SIZE, seed=42, scale=0.5)
        batched = kimi3_qkvfab_projection(x, w)
        torch.cuda.synchronize()
        for j in (0, 1, 63):
            alone = kimi3_qkvfab_projection(x[j : j + 1].contiguous(), w)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                alone[0].float(),
                batched[j].float(),
                atol=self.ATOL,
                rtol=self.RTOL,
            )

    @pytest.mark.parametrize("m", PROJ_BATCHES)
    def test_latent_projection_add3_matches_fp32(self, m):
        """``m == 1`` fuses ``prefix + x@W.T + shared`` in one kernel.

        ``m > 1`` composes: the projection rounds to bf16 first, then a
        separate add3 sums three bf16 values. The fused kernel adds in fp32
        before its single rounding, so the two paths differ by *one* bf16
        rounding of the projection term -- expected, and bounded below.
        This is the K3 MoE residual backbone, so the drift compounds over
        layers if it is ever larger than that.
        """
        w = _bf16(KIMI3_HIDDEN_SIZE, KIMI3_LATENT_SIZE, seed=51, scale=0.02)
        x = _bf16(m, KIMI3_LATENT_SIZE, seed=52 + m, scale=0.5)
        prefix = _bf16(m, KIMI3_HIDDEN_SIZE, seed=53 + m, scale=0.5)
        shared = _bf16(m, KIMI3_HIDDEN_SIZE, seed=54 + m, scale=0.5)
        got = kimi3_latent_projection_add3(x, w, prefix, shared)
        ref = (self._ref(x, w) + prefix.float() + shared.float()).to(torch.bfloat16)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            got.float(), ref.float(), atol=self.ATOL, rtol=self.RTOL
        )

    def test_latent_projection_add3_fused_vs_composed(self):
        """Force both solutions at ``m == 1`` and bound their difference.

        This is the only place the two implementations can be compared
        directly; in production they are selected by batch size and never
        run side by side.
        """
        w = _bf16(KIMI3_HIDDEN_SIZE, KIMI3_LATENT_SIZE, seed=61, scale=0.02)
        x = _bf16(1, KIMI3_LATENT_SIZE, seed=62, scale=0.5)
        prefix = _bf16(1, KIMI3_HIDDEN_SIZE, seed=63, scale=0.5)
        shared = _bf16(1, KIMI3_HIDDEN_SIZE, seed=64, scale=0.5)
        fused = kimi3_latent_projection_add3(
            x, w, prefix, shared, solution="rowcta_gemv"
        )
        composed = kimi3_latent_projection_add3(
            x, w, prefix, shared, solution="composed"
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            fused.float(), composed.float(), atol=self.ATOL, rtol=self.RTOL
        )

    @pytest.mark.parametrize("m", [1, 8, 32, 33, 64])
    def test_mla_qkv_gate_projection_across_the_m32_threshold(self, m):
        """``m <= 32`` -> one fused GEMM then a split; ``m > 32`` -> two GEMMs.

        The ``packed`` field also disappears above the threshold, which
        changes what the caller can hand to the communication layer. Assert
        both the numerics and the structural contract, because a caller that
        silently stops taking the packed path is a behaviour change nobody
        would notice from the outputs alone.
        """
        qkv_width = 2048
        out_width = qkv_width + 512
        w = _bf16(out_width, KIMI3_HIDDEN_SIZE, seed=71, scale=0.02)
        x = _bf16(m, KIMI3_HIDDEN_SIZE, seed=72 + m, scale=0.5)
        got = kimi3_mla_qkv_gate_projection(x, w, qkv_width)
        ref = self._ref(x, w).to(torch.bfloat16)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            got.qkv.float(), ref[:, :qkv_width].float(), atol=self.ATOL, rtol=self.RTOL
        )
        torch.testing.assert_close(
            got.gate.float(), ref[:, qkv_width:].float(), atol=self.ATOL, rtol=self.RTOL
        )
        if m <= 32:
            assert got.packed is not None, "fused schedule must expose packed rows"
            assert got.packed.shape == (m, out_width)
        else:
            assert got.packed is None, "split schedule must not expose packed rows"

    def test_mla_qkv_gate_fused_vs_split_agree(self):
        """Force both schedules at one ``m`` and bound their difference."""
        m, qkv_width = 32, 2048
        out_width = qkv_width + 512
        w = _bf16(out_width, KIMI3_HIDDEN_SIZE, seed=81, scale=0.02)
        x = _bf16(m, KIMI3_HIDDEN_SIZE, seed=82, scale=0.5)
        fused = kimi3_mla_qkv_gate_projection(x, w, qkv_width, solution="fused")
        split = kimi3_mla_qkv_gate_projection(x, w, qkv_width, solution="split")
        torch.cuda.synchronize()
        torch.testing.assert_close(
            fused.qkv.float(), split.qkv.float(), atol=self.ATOL, rtol=self.RTOL
        )
        torch.testing.assert_close(
            fused.gate.float(), split.gate.float(), atol=self.ATOL, rtol=self.RTOL
        )


# ---------------------------------------------------------------------------
# 4. MoE routing: the exactly-one-token top-k swap
# ---------------------------------------------------------------------------

from tokenspeed_kernel.ops.moe.sigmoid_topk import (  # noqa: E402
    moe_sigmoid_bias_topk,
)

K3_EXPERTS, K3_TOPK = 896, 16


def _routing_inputs(tokens: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(
        tokens, K3_EXPERTS, generator=g, dtype=torch.float32, device="cuda"
    )
    bias = (
        torch.randn(K3_EXPERTS, generator=g, dtype=torch.float32, device="cuda") * 0.05
    )
    return logits, bias


class TestMoeRoutingAcrossTokenCount:
    """``moe_sigmoid_bias_topk`` swaps algorithm at *exactly one token*.

    ``ops/moe/sigmoid_topk.py:86-92``: when
    ``router_logits.shape == (1, 896)`` and the dtypes/topk match, routing
    runs the packed-key single-CTA bitonic top-16. Any other token count
    falls through to ``select_kernel``, which on NVIDIA picks
    ``triton_minimax_sigmoid_bias_topk`` (the grouped minimax kernel with
    ``num_expert_group=1``).

    Top-k is a *categorical* output. A one-ulp logit difference or a
    different tie-break rule does not produce a small numerical error, it
    sends the token to a different expert. So unlike every other test in
    this module, the assertion here is exact set equality of the selected
    experts, with an explicit escape only for genuine numerical ties.
    """

    @staticmethod
    def _route(logits, bias):
        return moe_sigmoid_bias_topk(
            logits,
            bias,
            topk=K3_TOPK,
            routed_scaling_factor=1.0,
            normalize_topk_weights=True,
        )

    @pytest.mark.parametrize("tokens", [1, 2, 3, 4, 8, 16, 32, 64])
    def test_selected_experts_match_a_torch_reference(self, tokens):
        logits, bias = _routing_inputs(tokens, seed=5000 + tokens)
        weights, ids = self._route(logits, bias)
        torch.cuda.synchronize()

        scores = logits.sigmoid()
        ref_ids = torch.topk(scores + bias.unsqueeze(0), K3_TOPK, dim=-1).indices
        for t in range(tokens):
            got = set(ids[t].tolist())
            want = set(ref_ids[t].tolist())
            if got == want:
                continue
            # Only tolerate a swap between candidates that are numerically
            # tied at the k-th boundary.
            biased = (scores[t] + bias).sort(descending=True).values
            margin = (biased[K3_TOPK - 1] - biased[K3_TOPK]).item()
            assert margin < 1e-6, (
                f"token {t} at tokens={tokens}: selected experts differ from "
                f"the reference with a decision margin of {margin:.3e}; "
                f"symmetric difference {sorted(got ^ want)}"
            )

        # Weights are the sigmoid scores of the selected experts, renormalized.
        gathered = scores.gather(1, ids.long())
        expected = gathered / gathered.sum(-1, keepdim=True)
        torch.testing.assert_close(
            weights.float(), expected.float(), atol=1e-5, rtol=1e-5
        )

    def test_one_token_routing_matches_the_same_row_in_a_batch(self):
        """The single-token kernel and the batched kernel must agree.

        This is the direct cross-batch assertion for the ``shape == (1, 896)``
        special case: expert selection for a given row must not depend on
        how many other rows travel with it.
        """
        tokens = 64
        logits, bias = _routing_inputs(tokens, seed=6001)
        _, ids_batched = self._route(logits, bias)
        torch.cuda.synchronize()

        scores = logits.sigmoid()
        for t in (0, 1, 31, 63):
            row = logits[t : t + 1].contiguous()
            _, ids_alone = self._route(row, bias)
            torch.cuda.synchronize()
            got = set(ids_alone[0].tolist())
            want = set(ids_batched[t].tolist())
            if got == want:
                continue
            biased = (scores[t] + bias).sort(descending=True).values
            margin = (biased[K3_TOPK - 1] - biased[K3_TOPK]).item()
            assert margin < 1e-6, (
                f"token {t}: the one-token packed-key top-k and the batched "
                f"minimax top-k select different experts with a decision "
                f"margin of {margin:.3e} -- the two routing kernels disagree "
                f"structurally. Symmetric difference {sorted(got ^ want)}"
            )

    def test_tie_break_rule_is_the_same_on_both_sides(self):
        """Force exact ties and require an identical, deterministic choice.

        With every biased score equal, the packed-key kernel breaks ties by
        lowest expert id (it packs ``num_experts - expert`` into the low
        bits). The batched kernel must do the same, or a batch of duplicate
        tokens routes differently from a single token.
        """
        tokens = 4
        logits = torch.zeros(tokens, K3_EXPERTS, dtype=torch.float32, device="cuda")
        bias = torch.zeros(K3_EXPERTS, dtype=torch.float32, device="cuda")
        _, ids_batched = self._route(logits, bias)
        _, ids_one = self._route(logits[:1].contiguous(), bias)
        torch.cuda.synchronize()

        want = set(range(K3_TOPK))  # lowest ids win an all-way tie
        assert set(ids_one[0].tolist()) == want, (
            "one-token routing does not break an all-way tie by lowest "
            f"expert id: got {sorted(ids_one[0].tolist())}"
        )
        for t in range(tokens):
            assert set(ids_batched[t].tolist()) == want, (
                f"batched routing row {t} breaks an all-way tie differently "
                f"from the one-token kernel: got {sorted(ids_batched[t].tolist())}"
            )

    def test_routing_is_deterministic_across_repeats(self):
        """Same logits, many launches, identical ids -- both sides."""
        for tokens in (1, 8):
            logits, bias = _routing_inputs(tokens, seed=7000 + tokens)
            _, first = self._route(logits, bias)
            first = first.clone()
            torch.cuda.synchronize()
            for _ in range(32):
                _, again = self._route(logits, bias)
                torch.cuda.synchronize()
                assert torch.equal(
                    first, again
                ), f"routing at tokens={tokens} is not deterministic"


# ---------------------------------------------------------------------------
# 5. Threshold constants that other code depends on (no GPU work)
# ---------------------------------------------------------------------------


class TestBatchThresholdConstants:
    """Pin the token-count thresholds that are duplicated across packages.

    None of these can be checked at runtime on a single GPU (the latent tail
    needs a TP8 symmetric-memory rendezvous), but every one of them is a
    number that two files agree on by convention only. If a retune moves one
    side, this test fails instead of a 24 h soak producing wrong tokens.
    """

    def test_latent_tail_token_window(self):
        from tokenspeed_kernel.ops.moe import latent_tail

        # kimi_k3.py gates the multicast tail on
        #   1 <= num_tokens <= self._latent_tail.max_num_tokens
        # and the kernel raises above max_m, so these must be the same number.
        assert latent_tail._MAX_NUM_TOKENS == 16
        # M <= 5 compiles a static-M SIMT skinny GEMM (one CUBIN per M);
        # M >= 6 uses the dynamic-M tensor-core MMA GEMM. The two are
        # different implementations of the same up-projection.
        assert latent_tail._SKINNY_MAX_NUM_TOKENS == 5
        assert (
            0
            <= latent_tail._SKINNY_MAX_NUM_TOKENS
            <= min(8, latent_tail._MAX_NUM_TOKENS)
        ), "AdaptiveUpProjectionKernel rejects skinny_max_m outside [0, min(8, max_m)]"

    def test_attn_res_eligibility_window(self):
        from tokenspeed_kernel.ops import attn_res

        # Hand-maintained duplicate of the TVM_FFI_ICHECK bounds in
        # csrc/attn_res_binding.cu. Keep them in lockstep.
        assert attn_res._MAX_T == 16384
        assert attn_res._MAX_N == 12
        assert 7168 in attn_res._SUPPORTED_H

    def test_kda_nv_band_constant(self):
        # _pick_nv's 96-block floor is what makes B=8 the crossover at HV=12.
        assert _pick_nv(7, HV) == 2 and _pick_nv(8, HV) == 1

    def test_thirty_two_is_three_unrelated_constants(self):
        """Three independent thresholds all happen to equal 32.

        ``ATTNRES_FAST_PATH_MAX_TOKENS`` (runtime), the MLA gate
        projection's ``m > 32`` split, and ``attn_res_fwd``'s
        ``large_prefill: T > 32`` trait. None of them references the others,
        so tuning one silently desynchronizes the pair of code paths that
        flip together today. Pin them so a divergence is a test failure and
        a deliberate decision rather than a surprise.
        """
        import inspect

        from tokenspeed_kernel.ops.attn_res import attn_res_fwd
        from tokenspeed_kernel.ops.gemm import kimi3 as gemm_kimi3

        assert 'solution = "split" if m > 32 else "fused"' in inspect.getsource(
            gemm_kimi3.kimi3_mla_qkv_gate_projection
        )
        assert '"large_prefill": T > 32' in inspect.getsource(attn_res_fwd)
