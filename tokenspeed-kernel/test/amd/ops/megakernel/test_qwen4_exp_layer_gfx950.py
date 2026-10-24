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

"""A whole Qwen3.8-Flash-Next Gated DeltaNet layer as one megakernel launch."""

from __future__ import annotations

import functools

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from megakernel_testing import (  # noqa: E402
    random_mxfp4,
    round_mxfp4,
    softplus,
)
from qwen4_exp_reference import bf  # noqa: E402
from qwen4_exp_reference import cos_sin as _cos_sin  # noqa: E402
from qwen4_exp_reference import gemma, index_norm, rotate  # noqa: E402
from tokenspeed_kernel.ops.attention.gdn import gdn_replay_commit  # noqa: E402
from tokenspeed_kernel.ops.megakernel import (  # noqa: E402
    prepare_qwen4_exp_delta_layer,
    prepare_qwen4_exp_qsa_layer,
    qwen4_exp_delta_layer_decode,
    qwen4_exp_qsa_layer_decode,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.weights import (  # noqa: E402
    dequantize_mxfp4,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.n16_weights import (  # noqa: E402
    preprocess_n16_mxfp4_weights,
)

HIDDEN, WIDTH, LOWRANK = 2560, 10240, 320
QKV, VALUE, HEAD = 10240, 6144, 128
EXPERTS, INTER = 512, 640


def _bf16_weight(rows: int, cols: int, g, scale: float = 0.02) -> torch.Tensor:
    return (torch.randn(rows, cols, generator=g) * scale).bfloat16().cuda()


class _Mixer:
    """A random ``GatedResidualSimple`` and TokenSpeed's two-launch mix."""

    def __init__(self, g):
        self.norm_w = (torch.randn(WIDTH, generator=g) * 0.1).bfloat16().cuda()
        self.down = _bf16_weight(LOWRANK, WIDTH, g, 0.01)
        self.inject = _bf16_weight(4, WIDTH, g, 0.01)
        self.up = _bf16_weight(WIDTH, LOWRANK, g, 0.05)
        # As TokenSpeed's GatedResidualSimple holds them (storage padded to
        # whole 8-row groups).
        pad = torch.zeros_like(self.inject)
        self.raw = (self.norm_w, torch.cat([self.down, self.inject, pad]), self.up)

    def __call__(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mixed row, per-stream output gates)`` of streams ``h``."""
        streams = h.float().view(4, HIDDEN)
        inv = torch.rsqrt(streams.square().mean(-1) + 1e-6)
        scale = 1.0 + self.norm_w.float().view(4, HIDDEN)
        y = bf(streams * inv[:, None] * scale).flatten()
        halves = (slice(0, 2 * HIDDEN), slice(2 * HIDDEN, WIDTH))
        inj = sum(self.inject.float()[:, c] @ y[c] for c in halves)
        value = bf(self.down.float() @ y)
        lowrank = bf(value * torch.sigmoid(value))
        gate = torch.sigmoid(bf(self.up.float() @ lowrank)).view(4, HIDDEN)
        mixed = bf((gate * y.view(4, HIDDEN)).mean(0))
        return mixed, 2 * torch.sigmoid(bf(inj))


class _Moe:
    """A random sparse MoE block, TokenSpeed's banks of it and its math."""

    def __init__(self, g):
        self.router_w = (
            (torch.randn(EXPERTS, HIDDEN, generator=g) * 0.02).bfloat16().cuda()
        )
        self.gate_w = (torch.randn(HIDDEN, generator=g) * 0.02).bfloat16().cuda()
        self.w13, self.s13 = random_mxfp4((EXPERTS, 2 * INTER, HIDDEN), g)
        self.w2, self.s2 = random_mxfp4((EXPERTS, HIDDEN, INTER), g)
        self.gu, self.gu_s = random_mxfp4((2 * INTER, HIDDEN), g)
        self.dn, self.dn_s = random_mxfp4((HIDDEN, INTER), g)
        bank = torch.nn.Module()
        for name, value in (
            ("w13_weight", self.w13),
            ("w13_weight_scale", self.s13),
            ("w2_weight", self.w2),
            ("w2_weight_scale", self.s2),
        ):
            bank.register_parameter(
                name, torch.nn.Parameter(value.clone(), requires_grad=False)
            )
        preprocess_n16_mxfp4_weights(bank)
        self.tensors = {
            "router_w": self.router_w,
            "shared_gate_w": self.gate_w[None, :],
            "experts": "mxfp4",
            "banks": (
                bank.w13_weight,
                bank.w13_weight_scale,
                bank.w2_weight,
                bank.w2_weight_scale,
            ),
            "shared_gu": (self.gu, self.gu_s),
            "shared_dn": (self.dn, self.dn_s),
        }

    def __call__(self, mixed: torch.Tensor) -> torch.Tensor:
        logits = bf(self.router_w.float() @ mixed)
        top_w, top_e = torch.topk(torch.softmax(logits, 0), 10)
        top_w = top_w / top_w.sum()

        def expert(wgu, sgu, wdn, sdn, routed):
            x = bf(dequantize_mxfp4(wgu, sgu) @ round_mxfp4(mixed[None])[0])
            silu = torch.nn.functional.silu(x[:INTER])
            # The routed epilogue rounds silu(gate); the shared expert's not.
            x = bf((bf(silu) if routed else silu) * x[INTER:])
            return bf(dequantize_mxfp4(wdn, sdn) @ round_mxfp4(x[None])[0])

        routed = bf(
            sum(
                top_w[i]
                * expert(self.w13[e], self.s13[e], self.w2[e], self.s2[e], True)
                for i, e in enumerate(top_e.tolist())
            )
        )
        gate = torch.sigmoid(self.gate_w.float() @ mixed)
        shared = expert(self.gu, self.gu_s, self.dn, self.dn_s, False)
        return bf(routed + gate * shared)


@functools.cache
def _bf16_experts() -> dict:
    """Random BF16 experts (about 6.7 GB), as the MTP draft layer of the
    Quark checkpoint holds them: the prepare ops' ``"bf16"`` MoE tensors."""
    g = torch.Generator(device="cuda").manual_seed(5)

    def weight(*shape, scale):
        return (torch.randn(*shape, generator=g, device="cuda") * scale).bfloat16()

    return {
        "router_w": weight(EXPERTS, HIDDEN, scale=0.02),
        "shared_gate_w": weight(1, HIDDEN, scale=0.02),
        "experts": "bf16",
        "w13": weight(EXPERTS, 2 * INTER, HIDDEN, scale=0.02),
        "w2": weight(EXPERTS, HIDDEN, INTER, scale=0.04),
        "shared_gu": weight(2 * INTER, HIDDEN, scale=0.02),
        "shared_dn": weight(HIDDEN, INTER, scale=0.04),
    }


def _mlp_half(w, mlp_mixer, moe):
    """The output streams from the attention half's streams ``w``."""
    mixed, gate = mlp_mixer(w.flatten())
    return bf(w + moe(mixed)[None, :] * gate[:, None])


class _Ple:
    """A random PLE block (one step of it) and TokenSpeed's fused-kernel math."""

    EOS = 9

    def __init__(self, g, token: int, context: tuple[int, int]):
        self.sizes = torch.tensor([97 + 2 * i for i in range(16)], dtype=torch.int64)
        self.offsets = torch.cumsum(self.sizes, 0) - self.sizes
        self.multipliers = torch.tensor([1000003, 999983, 1000033], dtype=torch.int64)
        rows = int(self.sizes.sum())
        self.table = (torch.randn(rows, 160, generator=g) * 0.5).bfloat16().cuda()
        self.kv = _bf16_weight(5 * HIDDEN, HIDDEN, g)
        self.gk, self.gq, self.gc = (
            (1 + 0.1 * torch.randn(WIDTH, generator=g)).bfloat16().cuda()
            for _ in range(3)
        )
        self.conv_w = (torch.randn(WIDTH, 1, 4, generator=g) * 0.5).bfloat16().cuda()
        self.state = torch.randn(4, WIDTH, 9, generator=g).bfloat16().cuda()
        self.context = torch.zeros(4, 2, dtype=torch.int64).cuda()
        self.context[1] = torch.tensor(context)
        self.token = torch.tensor([token], dtype=torch.int64).cuda()
        self.page_in = torch.tensor([1], dtype=torch.int32).cuda()
        self.page_out = torch.tensor([2], dtype=torch.int32).cuda()
        self.tensors = {
            "kv": self.kv,
            "norm_key": self.gk,
            "norm_query": self.gq,
            "norm_conv": self.gc,
            "conv_w": self.conv_w,
            "table": self.table,
            "multipliers": self.multipliers.cuda(),
            "sizes": self.sizes.cuda(),
            "offsets": self.offsets.cuda(),
            "eos": self.EOS,
        }
        self.initial = self.state[1].clone()
        self.ctx = context

    def step(self):
        return {
            "token": self.token,
            "context": self.context,
            "conv_state": self.state,
            "page_in": self.page_in,
            "page_out": self.page_out,
            "context_out": self.context,
            "conv_out": self.state,
        }

    def embedding(self) -> torch.Tensor:
        """TokenSpeed's n-gram ids and table gather, in Python integers."""
        t = int(self.token[0])
        older, newer = self.ctx
        if newer == self.EOS:
            older = self.EOS
        m0, m1, m2 = (int(m) for m in self.multipliers)
        rows = []
        for h in range(16):
            mixed = (t * m0) ^ (newer * m1)
            if h >= 8:
                mixed ^= older * m2
            rows.append(mixed % int(self.sizes[h]) + int(self.offsets[h]))
        return self.table[torch.tensor(rows).cuda()].flatten().float()

    def __call__(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(new streams, next conv state)`` of streams ``h``."""
        kv = bf(self.kv.float() @ self.embedding())
        value = kv[WIDTH:]
        streams, nrms = [], []
        for b in range(4):
            sl = slice(b * HIDDEN, (b + 1) * HIDDEN)

            def norm(x, gamma):
                return x * torch.rsqrt(x.square().mean() + 1e-6) * gamma[sl].float()

            kn = bf(norm(kv[sl], self.gk))
            qn = bf(norm(h[sl].float(), self.gq))
            gg = (kn * qn).sum() * (1.0 / HIDDEN**0.5)
            gg = torch.sign(gg) * torch.sqrt(torch.clamp(gg.abs(), min=1e-6))
            gated = bf(torch.sigmoid(gg) * value)
            nrm = bf(norm(gated, self.gc))
            st = self.initial[sl].float()
            w = self.conv_w[sl, 0].float()
            acc = (
                w[:, 0] * st[:, 0]
                + w[:, 1] * st[:, 3]
                + w[:, 2] * st[:, 6]
                + w[:, 3] * nrm
            )
            act = bf(bf(torch.nn.functional.silu(acc)) + gated)
            streams.append(bf(act + h[sl].float()))
            nrms.append(nrm)
        next_state = torch.cat(
            [self.initial[:, 1:].float(), torch.cat(nrms)[:, None]], 1
        )
        return torch.cat(streams), next_state


# The PLE context is (t-2, t-1). An EOS t-1 blocks t-2 in the trigram heads.
@pytest.mark.parametrize("ple_context", [None, (1717, 3131), (1717, _Ple.EOS)])
def test_delta_layer_matches_tokenspeed_math(ple_context):
    with_ple = ple_context is not None
    g = torch.Generator().manual_seed(0)
    dev = "cuda"
    h = (torch.randn(WIDTH, generator=g) * 3).bfloat16().cuda()
    attn_mixer, mlp_mixer = _Mixer(g), _Mixer(g)
    qkvzba = _bf16_weight(QKV + VALUE + 96, HIDDEN, g)
    conv_w = (torch.randn(QKV, 4, generator=g) * 0.5).bfloat16().cuda()
    conv_state = torch.randn(3, QKV, 3, generator=g).bfloat16().cuda()
    a_log = (torch.randn(48, generator=g) * 0.5).cuda()
    dt_bias = (torch.randn(48, generator=g) * 0.5).cuda()
    gdn_norm = (1 + 0.1 * torch.randn(HEAD, generator=g)).bfloat16().cuda()
    ssm = (torch.randn(3, 48, HEAD, HEAD, generator=g) * 0.1).cuda()
    slots = torch.tensor([1, 2], dtype=torch.int32, device=dev)
    out_w = _bf16_weight(HIDDEN, VALUE, g)
    moe = _Moe(g)
    ple_block = _Ple(g, 4242, ple_context) if with_ple else None
    conv_before, ssm_before = conv_state.clone(), ssm.clone()

    tensors = {
        "attn_mixer": attn_mixer.raw,
        "qkvzba": qkvzba,
        "conv_w": conv_w,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "norm_w": gdn_norm,
        "out": out_w,
        "mlp_mixer": mlp_mixer.raw,
        **moe.tensors,
    }
    if with_ple:
        tensors["ple"] = ple_block.tensors
    assert (
        prepare_qwen4_exp_delta_layer(max_rows=5, tensors=tensors, dense_tiled=False)
        is None
    )
    fused = prepare_qwen4_exp_delta_layer(
        max_rows=1, tensors=tensors, dense_tiled=False
    )
    scratch = torch.zeros(fused.scratch_pairs, dtype=torch.int64, device=dev)
    h_out = torch.empty_like(h)
    qwen4_exp_delta_layer_decode(
        fused,
        h[None, :],
        h_out[None, :],
        chain=1,
        conv_state=conv_state,
        ssm_state=ssm,
        state_in=slots[:1],
        state_out=slots[1:],
        conv_out=conv_state,
        conv_out_rows=slots[1:],
        replay=None,
        scratch=scratch,
        step=torch.full((1,), 3, dtype=torch.int32, device=dev),
        layer_index=0,
        ple=ple_block.step() if with_ple else None,
    )
    torch.cuda.synchronize()
    h_in = h
    if with_ple:
        h, next_state = ple_block(h_in)
        torch.testing.assert_close(
            ple_block.state[2].float(), next_state, rtol=2e-2, atol=2e-2
        )
        assert ple_block.context[2].tolist() == [ple_context[1], 4242]

    # Attention half: mixer, Gated DeltaNet, output projection.
    mixed, attn_gate = attn_mixer(h)
    proj = bf(qkvzba.float() @ mixed)
    window = torch.cat([conv_before[1].float(), proj[:QKV, None]], 1)
    conv = (window * conv_w.float()).sum(1)
    q, k, v = bf(conv * torch.sigmoid(conv)).split([2048, 2048, VALUE])
    z = proj[QKV : QKV + VALUE]
    b, a = proj[QKV + VALUE :].split(48)
    state = ssm_before[1].clone()
    outputs = []
    for vh in range(48):
        kh = vh // 3
        qn = q[kh * HEAD : (kh + 1) * HEAD]
        kn = k[kh * HEAD : (kh + 1) * HEAD]
        qn = qn / torch.sqrt((qn * qn).sum() + 1e-6)
        kn = kn / torch.sqrt((kn * kn).sum() + 1e-6)
        decay = torch.exp(-torch.exp(a_log[vh]) * softplus(a[vh] + dt_bias[vh]))
        st = state[vh] * decay
        delta = torch.sigmoid(b[vh]) * (v[vh * HEAD : (vh + 1) * HEAD] - st @ kn)
        state[vh] = st + delta[:, None] * kn[None, :]
        o = bf((state[vh] @ qn) * HEAD**-0.5)
        o = o * torch.rsqrt(o.square().mean() + 1e-6)
        zz = z[vh * HEAD : (vh + 1) * HEAD]
        outputs.append(bf(o * torch.sigmoid(zz) * gdn_norm.float()))
    attn = bf(out_w.float() @ torch.cat(outputs))
    w = bf(h.float().view(4, HIDDEN) + attn[None, :] * attn_gate[:, None])

    ref = _mlp_half(w, mlp_mixer, moe)

    # With PLE the recurrence starts from streams that already carry BF16
    # rounding differences of the PLE block.
    atol = 1e-2 if with_ple else 1e-4
    torch.testing.assert_close(ssm[2], state, rtol=1e-3, atol=atol)
    assert torch.equal(ssm[1], ssm_before[1])
    assert torch.equal(conv_state[1], conv_before[1])
    # MXFP4 codes may round apart on a few elements. Compare streams in bulk.
    got = h_out.float().view(4, HIDDEN)
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=5e-2)


# L = 999 completes compressed block 249. 250 blocks need no top-k. L = 2998
# has 749 complete blocks, all with score 0 (the compressed cache is zero).
# Thus the top-k keeps blocks 0 .. 511 (ties go to the lower block). With
# ``final``, the layer also runs the model's final mixer (the last layer).
@pytest.mark.parametrize(("L", "final"), [(999, False), (2998, False), (999, True)])
def test_qsa_layer_matches_tokenspeed_math(L, final):
    page, max_pos = 256, 4096
    g = torch.Generator().manual_seed(1)
    dev = "cuda"
    h = (torch.randn(WIDTH, generator=g) * 3).bfloat16().cuda()
    attn_mixer, mlp_mixer = _Mixer(g), _Mixer(g)
    qkv = _bf16_weight(13312, HIDDEN, g)
    idx = _bf16_weight(640, HIDDEN, g)
    out_w = _bf16_weight(HIDDEN, 6144, g)
    q_norm_w = (torch.randn(256, generator=g) * 0.1).bfloat16().cuda()
    k_norm_w = (torch.randn(256, generator=g) * 0.1).bfloat16().cuda()
    iq_w = (torch.randn(128, generator=g) * 0.1).bfloat16().cuda()
    ik_w = (torch.randn(128, generator=g) * 0.1).bfloat16().cuda()
    moe = _Moe(g)
    table = _cos_sin(max_pos)
    n_pages = max_pos // page
    full_table = (torch.randperm(n_pages, generator=g) + 1).int().cuda()
    k_cache = torch.randn((n_pages + 1) * page, 2, 256, generator=g).bfloat16().cuda()
    v_cache = torch.randn((n_pages + 1) * page, 2, 256, generator=g).bfloat16().cuda()
    qsa_table = (torch.randperm(max_pos // 256, generator=g) + 1).int().cuda()
    compressed = torch.zeros(
        max_pos // 256 + 1, 64, 1, 128, dtype=torch.bfloat16
    ).cuda()
    recent_table = (torch.randperm(max_pos // 64, generator=g) + 1).int().cuda()
    raw_key = torch.randn(max_pos // 64 + 1, 4, 1, 128, generator=g).bfloat16().cuda()
    rope_position = torch.zeros(max_pos // 64 + 1, 3, dtype=torch.int64, device=dev)
    ring = int(recent_table[L // 64])
    rope_position[ring] = (L // 4) * 4
    raw_before = raw_key.clone()
    slot = lambda t: int(full_table[t // page]) * page + t % page  # noqa: E731
    complete = (L + 1) // 4
    if complete <= 512:
        attended = range(L)
    else:
        attended = [*range(512 * 4), *range(complete * 4, L)]
    kt = torch.stack([k_cache[slot(t)].float() for t in attended])
    vt = torch.stack([v_cache[slot(t)].float() for t in attended])

    tensors = {
        "attn_mixer": attn_mixer.raw,
        "qkv": qkv,
        "index_qk": idx,
        "q_norm_w": q_norm_w,
        "k_norm_w": k_norm_w,
        "iq_norm_w": iq_w,
        "ik_norm_w": ik_w,
        "out": out_w,
        "mlp_mixer": mlp_mixer.raw,
        **moe.tensors,
    }
    final_mixer = _Mixer(g) if final else None
    if final:
        # Like the model's final mixer: no inject rows in the projection.
        norm_w, projection, up = final_mixer.raw
        tensors["final_mixer"] = (norm_w, projection[:LOWRANK], up)
    fused = prepare_qwen4_exp_qsa_layer(max_rows=1, tensors=tensors, dense_tiled=False)
    scratch = torch.zeros(fused.scratch_pairs, dtype=torch.int64, device=dev)
    h_out = torch.empty_like(h)
    mixed_out = torch.empty(1, HIDDEN, dtype=torch.bfloat16, device=dev)
    step = torch.tensor([5, 0], dtype=torch.int32, device=dev)
    qwen4_exp_qsa_layer_decode(
        fused,
        h[None, :],
        h_out[None, :],
        chain=1,
        cos_sin=table,
        positions=torch.full((3, 1), L, dtype=torch.int64, device=dev),
        seq_len=torch.tensor([L + 1], dtype=torch.int32, device=dev),
        full_table=full_table[None],
        page_size=page,
        k_cache=k_cache,
        v_cache=v_cache,
        qsa_table=qsa_table[None],
        recent_table=recent_table[None],
        compressed=compressed,
        raw_key=raw_key,
        rope_position=rope_position,
        staging=None,
        draft=None,
        accepted=None,
        scratch=scratch,
        step=step,
        layer_index=3,
        mixed_out=mixed_out if final else None,
        advance_step=final,
    )
    torch.cuda.synchronize()
    # The final layer's launch (a forward's last) moves the step on as it
    # ends; the count of its finished CTAs is 0 again.
    assert step.tolist() == ([6, 0] if final else [5, 0])

    # Attention half: mixer, projections, prologue, attention, output gate.
    mixed, attn_gate = attn_mixer(h)
    proj = bf(qkv.float() @ mixed)
    idx_proj = bf(idx.float() @ mixed)
    q = [
        bf(rotate(gemma(proj[i * 512 : i * 512 + 256], q_norm_w), table, L))
        for i in range(24)
    ]
    k_new = torch.stack(
        [
            bf(
                rotate(
                    gemma(proj[12288 + i * 256 : 12544 + i * 256], k_norm_w), table, L
                )
            )
            for i in range(2)
        ]
    )
    v_new = proj[12800:13312].view(2, 256)
    keys = torch.cat([kt, k_new[None]])
    values = torch.cat([vt, v_new[None]])
    outs = []
    for i in range(24):
        p = torch.softmax(keys[:, i // 12] @ q[i] / 16.0, 0)
        a = bf(p @ values[:, i // 12])
        outs.append(bf(a * torch.sigmoid(proj[i * 512 + 256 : i * 512 + 512])))
    attn = bf(out_w.float() @ torch.cat(outs))
    w = bf(h.float().view(4, HIDDEN) + attn[None, :] * attn_gate[:, None])
    ref = _mlp_half(w, mlp_mixer, moe)

    # Caches: the token's KV, its raw index key, the block it completes.
    torch.testing.assert_close(k_cache[slot(L)].float(), k_new, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(v_cache[slot(L)].float(), v_new, rtol=2e-2, atol=2e-2)
    k_raw = idx_proj[512:]
    torch.testing.assert_close(
        raw_key[ring, L % 4, 0].float(), k_raw, rtol=2e-2, atol=2e-2
    )
    if (L + 1) % 4 == 0:
        members = [raw_before[ring, (L - o) % 4, 0].float() for o in (3, 2, 1)]
        pooled = (
            ((members[0] + members[1]) + members[2]) + raw_key[ring, L % 4, 0].float()
        ) / 4
        ck = bf(
            rotate(index_norm(pooled, (ik_w + 1.0).bfloat16()), table, (L // 4) * 4)
        )
        loc = int(qsa_table[L // 256]) * 256 + L % 256
        row = (loc // 256) * 64 + (loc % 256) // 4
        torch.testing.assert_close(
            compressed[row // 64, row % 64, 0].float(), ck, rtol=2e-2, atol=2e-2
        )
    got = h_out.float().view(4, HIDDEN)
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=5e-2)
    if final:
        # The final mixer on the layer's own output streams.
        ref_mixed, _ = final_mixer(h_out)
        torch.testing.assert_close(
            mixed_out[0].float(), ref_mixed.float(), rtol=1e-2, atol=1e-2
        )


def _delta_tensors(g, with_ple: bool) -> dict:
    """A random Gated DeltaNet layer's ``prepare`` tensors."""
    attn_mixer, mlp_mixer, moe = _Mixer(g), _Mixer(g), _Moe(g)
    tensors = {
        "attn_mixer": attn_mixer.raw,
        "qkvzba": _bf16_weight(QKV + VALUE + 96, HIDDEN, g),
        "conv_w": (torch.randn(QKV, 4, generator=g) * 0.5).bfloat16().cuda(),
        "a_log": (torch.randn(48, generator=g) * 0.5).cuda(),
        "dt_bias": (torch.randn(48, generator=g) * 0.5).cuda(),
        "norm_w": (1 + 0.1 * torch.randn(HEAD, generator=g)).bfloat16().cuda(),
        "out": _bf16_weight(HIDDEN, VALUE, g),
        "mlp_mixer": mlp_mixer.raw,
        **moe.tensors,
    }
    if with_ple:
        tensors["ple"] = _Ple(g, 4242, (1717, 3131)).tensors
    return tensors


@pytest.mark.parametrize("with_ple", [False, True])
def test_one_row_input_equals_repeated_streams(with_ple):
    """A layer given one row (stream stride 0, the first layer's embedding)
    computes exactly what it computes from the row repeated as four streams."""
    g = torch.Generator().manual_seed(2)
    dev = "cuda"
    tensors = _delta_tensors(g, with_ple)
    ple_block = _Ple(g, 4242, (1717, 3131)) if with_ple else None
    if with_ple:
        tensors["ple"] = ple_block.tensors
    fused = prepare_qwen4_exp_delta_layer(
        max_rows=1, tensors=tensors, dense_tiled=False
    )
    scratch = torch.zeros(fused.scratch_pairs, dtype=torch.int64, device=dev)
    conv0 = torch.randn(3, QKV, 3, generator=g).bfloat16().cuda()
    ssm0 = (torch.randn(3, 48, HEAD, HEAD, generator=g) * 0.1).cuda()
    row = (torch.randn(1, HIDDEN, generator=g) * 3).bfloat16().cuda()
    slots = torch.tensor([1, 2], dtype=torch.int32, device=dev)
    results = []
    for step, streams in enumerate((row, row.repeat(1, 4))):
        conv, ssm = conv0.clone(), ssm0.clone()
        # Both runs read the same input pages (they write page 2 only).
        ple_step = ple_block.step() if with_ple else None
        h_out = torch.empty(1, WIDTH, dtype=torch.bfloat16, device=dev)
        qwen4_exp_delta_layer_decode(
            fused,
            streams,
            h_out,
            chain=1,
            conv_state=conv,
            ssm_state=ssm,
            state_in=slots[:1],
            state_out=slots[1:],
            conv_out=conv,
            conv_out_rows=slots[1:],
            replay=None,
            scratch=scratch,
            step=torch.full((1,), 7 + step, dtype=torch.int32, device=dev),
            layer_index=0,
            ple=ple_step,
        )
        torch.cuda.synchronize()
        results.append((h_out, ssm[2], conv[2]))
    for one_row, repeated in zip(*results):
        assert torch.equal(one_row, repeated)


@pytest.mark.parametrize(
    "requests, chain, with_ple",
    [
        (1, 2, False),
        (1, 3, False),
        (1, 4, False),
        (2, 2, False),
        (1, 4, True),
        (2, 2, True),
    ],
)
def test_delta_layer_verify_equals_decode_steps(requests, chain, with_ple):
    """A verify launch of ``chain`` tokens per request computes what as many
    one-token decode steps do. It gives:

    * the same output streams,
    * each token's conv windows (and PLE context) in its scratch rows,
    * a replay payload from which TokenSpeed's replay commit rebuilds the
      state after the last token.

    It writes no state page."""
    g = torch.Generator().manual_seed(40 + requests * chain)
    dev = "cuda"
    rows = requests * chain
    tensors = _delta_tensors(g, with_ple)
    fused = prepare_qwen4_exp_delta_layer(
        max_rows=rows, tensors=tensors, dense_tiled=False
    )
    scratch = torch.zeros(fused.scratch_pairs, dtype=torch.int64, device=dev)
    # Request r reads page base[r]. Decode step t writes page base[r] + 1 + t.
    base = torch.arange(requests, dtype=torch.int32, device=dev) * (chain + 1) + 1
    pages = requests * (chain + 1) + 1
    conv0 = torch.randn(pages, QKV, 3, generator=g).bfloat16().cuda()
    ssm0 = (torch.randn(pages, 48, HEAD, HEAD, generator=g) * 0.1).cuda()
    h = (torch.randn(rows, WIDTH, generator=g) * 3).bfloat16().cuda()
    t = torch.arange(chain, dtype=torch.int32, device=dev)
    scratch_rows = ((base - 1)[:, None] + 1 + t[None, :]).flatten()
    tokens = torch.randint(10, 9000, (rows,), generator=g).cuda()
    context0 = torch.randint(10, 9000, (pages, 2), generator=g).cuda()
    ple0 = torch.randn(pages, WIDTH, 9, generator=g).bfloat16().cuda()
    context, ple_state = context0.clone(), ple0.clone()
    context_windows = torch.zeros(
        requests * (chain + 1), 2, dtype=torch.int64, device=dev
    )
    ple_windows = torch.zeros(
        requests * (chain + 1), WIDTH, 9, dtype=torch.bfloat16, device=dev
    )

    conv, ssm = conv0.clone(), ssm0.clone()
    windows = torch.zeros(
        requests * (chain + 1), QKV, 3, dtype=torch.bfloat16, device=dev
    )
    payload = torch.zeros(rows, QKV - 2048 + 96, dtype=torch.bfloat16, device=dev)
    h_verify = torch.empty(rows, WIDTH, dtype=torch.bfloat16, device=dev)
    qwen4_exp_delta_layer_decode(
        fused,
        h,
        h_verify,
        chain=chain,
        conv_state=conv,
        ssm_state=ssm,
        state_in=base,
        state_out=torch.full((requests,), -1, dtype=torch.int32, device=dev),
        conv_out=windows,
        conv_out_rows=scratch_rows,
        replay=payload,
        scratch=scratch,
        step=torch.full((1,), 3, dtype=torch.int32, device=dev),
        layer_index=0,
        ple=(
            {
                "token": tokens,
                "context": context,
                "conv_state": ple_state,
                "page_in": base,
                "page_out": scratch_rows,
                "context_out": context_windows,
                "conv_out": ple_windows,
            }
            if with_ple
            else None
        ),
    )
    torch.cuda.synchronize()
    assert torch.equal(conv, conv0) and torch.equal(ssm, ssm0)
    assert torch.equal(context, context0) and torch.equal(ple_state, ple0)

    h_steps = torch.empty_like(h_verify)
    for step in range(chain):
        token = torch.arange(requests, device=dev) * chain + step
        h_out = torch.empty(requests, WIDTH, dtype=torch.bfloat16, device=dev)
        qwen4_exp_delta_layer_decode(
            fused,
            h[token],
            h_out,
            chain=1,
            conv_state=conv,
            ssm_state=ssm,
            state_in=base + step,
            state_out=base + step + 1,
            conv_out=conv,
            conv_out_rows=base + step + 1,
            replay=None,
            scratch=scratch,
            step=torch.full((1,), 100 + step, dtype=torch.int32, device=dev),
            layer_index=0,
            ple=(
                {
                    "token": tokens[token],
                    "context": context,
                    "conv_state": ple_state,
                    "page_in": base + step,
                    "page_out": base + step + 1,
                    "context_out": context,
                    "conv_out": ple_state,
                }
                if with_ple
                else None
            ),
        )
        h_steps[token] = h_out
    torch.cuda.synchronize()
    assert torch.equal(h_verify, h_steps)
    for r in range(requests):
        first = int(base[r])
        for step in range(chain):
            row = (first - 1) + 1 + step
            assert torch.equal(windows[row], conv[first + 1 + step])
            if with_ple:
                assert torch.equal(context_windows[row], context[first + 1 + step])
                assert torch.equal(ple_windows[row], ple_state[first + 1 + step])

    replayed = ssm0.clone()
    gdn_replay_commit(
        payload[None],
        torch.stack((tensors["a_log"], tensors["dt_bias"]))[None].float(),
        state_addresses=torch.tensor(
            [replayed.data_ptr()], dtype=torch.uint64, device=dev
        ),
        state_row_strides=torch.tensor(
            [replayed.stride(0)], dtype=torch.int64, device=dev
        ),
        read_indices=base[None],
        write_indices=(base + chain)[None],
        accepted_length=torch.full((requests,), chain, dtype=torch.int32, device=dev),
        draft_token_num=chain,
        geometry=(16, 48, HEAD, HEAD),
        state_dtype=torch.float32,
    )
    final = (base + chain).long()
    torch.testing.assert_close(replayed[final], ssm[final], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("rows", [2, 3, 4])
@pytest.mark.parametrize("with_ple", [False, True])
def test_delta_layer_rows_equal_one_row_launches(rows, with_ple):
    """Each request row of a multi-row launch computes and writes exactly
    what a launch of that row alone does."""
    g = torch.Generator().manual_seed(10 + rows)
    dev = "cuda"
    tensors = _delta_tensors(g, with_ple)
    fused = prepare_qwen4_exp_delta_layer(
        max_rows=rows, tensors=tensors, dense_tiled=False
    )
    scratch = torch.zeros(fused.scratch_pairs, dtype=torch.int64, device=dev)
    pages = 2 * rows + 1
    # Row r reads page 1 + 2 r and writes page 2 + 2 r.
    page_in = torch.arange(rows, dtype=torch.int32, device=dev) * 2 + 1
    page_out = page_in + 1
    h = (torch.randn(rows, WIDTH, generator=g) * 3).bfloat16().cuda()
    initial = {
        "conv": torch.randn(pages, QKV, 3, generator=g).bfloat16().cuda(),
        "ssm": (torch.randn(pages, 48, HEAD, HEAD, generator=g) * 0.1).cuda(),
        "ple_state": torch.randn(pages, WIDTH, 9, generator=g).bfloat16().cuda(),
        "context": torch.randint(10, 9000, (pages, 2), generator=g).cuda(),
    }
    tokens = torch.randint(10, 9000, (rows,), generator=g).cuda()

    def run(state, rows_of_launch, step):
        for first in range(0, rows, rows_of_launch):
            r = slice(first, first + rows_of_launch)
            h_out = state["h_out"][r]
            ple = (
                {
                    "token": tokens[r],
                    "context": state["context"],
                    "conv_state": state["ple_state"],
                    "page_in": page_in[r],
                    "page_out": page_out[r],
                    "context_out": state["context"],
                    "conv_out": state["ple_state"],
                }
                if with_ple
                else None
            )
            qwen4_exp_delta_layer_decode(
                fused,
                h[r],
                h_out,
                chain=1,
                conv_state=state["conv"],
                ssm_state=state["ssm"],
                state_in=page_in[r],
                state_out=page_out[r],
                conv_out=state["conv"],
                conv_out_rows=page_out[r],
                replay=None,
                scratch=scratch,
                step=torch.full((1,), step + first, dtype=torch.int32, device=dev),
                layer_index=0,
                ple=ple,
            )
        torch.cuda.synchronize()

    together = {k: v.clone() for k, v in initial.items()}
    alone = {k: v.clone() for k, v in initial.items()}
    together["h_out"] = torch.empty(rows, WIDTH, dtype=torch.bfloat16, device=dev)
    alone["h_out"] = torch.empty_like(together["h_out"])
    run(together, rows, 3)
    run(alone, 1, 100)
    for name in together:
        assert torch.equal(together[name], alone[name]), name


def _qsa_tensors(g, final: bool, experts: str) -> dict:
    """A random QSA layer's ``prepare`` tensors, with MXFP4 or BF16
    (``experts``) experts."""
    attn_mixer, mlp_mixer = _Mixer(g), _Mixer(g)
    moe = _Moe(g).tensors if experts == "mxfp4" else _bf16_experts()
    tensors = {
        "attn_mixer": attn_mixer.raw,
        "qkv": _bf16_weight(13312, HIDDEN, g),
        "index_qk": _bf16_weight(640, HIDDEN, g),
        "q_norm_w": (torch.randn(256, generator=g) * 0.1).bfloat16().cuda(),
        "k_norm_w": (torch.randn(256, generator=g) * 0.1).bfloat16().cuda(),
        "iq_norm_w": (torch.randn(128, generator=g) * 0.1).bfloat16().cuda(),
        "ik_norm_w": (torch.randn(128, generator=g) * 0.1).bfloat16().cuda(),
        "out": _bf16_weight(HIDDEN, 6144, g),
        "mlp_mixer": mlp_mixer.raw,
        **moe,
    }
    if final:
        norm_w, projection, up = _Mixer(g).raw
        tensors["final_mixer"] = (norm_w, projection[:LOWRANK], up)
    return tensors


# Rows at different positions. 2998 needs the top-k (749 complete blocks).
@pytest.mark.parametrize(
    ("lengths", "final", "experts"),
    [
        ((999, 37), False, "mxfp4"),
        ((2998, 1500, 400), False, "mxfp4"),
        ((999, 2998, 64, 1203), True, "mxfp4"),
        ((999, 2998, 64, 1203), True, "bf16"),
    ],
)
def test_qsa_layer_rows_equal_one_row_launches(lengths, final, experts):
    """Each request row of a multi-row launch computes and writes exactly
    what a launch of that row alone does."""
    rows, page, max_pos = len(lengths), 256, 4096
    g = torch.Generator().manual_seed(20 + rows)
    dev = "cuda"
    tensors = _qsa_tensors(g, final, experts)
    fused = prepare_qwen4_exp_qsa_layer(
        max_rows=rows, tensors=tensors, dense_tiled=False
    )
    scratch = torch.zeros(fused.scratch_pairs, dtype=torch.int64, device=dev)
    h = (torch.randn(rows, WIDTH, generator=g) * 3).bfloat16().cuda()
    L = torch.tensor(lengths, device=dev)

    def tables(per_row: int) -> torch.Tensor:
        """Disjoint page tables of ``per_row`` pages per row (page 0 unused)."""
        perm = torch.randperm(rows * per_row, generator=g) + 1
        return perm.view(rows, per_row).int().cuda()

    full_table, qsa_table, recent_table = (
        tables(max_pos // page),
        tables(max_pos // 256),
        tables(max_pos // 64),
    )
    ring_pages = rows * (max_pos // 64) + 1
    initial = {
        "k_cache": torch.randn((rows * max_pos // page + 1) * page, 2, 256, generator=g)
        .bfloat16()
        .cuda(),
        "v_cache": torch.randn((rows * max_pos // page + 1) * page, 2, 256, generator=g)
        .bfloat16()
        .cuda(),
        "compressed": (
            torch.randn(rows * max_pos // 256 + 1, 64, 1, 128, generator=g) * 0.1
        )
        .bfloat16()
        .cuda(),
        "raw_key": torch.randn(ring_pages, 4, 1, 128, generator=g).bfloat16().cuda(),
        "rope_position": torch.zeros(ring_pages, 3, dtype=torch.int64, device=dev),
    }
    for r, length in enumerate(lengths):
        ring = int(recent_table[r, length // 64])
        initial["rope_position"][ring] = (length // 4) * 4

    def run(state, rows_of_launch, step):
        for first in range(0, rows, rows_of_launch):
            r = slice(first, first + rows_of_launch)
            qwen4_exp_qsa_layer_decode(
                fused,
                h[r],
                state["h_out"][r],
                chain=1,
                cos_sin=_cos_sin(max_pos),
                positions=L[r][None].expand(3, -1).contiguous(),
                seq_len=(L[r] + 1).int(),
                full_table=full_table[r],
                page_size=page,
                k_cache=state["k_cache"],
                v_cache=state["v_cache"],
                qsa_table=qsa_table[r],
                recent_table=recent_table[r],
                compressed=state["compressed"],
                raw_key=state["raw_key"],
                rope_position=state["rope_position"],
                staging=None,
                draft=None,
                accepted=None,
                scratch=scratch,
                step=torch.full((1,), step + first, dtype=torch.int32, device=dev),
                layer_index=3,
                mixed_out=state["mixed"][r] if final else None,
                advance_step=False,
            )
        torch.cuda.synchronize()

    together = {k: v.clone() for k, v in initial.items()}
    alone = {k: v.clone() for k, v in initial.items()}
    for state in (together, alone):
        state["h_out"] = torch.empty(rows, WIDTH, dtype=torch.bfloat16, device=dev)
        state["mixed"] = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
    run(together, rows, 5)
    run(alone, 1, 200)
    for name in together:
        if name != "mixed" or final:
            assert torch.equal(together[name], alone[name]), name


# Requests' first verified positions are block-aligned and not, and before
# and past the top-k (2998 has 749 complete blocks). One case also
# completes a block inside the verify window.
@pytest.mark.parametrize(
    ("starts", "chain", "final"),
    [
        ((997,), 4, False),
        ((2996,), 4, False),
        ((1001, 38), 2, True),
        ((2998,), 3, False),
    ],
)
def test_qsa_layer_verify_equals_decode_steps(starts, chain, final):
    """A verify launch of ``chain`` tokens per request computes what as many
    one-token decode steps do (outputs, KV and compressed caches). It also
    stages each token's raw index key, positions and ring location instead
    of writing the recent ring."""
    requests, page, max_pos = len(starts), 256, 4096
    rows = requests * chain
    g = torch.Generator().manual_seed(60 + rows)
    dev = "cuda"
    tensors = _qsa_tensors(g, final, "mxfp4")
    fused = prepare_qwen4_exp_qsa_layer(
        max_rows=rows, tensors=tensors, dense_tiled=False
    )
    scratch = torch.zeros(fused.scratch_pairs, dtype=torch.int64, device=dev)
    h = (torch.randn(rows, WIDTH, generator=g) * 3).bfloat16().cuda()

    def tables(per_row: int) -> torch.Tensor:
        perm = torch.randperm(requests * per_row, generator=g) + 1
        return perm.view(requests, per_row).int().cuda()

    full_table, qsa_table, recent_table = (
        tables(max_pos // page),
        tables(max_pos // 256),
        tables(max_pos // 64),
    )
    ring_pages = requests * (max_pos // 64) + 1
    initial = {
        "k_cache": torch.randn(
            (requests * max_pos // page + 1) * page, 2, 256, generator=g
        )
        .bfloat16()
        .cuda(),
        "v_cache": torch.randn(
            (requests * max_pos // page + 1) * page, 2, 256, generator=g
        )
        .bfloat16()
        .cuda(),
        "compressed": (
            torch.randn(requests * max_pos // 256 + 1, 64, 1, 128, generator=g) * 0.1
        )
        .bfloat16()
        .cuda(),
        "raw_key": torch.randn(ring_pages, 4, 1, 128, generator=g).bfloat16().cuda(),
        "rope_position": torch.zeros(ring_pages, 3, dtype=torch.int64, device=dev),
    }
    for r, start in enumerate(starts):
        ring = int(recent_table[r, start // 64])
        initial["rope_position"][ring] = (start // 4) * 4
    # Row r * chain + t: token t of request r, at starts[r] + t.
    L = (
        torch.tensor(starts, device=dev)[:, None] + torch.arange(chain, device=dev)
    ).flatten()
    request = torch.arange(requests, device=dev).repeat_interleave(chain)

    def launch(state, row, n, step, staging):
        # Rows by index: the kernel writes copies, and the code below scatters
        # them back.
        h_out = torch.empty(len(row), WIDTH, dtype=torch.bfloat16, device=dev)
        mixed = torch.empty(len(row), HIDDEN, dtype=torch.bfloat16, device=dev)
        qwen4_exp_qsa_layer_decode(
            fused,
            h[row],
            h_out,
            chain=n,
            cos_sin=_cos_sin(max_pos),
            positions=L[row][None].expand(3, -1).contiguous(),
            # Per request: its last row's L + 1, and its tables.
            seq_len=(L[row].view(-1, n)[:, -1] + 1).int(),
            full_table=full_table[request[row].view(-1, n)[:, 0]],
            page_size=page,
            k_cache=state["k_cache"],
            v_cache=state["v_cache"],
            qsa_table=qsa_table[request[row].view(-1, n)[:, 0]],
            recent_table=recent_table[request[row].view(-1, n)[:, 0]],
            compressed=state["compressed"],
            raw_key=state["raw_key"],
            rope_position=state["rope_position"],
            staging=staging,
            draft=None,
            accepted=None,
            scratch=scratch,
            step=torch.full((1,), step, dtype=torch.int32, device=dev),
            layer_index=3,
            mixed_out=mixed if final else None,
            advance_step=False,
        )
        torch.cuda.synchronize()
        state["h_out"][row] = h_out
        state["mixed"][row] = mixed

    verify = {k: v.clone() for k, v in initial.items()}
    steps = {k: v.clone() for k, v in initial.items()}
    for state in (verify, steps):
        state["h_out"] = torch.empty(rows, WIDTH, dtype=torch.bfloat16, device=dev)
        state["mixed"] = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
    staged = (
        torch.zeros(rows, 128, dtype=torch.bfloat16, device=dev),
        torch.zeros(rows, 3, dtype=torch.int64, device=dev),
        torch.zeros(rows, dtype=torch.int64, device=dev),
        torch.zeros(rows, dtype=torch.int32, device=dev),
    )
    launch(verify, torch.arange(rows, device=dev), chain, 5, staged)
    for t in range(chain):
        launch(steps, torch.arange(requests, device=dev) * chain + t, 1, 100 + t, None)

    for name in ("h_out", "k_cache", "v_cache", "compressed") + (
        ("mixed",) if final else ()
    ):
        assert torch.equal(verify[name], steps[name]), name
    # The verify writes no ring. It stages what the decode steps wrote there.
    assert torch.equal(verify["raw_key"], initial["raw_key"])
    assert torch.equal(verify["rope_position"], initial["rope_position"])
    ring = recent_table[request, L // 64].long()
    assert torch.equal(staged[0], steps["raw_key"][ring, L % 4, 0])
    assert torch.equal(staged[1], L[:, None].expand(-1, 3))
    assert torch.equal(staged[2], L)
    assert torch.equal(staged[3], (ring * 64 + L % 64).int())


# Four draft positions from a block's first token (before and past the top-k),
# and from its third token (the completed block mixes ring and draft members).
@pytest.mark.parametrize("experts", ["mxfp4", "bf16"])
@pytest.mark.parametrize("start", [1000, 1002, 2996])
def test_qsa_layer_draft_steps_equal_decode_steps(start, experts):
    """MTP draft steps (one row, ``draft`` scratch) compute what decode steps
    compute (outputs, KV and compressed caches). They stage each index key
    and its group-start position in the draft scratch, which the next steps'
    compression reads, and write no ring."""
    page, max_pos = 256, 4096
    g = torch.Generator().manual_seed(80 + start % 7)
    dev = "cuda"
    tensors = _qsa_tensors(g, False, experts)
    fused = prepare_qwen4_exp_qsa_layer(max_rows=1, tensors=tensors, dense_tiled=False)
    scratch = torch.zeros(fused.scratch_pairs, dtype=torch.int64, device=dev)
    h = (torch.randn(4, WIDTH, generator=g) * 3).bfloat16().cuda()

    def table(pages: int) -> torch.Tensor:
        return (torch.randperm(pages, generator=g) + 1).view(1, pages).int().cuda()

    full_table, qsa_table, recent_table = (
        table(max_pos // page),
        table(max_pos // 256),
        table(max_pos // 64),
    )
    ring_pages = max_pos // 64 + 1
    initial = {
        "k_cache": torch.randn((max_pos // page + 1) * page, 2, 256, generator=g)
        .bfloat16()
        .cuda(),
        "v_cache": torch.randn((max_pos // page + 1) * page, 2, 256, generator=g)
        .bfloat16()
        .cuda(),
        "compressed": (torch.randn(max_pos // 256 + 1, 64, 1, 128, generator=g) * 0.1)
        .bfloat16()
        .cuda(),
        "raw_key": torch.randn(ring_pages, 4, 1, 128, generator=g).bfloat16().cuda(),
        "rope_position": torch.zeros(ring_pages, 3, dtype=torch.int64, device=dev),
    }
    draft = (
        torch.zeros(1, 4, 1, 128, dtype=torch.bfloat16, device=dev),
        torch.zeros(1, 3, dtype=torch.int64, device=dev),
        torch.full((1, 4), -1, dtype=torch.int64, device=dev),
    )

    def launch(state, t, draft_scratch):
        L = start + t
        h_out = torch.empty(1, WIDTH, dtype=torch.bfloat16, device=dev)
        qwen4_exp_qsa_layer_decode(
            fused,
            h[t : t + 1],
            h_out,
            chain=1,
            cos_sin=_cos_sin(max_pos),
            positions=torch.full((3, 1), L, dtype=torch.int64, device=dev),
            seq_len=torch.full((1,), L + 1, dtype=torch.int32, device=dev),
            full_table=full_table,
            page_size=page,
            k_cache=state["k_cache"],
            v_cache=state["v_cache"],
            qsa_table=qsa_table,
            recent_table=recent_table,
            compressed=state["compressed"],
            raw_key=state["raw_key"],
            rope_position=state["rope_position"],
            staging=None,
            draft=draft_scratch,
            accepted=None,
            scratch=scratch,
            step=torch.full((1,), 10 + t, dtype=torch.int32, device=dev),
            layer_index=0,
            mixed_out=None,
            advance_step=False,
        )
        torch.cuda.synchronize()
        return h_out

    decode = {k: v.clone() for k, v in initial.items()}
    drafted = {k: v.clone() for k, v in initial.items()}
    for t in range(4):
        assert torch.equal(launch(decode, t, None), launch(drafted, t, draft)), t
        for name in ("k_cache", "v_cache", "compressed"):
            assert torch.equal(decode[name], drafted[name]), (name, t)
    assert torch.equal(drafted["raw_key"], initial["raw_key"])
    assert torch.equal(drafted["rope_position"], initial["rope_position"])
    for t in range(4):
        L = start + t
        ring = int(recent_table[0, L // 64])
        assert torch.equal(draft[0][0, L % 4], decode["raw_key"][ring, L % 4])
        assert int(draft[2][0, L % 4]) == L
        if L % 4 == 0:
            assert torch.equal(draft[1][0], decode["rope_position"][ring])


# Rows 997-1000: 999 completes a block. Accepted lengths before, at and past it.
@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("experts", ["mxfp4", "bf16"])
@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
def test_qsa_layer_draft_catch_up_writes_accepted_index_keys(accepted, experts, final):
    """The MTP draft's first step runs the verified rows as a chain. Every row
    computes and writes K / V as a decode step does. Only the accepted rows
    write index keys and compressed keys (TokenSpeed's draft write mask).
    Only the live row, the last accepted one, gives an output (TokenSpeed's
    draft narrowing): its decode step's output and final mixer output."""
    start, chain, page, max_pos = 997, 4, 256, 4096
    g = torch.Generator().manual_seed(90 + accepted)
    dev = "cuda"
    tensors = _qsa_tensors(g, final, experts)
    fused = prepare_qwen4_exp_qsa_layer(
        max_rows=chain, tensors=tensors, dense_tiled=False
    )
    scratch = torch.zeros(fused.scratch_pairs, dtype=torch.int64, device=dev)
    h = (torch.randn(chain, WIDTH, generator=g) * 3).bfloat16().cuda()

    def table(pages: int) -> torch.Tensor:
        return (torch.randperm(pages, generator=g) + 1).view(1, pages).int().cuda()

    full_table, qsa_table, recent_table = (
        table(max_pos // page),
        table(max_pos // 256),
        table(max_pos // 64),
    )
    ring_pages = max_pos // 64 + 1
    initial = {
        "k_cache": torch.randn((max_pos // page + 1) * page, 2, 256, generator=g)
        .bfloat16()
        .cuda(),
        "v_cache": torch.randn((max_pos // page + 1) * page, 2, 256, generator=g)
        .bfloat16()
        .cuda(),
        "compressed": (torch.randn(max_pos // 256 + 1, 64, 1, 128, generator=g) * 0.1)
        .bfloat16()
        .cuda(),
        "raw_key": torch.randn(ring_pages, 4, 1, 128, generator=g).bfloat16().cuda(),
        "rope_position": torch.zeros(ring_pages, 3, dtype=torch.int64, device=dev),
    }
    ring = int(recent_table[0, start // 64])
    initial["rope_position"][ring] = (start // 4) * 4

    def launch(state, first, n, step, limit):
        L = torch.arange(start + first, start + first + n, device=dev)
        out_rows = n if limit is None else 1
        h_out = torch.empty(out_rows, WIDTH, dtype=torch.bfloat16, device=dev)
        mixed = (
            torch.empty(out_rows, HIDDEN, dtype=torch.bfloat16, device=dev)
            if final
            else None
        )
        qwen4_exp_qsa_layer_decode(
            fused,
            h[first : first + n],
            h_out,
            chain=n,
            cos_sin=_cos_sin(max_pos),
            positions=L[None].expand(3, -1).contiguous(),
            # One request: its last token's L + 1, and its tables.
            seq_len=(L[-1:] + 1).int(),
            full_table=full_table,
            page_size=page,
            k_cache=state["k_cache"],
            v_cache=state["v_cache"],
            qsa_table=qsa_table,
            recent_table=recent_table,
            compressed=state["compressed"],
            raw_key=state["raw_key"],
            rope_position=state["rope_position"],
            staging=None,
            draft=None,
            accepted=limit,
            scratch=scratch,
            step=torch.full((1,), step, dtype=torch.int32, device=dev),
            layer_index=0,
            mixed_out=mixed,
            advance_step=False,
        )
        torch.cuda.synchronize()
        return h_out, mixed

    caught_up = {k: v.clone() for k, v in initial.items()}
    out = launch(
        caught_up,
        0,
        chain,
        5,
        torch.full((1,), start + accepted, dtype=torch.int32, device=dev),
    )
    every_step = {k: v.clone() for k, v in initial.items()}
    accepted_steps = {k: v.clone() for k, v in initial.items()}
    for t in range(chain):
        decode = launch(every_step, t, 1, 100 + t, None)
        if t == accepted - 1:
            assert torch.equal(out[0], decode[0])
            if final:
                assert torch.equal(out[1], decode[1])
        if t < accepted:
            launch(accepted_steps, t, 1, 200 + t, None)
    for name in ("k_cache", "v_cache"):
        assert torch.equal(caught_up[name], every_step[name]), name
    for name in ("raw_key", "rope_position", "compressed"):
        assert torch.equal(caught_up[name], accepted_steps[name]), name
