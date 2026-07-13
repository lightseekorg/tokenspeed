from __future__ import annotations

import math

import pytest
import torch


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


_IS_GFX950 = _is_gfx950()
if not _IS_GFX950:
    pytest.skip(
        "Gluon MLA prefill kernel is gfx950 (CDNA4) only",
        allow_module_level=True,
    )

from tokenspeed_kernel import mla_prefill

_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_QK_HEAD_DIM = _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM  # 192
_V_HEAD_DIM = 128


def _cu_seqlens(lens: list[int], device: str) -> torch.Tensor:
    out = torch.zeros(len(lens) + 1, dtype=torch.int32, device=device)
    out[1:] = torch.tensor(lens, dtype=torch.int32, device=device).cumsum(0)
    return out


def _torch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    scale: float,
    is_causal: bool,
    kv_group_num: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, n_heads, _ = q.shape
    out = torch.zeros(
        total_q, n_heads, _V_HEAD_DIM, device=q.device, dtype=torch.float32
    )
    lse = torch.zeros(total_q, n_heads, device=q.device, dtype=torch.float32)
    batch = cu_q.numel() - 1
    for b in range(batch):
        qs, qe = int(cu_q[b]), int(cu_q[b + 1])
        ks, ke = int(cu_kv[b]), int(cu_kv[b + 1])
        q_len = qe - qs
        kv_len = ke - ks
        q_causal_start = max(kv_len - q_len, 0)
        for h in range(n_heads):
            kh = h // kv_group_num
            qb = q[qs:qe, h].float()  # [q_len, 192]
            kb = k[ks:ke, kh].float()  # [kv_len, 192]
            vb = v[ks:ke, kh].float()  # [kv_len, 128]
            scores = (qb @ kb.transpose(0, 1)) * scale  # [q_len, kv_len]
            if is_causal:
                qpos = q_causal_start + torch.arange(q_len, device=q.device)[:, None]
                kpos = torch.arange(kv_len, device=q.device)[None, :]
                scores = scores.masked_fill(kpos > qpos, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[qs:qe, h] = probs @ vb
            lse[qs:qe, h] = torch.logsumexp(scores, dim=-1)
    return out, lse


@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize(
    "q_lens,kv_lens,num_heads,num_kv_heads",
    [
        # Causal self-attention (extend): q_len == kv_len.
        ([200, 64, 129, 1], [200, 64, 129, 1], 16, 16),
        ([300, 5], [300, 5], 8, 8),
        # Ragged cross / right-aligned causal: kv_len >= q_len (prefix + extend).
        ([64, 100], [320, 250], 16, 16),
        ([130], [600], 4, 4),
        # GQA-ish grouping.
        ([128, 96], [128, 96], 16, 4),
    ],
)
def test_gluon_mla_prefill_parity(
    is_causal: bool,
    q_lens: list[int],
    kv_lens: list[int],
    num_heads: int,
    num_kv_heads: int,
) -> None:
    device = "cuda"
    torch.manual_seed(0)
    cu_q = _cu_seqlens(q_lens, device)
    cu_kv = _cu_seqlens(kv_lens, device)
    total_q = int(cu_q[-1])
    total_kv = int(cu_kv[-1])
    kv_group_num = num_heads // num_kv_heads

    q = torch.randn(
        total_q, num_heads, _QK_HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    k = torch.randn(
        total_kv, num_kv_heads, _QK_HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    v = torch.randn(
        total_kv, num_kv_heads, _V_HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    scale = 1.0 / math.sqrt(_QK_HEAD_DIM)

    out_g, lse_g = mla_prefill(
        q,
        k,
        v,
        cu_q,
        cu_kv,
        max(q_lens),
        max(kv_lens),
        scale,
        is_causal=is_causal,
        return_lse=True,
        solution="gluon",
    )
    out_ref, lse_ref = _torch_ref(q, k, v, cu_q, cu_kv, scale, is_causal, kv_group_num)

    assert out_g.shape == (total_q, num_heads, _V_HEAD_DIM)
    assert lse_g.shape == (total_q, num_heads)
    torch.testing.assert_close(out_g.float(), out_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse_g.float(), lse_ref, rtol=2e-2, atol=2e-2)


def test_gluon_mla_prefill_is_selected() -> None:
    """The gluon kernel wins dispatch for the 192/128 bf16 MLA prefill shape and
    falls back to triton outside its supported regime."""
    from tokenspeed_kernel.ops.attention import _attention_format_signature
    from tokenspeed_kernel.selection import select_kernel

    def pick(qk_head_dim: int, v_head_dim: int, dtype: torch.dtype) -> str:
        q = torch.empty(4, 16, qk_head_dim, dtype=dtype, device="cuda")
        k = torch.empty(4, 16, qk_head_dim, dtype=dtype, device="cuda")
        v = torch.empty(4, 16, v_head_dim, dtype=dtype, device="cuda")
        sig = _attention_format_signature(q=q, k=k, v=v)
        traits = {
            "qk_head_dim": qk_head_dim,
            "v_head_dim": v_head_dim,
            "is_causal": True,
            "support_logit_cap": False,
            "return_lse": False,
        }
        return select_kernel("attention", "mla_prefill", sig, traits=traits).name

    assert pick(192, 128, torch.bfloat16) == "gluon_mla_prefill_bf16_gfx950"
    # Unsupported dtype and head dims fall back to the portable triton kernel.
    assert pick(192, 128, torch.float16) == "triton_mla_prefill"
    assert pick(576, 512, torch.bfloat16) == "triton_mla_prefill"
