import torch
from tokenspeed_kernel.registry import error_fn

from tokenspeed.runtime.models import deepseek_v3


def test_portable_mla_kv_fp8_pack_matches_torch_reference(monkeypatch) -> None:
    monkeypatch.setattr(deepseek_v3, "mla_kv_pack_quantize_fp8", error_fn)
    torch.manual_seed(7)
    k_nope = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    k_pe = torch.randn(2, 1, 2, dtype=torch.bfloat16)
    v = torch.randn(2, 3, 5, dtype=torch.bfloat16)

    k, v_out = deepseek_v3._pack_mla_kv_fp8(
        k_nope,
        k_pe,
        v,
        k_scale_inv=0.5,
        v_scale_inv=1.5,
    )
    expected_k = torch.cat([k_nope, k_pe.expand(-1, 3, -1)], dim=-1)
    expected_k = (expected_k * 0.5).to(torch.float8_e4m3fn)
    expected_v = (v * 1.5).to(torch.float8_e4m3fn)
    assert torch.equal(k, expected_k)
    assert torch.equal(v_out, expected_v)


def test_portable_mla_kv_fp8_pack_accepts_2d_rope(monkeypatch) -> None:
    monkeypatch.setattr(deepseek_v3, "mla_kv_pack_quantize_fp8", error_fn)
    k_nope = torch.zeros(1, 2, 3, dtype=torch.bfloat16)
    k_pe = torch.ones(1, 2, dtype=torch.bfloat16)
    v = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    k, _ = deepseek_v3._pack_mla_kv_fp8(k_nope, k_pe, v)
    assert k.shape == (1, 2, 5)
