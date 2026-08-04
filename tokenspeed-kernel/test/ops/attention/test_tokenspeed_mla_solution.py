import importlib
import math
from pathlib import Path

import pytest
import tokenspeed_kernel.ops.attention.mla.tokenspeed_mla as tokenspeed_mla
import tokenspeed_kernel.ops.attention.mla.tokenspeed_mla.decode as decode
import tokenspeed_kernel.ops.attention.mla.tokenspeed_mla.prefill as prefill
import torch
from tokenspeed_kernel.platform import ArchVersion
from tokenspeed_kernel.registry import KernelRegistry, Priority


def test_tokenspeed_mla_is_solution_package() -> None:
    module_path = Path(tokenspeed_mla.__file__)

    assert module_path.name == "__init__.py"
    assert module_path.parent.name == "tokenspeed_mla"
    assert callable(tokenspeed_mla.tokenspeed_mla_decode)
    assert callable(tokenspeed_mla.tokenspeed_mla_prefill)
    assert callable(tokenspeed_mla.mla_kv_pack_quantize_fp8)


def test_tokenspeed_mla_registry_specs(fresh_registry) -> None:
    importlib.reload(decode)
    importlib.reload(prefill)

    registry = KernelRegistry.get()
    decode_spec = next(
        spec
        for spec in registry.list_kernels("attention", "mla_decode_with_kvcache")
        if spec.name == "tokenspeed_mla_decode_with_kvcache"
    )
    prefill_spec = next(
        spec
        for spec in registry.list_kernels("attention", "mla_prefill")
        if spec.name == "tokenspeed_mla_prefill"
    )

    for spec in (decode_spec, prefill_spec):
        assert spec.solution == "tokenspeed_mla"
        assert spec.priority == Priority.SPECIALIZED
        assert spec.capability.vendors == frozenset({"nvidia"})
        assert spec.capability.min_arch_version == ArchVersion(10, 0)
        assert spec.capability.max_arch_version == ArchVersion(10, 3)


def test_decode_adapter_normalizes_cache_and_lse(monkeypatch) -> None:
    captured = {}
    workspace = torch.empty(1, dtype=torch.int8)

    def fake_decode(**kwargs):
        captured.update(kwargs)
        output = torch.empty((*kwargs["query"].shape[:-1], 512))
        lse = torch.full(kwargs["query"].shape[:-1], 2.0)
        return output, lse

    monkeypatch.setattr(decode, "tokenspeed_mla_decode", fake_decode)
    monkeypatch.setattr(decode, "_workspace", lambda *_args: workspace)

    q = torch.empty((2, 1, 4, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((3, 32, 1, 576), dtype=torch.bfloat16)
    page_table = torch.zeros((2, 1), dtype=torch.int32)
    cache_seqlens = torch.ones(2, dtype=torch.int32)

    output, lse = decode.tokenspeed_mla_decode_with_kvcache(
        q,
        kv_cache,
        page_table,
        cache_seqlens,
        max_seqlen_k=32,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=0.125,
        return_lse=True,
    )

    assert output.shape == (2, 1, 4, 512)
    assert captured["kv_cache"].shape == (3, 32, 576)
    assert captured["workspace_buffer"] is workspace
    assert captured["block_tables"] is page_table
    assert captured["seq_lens"] is cache_seqlens
    torch.testing.assert_close(lse, torch.full_like(lse, 2.0 * math.log(2.0)))


def test_decode_adapter_rejects_logit_cap() -> None:
    with pytest.raises(ValueError, match="does not support logit_cap"):
        decode.tokenspeed_mla_decode_with_kvcache(
            torch.empty((1, 1, 1, 576)),
            torch.empty((1, 32, 1, 576)),
            torch.zeros((1, 1), dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            max_seqlen_k=32,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            softmax_scale=1.0,
            logit_cap=1.0,
        )


def test_prefill_adapter_maps_ragged_inputs_and_lse(monkeypatch) -> None:
    captured = {}

    def fake_prefill(**kwargs):
        captured.update(kwargs)
        output = torch.empty(
            (kwargs["query"].shape[0], kwargs["query"].shape[1], 128),
            dtype=torch.bfloat16,
        )
        lse = torch.full(kwargs["query"].shape[:2], 3.0)
        return output, lse

    monkeypatch.setattr(prefill, "tokenspeed_mla_prefill", fake_prefill)

    q = torch.empty((5, 4, 384), dtype=torch.float8_e4m3fn)[..., ::2]
    k = torch.empty((7, 4, 384), dtype=torch.float8_e4m3fn)[..., ::2]
    v = torch.empty((7, 4, 256), dtype=torch.float8_e4m3fn)[..., ::2]
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_kv = torch.tensor([0, 3, 7], dtype=torch.int32)

    output, lse = prefill.tokenspeed_mla_prefill_adapter(
        q,
        k,
        v,
        cu_q,
        cu_kv,
        max_seqlen_q=3,
        max_seqlen_kv=4,
        softmax_scale=0.25,
        is_causal=False,
        return_lse=True,
    )

    assert output.shape == (5, 4, 128)
    assert captured["query"].is_contiguous()
    assert captured["key"].is_contiguous()
    assert captured["value"].is_contiguous()
    assert captured["batch_size"] == 2
    assert captured["max_seq_len"] == 4
    assert captured["max_seq_len_q"] == 3
    assert torch.equal(captured["seq_lens"], torch.tensor([3, 4]))
    torch.testing.assert_close(lse, torch.full_like(lse, 3.0 * math.log(2.0)))


def test_prefill_adapter_rejects_logit_cap() -> None:
    with pytest.raises(ValueError, match="does not support logit_cap"):
        prefill.tokenspeed_mla_prefill_adapter(
            torch.empty((1, 1, 192)),
            torch.empty((1, 1, 192)),
            torch.empty((1, 1, 128)),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            max_seqlen_q=1,
            max_seqlen_kv=1,
            softmax_scale=1.0,
            logit_cap=1.0,
        )
