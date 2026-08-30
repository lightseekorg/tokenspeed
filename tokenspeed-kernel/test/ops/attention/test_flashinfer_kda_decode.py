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

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch
from tokenspeed_kernel.ops.attention.flashinfer import kda_decode


def _enable_fake_fused_decode(monkeypatch):
    calls = {}

    def fake_fused_decode(**kwargs):
        calls.update(kwargs)
        return kwargs["x"]

    monkeypatch.setattr(kda_decode, "_fused_kda_decode", fake_fused_decode)
    monkeypatch.setattr(
        kda_decode,
        "_fused_kda_decode_available",
        lambda: True,
    )
    return calls


def test_fused_decode_capability_requires_real_backend(monkeypatch) -> None:
    flashinfer = ModuleType("flashinfer")
    flashinfer.__path__ = []
    module = ModuleType("flashinfer.kda_decode")
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.kda_decode", module)

    def in_place_only(state_indices):
        return state_indices

    module.fused_kda_decode = in_place_only
    module._FUSED_KDA_DECODE_AVAILABLE = False
    assert kda_decode._load_fused_kda_decode() is None
    module._FUSED_KDA_DECODE_AVAILABLE = True
    assert kda_decode._load_fused_kda_decode() is in_place_only


def test_prepare_kda_weights_preserves_qkv_tap_order(monkeypatch) -> None:
    _enable_fake_fused_decode(monkeypatch)
    heads = 12
    hidden = heads * 128
    source = torch.arange(3 * hidden * 4, dtype=torch.bfloat16).view(3 * hidden, 4)
    norm = torch.arange(128, dtype=torch.bfloat16)

    prepared = kda_decode.prepare_flashinfer_kda_decode_weights(source, norm)

    assert prepared is not None
    assert prepared.conv.shape == (3, 4, hidden)
    assert prepared.conv.dtype == torch.float32
    assert prepared.conv.is_contiguous()
    assert prepared.norm.dtype == torch.float32
    for qkv in range(3):
        for tap in range(4):
            torch.testing.assert_close(
                prepared.conv[qkv, tap],
                source.view(3, hidden, 4)[qkv, :, tap].float(),
            )


def test_prepare_kda_weights_refit_preserves_addresses(monkeypatch) -> None:
    _enable_fake_fused_decode(monkeypatch)
    hidden = 12 * 128
    source = torch.randn(3 * hidden, 4, dtype=torch.bfloat16)
    norm = torch.randn(128, dtype=torch.bfloat16)
    prepared = kda_decode.prepare_flashinfer_kda_decode_weights(source, norm)
    assert prepared is not None
    conv_ptr = prepared.conv.data_ptr()
    norm_ptr = prepared.norm.data_ptr()

    source.fill_(0.25)
    norm.fill_(0.5)
    refreshed = kda_decode.prepare_flashinfer_kda_decode_weights(
        source,
        norm,
        prepared,
    )

    assert refreshed is prepared
    assert prepared.conv.data_ptr() == conv_ptr
    assert prepared.norm.data_ptr() == norm_ptr
    torch.testing.assert_close(prepared.conv, torch.full_like(prepared.conv, 0.25))
    torch.testing.assert_close(prepared.norm, torch.full_like(prepared.norm, 0.5))


def test_adapter_uses_staged_write_blocks(monkeypatch) -> None:
    calls = _enable_fake_fused_decode(monkeypatch)
    batch = 2
    heads = 12
    head_dim = 128
    hidden = heads * head_dim
    conv_weights = torch.randn(3 * hidden, 4, dtype=torch.bfloat16)
    norm_weight = torch.randn(head_dim, dtype=torch.bfloat16)
    prepared = kda_decode.prepare_flashinfer_kda_decode_weights(
        conv_weights, norm_weight
    )
    physical_conv = torch.zeros(4, 3, 3 * hidden, dtype=torch.bfloat16)
    conv_states = physical_conv.transpose(1, 2)
    mixed_qkv = torch.randn(batch, 3 * hidden, dtype=torch.bfloat16)
    f_a_out = torch.randn(batch, head_dim, dtype=torch.bfloat16)
    f_b_weight = torch.randn(hidden, head_dim, dtype=torch.bfloat16)
    write_indices = torch.tensor([2, 3], dtype=torch.int32)

    result = kda_decode._flashinfer_kda_fused_paged_decode(
        mixed_qkv,
        conv_weights,
        conv_states,
        f_a_out,
        f_b_weight,
        torch.randn(batch, heads, dtype=torch.bfloat16),
        torch.randn(heads, dtype=torch.float32),
        torch.randn(hidden, dtype=torch.float32),
        state_pool=torch.empty(1),
        read_indices=write_indices,
        write_indices=write_indices,
        num_heads=heads,
        head_dim=head_dim,
        cu_seqlens=torch.arange(batch + 1, dtype=torch.int32),
        lower_bound=-5.0,
        output_gate=torch.randn(batch, hidden, dtype=torch.bfloat16),
        norm_weight=norm_weight,
        norm_eps=1e-5,
        prepared_weights=prepared,
    )

    assert result is mixed_qkv
    assert calls["state_indices"] is write_indices
    assert "write_state_indices" not in calls
    assert calls["conv_state"].stride()[1:] == (1, 3 * hidden)


def test_adapter_rejects_unstaged_dual_indices(monkeypatch) -> None:
    _enable_fake_fused_decode(monkeypatch)
    batch = 1
    heads = 12
    hidden = heads * 128
    conv_weights = torch.randn(3 * hidden, 4, dtype=torch.bfloat16)
    norm_weight = torch.randn(128, dtype=torch.bfloat16)
    prepared = kda_decode.prepare_flashinfer_kda_decode_weights(
        conv_weights, norm_weight
    )
    physical_conv = torch.zeros(2, 3, 3 * hidden, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="copy-on-write staging"):
        kda_decode._flashinfer_kda_fused_paged_decode(
            torch.randn(batch, 3 * hidden, dtype=torch.bfloat16),
            conv_weights,
            physical_conv.transpose(1, 2),
            torch.randn(batch, 128, dtype=torch.bfloat16),
            torch.randn(hidden, 128, dtype=torch.bfloat16),
            torch.randn(batch, heads, dtype=torch.bfloat16),
            torch.randn(heads, dtype=torch.float32),
            torch.randn(hidden, dtype=torch.float32),
            state_pool=torch.empty(1),
            read_indices=torch.tensor([0], dtype=torch.int32),
            write_indices=torch.tensor([1], dtype=torch.int32),
            num_heads=heads,
            head_dim=128,
            cu_seqlens=torch.arange(batch + 1, dtype=torch.int32),
            lower_bound=-5.0,
            output_gate=torch.randn(batch, hidden, dtype=torch.bfloat16),
            norm_weight=norm_weight,
            norm_eps=1e-5,
            prepared_weights=prepared,
        )
