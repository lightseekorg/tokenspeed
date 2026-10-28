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

"""State-scan extension compatibility and shared-forward layer isolation.

The test provider executes a small CPU delta-rule scan, including state reads
and writes. Shared-forward tests replace only GPU tensor leaves with CPU tensor
implementations; they check computed outputs and state, not call records.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.state import mamba
from tokenspeed.runtime.layers.attention.backends.state.mamba import (
    MambaAttnBackend,
    MambaForwardMetadata,
    StateScanContext,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig

register_cuda_ci(est_time=5, suite="runtime-1gpu")


class _CpuScanBackend(MambaAttnBackend):
    """A family provider using only the pre-existing scan override interfaces."""

    def __init__(self):
        spec = MHAConfig(
            num_attention_heads=1, num_kv_heads=1, head_dim=2, attn_tp_size=1
        )
        linear = LinearAttnConfig(
            num_k_heads=1,
            num_v_heads=1,
            head_k_dim=2,
            head_v_dim=2,
            conv_kernel_size=1,
            layer_ids=(3, 7),
            tp_size=1,
        )
        config = AttnConfig(
            device="cpu",
            dtype=torch.float32,
            kv_cache_dtype=torch.float32,
            kv_cache_quant_method="none",
            prefix_granularity=16,
            context_len=16,
            max_bs=1,
            components=(spec, linear),
        )
        super().__init__(config, spec)
        self.components: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_id in (3, 7):
            conv = torch.empty(3, 6, 0)
            state = torch.zeros(3, 1, 2, 2)
            state[1, 0] = torch.diag(torch.tensor([2.0, 3.0]))
            self.components[layer_id] = conv, state

    def _state_group_for(self, layer_id):
        if layer_id not in self.components:
            raise ValueError(f"Unknown layer {layer_id}")
        return "state"

    def _state_components(self, layer_id):
        return self.components[layer_id]

    @staticmethod
    def _scan(
        query, key, value, initial, boundaries, *, A_log, dt_bias, g_raw, beta_raw
    ):
        heads, dim = query.shape[-2:]
        q, k, v = (x.reshape(-1, heads, dim) for x in (query, key, value))
        gates = (g_raw.reshape(-1, heads, dim) + dt_bias) * A_log.exp()[:, None]
        beta = beta_raw.reshape(-1, heads).sigmoid()
        output = torch.empty_like(v)
        final = initial.clone()
        for row, (begin, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            state = initial[row].clone()
            for token in range(int(begin), int(end)):
                state = state * gates[token].exp()[:, None, :]
                residual = v[token] - (state * k[token, :, None, :]).sum(-1)
                state = (
                    state
                    + (residual * beta[token, :, None])[:, :, None]
                    * k[token, :, None, :]
                )
                output[token] = (state * q[token, :, None, :]).sum(-1)
            final[row] = state
        return output, final

    def _decode_scan(
        self,
        query,
        key,
        value,
        ssm_states,
        read_indices,
        write_indices,
        *,
        A_log,
        dt_bias,
        D,
        a,
        b,
        g_raw,
        f_a_out,
        f_b_weight,
        beta_raw,
        lower_bound,
    ):
        assert D is None and a is None and b is None
        assert f_a_out is None and f_b_weight is None and lower_bound is None
        boundaries = torch.arange(query.shape[0] + 1, dtype=torch.int32)
        output, final = self._scan(
            query,
            key,
            value,
            ssm_states[read_indices.long()],
            boundaries,
            A_log=A_log,
            dt_bias=dt_bias,
            g_raw=g_raw,
            beta_raw=beta_raw,
        )
        ssm_states.index_copy_(0, write_indices.long(), final)
        return output.unsqueeze(0)

    def _prefill_scan(
        self,
        query,
        key,
        value,
        recurrent_state,
        query_start_loc,
        *,
        A_log,
        dt_bias,
        D,
        a,
        b,
        g_raw,
        f_a_out,
        f_b_weight,
        beta_raw,
        seq_len,
        num_real_tokens,
        lower_bound,
        cu_seqlens_cpu=None,
        state_checkpoints=None,
    ):
        assert D is None and a is None and b is None
        assert f_a_out is None and f_b_weight is None and lower_bound is None
        assert state_checkpoints is None
        assert seq_len == num_real_tokens == query.shape[1]
        if cu_seqlens_cpu is not None:
            if not torch.equal(query_start_loc, cu_seqlens_cpu):
                raise ValueError("Host boundaries differ from the scan boundaries")
        return self._scan(
            query,
            key,
            value,
            recurrent_state,
            query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            g_raw=g_raw,
            beta_raw=beta_raw,
        )


class _LayerStateBackend(_CpuScanBackend):
    """A new provider whose extra state is selected by explicit layer identity."""

    def __init__(self):
        super().__init__()
        self.layer_state = {
            3: torch.tensor([10.0, 20.0]),
            7: torch.tensor([100.0, 200.0]),
        }

    def _with_layer_state(self, context, output):
        state = self.layer_state[context.layer_id]
        state.add_(output.reshape(-1, 2).sum(0))
        return output + state

    def _decode_scan_with_context(self, scan_context, *args, **kwargs):
        output = super()._decode_scan_with_context(scan_context, *args, **kwargs)
        return self._with_layer_state(scan_context, output)

    def _prefill_scan_with_context(self, scan_context, *args, **kwargs):
        output, final = super()._prefill_scan_with_context(
            scan_context, *args, **kwargs
        )
        return self._with_layer_state(scan_context, output), final


def _scan_inputs(tokens):
    query = torch.ones(1, tokens, 1, 2)
    key = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])[:, :tokens].clone()
    value = torch.tensor([[[[6.0, 8.0]], [[1.0, 2.0]]]])[:, :tokens].clone()
    arguments = dict(
        A_log=torch.zeros(1),
        dt_bias=torch.full((1, 2), math.log(2.0)),
        D=None,
        a=None,
        b=None,
        g_raw=torch.zeros(tokens, 1, 2),
        f_a_out=None,
        f_b_weight=None,
        beta_raw=torch.zeros(tokens, 1),
        lower_bound=None,
    )
    return query, key, value, arguments


def test_legacy_decode_override_preserves_arguments_and_state_writes():
    backend = _CpuScanBackend()
    _, state = backend.components[3]
    query, key, value, arguments = _scan_inputs(1)
    output = backend._decode_scan_with_context(
        StateScanContext(layer_id=3),
        query,
        key,
        value,
        state,
        torch.tensor([1]),
        torch.tensor([2]),
        **arguments,
    )
    torch.testing.assert_close(output.reshape(-1), torch.tensor([5.0, 10.0]))
    torch.testing.assert_close(state[2, 0], torch.tensor([[5.0, 0.0], [4.0, 6.0]]))
    torch.testing.assert_close(state[1, 0], torch.diag(torch.tensor([2.0, 3.0])))


def test_legacy_prefill_override_preserves_optional_inputs_and_final_state():
    backend = _CpuScanBackend()
    initial = backend.components[3][1][1:2].clone()
    query, key, value, arguments = _scan_inputs(2)
    boundaries = torch.tensor([0, 2], dtype=torch.int64)
    output, final = backend._prefill_scan_with_context(
        StateScanContext(layer_id=3),
        query,
        key,
        value,
        initial,
        boundaries,
        seq_len=2,
        num_real_tokens=2,
        cu_seqlens_cpu=boundaries.clone(),
        state_checkpoints=None,
        **arguments,
    )
    torch.testing.assert_close(output[:, 0], torch.tensor([[5.0, 10.0], [10.5, 15.0]]))
    torch.testing.assert_close(final[0, 0], torch.tensor([[10.0, 0.5], [8.0, 7.0]]))
    torch.testing.assert_close(initial[0, 0], torch.diag(torch.tensor([2.0, 3.0])))


def test_legacy_prefill_rejects_a_mismatched_explicit_host_hint():
    backend = _CpuScanBackend()
    initial = backend.components[3][1][1:2].clone()
    query, key, value, arguments = _scan_inputs(2)
    with pytest.raises(ValueError, match="Host boundaries differ"):
        backend._prefill_scan_with_context(
            StateScanContext(layer_id=3),
            query,
            key,
            value,
            initial,
            torch.tensor([0, 2]),
            seq_len=2,
            num_real_tokens=2,
            cu_seqlens_cpu=torch.tensor([0, 1]),
            state_checkpoints=None,
            **arguments,
        )


@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_unsupported_scan_arguments_are_not_discarded(phase):
    backend = _CpuScanBackend()
    query, key, value, arguments = _scan_inputs(1)
    context = StateScanContext(layer_id=3)
    state = backend.components[3][1]
    with pytest.raises(TypeError, match="unsupported"):
        if phase == "decode":
            backend._decode_scan_with_context(
                context,
                query,
                key,
                value,
                state,
                torch.tensor([1]),
                torch.tensor([2]),
                unsupported=True,
                **arguments,
            )
        else:
            backend._prefill_scan_with_context(
                context,
                query,
                key,
                value,
                state[1:2],
                torch.tensor([0, 1]),
                seq_len=1,
                num_real_tokens=1,
                unsupported=True,
                **arguments,
            )


def _cpu_conv_update(
    x,
    state,
    weight,
    bias,
    activation,
    *,
    conv_state_indices,
    output_state_indices,
    parent_indices,
):
    assert state.shape[-1] == 0 and weight.shape[-1] == 1
    assert bias is None and activation is None and parent_indices is None
    assert conv_state_indices.numel() == output_state_indices.numel() == x.shape[0]
    return x * weight[:, 0]


def _cpu_prepare(conv, state, read_indices, write_indices):
    conv.index_copy_(0, write_indices.long(), conv[read_indices.long()])
    return state[read_indices.long()].clone(), read_indices > 0


def _cpu_conv_prefill(
    x,
    weight,
    bias,
    *,
    activation,
    conv_states,
    has_initial_state,
    cache_indices,
    query_start_loc,
    prefill_metadata,
):
    assert conv_states.shape[-1] == 0 and weight.shape[-1] == 1
    assert bias is None and activation is None and prefill_metadata is None
    assert (
        has_initial_state.numel()
        == cache_indices.numel()
        == query_start_loc.numel() - 1
    )
    return x * weight[:, :1]


def _cpu_split(
    x,
    *,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    head_q,
    head_k,
    head_v,
    replay,
    valid_tokens,
):
    assert replay is None
    if valid_tokens is not None:
        assert int(valid_tokens[0]) == x.shape[0]
    widths = (num_q_heads * head_q, num_k_heads * head_k, num_v_heads * head_v)
    return tuple(
        part.reshape(1, x.shape[0], heads, dim)
        for part, heads, dim in zip(
            x.split(widths, dim=-1),
            (num_q_heads, num_k_heads, num_v_heads),
            (head_q, head_k, head_v),
        )
    )


def _cpu_store(source, destination, indices):
    destination.index_copy_(0, indices.long(), source)


@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_shared_forward_passes_layer_identity_without_cross_layer_state(
    phase, monkeypatch
):
    # These are unavailable GPU tensor leaves, replaced by actual CPU math.
    monkeypatch.setattr(mamba, "causal_conv1d_update", _cpu_conv_update)
    monkeypatch.setattr(mamba, "_prepare_cache_prefill_state_inputs", _cpu_prepare)
    monkeypatch.setattr(mamba, "causal_conv1d_fn", _cpu_conv_prefill)
    monkeypatch.setattr(mamba, "fused_qkv_split_gdn_prefill", _cpu_split)
    monkeypatch.setattr(mamba, "store_state_rows", _cpu_store)
    backend = _LayerStateBackend()
    boundaries = torch.tensor([0, 1], dtype=torch.int32)
    backend.forward_metadata = MambaForwardMetadata(
        query_start_loc=boundaries,
        scan_query_start_loc=boundaries,
        state_in_blocks_by_group={"state": torch.tensor([1], dtype=torch.int32)},
        state_out_blocks_by_group={"state": torch.tensor([2], dtype=torch.int32)},
    )
    _, _, _, scan_arguments = _scan_inputs(1)
    arguments = dict(
        mixed_qkv=torch.tensor([[1.0, 1.0, 1.0, 0.0, 6.0, 8.0]]),
        conv_weights=torch.ones(6, 1),
        bias=None,
        activation=None,
        key_dim=2,
        value_dim=2,
        attention_tp_size=1,
        head_k_dim=2,
        head_v_dim=2,
        seq_len=1,
        **scan_arguments,
    )
    for layer_id, expected in (
        (7, [110.0, 220.0]),
        (3, [20.0, 40.0]),
        (7, [115.0, 230.0]),
    ):
        if phase == "decode":
            output = backend.forward_decode(
                None,
                None,
                None,
                None,
                None,
                1,
                True,
                layer_id=layer_id,
                **arguments,
            )
        else:
            output = backend.forward_extend(
                None,
                None,
                None,
                None,
                None,
                1,
                ForwardMode.EXTEND,
                True,
                layer_id=layer_id,
                **arguments,
            )
        torch.testing.assert_close(output.reshape(-1), torch.tensor(expected))
        state = backend.components[layer_id][1]
        torch.testing.assert_close(state[2, 0], torch.tensor([[5.0, 0.0], [4.0, 6.0]]))
        torch.testing.assert_close(state[1, 0], torch.diag(torch.tensor([2.0, 3.0])))
    torch.testing.assert_close(backend.layer_state[3], torch.tensor([15.0, 30.0]))
    torch.testing.assert_close(backend.layer_state[7], torch.tensor([110.0, 220.0]))
