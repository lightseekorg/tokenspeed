from types import SimpleNamespace

import torch
from torch import nn

from tokenspeed.runtime.models.qwen3 import Qwen3DecoderLayer
from tokenspeed.runtime.utils.env import global_server_args_dict


class _InputNorm(nn.Module):
    def forward(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states
        complete_residual = hidden_states + residual + 10
        return torch.zeros_like(hidden_states), complete_residual

    def forward_with_allreduce_fusion(
        self,
        tp_rank,
        tp_group,
        hidden_states,
        residual,
    ):
        del tp_rank, tp_group
        complete_residual = hidden_states + residual + 10
        return torch.zeros_like(hidden_states), complete_residual, None


class _PostAttentionNorm(nn.Module):
    def forward_with_allreduce_fusion(
        self,
        tp_rank,
        tp_group,
        hidden_states,
        residual,
    ):
        del tp_rank, tp_group
        return hidden_states, residual, None


class _IdentityAttention(nn.Module):
    def forward(self, *, hidden_states, **kwargs):
        del kwargs
        return hidden_states


def _decoder_layer() -> Qwen3DecoderLayer:
    layer = Qwen3DecoderLayer.__new__(Qwen3DecoderLayer)
    nn.Module.__init__(layer)
    parallel = SimpleNamespace(tp_rank=0, tp_group=(0, 1))
    layer.mapping = SimpleNamespace(dense=parallel, attn=parallel)
    layer.input_layernorm = _InputNorm()
    layer.post_attention_layernorm = _PostAttentionNorm()
    layer.self_attn = _IdentityAttention()
    layer.mlp = nn.Identity()
    return layer


def test_capture_uses_tp_reduced_input_residual(monkeypatch):
    monkeypatch.setitem(global_server_args_dict, "comm_fusion_max_num_tokens", 1024)
    layer = _decoder_layer()
    hidden_states = torch.tensor([[1.0, 2.0]])
    residual = torch.tensor([[3.0, 4.0]])

    _, _, captured = layer.forward_with_input_residual(
        positions=torch.tensor([0]),
        hidden_states=hidden_states,
        ctx=SimpleNamespace(input_num_tokens=1),
        out_cache_loc=torch.tensor([0]),
        residual=residual,
        cos_sin=None,
    )

    torch.testing.assert_close(captured, hidden_states + residual + 10)
