"""GLM-5.3-Flash loading of an FP8 indexer wk into the bf16 fused projection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tokenspeed.runtime.models.glm5 import GlmDsaIndexer
from tokenspeed.runtime.models.glm53_flash import load_glm53_flash_text_weights

PREFIX = "model.layers.0.self_attn.indexer"
HEAD_DIM, N_HEADS, HIDDEN = 128, 4, 256


def _param(*shape: int, dtype: torch.dtype) -> nn.Parameter:
    return nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)


def _fused_loader(param: nn.Parameter, weight: torch.Tensor, shard_id: int) -> None:
    offset, size = ((0, HEAD_DIM), (HEAD_DIM, N_HEADS))[shard_id]
    param.data[offset : offset + size].copy_(weight)


class _Indexer(GlmDsaIndexer):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.wk = nn.Module()
        self.wk.weight = _param(HEAD_DIM, HIDDEN, dtype=torch.float8_e4m3fn)
        self.wk.weight_scale_inv = _param(1, HIDDEN // 128, dtype=torch.float32)
        self.weights_proj = nn.Module()
        self.weights_proj.weight = _param(N_HEADS, HIDDEN, dtype=torch.bfloat16)
        self.wk_weights_proj = nn.Module()
        self.wk_weights_proj.weight = _param(
            HEAD_DIM + N_HEADS, HIDDEN, dtype=torch.bfloat16
        )
        self.wk_weights_proj.weight.weight_loader = _fused_loader
        self._wk_weights_proj_loaded = False


def _owner() -> nn.Module:
    owner = nn.Module()
    owner.model = nn.Module()
    owner.model.layers = nn.ModuleList([nn.Module()])
    owner.model.layers[0].self_attn = nn.Module()
    owner.model.layers[0].self_attn.indexer = _Indexer()
    owner.config = SimpleNamespace(n_routed_experts=1, q_lora_rank=64)
    owner.mapping = SimpleNamespace(moe=SimpleNamespace(ep_rank=0, ep_size=1))
    owner.quant_config = SimpleNamespace(weight_block_size=[128, 128])
    return owner


def _checkpoint() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        f"{PREFIX}.wk.weight": torch.randn(HEAD_DIM, HIDDEN).to(torch.float8_e4m3fn),
        f"{PREFIX}.wk.weight_scale_inv": torch.tensor([[0.5, 2.0]]),
        f"{PREFIX}.weights_proj.weight": torch.randn(N_HEADS, HIDDEN).bfloat16(),
    }


@pytest.mark.parametrize("scale_first", [False, True])
def test_fp8_wk_is_dequantized_into_the_fused_projection(scale_first: bool) -> None:
    owner = _owner()
    checkpoint = _checkpoint()
    order = list(checkpoint.items())
    if scale_first:
        order[0], order[1] = order[1], order[0]

    load_glm53_flash_text_weights(owner, order)

    indexer = owner.model.layers[0].self_attn.indexer
    scale = checkpoint[f"{PREFIX}.wk.weight_scale_inv"].repeat_interleave(128, dim=1)
    expected_wk = (checkpoint[f"{PREFIX}.wk.weight"].float() * scale).bfloat16()
    fused = indexer.wk_weights_proj.weight
    torch.testing.assert_close(fused[:HEAD_DIM], expected_wk, rtol=0, atol=0)
    torch.testing.assert_close(
        fused[HEAD_DIM:],
        checkpoint[f"{PREFIX}.weights_proj.weight"],
        rtol=0,
        atol=0,
    )
    assert indexer._wk_weights_proj_loaded


def test_bf16_wk_is_copied_into_the_fused_projection() -> None:
    owner = _owner()
    checkpoint = _checkpoint()
    checkpoint[f"{PREFIX}.wk.weight"] = torch.randn(HEAD_DIM, HIDDEN).bfloat16()
    del checkpoint[f"{PREFIX}.wk.weight_scale_inv"]

    load_glm53_flash_text_weights(owner, checkpoint.items())

    indexer = owner.model.layers[0].self_attn.indexer
    torch.testing.assert_close(
        indexer.wk_weights_proj.weight[:HEAD_DIM],
        checkpoint[f"{PREFIX}.wk.weight"],
        rtol=0,
        atol=0,
    )
    assert indexer._wk_weights_proj_loaded


def test_fp8_wk_without_its_scale_fails() -> None:
    checkpoint = _checkpoint()
    del checkpoint[f"{PREFIX}.wk.weight_scale_inv"]

    with pytest.raises(RuntimeError, match="missing its partner"):
        load_glm53_flash_text_weights(_owner(), checkpoint.items())
