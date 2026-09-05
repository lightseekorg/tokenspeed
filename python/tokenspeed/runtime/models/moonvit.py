# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
#
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

"""Shared MoonViT vision tower, projector, and execution path."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import activations

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.conv import Conv2dLayer
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig

try:
    from transformers.activations import PytorchGELUTanh
except ImportError:
    from transformers.activations import GELUTanh

    activations.PytorchGELUTanh = GELUTanh
    PytorchGELUTanh = GELUTanh

from tokenspeed.runtime.layers.attention.mm_encoder_attention import VisionAttention
from tokenspeed.runtime.layers.linear import ReplicatedLinear

try:
    from tokenspeed.runtime.layers.quantization.modelslim.modelslim import (
        ModelSlimConfig,
    )
except ImportError:

    class ModelSlimConfig:
        pass


from tokenspeed.runtime.multimodal.encoder_cudagraph import (
    EncoderForwardStepRunner,
    VisionEncoderCudaGraphAdapter,
)
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem
from tokenspeed.runtime.utils import add_prefix


@dataclass(frozen=True)
class MoonViTConfig:
    """Normalized MoonViT tower and projector configuration."""

    patch_size: int | tuple[int, int]
    init_pos_emb_height: int
    init_pos_emb_width: int
    init_pos_emb_time: int
    pos_emb_type: str
    pos_emb_interpolation_mode: str
    num_attention_heads: int
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    qkv_hidden_size: int | None
    merge_kernel_size: tuple[int, int]
    video_attn_type: str
    mm_projector_type: str
    projector_vision_hidden_size: int
    mm_hidden_size: int
    projector_hidden_act: str
    projector_ln_eps: float
    text_hidden_size: int
    norm_type: str
    attn_bias: bool
    patch_embed_proj_bias: bool
    mlp_type: str
    linear_bias: bool
    activation_func: str

    @classmethod
    def from_config(cls, config) -> "MoonViTConfig":
        if isinstance(config, cls):
            return config

        uses_prefixed_fields = hasattr(config, "vt_num_attention_heads")
        if uses_prefixed_fields:
            hidden_size = getattr(config, "vt_hidden_size", None)
            num_attention_heads = getattr(config, "vt_num_attention_heads", None)
            num_hidden_layers = getattr(config, "vt_num_hidden_layers", None)
            intermediate_size = getattr(config, "vt_intermediate_size", None)
        else:
            hidden_size = getattr(config, "hidden_size", None)
            num_attention_heads = getattr(config, "num_attention_heads", None)
            num_hidden_layers = getattr(config, "num_hidden_layers", None)
            intermediate_size = getattr(config, "intermediate_size", None)

        if hidden_size is None:
            raise ValueError("MoonViT config must define a vision hidden size.")
        if None in (num_attention_heads, num_hidden_layers, intermediate_size):
            raise ValueError(
                "MoonViT config must define attention heads, layers, and "
                "intermediate size."
            )

        projector_vision_hidden_size = getattr(config, "vt_hidden_size", hidden_size)
        mm_hidden_size = getattr(config, "mm_hidden_size", None)
        return cls(
            patch_size=config.patch_size,
            init_pos_emb_height=config.init_pos_emb_height,
            init_pos_emb_width=config.init_pos_emb_width,
            init_pos_emb_time=config.init_pos_emb_time,
            pos_emb_type=config.pos_emb_type,
            pos_emb_interpolation_mode=getattr(
                config, "pos_emb_interpolation_mode", "bicubic"
            ),
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            qkv_hidden_size=getattr(config, "qkv_hidden_size", None),
            merge_kernel_size=tuple(config.merge_kernel_size),
            video_attn_type=config.video_attn_type,
            mm_projector_type=getattr(config, "mm_projector_type", "patchmerger"),
            projector_vision_hidden_size=projector_vision_hidden_size,
            mm_hidden_size=(
                projector_vision_hidden_size
                if mm_hidden_size is None
                else mm_hidden_size
            ),
            projector_hidden_act=getattr(config, "projector_hidden_act", "gelu"),
            projector_ln_eps=getattr(config, "projector_ln_eps", 1e-5),
            text_hidden_size=config.text_hidden_size,
            norm_type=getattr(config, "norm_type", "layernorm"),
            attn_bias=getattr(config, "attn_bias", True),
            patch_embed_proj_bias=getattr(config, "patch_embed_proj_bias", True),
            mlp_type=getattr(config, "mlp_type", "mlp2"),
            linear_bias=getattr(config, "linear_bias", True),
            activation_func=getattr(config, "activation_func", "gelu_pytorch_tanh"),
        )


class MLP2(nn.Module):
    """
    Two-layer MLP helper used by the Kimi MoonViT blocks.

    Kimi-K2.5 and Kimi-K3 share this implementation while selecting their
    dimensions and bias policy through the vision config.
    """

    def __init__(
        self,
        dims: list[int],
        activation,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        if len(dims) != 3:
            raise ValueError(f"dims must have length 3, got {len(dims)}.")

        self.quant_config = quant_config
        if isinstance(self.quant_config, ModelSlimConfig):
            self.fc0 = ReplicatedLinear(
                dims[0],
                dims[1],
                bias=bias,
                quant_config=quant_config,
                prefix=add_prefix("fc0", prefix),
            )
            self.fc1 = ReplicatedLinear(
                dims[1],
                dims[2],
                bias=bias,
                quant_config=quant_config,
                prefix=add_prefix("fc1", prefix),
            )
        else:
            self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
            self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
            for module in (self.fc0, self.fc1):
                nn.init.trunc_normal_(
                    module.weight, std=math.sqrt(2 / module.in_features)
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.quant_config, ModelSlimConfig):
            x = x.flatten(0, 1)
            x, _ = self.fc0(x)
            x = self.activation(x)
            x, _ = self.fc1(x)
        else:
            x = self.fc0(x)
            x = self.activation(x)
            x = self.fc1(x)
        return x


def _make_vision_norm(norm_type: str, hidden_dim: int) -> nn.Module:
    """Build the normalization used by a MoonViT encoder block."""
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_dim)
    if norm_type == "rmsnorm":
        return nn.RMSNorm(hidden_dim)
    raise NotImplementedError(f"Not support norm_type: {norm_type}")


def _get_vision_activation(name: str) -> nn.Module:
    """Resolve the activation name stored in MoonViT configs."""
    if name == "gelu_pytorch_tanh":
        return PytorchGELUTanh()
    try:
        return activations.get_activation(name)
    except KeyError as exc:
        raise ValueError(f"Unsupported MoonViT activation: {name}") from exc


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, x_shape=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """

    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
    # ..., num_heads, head_dim/2
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return xq_out.type_as(xq), xk_out.type_as(xk)


def tpool_patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[torch.Tensor]:
    d_model = x.size(-1)

    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        seq = x[pre_sum : pre_sum + t * h * w]
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = h // kernel_height, w // kernel_width
        reshaped_seq = seq.view(
            t, new_height, kernel_height, new_width, kernel_width, d_model
        )
        reshaped_seq = (
            reshaped_seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        )  # temporal pooling
        padded_seq = reshaped_seq.view(
            new_height * new_width, kernel_height * kernel_width, -1
        )
        outputs.append(padded_seq)
        pre_sum += t * h * w

    return outputs


class MoonViTEncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        mapping: Mapping,
        *,
        activation=F.gelu,
        attn_bias: bool = False,
        qkv_hidden_size: int | None = None,
        norm_type: str = "layernorm",
        mlp_type: str = "mlp2",
        linear_bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mm_attention_backend: str | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.qkv_hidden_size = (
            hidden_dim if qkv_hidden_size is None else qkv_hidden_size
        )
        if self.qkv_hidden_size % self.num_heads != 0:
            raise ValueError(
                "qkv_hidden_size must be divisible by num_heads, got "
                f"{self.qkv_hidden_size} and {self.num_heads}."
            )
        self.hidden_size_per_attention_head = self.qkv_hidden_size // self.num_heads

        self.norm0 = _make_vision_norm(norm_type, hidden_dim)
        self.norm1 = _make_vision_norm(norm_type, hidden_dim)

        if mlp_type != "mlp2":
            raise NotImplementedError(f"Not support mlp_type: {mlp_type}")
        self.mlp = MLP2(
            [hidden_dim, mlp_dim, hidden_dim],
            activation,
            bias=linear_bias,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.attn = VisionAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            head_size=self.hidden_size_per_attention_head,
            qkv_bias=attn_bias,
            proj_bias=attn_bias,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            customized_position_embedding_applier=apply_rope,
            position_embedding_mode="complex_rope",
            mapping=mapping,
            mm_attention_backend=mm_attention_backend,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_freqs_cis: torch.Tensor | None = None,
    ):
        if not isinstance(max_seqlen, int):
            raise TypeError(
                f"max_seqlen must be a Python int for capture-safety, got {type(max_seqlen)}"
            )
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)

        hidden_states = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=rope_freqs_cis,
            max_seqlen=max_seqlen,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def get_rope_shape_decorate(func):
    _get_rope_shape_first_call_flag = set()

    def wrapper(org, interpolation_mode, shape):
        key = (org.requires_grad, torch.is_grad_enabled(), interpolation_mode)
        if key not in _get_rope_shape_first_call_flag:
            _get_rope_shape_first_call_flag.add(key)
            _ = func(org, interpolation_mode, shape=(64, 64))
        return func(org, interpolation_mode, shape)

    return wrapper


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    From:
    https://github.com/OpenGVLab/InternVideo/blob/421f6d2361fc8f61a3394244571f2601a4e99e29/InternVideo2/multi_modality/models/backbones/internvideo2/pos_embed.py#L86
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}.")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


@get_rope_shape_decorate
@torch.compile(dynamic=True)
def get_rope_shape(org, interpolation_mode, shape):
    return (
        F.interpolate(
            org.permute((2, 0, 1)).unsqueeze(0),
            size=shape,
            mode=interpolation_mode,
        )
        .squeeze(0)
        .permute((1, 2, 0))
        .flatten(end_dim=1)
    )


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """
    t_size: int of the temporal size
    return:
    pos_embed: [t_size, embed_dim] or [1+t_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class Learnable2DInterpPosEmbDivided_fixed(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bicubic",
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        time_weight = (
            torch.from_numpy(get_1d_sincos_pos_embed(self.dim, self.num_frames))
            .float()
            .unsqueeze(1)
        )
        self.register_buffer(
            "time_weight",
            time_weight.to(device=self.weight.device, dtype=self.weight.dtype),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            if t > self.num_frames:
                raise ValueError(f"t:{t} > self.num_frames:{self.num_frames}")
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = get_rope_shape(
                    self.weight,
                    interpolation_mode=self.interpolation_mode,
                    shape=(h, w),
                )

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = (
                    pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]
                )

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        out = x + torch.cat(pos_embs)
        return out


class Rope2DPosEmbRepeated(nn.Module):
    """2D rotary position embedding with multi-resolution support.

    Lifecycle:
    1. At construction, precompute and hold the cis tensor.
    2. Before each forward pass, call ``get_freqs_cis_by_*`` to get the
       ``freqs_cis`` tensor for this iteration.
    3. During the forward pass, pass ``freqs_cis`` to each attention layer
       and call ``apply`` just before each attention op. Rope is shared
       across all attention layers and all heads.

    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py

    Args:
        dim (int): usually the multi-head attention dimension; must be divisible by 4.
        max_height (int): the maximum height of the 2D grid.
        max_width (int): the maximum width of the 2D grid.
        theta_base (float): the base of the theta.
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        if self.dim % 4 != 0:
            raise ValueError("dim must be divisible by 4")
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def extra_repr(self):
        return f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.
        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = (
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(
        self, grid_thws: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Args:
            grid_thws (torch.Tensor): grid time, height and width
        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if not hasattr(self, "freqs_cis"):
            self.register_buffer(
                "freqs_cis", self._precompute_freqs_cis(device), persistent=False
            )

        shapes = grid_thws.tolist()
        if not all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes
        ):
            raise ValueError(
                f"grid shape out of range: {shapes}, max_height={self.max_height}, "
                f"max_width={self.max_width}"
            )
        freqs_cis = torch.cat(
            [
                self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
                for t, h, w in shapes
            ],
            dim=0,
        )
        return freqs_cis


class MoonVision3dPatchEmbed(nn.Module):
    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
        pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        patch_embed_proj_bias: bool = True,
        pos_emb_interpolation_mode: str = "bicubic",
    ):
        super().__init__()
        if not isinstance(patch_size, int | Sequence):
            raise TypeError(f"Invalid patch_size type: {type(patch_size)}")
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if len(patch_size) != 2:
            raise ValueError(
                f"Expected patch_size to be a tuple of 2, got {patch_size}"
            )
        self.patch_size = patch_size

        self.proj = Conv2dLayer(
            in_dim,
            out_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=patch_embed_proj_bias,
        )

        if pos_emb_type == "divided_fixed":
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=pos_emb_height,
                width=pos_emb_width,
                num_frames=pos_emb_time,
                dim=out_dim,
                interpolation_mode=pos_emb_interpolation_mode,
            )
        else:
            raise NotImplementedError(f"Not support pos_emb_type: {pos_emb_type}")

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (L, Channels): input tensor
            grid_thws (N, 3): temporal, height and width
        Returns:
            (L, Cout) tensor
        """
        x = self.proj(x).view(x.size(0), -1)
        # apply positional embedding
        x = self.pos_emb(x, grid_thws)
        return x


class MoonViT3dEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
        video_attn_type: str = "spatial_temporal",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        if video_attn_type != "spatial_temporal":
            raise ValueError(
                f'video_attn_type must be "spatial_temporal", got {video_attn_type}'
            )
        self.video_attn_type = video_attn_type
        qkv_hidden_size = block_cfg.get("qkv_hidden_size") or block_cfg["hidden_dim"]
        num_heads = block_cfg["num_heads"]
        if qkv_hidden_size % num_heads != 0:
            raise ValueError(
                "qkv_hidden_size must be divisible by num_heads, got "
                f"{qkv_hidden_size} and {num_heads}."
            )
        self.rope_2d = Rope2DPosEmbRepeated(qkv_hidden_size // num_heads, 512, 512)
        self.blocks = nn.ModuleList(
            [
                MoonViTEncoderLayer(
                    **block_cfg,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{layer_idx}", prefix),
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.final_layernorm = _make_vision_norm(
            block_cfg.get("norm_type", "layernorm"), hidden_dim
        )

    def prepare_metadata(
        self, grid_thws: torch.Tensor, device: torch.device | None = None
    ) -> dict[str, torch.Tensor | int]:
        """Eager metadata pass: everything with a GPU->CPU sync or a
        data-dependent shape lives here, outside the capture-safe block loop.

        Returns the ``rope_freqs_cis`` / ``cu_seqlens`` tensors plus
        ``max_seqlen`` as a Python int (see ``MoonViTEncoderLayer.forward``).
        ``max_seqlen`` is materialized numpy-side so the block loop never hits
        a ``.item()`` host sync on cudagraph replay.
        """
        if device is None:
            device = self.final_layernorm.weight.device
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thws=grid_thws, device=device)

        grid_thws_np = grid_thws.cpu().numpy()
        real_seq_lens = grid_thws_np[:, 0] * grid_thws_np[:, 1] * grid_thws_np[:, 2]
        max_seqlen = int(real_seq_lens.max()) if real_seq_lens.size > 0 else 0
        cu_seqlens_np = np.concatenate(
            [np.zeros(1, dtype=np.int32), real_seq_lens.cumsum(dtype=np.int32)]
        )
        cu_seqlens = torch.from_numpy(cu_seqlens_np).to(
            device=device, dtype=torch.int32, non_blocking=True
        )

        return {
            "rope_freqs_cis": rope_freqs_cis,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        }

    def forward_blocks(
        self,
        hidden_states: torch.Tensor,
        metadata: dict[str, torch.Tensor | int],
    ) -> torch.Tensor:
        """Capture-safe encoder body: the block loop + final norm. No host
        syncs and no data-dependent control flow, so this region is safe to
        record into a CUDA graph. ``metadata`` comes from
        :meth:`prepare_metadata`."""
        rope_freqs_cis = metadata["rope_freqs_cis"]
        cu_seqlens = metadata["cu_seqlens"]
        max_seqlen = metadata["max_seqlen"]

        for block in self.blocks:
            hidden_states = block(
                hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis=rope_freqs_cis
            )

        return self.final_layernorm(hidden_states)


class MoonViT3dPretrainedModel(nn.Module):
    def __init__(
        self,
        config,
        mapping: Mapping,
        *inputs,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mm_attention_backend: str | None = None,
        **kwargs,
    ):
        super().__init__()
        config = MoonViTConfig.from_config(config)
        self.config = config
        self.merge_kernel_size = config.merge_kernel_size

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
            pos_emb_type=config.pos_emb_type,
            patch_embed_proj_bias=config.patch_embed_proj_bias,
            pos_emb_interpolation_mode=config.pos_emb_interpolation_mode,
        )

        self.encoder = MoonViT3dEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "qkv_hidden_size": config.qkv_hidden_size,
                "mlp_dim": config.intermediate_size,
                "activation": _get_vision_activation(config.activation_func),
                "attn_bias": config.attn_bias,
                "norm_type": config.norm_type,
                "mlp_type": config.mlp_type,
                "linear_bias": config.linear_bias,
                "mapping": mapping,
                "mm_attention_backend": mm_attention_backend,
            },
            video_attn_type=config.video_attn_type,
            quant_config=quant_config,
            prefix=add_prefix("encoder", prefix),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device


class MoonViTMultiModalProjector(nn.Module):
    """MoonViT patch-merging projector shared by Kimi-K2.5 and Kimi-K3."""

    def __init__(
        self,
        config: MoonViTConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = MoonViTConfig.from_config(config)
        self.mm_projector_type = config.mm_projector_type
        if self.mm_projector_type not in {"patchmerger", "patchmergerv2"}:
            raise NotImplementedError(
                f"Not support mm_projector_type: {self.mm_projector_type}"
            )

        # Hidden size after patch merging
        merge_h, merge_w = config.merge_kernel_size
        vision_hidden_size = config.projector_vision_hidden_size
        projector_input_size = config.mm_hidden_size
        if projector_input_size != vision_hidden_size:
            raise ValueError(
                "mm_hidden_size must match the MoonViT hidden size, got "
                f"{projector_input_size} and {vision_hidden_size}."
            )
        self.hidden_size = projector_input_size * merge_h * merge_w
        output_hidden_size = config.text_hidden_size

        if config.projector_hidden_act != "gelu":
            raise NotImplementedError(
                "Only the GELU MoonViT projector activation is supported, got "
                f"{config.projector_hidden_act}."
            )

        if self.mm_projector_type == "patchmergerv2":
            self.linear_1 = ReplicatedLinear(
                self.hidden_size,
                self.hidden_size,
                bias=False,
                prefix=add_prefix("linear_1", prefix),
            )
            self.linear_2 = ReplicatedLinear(
                self.hidden_size,
                output_hidden_size,
                bias=False,
                prefix=add_prefix("linear_2", prefix),
            )
            self.post_norm = nn.RMSNorm(output_hidden_size, eps=config.projector_ln_eps)
            self.act = nn.GELU()
            return

        self.pre_norm = nn.LayerNorm(vision_hidden_size, eps=1e-5)
        self.linear_1 = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            prefix=add_prefix("linear_1", prefix),
        )
        self.linear_2 = ReplicatedLinear(
            self.hidden_size,
            output_hidden_size,
            bias=True,
            prefix=add_prefix("linear_2", prefix),
        )
        self.act = nn.GELU()

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        if self.mm_projector_type == "patchmergerv2":
            hidden_states = image_features.view(image_features.shape[0], -1)
            hidden_states, _ = self.linear_1(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states, _ = self.linear_2(hidden_states)
            return self.post_norm(hidden_states)

        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
        hidden_states, _ = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


@torch.inference_mode()
def mm_projection_auto(
    mm_projector: torch.nn.Module | None, vt_output: list[torch.Tensor]
):
    """Apply MM projector to vision tower outputs."""
    if mm_projector is None:
        return vt_output

    num_embedding_list = [x.shape[0] for x in vt_output]
    batched = torch.cat(vt_output, dim=0)
    proj_out = mm_projector(batched)
    proj_out = proj_out.reshape(-1, proj_out.shape[-1])
    proj_out = torch.split(proj_out, num_embedding_list)
    return proj_out


class MoonViTVisionPath(nn.Module):
    """Shared MoonViT tower/projector lifecycle for Kimi multimodal models."""

    def __init__(
        self,
        vision_config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        mm_attention_backend: str | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.vision_config = vision_config
        self.vision_spec = MoonViTConfig.from_config(vision_config) if enabled else None
        if not enabled:
            self.vision_tower = None
            self.mm_projector = None
            return

        self.vision_tower = MoonViT3dPretrainedModel(
            self.vision_spec,
            mapping=mapping,
            quant_config=(
                quant_config if isinstance(quant_config, ModelSlimConfig) else None
            ),
            prefix="vision_tower",
            mm_attention_backend=mm_attention_backend,
        )
        self.mm_projector = MoonViTMultiModalProjector(self.vision_spec)

    def make_image_warmup_items(self) -> list[MultimodalDataItem]:
        """Build a synthetic image at MoonViT's native position grid."""
        grid_height = int(self.vision_spec.init_pos_emb_height)
        grid_width = int(self.vision_spec.init_pos_emb_width)
        merge_h, merge_w = self.vision_tower.merge_kernel_size
        if grid_height % merge_h or grid_width % merge_w:
            raise ValueError(
                "MoonViT's native position grid "
                f"{(grid_height, grid_width)} must be divisible by "
                f"merge_kernel_size={(merge_h, merge_w)}"
            )
        patch_embed = self.vision_tower.patch_embed
        patch_height, patch_width = patch_embed.patch_size
        feature = torch.zeros(
            grid_height * grid_width,
            3,
            patch_height,
            patch_width,
            dtype=patch_embed.proj.weight.dtype,
        )
        grid_thws = torch.tensor([[1, grid_height, grid_width]], dtype=torch.long)
        return [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=feature,
                model_specific_data={"grid_thws": grid_thws},
            )
        ]

    def pre_encode(
        self, items: list[MultimodalDataItem]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.vision_tower.device
        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = torch.cat(
            [item.feature.to(device, non_blocking=True) for item in items], dim=0
        ).to(dtype=target_dtype)
        grid_thws = torch.cat([item.grid_thws for item in items], dim=0).to(device)
        hidden_states = self.vision_tower.patch_embed(pixel_values, grid_thws)
        return hidden_states, grid_thws

    def post_encode(
        self, encoder_outs: list[torch.Tensor], grid_thws: torch.Tensor
    ) -> torch.Tensor:
        merged = tpool_patch_merger(
            torch.cat(encoder_outs, dim=0),
            grid_thws,
            merge_kernel_size=self.vision_tower.merge_kernel_size,
        )
        projected = mm_projection_auto(self.mm_projector, merged)
        return torch.cat(projected, dim=0)

    def embed_media(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        tokens, grid_thws = self.pre_encode(items)
        encoder = self.vision_tower.encoder
        encoded = encoder.forward_blocks(tokens, encoder.prepare_metadata(grid_thws))
        return self.post_encode([encoded.squeeze(0)], grid_thws)

    def make_encoder_cudagraph_wrapper(
        self, mapping: Mapping
    ) -> EncoderForwardStepRunner:
        return EncoderForwardStepRunner(
            adapter=VisionEncoderCudaGraphAdapter(
                tower=self.vision_tower.encoder,
                pre_encode=self.pre_encode,
                post_encode=self.post_encode,
                out_div=1,
                merge=1,
                input_feature_shape=(self.vision_spec.hidden_size,),
                modality_name="image",
                out_squeeze_dim=0,
                capture_tp_size=mapping.vision.tp_size,
                capture_tp_group=mapping.vision.tp_group,
            ),
            budget_range=(256, 16384),
        )
