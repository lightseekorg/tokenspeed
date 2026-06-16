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

"""Inference-only Kimi-K2.5 VLM (DeepseekV3 LM + MoonViT vision tower) compatible with HuggingFace weights."""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import activations

from tokenspeed.runtime.configs.kimi_k25_config import (
    KimiK25Config,
    KimiK25VisionConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.conv import Conv2dLayer
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.moe.expert_location import ModelConfigForExpertLocation
from tokenspeed.runtime.multimodal.embedder import (
    EncoderSpec,
    VisionEmbedder,
    pad_input_tokens,
)

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


try:
    from tokenspeed.runtime.layers.quantization.quark.quark import QuarkConfig
except ImportError:

    class QuarkConfig:
        pass


from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3ForCausalLM
from tokenspeed.runtime.multimodal.encoder_cudagraph import EncoderCudaGraphWrapper
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from tokenspeed.runtime.utils import add_prefix

logger = logging.getLogger(__name__)


class MLP2(nn.Module):
    """
    Two-layer MLP helper used by the Kimi-K2.5 MoonViT blocks.

    This helper is inlined so the TokenSpeed VLM snapshot can keep only the
    Kimi-K2.5 target model.
    """

    def __init__(
        self,
        dims: list[int],
        activation,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        assert len(dims) == 3

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
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        mm_attention_backend: str | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.mlp = MLP2(
            [hidden_dim, mlp_dim, hidden_dim],
            activation,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.attn = VisionAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=attn_bias,
            proj_bias=attn_bias,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            customized_position_embedding_applier=apply_rope,
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
        assert isinstance(max_seqlen, int), (
            f"max_seqlen must be a Python int for capture-safety, "
            f"got {type(max_seqlen)}"
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
    assert embed_dim % 2 == 0
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
        self.register_buffer(
            "time_weight",
            torch.from_numpy(get_1d_sincos_pos_embed(self.dim, self.num_frames))
            .float()
            .unsqueeze(1),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            assert t <= self.num_frames, f"t:{t} > self.num_frames:{self.num_frames}"
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
        assert self.dim % 4 == 0, "dim must be divisible by 4"
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
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
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
    ):
        super().__init__()
        assert isinstance(
            patch_size, int | Sequence
        ), f"Invalid patch_size type: {type(patch_size)}"
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert (
            len(patch_size) == 2
        ), f"Expected patch_size to be a tuple of 2, got {patch_size}"
        self.patch_size = patch_size

        self.proj = Conv2dLayer(
            in_dim, out_dim, kernel_size=patch_size, stride=patch_size
        )

        if pos_emb_type == "divided_fixed":
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=pos_emb_height,
                width=pos_emb_width,
                num_frames=pos_emb_time,
                dim=out_dim,
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
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        assert (
            video_attn_type == "spatial_temporal"
        ), f'video_attn_type must be "spatial_temporal", got {video_attn_type}'
        self.video_attn_type = video_attn_type
        self.rope_2d = Rope2DPosEmbRepeated(
            block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512
        )
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
        self.final_layernorm = nn.LayerNorm(hidden_dim)

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
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        mm_attention_backend: str | None = None,
        **kwargs,
    ):
        super().__init__()
        config = deepcopy(config)
        self.config = config
        self.merge_kernel_size = config.merge_kernel_size

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
            pos_emb_type=config.pos_emb_type,
        )

        self.encoder = MoonViT3dEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "mlp_dim": config.intermediate_size,
                "activation": PytorchGELUTanh(),
                "attn_bias": True,
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


class K2VLMultiModalProjector(nn.Module):
    """Multi-modal projector with patch merging for K2-VL."""

    def __init__(
        self,
        config: KimiK25VisionConfig,
        prefix: str = "",
    ):
        super().__init__()

        # Hidden size after patch merging
        merge_h, merge_w = config.merge_kernel_size
        self.hidden_size = config.vt_hidden_size * merge_h * merge_w

        self.pre_norm = torch.nn.LayerNorm(config.vt_hidden_size, eps=1e-5)
        self.linear_1 = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            prefix=add_prefix("linear_1", prefix),
        )
        self.linear_2 = ReplicatedLinear(
            self.hidden_size,
            config.text_hidden_size,
            bias=True,
            prefix=add_prefix("linear_2", prefix),
        )
        self.act = nn.GELU()

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
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


class KimiK25ForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: KimiK25Config,
        mapping: Mapping,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
        **kwargs,  # fix init_tts argument error
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config
        self.is_multimodal_active = is_multimodal_active
        if not self.is_multimodal_active:
            self.vision_tower = None
            self.mm_projector = None
        else:
            self.vision_tower = MoonViT3dPretrainedModel(
                config.vision_config,
                quant_config=(
                    quant_config if isinstance(quant_config, ModelSlimConfig) else None
                ),
                prefix="vision_tower",
                mapping=mapping,
                mm_attention_backend=mm_attention_backend,
            )
            self.mm_projector = K2VLMultiModalProjector(config.vision_config)

        self.language_model = None
        if not getattr(config, "encoder_only", False):
            self.language_model = DeepseekV3ForCausalLM(
                config.text_config,
                mapping=mapping,
                quant_config=quant_config,
                prefix=(
                    "language_model"
                    if isinstance(quant_config, (ModelSlimConfig, QuarkConfig))
                    else ""
                ),
            )

        if self.is_multimodal_active:
            # Match vision-tower / mm-projector dtype to language-model dtype;
            # the vision tower defaults to float32 while the LM may be bf16 / fp8.
            if self.language_model is not None and hasattr(
                self.language_model, "dtype"
            ):
                target_dtype = self.language_model.dtype
                self.vision_tower = self.vision_tower.to(dtype=target_dtype)
                self.mm_projector = self.mm_projector.to(dtype=target_dtype)

            # image_encoder may be swapped to a cudagraph wrapper by ModelExecutor.
            self.vision_embedder = VisionEmbedder()
            self.image_encoder = self.get_image_feature
        else:
            self.vision_embedder = None
            self.image_encoder = None

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Eager image encode via the same ``pre_encode`` / ``forward_blocks``
        / ``post_encode`` decomposition the cudagraph wrapper uses, so the
        eager and captured paths share a single source of truth."""
        tokens, grid_thws = self.pre_encode(items)
        encoder = self.vision_tower.encoder
        encoded = encoder.forward_blocks(tokens, encoder.prepare_metadata(grid_thws))
        # forward_blocks keeps a leading batch dim of 1; squeeze it for per-image
        # consumption (mirrors ``out_squeeze_dim=0`` in
        # ``make_encoder_cudagraph_wrapper``).
        return self.post_encode([encoded.squeeze(0)], grid_thws)

    def pre_encode(
        self, items: List[MultimodalDataItem]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eager patch-embed before the captured region; returns (tokens, grid).

        Reads HF-native ``grid_thws`` on each item (matches the SMG gateway's
        Kimi-K2.5 processor).
        """
        device = self.vision_tower.device
        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            device=device, dtype=target_dtype
        )
        grid_thws = torch.concat([item.grid_thws for item in items], dim=0).to(device)
        hidden_states = self.vision_tower.patch_embed(pixel_values, grid_thws)
        return hidden_states, grid_thws

    def post_encode(
        self, encoder_outs: list[torch.Tensor], grid_thws: torch.Tensor
    ) -> torch.Tensor:
        """Eager merge + projection after the captured region; returns features."""
        merged = tpool_patch_merger(
            torch.cat(encoder_outs, dim=0),
            grid_thws,
            merge_kernel_size=self.vision_tower.merge_kernel_size,
        )
        proj_out = mm_projection_auto(self.mm_projector, merged)
        return torch.cat(proj_out, dim=0)

    def make_encoder_cudagraph_wrapper(self, mapping):
        # Captured region is ``MoonViT3dEncoder.forward_blocks`` (token-preserving
        # block loop); spatial/temporal merge lives in ``post_encode``, so
        # budgets are encoder-input patch counts (``out_div=1``). ``forward_blocks``
        # keeps a leading batch dim of 1 -- ``out_squeeze_dim=0`` drops it before
        # per-item slicing.
        return EncoderCudaGraphWrapper(
            mapping=mapping,
            tower=self.vision_tower.encoder,
            pre_encode=self.pre_encode,
            post_encode=self.post_encode,
            out_div=1,
            merge=1,
            budget_range=(256, 16384),
            input_feature_shape=(self.config.vision_config.hidden_size,),
            out_squeeze_dim=0,
        )

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return pad_input_tokens(input_ids, mm_inputs)

    @property
    def start_layer(self) -> int:
        return self.language_model.start_layer if self.language_model is not None else 0

    @property
    def end_layer(self) -> int:
        if self.language_model is not None:
            return self.language_model.end_layer
        text_config = getattr(self.config, "text_config", None)
        return int(getattr(text_config, "num_hidden_layers", 0))

    @property
    def routed_experts_weights_of_layer(self):
        return (
            self.language_model._routed_experts_weights_of_layer.value
            if self.language_model is not None
            else {}
        )

    @torch.no_grad()
    def forward(
        self,
        ctx,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        **kwargs,
    ):
        if self.language_model is None:
            raise RuntimeError("KimiK25 language_model is not initialized.")
        multimodal_context = kwargs.pop("multimodal_context", None)
        if (
            multimodal_context is not None
            and multimodal_context.has_extend_inputs()
            and not ctx.forward_mode.is_decode_or_idle()
        ):
            input_embeds, model_kwargs = self.vision_embedder.apply(
                input_ids=input_ids,
                text_embedding=self.get_input_embeddings(),
                ctx=multimodal_context,
                encoders={Modality.IMAGE: EncoderSpec(self.image_encoder)},
                multimodal_model=self,
                is_decode_or_idle=ctx.forward_mode.is_decode_or_idle(),
            )
            kwargs.update(model_kwargs)
            if input_embeds is not None:
                kwargs["input_embeds"] = input_embeds
        return self.language_model.forward(
            ctx,
            input_ids,
            positions,
            out_cache_loc,
            **kwargs,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights, streaming language weights to the language model.

        The language weights are forwarded to ``language_model.load_weights``
        lazily (as a generator) instead of being collected into a list first.
        Materializing the whole iterator would keep every loaded tensor alive
        at once; that is harmless for CPU-staged loaders but OOMs GPU-direct
        loaders (e.g. ``--load-format instanttensor``), which would then hold
        the entire model on the device during loading. Vision weights are
        small and are still collected, then loaded after the language model.
        """
        vision_weights: list[Tuple[str, torch.Tensor]] = []
        encoder_only = getattr(self.config, "encoder_only", False)
        load_vision = self.is_multimodal_active and not getattr(
            self.config, "language_only", False
        )

        def language_weights() -> Iterable[Tuple[str, torch.Tensor]]:
            for name, loaded_weight in weights:
                # nvidia/Kimi-K2.5-NVFP4 stores decoder layers under
                # language_model.layers.*, while TokenSpeed's DeepSeek module
                # expects model.layers.* after stripping language_model.
                if name.startswith("language_model.layers."):
                    name = name.replace(
                        "language_model.layers.", "language_model.model.layers.", 1
                    )

                if "vision_tower" in name or "mm_projector" in name:
                    name = name.replace(r"wqkv.", r"attn.qkv_proj.")
                    name = name.replace(r"wo.", r"attn.proj.")
                    name = name.replace("mm_projector.proj.0", "mm_projector.linear_1")
                    name = name.replace("mm_projector.proj.2", "mm_projector.linear_2")
                    if load_vision:
                        vision_weights.append((name, loaded_weight))
                else:
                    yield name.replace("language_model.", ""), loaded_weight

        if not encoder_only:
            # Consumes the iterator lazily; fills vision_weights as a side
            # effect for the multimodal branch below.
            self.language_model.load_weights(language_weights())
        elif load_vision:
            # Encoder-only: still drain the iterator to collect vision weights.
            for _ in language_weights():
                pass

        if load_vision:
            params_dict = dict(self.named_parameters(remove_duplicate=False))
            for name, loaded_weight in vision_weights:
                if name not in params_dict:
                    raise ValueError(f"Weight {name} not found in params_dict")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    @classmethod
    def get_model_config_for_expert_location(cls, config: KimiK25Config):
        text_config = config.text_config
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=text_config.n_routed_experts,
            num_groups=text_config.n_group,
        )

    def set_eagle3_layers_to_capture(
        self, layer_ids: Optional[List[int]] = None
    ) -> None:
        """Set the layers to capture for EAGLE3 speculative decoding."""
        if self.language_model is None or not hasattr(
            self.language_model, "set_eagle3_layers_to_capture"
        ):
            raise AttributeError(
                "language_model does not support EAGLE3 speculative decoding."
            )

        self.language_model.set_eagle3_layers_to_capture(layer_ids)

    def set_dflash_layers_to_capture(self, layer_ids: List[int]) -> None:
        """Set the layers to capture for DFLASH draft model training."""
        if not hasattr(self.language_model, "set_dflash_layers_to_capture"):
            raise AttributeError(
                "language_model does not support DFLASH layer capture."
            )

        self.language_model.set_dflash_layers_to_capture(layer_ids)

    def get_input_embeddings(self):
        if hasattr(self.language_model, "get_input_embeddings"):
            return self.language_model.get_input_embeddings()
        if hasattr(self.language_model, "model") and hasattr(
            self.language_model.model, "embed_tokens"
        ):
            return self.language_model.model.embed_tokens
        raise AttributeError("language_model does not support get_input_embeddings().")

    @property
    def lm_head(self):
        if not hasattr(self.language_model, "lm_head"):
            raise AttributeError("language_model does not expose lm_head.")

        return self.language_model.lm_head

    @property
    def logits_processor(self):
        if self.language_model is None or not hasattr(
            self.language_model, "logits_processor"
        ):
            raise AttributeError("language_model does not expose logits_processor.")

        return self.language_model.logits_processor

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embedding and LM head weights for speculative decoding."""
        if self.language_model is None or not hasattr(
            self.language_model, "get_embed_and_head"
        ):
            raise AttributeError(
                "language_model does not support get_embed_and_head()."
            )

        return self.language_model.get_embed_and_head()

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        """Set embedding and LM head weights for speculative decoding."""
        if self.language_model is None or not hasattr(
            self.language_model, "set_embed_and_head"
        ):
            raise AttributeError(
                "language_model does not support set_embed_and_head()."
            )

        self.language_model.set_embed_and_head(embed, head)


EntryClass = [KimiK25ForConditionalGeneration]
