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

"""Cache-less multi-headed attention used by VLM vision encoders.

Encode-only attention layer used by Qwen3.5-VL / Kimi-K2.5-VL vision towers,
plus its backend dispatch table. Backends are free functions (not
``AttentionBackend`` subclasses) because the vision encoder is single-shot
with no KV cache, no decode/extend split, and no graph capture protocol --
the ``AttentionBackend`` ABC's prefill/decode/extend lifecycle does not
apply. This file is therefore kept out of ``backends/`` (the home of
``AttentionBackend`` subclasses) and lives at the ``layers/attention/`` top
level alongside ``registry.py`` / ``utils.py``.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed import utils as dist_utils
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.quantization import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import apply_rotary_pos_emb_native
from tokenspeed.runtime.utils import add_prefix, round_up
from tokenspeed.runtime.utils.env import envs, get_global_server_args

logger = logging.getLogger(__name__)

_platform = current_platform()
_is_nvidia = _platform.is_nvidia
_is_amd = _platform.is_amd

if _is_nvidia:
    from tokenspeed_kernel.ops.attention.flash_attn import flash_attn_varlen_func
    from tokenspeed_kernel.ops.attention.flashinfer import (
        cudnn_batch_prefill_with_kv_cache,
    )

from tokenspeed_kernel.ops.attention.triton.context import context_attention_fwd

# CUDA-graph bucketing for the cuDNN vision prefill backend: batch and max
# seqlen are quantized so a small set of captured graphs covers the request
# distribution. The consts are consumed by VLM tower models, not by
# ``VisionAttention`` itself.
VIT_CUDNN_WORKSPACE_BYTES = 128 * 1024 * 1024
VIT_CUDNN_BATCH_BUCKETS: tuple[int, ...] = (8, 16, 32, 64)
VIT_CUDNN_SEQLEN_BUCKETS: tuple[int, ...] = (4096, 8192, 16384, 32768, 65536, 131072)


def round_up_to_bucket(value: int, buckets: tuple[int, ...]) -> int:
    """Smallest bucket >= value; values past the last bucket round up to a
    multiple of it. Used by vision tower code to pad batch size and max-seqlen
    into a finite set of captured cuDNN graph shapes.
    """
    if value <= 0:
        return buckets[0]
    for bucket in buckets:
        if bucket >= value:
            return bucket
    return round_up(value, buckets[-1])


# === Backend dispatch ===
# The dispatcher always passes the full kwarg set (cu_seqlens / bsz / seq_len /
# softmax_scale / max_seqlen / sequence_lengths / workspace_buffer); each
# backend declares the subset it uses and absorbs the rest via ``**_``.


def _varlen_metadata(
    cu_seqlens: torch.Tensor | None,
    bsz: int,
    seq_len: int,
    *,
    device: torch.device,
    max_seqlen: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Resolve cu_seqlens / seq_lens / max_seqlen shared by the varlen backends.

    ``max_seqlen`` is honored when the caller supplies it (the capture-safe
    path); only the eager fallback derives it via ``.item()``, which forces a
    GPU->CPU sync that is illegal inside a captured CUDA graph. Deriving it
    once here keeps every varlen backend capture-safe instead of each kernel
    wrapper re-deriving (and re-syncing) it.
    """
    if cu_seqlens is None:
        cu_seqlens = torch.arange(
            0, (bsz + 1) * seq_len, step=seq_len, dtype=torch.int32, device=device
        )
    else:
        cu_seqlens = cu_seqlens.to(dtype=torch.int32, device=device)
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    if max_seqlen is None:
        max_seqlen = int(seq_lens.max().item())
    return cu_seqlens, seq_lens, int(max_seqlen)


def vision_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None,
    bsz: int,
    seq_len: int,
    softmax_scale: Optional[float] = None,
    max_seqlen: Optional[int] = None,
    **_: Any,
) -> torch.Tensor:
    """Triton context attention without a causal mask."""
    cu_seqlens, seq_lens, max_seqlen = _varlen_metadata(
        cu_seqlens, bsz, seq_len, device=q.device, max_seqlen=max_seqlen
    )
    output = torch.empty_like(q)
    context_attention_fwd(
        q,
        k,
        v,
        output,
        cu_seqlens,
        seq_lens,
        max_seqlen,
        is_causal=False,
        sm_scale=softmax_scale,
    )
    return output


def vision_attn_fa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None,
    bsz: int,
    seq_len: int,
    softmax_scale: Optional[float] = None,
    max_seqlen: Optional[int] = None,
    **_: Any,
) -> torch.Tensor:
    cu_seqlens, _, max_seqlen = _varlen_metadata(
        cu_seqlens, bsz, seq_len, device=q.device, max_seqlen=max_seqlen
    )
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=softmax_scale,
    )


def vision_attn_fa4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None,
    bsz: int,
    seq_len: int,
    softmax_scale: Optional[float] = None,
    max_seqlen: Optional[int] = None,
    **_: Any,
) -> torch.Tensor:
    cu_seqlens, _, max_seqlen = _varlen_metadata(
        cu_seqlens, bsz, seq_len, device=q.device, max_seqlen=max_seqlen
    )
    result = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=softmax_scale,
    )
    # FA4 CUTE returns (output, lse) in newer builds and bare output in older
    # ones; downstream callers only consume the tensor.
    return result[0] if isinstance(result, tuple) else result


def vision_attn_flashinfer_cudnn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None,
    softmax_scale: Optional[float] = None,
    max_seqlen: Any = None,
    sequence_lengths: Optional[torch.Tensor] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    **_: Any,
) -> torch.Tensor:
    """cuDNN prefill backend. The caller (vision tower with cuDNN graph capture)
    prepares ``cu_seqlens`` as three concatenated element-offset indptrs
    ``[qk | v | o]`` of length ``batch+1`` each, plus ``sequence_lengths`` per
    real (un-padded) sequence and ``max_seqlen`` as the bucketed budget.
    """
    assert (
        sequence_lengths is not None
        and max_seqlen is not None
        and isinstance(cu_seqlens, torch.Tensor)
    ), "flashinfer_cudnn needs sequence_lengths, max_seqlen, and packed indptrs"

    # cuDNN wants a python int for the seq budget.
    max_seqlen = int(
        max_seqlen.item() if isinstance(max_seqlen, torch.Tensor) else max_seqlen
    )

    # Flatten (b, s, h, d) -> (b*s, h, d) when the caller hands us 4-D.
    in_4d = q.dim() == 4
    if in_4d:
        b4 = q.shape[0]
        q, k, v = (rearrange(t, "b s ... -> (b s) ...") for t in (q, k, v))

    seq_lens = sequence_lengths.view(-1).to(device=q.device, dtype=torch.int32)
    batch = seq_lens.numel()
    packed = cu_seqlens.view(-1).to(device=q.device, dtype=torch.int32)
    assert packed.numel() == 3 * (batch + 1)
    chunk = batch + 1
    qk_off = packed[:chunk].view(chunk, 1, 1, 1)
    v_off = packed[chunk : 2 * chunk].view(chunk, 1, 1, 1)
    o_off = packed[2 * chunk :].view(chunk, 1, 1, 1)
    seq_lens_4d = seq_lens.view(batch, 1, 1, 1)

    head_size = q.shape[-1]
    scale = softmax_scale if softmax_scale is not None else head_size**-0.5

    output, _ = cudnn_batch_prefill_with_kv_cache(
        q,
        k,
        v,
        scale,
        workspace_buffer,
        max_token_per_sequence=max_seqlen,
        max_sequence_kv=max_seqlen,
        actual_seq_lens_q=seq_lens_4d,
        actual_seq_lens_kv=seq_lens_4d,
        causal=False,
        return_lse=True,
        batch_offsets_q=qk_off,
        batch_offsets_k=qk_off,
        batch_offsets_v=v_off,
        batch_offsets_o=o_off,
        is_cuda_graph_compatible=True,
    )
    if in_4d:
        output = rearrange(output, "(b s) h d -> b s h d", b=b4)
    return output


_BACKENDS: dict[str, Callable[..., torch.Tensor]] = {
    "triton_attn": vision_attn_triton,
    "fa3": vision_attn_fa3,
    "fa4": vision_attn_fa4,
    "flashinfer_cudnn": vision_attn_flashinfer_cudnn,
}


def _default_vision_attn_backend() -> str:
    """Platform default backend name."""
    if _is_nvidia:
        if _platform.arch_version.major == 9:  # Hopper SM90
            return "fa3"
        if _platform.arch_version.major == 10:  # Blackwell SM100
            return "fa4"
        return "triton_attn"
    if _is_amd:
        return "triton_attn"
    raise RuntimeError(
        f"No default vision attention backend for platform {_platform}; "
        "set --mm-attention-backend explicitly."
    )


@functools.lru_cache(maxsize=None)
def _resolve_backend(name: Optional[str]) -> Callable[..., torch.Tensor]:
    """Resolve a backend name to its dispatch function.

    ``None`` falls back to the platform default; an unknown or platform-
    incompatible name raises ValueError listing the registered backends.
    Cached so a process logs the chosen backend exactly once per name.
    """
    explicit = name is not None
    if name is None:
        name = _default_vision_attn_backend()
    fn = _BACKENDS.get(name)
    if fn is None:
        raise ValueError(
            f"Unknown vision attention backend {name!r} "
            f"(check --mm-attention-backend); available: {sorted(_BACKENDS)}"
        )
    if name in ("fa3", "fa4", "flashinfer_cudnn") and not _is_nvidia:
        raise ValueError(
            f"vision attention backend {name!r} is only available on NVIDIA CUDA"
        )
    if name == "fa3" and _platform.is_blackwell:
        raise ValueError("The 'fa3' backend is not supported on Blackwell GPUs")
    logger.info(
        f"vision attention backend: {name} ({'override' if explicit else 'auto'})"
    )
    return fn


class VisionAttention(nn.Module):
    r"""Multi-headed attention without any cache, mostly used for multimodal transformers."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mapping: Mapping,
        head_size: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        proj_bias: bool = True,
        qkv_bias: bool = True,
        customized_position_embedding_applier: Callable[
            [torch.Tensor, torch.Tensor, Any, Any], Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        workspace_buffer: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.vision = mapping.vision
        self.tp_size = self.vision.tp_size
        self.tp_rank = self.vision.tp_rank
        self.tp_group = self.vision.tp_group
        self.head_size = head_size if head_size is not None else embed_dim // num_heads
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size
        )
        self.num_attention_kv_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size
        )

        self.q_size = self.num_attention_heads_per_partition * self.head_size
        self.kv_size = self.num_attention_kv_heads_per_partition * self.head_size

        self.customized_position_embedding_applier = (
            customized_position_embedding_applier
        )
        self._backend_fn = _resolve_backend(
            get_global_server_args().mm_attention_backend
        )
        self._workspace_buffer = workspace_buffer

        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_size,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            tp_group=self.tp_group,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.proj = RowParallelLinear(
            input_size=num_heads * self.head_size,
            output_size=embed_dim,
            bias=proj_bias,
            quant_config=quant_config,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            tp_group=self.tp_group,
            prefix=add_prefix("proj", prefix),
            reduce_results=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            x: [b, s, embed_dim]
            cu_seqlens: [b]
        Returns:
             [b, s, head * head_size]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x_shape = x.shape
        bsz, s, _ = x_shape
        head = self.num_attention_heads_per_partition
        kv_head = self.num_attention_kv_heads_per_partition

        max_seqlen = kwargs["max_seqlen"] if "max_seqlen" in kwargs else None
        sequence_lengths = (
            kwargs["sequence_lengths"] if "sequence_lengths" in kwargs else None
        )

        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.reshape(bsz * s, head, -1).contiguous()
        k = k.reshape(bsz * s, kv_head, -1).contiguous()
        v = v.reshape(bsz * s, kv_head, -1).contiguous()

        cos = None
        sin = None

        if position_embeddings is not None:
            if self.customized_position_embedding_applier is not None:
                q, k = self.customized_position_embedding_applier(
                    q, k, position_embeddings, x_shape
                )
            else:
                cos, sin = position_embeddings
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            cos = rotary_pos_emb_cos
            sin = rotary_pos_emb_sin

        if cos is not None and sin is not None:
            original_shape = q.shape

            # [total_tokens, head, head_size]
            q = q.view(-1, head, self.head_size)
            k = k.view(-1, head, self.head_size)

            if cos.size(-1) * 2 == self.head_size:
                cos = torch.cat([cos, cos], dim=-1)
                sin = torch.cat([sin, sin], dim=-1)

            q, k = apply_rotary_pos_emb_native(q, k, cos, sin)
            q = q.view(original_shape)
            k = k.view(original_shape)

        q, k, v = [
            rearrange(t, "b s ... -> (b s) ...") if t.dim() == 4 else t
            for t in (q, k, v)
        ]

        output = self._backend_fn(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            bsz=bsz,
            seq_len=s,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
            workspace_buffer=self._workspace_buffer,
        )

        output = rearrange(output, "(b s) ... h d -> b s ... (h d)", b=bsz)
        output, _ = self.proj(output)

        return output
