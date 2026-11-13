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

"""Inkling2 relative GQA and convolution stream wiring.

The shared prologue owns KV writes. The existing Inkling cache recipe owns
raw convolution history and checkpoints for every stream.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import torch
from tokenspeed_kernel.ops.attention.prologue import (
    ConvStream,
    KVShortConv,
    RelativeHeadNorm,
)
from tokenspeed_kernel.ops.communication.mailbox import (
    MAILBOX_ROWS,
    PROJECTION_MAX_ROWS,
    MailboxTarget,
    publish_projection,
)
from tokenspeed_kernel.ops.conv import inkling_ring_sconv_mailbox
from tokenspeed_kernel.ops.gemm import mm as kernel_mm

# skinny_mm is not registered; skinny_mm_supported keeps other platforms on kernel_mm.
from tokenspeed_kernel.ops.gemm.triton_skinny import skinny_mm, skinny_mm_supported
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.configs.inkling2 import Inkling2ModelConfig
from tokenspeed.runtime.configs.inkling_config import (
    InklingConvStream,
    inkling_conv_stream_layout,
)
from tokenspeed.runtime.distributed.comm_ops import (
    all_reduce,
    all_reduce_sconv,
    supports_all_reduce_sconv,
)
from tokenspeed.runtime.distributed.mapping import Group, Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.dense import UnquantizedLinearMethod
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.paged_attention import (
    PagedAttention,
    hf_sliding_window_to_window_left,
)
from tokenspeed.runtime.models.inkling import (
    _sconv_apply,
    _sconv_stream,
    sconv_decodes_without_publish,
)
from tokenspeed.runtime.utils.common import set_weight_attrs


@contextmanager
def _fp32_split_k_sums():
    """Keep cuBLAS from summing split-K partials of BF16/FP16 outputs in the
    output dtype, which PyTorch allows by default."""
    matmul = torch.backends.cuda.matmul
    allowed = (
        matmul.allow_bf16_reduced_precision_reduction,
        matmul.allow_fp16_reduced_precision_reduction,
    )
    matmul.allow_bf16_reduced_precision_reduction = False
    matmul.allow_fp16_reduced_precision_reduction = False
    try:
        yield
    finally:
        (
            matmul.allow_bf16_reduced_precision_reduction,
            matmul.allow_fp16_reduced_precision_reduction,
        ) = allowed


class FP32AccumulationLinearMethod(UnquantizedLinearMethod):
    """Keep the dense result in FP32 until its final input-dtype store.

    This preserves ordinary weight loading and the layer's output dtype while
    avoiding an implicit low-precision intermediate in the selected GEMM.
    Bias-free decode-size inputs on NVIDIA take the GEMM facade's small-M
    BF16 path (FlashInfer's autotuned runner on Blackwell), biased ones
    ``skinny_mm``; both accumulate in FP32 and round once. Larger inputs of
    bias-free layers on NVIDIA take the facade's result in the input dtype
    with split-K partials summed in FP32; other inputs take an FP32 result.
    """

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if skinny_mm_supported(x, layer.weight):
            if bias is None:
                return kernel_mm(x, layer.weight, out_dtype=x.dtype)
            return skinny_mm(x, layer.weight, bias, out_dtype=x.dtype)
        # Keyed on the layer, not the call: a row-parallel layer passes its
        # bias on rank 0 only, and every rank must take the same GEMM (the
        # BF16 one runs collective tuning at startup).
        if (
            bias is None
            and layer.bias is None
            and x.is_cuda
            and current_platform().is_nvidia
        ):
            with _fp32_split_k_sums():
                return kernel_mm(x, layer.weight, out_dtype=x.dtype)
        output = kernel_mm(x, layer.weight, bias=bias, out_dtype=torch.float32)
        return output.to(x.dtype)

    def apply_into(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        block_scale: torch.Tensor | None,
        output_dtype: torch.dtype,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """``apply`` into ``out``; the GEMM writes it directly where ``apply``
        takes the input-dtype result above the small-M range."""
        if (
            block_scale is None
            and output_dtype == x.dtype
            and not skinny_mm_supported(x, layer.weight)
            and bias is None
            and layer.bias is None
            and x.is_cuda
            and current_platform().is_nvidia
        ):
            with _fp32_split_k_sums():
                kernel_mm(x, layer.weight, out=out, out_dtype=x.dtype)
            return out
        return super().apply_into(layer, x, bias, block_scale, output_dtype, out)


@dataclass(frozen=True)
class Inkling2DynamicTaps:
    """Per-head taps projected from a source row of ``source_width`` channels."""

    source_width: int
    head_dim: int


class Inkling2ShortConvolution(nn.Module):
    """Static or input-dependent FIR using the common raw-input state path."""

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        stream: InklingConvStream,
        layer_id: int,
        channel_offset: int,
        tp_rank: int,
        dynamic: Inkling2DynamicTaps | None,
    ) -> None:
        super().__init__()
        if dim <= 0 or (
            dynamic is not None and (dynamic.head_dim <= 0 or dim % dynamic.head_dim)
        ):
            raise ValueError("Convolution channels must contain complete heads")
        self.dim = dim
        self.kernel_size = kernel_size
        self.stream = stream
        self.layer_id = layer_id
        self.channel_offset = channel_offset
        self.tp_rank = tp_rank
        self.dynamic = dynamic is not None
        self.num_heads: int | None = (
            None if dynamic is None else dim // dynamic.head_dim
        )
        self.output_scale = (kernel_size * (2 if self.dynamic else 1) + 1) ** -0.5
        if dynamic is not None:
            self.register_parameter("weight", None)
            self.static_w = nn.Parameter(
                torch.empty(kernel_size, dim, dtype=torch.bfloat16), requires_grad=False
            )
            self.weight_proj = nn.Parameter(
                torch.empty(
                    self.num_heads * kernel_size,
                    dynamic.source_width,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            set_weight_attrs(
                self.static_w, {"weight_loader": partial(self._load_shard, axis=1)}
            )
            set_weight_attrs(
                self.weight_proj, {"weight_loader": partial(self._load_shard, axis=0)}
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(dim, 1, kernel_size, dtype=torch.bfloat16),
                requires_grad=False,
            )
            self.register_parameter("static_w", None)
            self.register_parameter("weight_proj", None)
            set_weight_attrs(
                self.weight, {"weight_loader": partial(self._load_shard, axis=0)}
            )

    def _load_shard(
        self, param: nn.Parameter, loaded: torch.Tensor, *, axis: int
    ) -> None:
        if loaded.dtype != param.dtype:
            raise ValueError("Convolution loading must preserve checkpoint precision")
        if loaded.shape != param.shape:
            if loaded.shape[axis] % param.shape[axis]:
                raise ValueError(
                    "Convolution tensor does not contain complete rank shards"
                )
            loaded = loaded.narrow(
                axis, self.tp_rank * param.shape[axis], param.shape[axis]
            )
        if loaded.shape != param.shape:
            raise ValueError(
                f"Convolution shape mismatch: {loaded.shape} != {param.shape}"
            )
        with torch.no_grad():
            param.copy_(loaded)

    def taps(self) -> torch.Tensor:
        """The static taps ``[dim, W]``."""
        return self.static_w.mT if self.dynamic else self.weight.squeeze(1)

    def sums_partial(self, x: torch.Tensor, ctx: ForwardContext, group: Group) -> bool:
        """Whether ``forward(x, ctx, all_reduce_group=group)`` can take the
        partial ``x``'s sum over ``group`` as ``all_reduce_sconv`` does:
        static taps, a decode batch without a live publish, and a group whose
        workspace serves ``x``. ``x`` only describes the partial's shape."""
        return (
            not self.dynamic
            and sconv_decodes_without_publish(ctx)
            and supports_all_reduce_sconv(x, self.taps(), group)
        )

    def consumes_mailbox(self, tokens: int, ctx: ForwardContext) -> bool:
        """Whether ``from_mailbox`` serves a batch of ``tokens``: static taps
        and a decode batch without a live publish, as for ``sums_partial``."""
        return (
            not self.dynamic
            and sconv_decodes_without_publish(ctx)
            and 0 < tokens <= MAILBOX_ROWS
        )

    def from_mailbox(
        self, source: MailboxTarget, tokens: int, ctx: ForwardContext
    ) -> torch.Tensor:
        """Convolve the TP sum the ranks' producers published into ``source``
        (admitted by ``consumes_mailbox``): the sum of ``all_reduce_sconv``,
        added in FP32 rank order and rounded once."""
        state, checkpoints, batch = _sconv_stream(
            ctx, self.layer_id, self.channel_offset, self.dim, self._group()
        )
        return inkling_ring_sconv_mailbox(
            source.mailbox.slots(source.parity),
            tokens,
            self.taps(),
            state,
            batch.cu_seqlens,
            batch.seq_idx,
            batch.cache_indices,
            batch.has_initial_state,
            batch.seq_lens,
            batch.block_table,
            checkpoints[0],
            output_scale=self.output_scale,
        )

    def forward(
        self,
        x: torch.Tensor,
        ctx: ForwardContext,
        *,
        all_reduce_group: Group | None,
        token_range: tuple[int, int] | None,
    ) -> torch.Tensor:
        """Convolve ``x`` with static taps; a dynamic convolution's coefficients
        are projected with its layer's K/V projection (Inkling2Attention).
        With ``all_reduce_group`` (admitted by ``sums_partial``) ``x`` is
        this rank's partial, summed over the group by ``all_reduce_sconv``.
        ``token_range`` computes only those prefill rows (a sequence-parallel
        rank's share, ``inkling_ring_sconv``)."""
        if self.dynamic:
            raise ValueError("dynamic convolutions run in the attention prologue")
        weight = self.taps()
        if all_reduce_group is not None:
            state, checkpoints, batch = _sconv_stream(
                ctx, self.layer_id, self.channel_offset, self.dim, self._group()
            )
            return all_reduce_sconv(
                x.view(x.shape[0], self.dim),
                weight,
                state,
                batch.cu_seqlens,
                batch.seq_idx,
                batch.cache_indices,
                batch.has_initial_state,
                batch.seq_lens,
                batch.block_table,
                checkpoints[0],
                self.output_scale,
                all_reduce_group,
            )
        return _sconv_apply(
            x,
            weight,
            ctx,
            self.layer_id,
            self.channel_offset,
            self.dim,
            conv_group=self._group(),
            dynamic_weights=None,
            output_scale=self.output_scale,
            token_range=token_range,
        )

    def _group(self) -> str:
        if self.stream in (InklingConvStream.K, InklingConvStream.V):
            return "kvconv"
        return "attnconv" if self.stream == InklingConvStream.ATTN else "mlpconv"


class Inkling2L2Clip(nn.Module):
    """Per-head clipping parameters, loaded with checkpoint-native scale names
    and TP slicing; the attention prologue applies the clip."""

    eps = 1e-6

    def __init__(
        self, heads: int, width: int, threshold: float, tp_rank: int, learned: bool
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.tp_rank = tp_rank
        if learned:
            self.scale = nn.Parameter(
                torch.empty(heads, width, dtype=torch.bfloat16), requires_grad=False
            )
            set_weight_attrs(self.scale, {"weight_loader": self._load_scale})
        else:
            self.register_parameter("scale", None)

    def _load_scale(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        if loaded.dtype != param.dtype:
            raise ValueError("Clipping scales must retain checkpoint precision")
        if loaded.shape != param.shape:
            loaded = loaded.narrow(0, self.tp_rank * param.shape[0], param.shape[0])
        if loaded.shape != param.shape:
            raise ValueError("Clipping scale shape does not match the local heads")
        with torch.no_grad():
            param.copy_(loaded)


class Inkling2RelativeProjection(nn.Module):
    """The relative-logit weight; the attention prologue applies it."""

    def __init__(self, width: int, extent: int) -> None:
        super().__init__()
        self.proj = nn.Parameter(
            torch.empty(width, extent, dtype=torch.bfloat16), requires_grad=False
        )


# Longer (compute-bound) batches project the Q, K and V sources with three GEMMs.
_STACKED_PROJECTION_MAX_TOKENS = 128


class Inkling2Attention(nn.Module):
    """Separate full-layer Q/K/V sources, relative bias and a zero-value sink."""

    def __init__(
        self,
        config: Inkling2ModelConfig,
        mapping: Mapping,
        layer_id: int,
        is_local: bool,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_local = is_local
        self.head_dim = config.swa_head_dim if is_local else config.head_dim
        self.d_rel = config.d_rel
        self.normalize_values = config.use_value_norm_no_gamma
        self.norm_eps = config.rms_norm_eps
        total_heads = (
            config.swa_num_attention_heads if is_local else config.num_attention_heads
        )
        total_kv_heads = (
            config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        )
        tp = mapping.attn.tp_size
        if total_heads % tp or total_kv_heads % tp:
            raise ValueError("Attention TP must divide both Q and KV head counts")
        self.num_heads = total_heads // tp
        self.num_kv_heads = total_kv_heads // tp
        self.split_sizes = [
            self.num_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.num_heads * self.d_rel,
            self.num_heads * self.head_dim,
            self.num_heads,
        ]
        # Full layers project their K and V convolution taps (one row per KV
        # head and tap, copied from the convolutions after loading) with the
        # rest of the projection, from the same K and V sources.
        taps = 0 if is_local else self.num_kv_heads * config.sconv_kernel_size
        self.tap_sizes = [taps, taps] if taps else []
        pad = -(sum(self.split_sizes) + sum(self.tap_sizes)) % 8
        self.projection_sizes = (
            self.split_sizes + self.tap_sizes + ([pad] if pad else [])
        )
        self.qkvr = MergedColumnParallelLinear(
            config.hidden_size,
            [size * tp for size in self.projection_sizes],
            bias=config.q_bias,
            params_dtype=torch.bfloat16,
            quant_config=None,
            tp_rank=mapping.attn.tp_rank,
            tp_size=tp,
            tp_group=mapping.attn.tp_group,
            prefix=f"{prefix}.qkvr",
        )
        self.qkvr.quant_method = FP32AccumulationLinearMethod()
        # Rows the checkpoint does not hold: the taps (derived after loading) and the pad.
        derived = sum(self.tap_sizes) + pad
        if derived:
            with torch.no_grad():
                self.qkvr.weight[-derived:].zero_()
        if self.qkvr.bias is not None:
            with torch.no_grad():
                self.qkvr.bias.zero_()
        self.wo_ud = RowParallelLinear(
            total_heads * self.head_dim,
            config.hidden_size,
            bias=config.o_bias,
            params_dtype=torch.bfloat16,
            quant_config=None,
            tp_rank=mapping.attn.tp_rank,
            tp_size=tp,
            tp_group=mapping.attn.tp_group,
            reduce_results=False,
            prefix=f"{prefix}.wo_ud",
        )
        self.tp_group = mapping.attn.tp_group if tp > 1 else None
        self.wo_ud.quant_method = FP32AccumulationLinearMethod()
        self.q_l2_clip = Inkling2L2Clip(
            self.num_heads,
            self.head_dim,
            config.qk_l2_clip_threshold,
            mapping.attn.tp_rank,
            True,
        )
        self.k_l2_clip = Inkling2L2Clip(
            self.num_kv_heads,
            self.head_dim,
            config.qk_l2_clip_threshold,
            mapping.attn.tp_rank,
            False,
        )
        self.rel_l2_clip = Inkling2L2Clip(
            self.num_heads,
            self.d_rel,
            config.rel_l2_clip_threshold,
            mapping.attn.tp_rank,
            True,
        )
        extent = config.sliding_window_size if is_local else config.rel_extent
        self.rel_logits_proj = Inkling2RelativeProjection(self.d_rel, extent)
        layout = inkling_conv_stream_layout(config, tp)
        args = dict(
            dim=self.num_kv_heads * self.head_dim,
            kernel_size=config.sconv_kernel_size,
            layer_id=layer_id,
            tp_rank=mapping.attn.tp_rank,
            dynamic=(
                None
                if is_local
                else Inkling2DynamicTaps(config.hidden_size, self.head_dim)
            ),
        )
        self.k_sconv = Inkling2ShortConvolution(
            stream=InklingConvStream.K,
            channel_offset=layout[InklingConvStream.K][0],
            **args,
        )
        self.v_sconv = Inkling2ShortConvolution(
            stream=InklingConvStream.V,
            channel_offset=layout[InklingConvStream.V][0],
            **args,
        )
        # Local K and V streams register as one over their adjacent channels.
        if (
            is_local
            and self.v_sconv.channel_offset != self.k_sconv.channel_offset + args["dim"]
        ):
            raise ValueError("Local K and V convolution streams must be adjacent")
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=(
                hf_sliding_window_to_window_left(config.sliding_window_size)
                if is_local
                else -1
            ),
            rotary_emb=None,
            qk_norm=None,
        )

    def derive_projected_taps(self) -> None:
        """Write the full layer's tap rows of the projection from the loaded
        convolutions; safe to repeat."""
        if not self.tap_sizes:
            return
        start = sum(self.split_sizes)
        with torch.no_grad():
            for conv, size in zip((self.k_sconv, self.v_sconv), self.tap_sizes):
                self.qkvr.weight[start : start + size].copy_(conv.weight_proj)
                start += size

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        projected, _ = self.qkvr(x)
        return projected.split(self.projection_sizes, dim=-1)[
            : len(self.split_sizes) + len(self.tap_sizes)
        ]

    def _project_full(
        self, sources: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> list[tuple[torch.Tensor, ...]]:
        """The Q, K and V projections of a full layer's three sources."""
        q_source = sources[0]
        tokens = q_source.shape[0]
        if tokens > _STACKED_PROJECTION_MAX_TOKENS:
            return [self._project(source) for source in sources]
        # One GEMM over all three sources reads the weight once.
        row_bytes = q_source.numel() * q_source.element_size()
        storage = q_source.untyped_storage().data_ptr()
        # as_strided reads in place only within one storage; adjacent allocations are copied.
        adjacent = all(
            source.is_contiguous() and source.untyped_storage().data_ptr() == storage
            for source in sources
        ) and all(
            sources[i + 1].data_ptr() - sources[i].data_ptr() == row_bytes
            for i in range(2)
        )
        # Explicit row-major strides: a one-row view is contiguous whatever its row stride.
        stacked, _ = self.qkvr(
            q_source.as_strided((3 * tokens, q_source.shape[1]), (q_source.shape[1], 1))
            if adjacent
            else torch.cat(sources)
        )
        return [
            stacked[part * tokens : (part + 1) * tokens].split(
                self.projection_sizes, dim=-1
            )[: len(self.split_sizes) + len(self.tap_sizes)]
            for part in range(3)
        ]

    def _kv_conv(
        self, ctx: ForwardContext, taps: tuple[torch.Tensor | None, torch.Tensor | None]
    ) -> KVShortConv:
        """This forward's K/V convolution, which the prologue applies to the
        K and V projections before their head steps; ``taps`` are the full
        layer's projected K and V coefficients (``None`` for static taps)."""
        convs = (self.k_sconv, self.v_sconv)
        if self.is_local:
            ring, fields, batch = _sconv_stream(
                ctx,
                self.layer_id,
                self.k_sconv.channel_offset,
                2 * self.k_sconv.dim,
                "kvconv",
            )
            dim = self.k_sconv.dim
            rings = (ring[:, :, :dim], ring[:, :, dim:])
        else:
            (k_ring, k_fields, batch), (v_ring, v_fields, _) = (
                _sconv_stream(ctx, self.layer_id, c.channel_offset, c.dim, "kvconv")
                for c in convs
            )
            rings, fields = (k_ring, v_ring), (*k_fields, *v_fields)
        k, v = (
            ConvStream(
                weight=conv.taps(),
                dynamic_weights=(
                    None
                    if projected is None
                    else projected.view(-1, conv.num_heads, conv.kernel_size)
                ),
                ring=ring,
                checkpoint=field,
            )
            for conv, projected, ring, field in zip(convs, taps, rings, fields)
        )
        return KVShortConv(
            k=k, v=v, batch=batch, output_scale=self.k_sconv.output_scale
        )

    def publishes(self, tokens: int) -> bool:
        """Whether ``forward`` can publish the output projection of a batch of
        ``tokens`` into a mailbox (``publish_into``)."""
        return self.wo_ud.bias is None and 0 < tokens <= PROJECTION_MAX_ROWS

    def forward(
        self,
        streams: tuple[torch.Tensor, ...],
        ctx: ForwardContext,
        log_scaling_tau: torch.Tensor | None,
        *,
        reduce_output: bool,
        publish_into: MailboxTarget | None,
        partial_out: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """The attention output; without ``reduce_output``, this rank's
        partial of it, which the caller sums over ``tp_group``, written into
        ``partial_out`` (BF16 ``[T, H]``) when given. With ``publish_into``
        (admitted by ``publishes``) the partial goes into that mailbox parity
        instead, for its consumer to sum (returns None): FP32 MMA chains over
        K rounded once, as ``wo_ud`` accumulates, in another order."""
        q_source = streams[0]
        k_source, v_source = (q_source, q_source) if self.is_local else streams[1:]
        tokens = q_source.shape[0]
        if self.is_local:
            q_projection = k_projection = v_projection = self._project(q_source)
        else:
            q_projection, k_projection, v_projection = self._project_full(
                (q_source, k_source, v_source)
            )
        q, _, _, relative, gate, sink = q_projection[:6]
        # A full layer's K taps come from the K source's row, V's from V's.
        taps = (None, None) if self.is_local else (k_projection[6], v_projection[7])
        tau = None if self.is_local else log_scaling_tau
        # The prologue convolves the K and V rows, clips the Q, K and relative
        # heads, normalizes the V heads, projects the relative logits and
        # writes the KV cache.
        prepared = self.attn.relative_prologue(
            q.view(tokens, self.num_heads, self.head_dim),
            k_projection[1],
            v_projection[2],
            ctx,
            RelativeHeadNorm(
                relative=relative.view(tokens, self.num_heads, self.d_rel),
                projection=self.rel_logits_proj.proj,
                q_scale=self.q_l2_clip.scale,
                k_scale=self.k_l2_clip.scale,
                rel_scale=self.rel_l2_clip.scale,
                q_threshold=self.q_l2_clip.threshold,
                k_threshold=self.k_l2_clip.threshold,
                rel_threshold=self.rel_l2_clip.threshold,
                clip_eps=Inkling2L2Clip.eps,
                value_norm_eps=self.norm_eps if self.normalize_values else None,
                q_output_scale=tau,
                kv_conv=self._kv_conv(ctx, taps),
            ),
        )
        # The backend gates the FP32 attention output by its LSE and the sink
        # (rel_mha_gate_sink), inside the decode kernel's reduction.
        gated = self.attn.attend(
            prepared.q,
            prepared.k,
            prepared.v,
            ctx,
            rel_logits=prepared.rel_logits,
            log_scaling_tau=None,
            gate=gate,
            sink=sink,
            output_dtype=q.dtype,
            return_lse=False,
            record_kv_cache=False,
        )
        if publish_into is not None:
            if reduce_output:
                raise ValueError(
                    "a published output projection is reduced by its consumer"
                )
            publish_projection(
                gated.reshape(tokens, -1).contiguous(),
                self.wo_ud.weight,
                publish_into.mailbox,
                publish_into.parity,
            )
            return None
        if partial_out is not None:
            if reduce_output:
                raise ValueError("a partial destination holds an unreduced output")
            return self.wo_ud.quant_method.apply_into(
                self.wo_ud, gated.view(tokens, -1), None, None, gated.dtype, partial_out
            )
        output, _ = self.wo_ud(gated.view(tokens, -1))
        if reduce_output and self.tp_group is not None:
            output = all_reduce(output, self.tp_group)
        return output
