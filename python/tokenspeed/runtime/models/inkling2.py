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

"""Inkling2 text target: depth mixtures, shared attention caches, and exact loading.

The target keeps its checkpoint precision and depth residuals within each
forward. Persistent convolution state belongs to the shared cache subsystem.
Image patches use the shared multimodal embedder. DSpark draft execution and
frontend image preprocessing are not implemented here.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

# tanh_softcap is outside the registry; its Triton kernel serves every supported vendor.
from tokenspeed_kernel.ops.activation.triton import (
    tanh_softcap,
    tanh_softcap_rejection,
)

# log_scaling_tau is outside the registry; its Triton kernel serves every supported vendor.
from tokenspeed_kernel.ops.attention.rmha.triton import log_scaling_tau
from tokenspeed_kernel.ops.communication.mailbox import (
    Mailbox,
    MailboxTarget,
    mailbox_supported,
)
from tokenspeed_kernel.ops.gemm import mm as kernel_mm
from torch import nn

from tokenspeed.runtime.configs.inkling2 import (
    Inkling2MMConfig,
    Inkling2ModelConfig,
    Inkling2MTPConfig,
)
from tokenspeed.runtime.configs.inkling_config import (
    InklingConvStream,
    inkling_conv_stream_layout,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention_residual import (
    AttentionResidual,
    AttentionResidualState,
    PrecisionRMSNorm,
    norm_gains,
)
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
from tokenspeed.runtime.layers.moe.inkling2 import Inkling2MoE
from tokenspeed.runtime.layers.quantization.inkling2 import Inkling2QuantConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
from tokenspeed.runtime.models.base.input_embedding import DecoderInputEmbedding
from tokenspeed.runtime.models.inkling2_attention import (
    Inkling2Attention,
    Inkling2ShortConvolution,
)
from tokenspeed.runtime.models.inkling2_sequence import (
    Inkling2SequenceParallel,
    SequenceShare,
)
from tokenspeed.runtime.models.inkling2_vision import Inkling2VisionEncoder
from tokenspeed.runtime.moe.expert_location import ModelConfigForExpertLocation
from tokenspeed.runtime.multimodal.embedder import (
    EncoderSpec,
    VisionEmbedder,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalForwardContext,
)
from tokenspeed.runtime.utils.env import global_server_args_dict


class Inkling2LMHeadMethod(UnquantizedEmbeddingMethod):
    """Project BF16 activations and weights into an FP32 result before softcap."""

    def apply(
        self, layer: ParallelLMHead, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        return kernel_mm(x, layer.weight, bias=bias, out_dtype=torch.float32)


class Inkling2LogitsProcessor(LogitsProcessor):
    """Use shared row/gather/vocabulary selection, then cap every logits request."""

    def __init__(self, config: Inkling2ModelConfig, mapping: Mapping) -> None:
        super().__init__(
            config,
            skip_all_gather=False,
            logit_scale=None,
            do_argmax=False,
            tp_rank=mapping.lm_head.tp_rank,
            tp_size=mapping.lm_head.tp_size,
            tp_group=mapping.lm_head.tp_group,
            dp_lm_head_tp=False,
            attn_tp_size=mapping.attn.tp_size,
        )
        self.logit_softcap = config.final_logit_softcapping
        # Apply the same cap to prompt logprobs and sampled logits.
        self.final_logit_softcapping = None

    def fuse_argmax(self) -> None:
        """Keep full logits so padded vocabulary rows cannot win an argmax."""

    def _vocab_logits(self, logits: torch.Tensor) -> torch.Tensor:
        cap = self.logit_softcap
        if cap is not None and cap > 0:
            if logits.dtype == torch.float32 and not tanh_softcap_rejection(
                logits, cap
            ):
                # The in-place statement below in one pass, byte for byte,
                # reading the column slice in place into a fresh tensor.
                return tanh_softcap(logits, cap)
            return logits.contiguous().div_(cap).tanh_().mul_(cap)
        return logits.contiguous()


class Inkling2DecoderLayer(nn.Module):
    """One depth-mixture decoder layer with full-width output convolutions."""

    def __init__(
        self,
        config: Inkling2ModelConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: Inkling2QuantConfig,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_local = layer_id in config.local_layer_ids
        prefix = f"model.llm.layers.{layer_id}"
        self.attn = Inkling2Attention(
            config, mapping, layer_id, self.is_local, f"{prefix}.attn"
        )
        self.mlp = Inkling2MoE(config, mapping, layer_id, quant_config, f"{prefix}.mlp")

        def norm() -> PrecisionRMSNorm:
            return PrecisionRMSNorm(
                config.hidden_size,
                config.rms_norm_eps,
                config.pre_norm_pre_affine,
                torch.bfloat16,
                True,
                False,
            )

        self.attn_norm = norm() if self.is_local else None
        self.q_input_norm = None if self.is_local else norm()
        self.k_input_norm = None if self.is_local else norm()
        self.v_input_norm = None if self.is_local else norm()
        self.mlp_norm = norm()
        layout = inkling_conv_stream_layout(config, mapping.attn.tp_size)
        self.attn_sconv = Inkling2ShortConvolution(
            config.hidden_size,
            config.sconv_kernel_size,
            InklingConvStream.ATTN,
            layer_id,
            layout[InklingConvStream.ATTN][0],
            tp_rank=0,
            dynamic=None,
        )
        self.mlp_sconv = Inkling2ShortConvolution(
            config.hidden_size,
            config.sconv_kernel_size,
            InklingConvStream.MLP,
            layer_id,
            layout[InklingConvStream.MLP][0],
            tp_rank=0,
            dynamic=None,
        )
        # The three full-layer input gains, stacked for the depth-mix kernel.
        self.register_buffer(
            "input_norm_gains",
            (
                None
                if self.is_local or config.pre_norm_pre_affine
                else torch.zeros((3, config.hidden_size), dtype=torch.bfloat16)
            ),
            persistent=False,
        )

    def input_norms(self) -> tuple[PrecisionRMSNorm, ...]:
        if self.is_local:
            return (self.attn_norm,)
        return (self.q_input_norm, self.k_input_norm, self.v_input_norm)

    def derive_input_norm_gains(self) -> None:
        """Write ``input_norm_gains`` from the loaded norms; safe to repeat."""
        if self.input_norm_gains is not None:
            with torch.no_grad():
                self.input_norm_gains.copy_(norm_gains(self.input_norms()))

    def forward(
        self,
        state: AttentionResidualState,
        ctx: ForwardContext,
        log_scaling_tau: torch.Tensor | None,
        mailbox: Mailbox | None,
        share: SequenceShare | None,
    ) -> None:
        norms = self.input_norms()
        normalized = state.attention_input(
            self.layer_id,
            norms,
            norm_gains(norms) if self.is_local else self.input_norm_gains,
        )
        tokens = normalized[0].shape[0]
        # With a mailbox, both sublayers' partials go straight to the ranks'
        # output convolutions (the attention's output projection and the
        # MoE's route fold publish them) or neither does: the two uses
        # alternate the mailbox's parities.
        if (
            mailbox is not None
            and self.attn.publishes(tokens)
            and self.mlp.all_reduces(tokens)
            and self.attn_sconv.consumes_mailbox(tokens, ctx)
            and self.mlp_sconv.consumes_mailbox(tokens, ctx)
        ):
            attention_out, mlp_out = MailboxTarget(mailbox, 0), MailboxTarget(
                mailbox, 1
            )
            self.attn(
                normalized,
                ctx,
                log_scaling_tau,
                reduce_output=False,
                publish_into=attention_out,
                partial_out=None,
            )
            attention = self.attn_sconv.from_mailbox(attention_out, tokens, ctx)
            mlp_input = state.mlp_input(self.layer_id, attention, self.mlp_norm)
            self.mlp(mlp_input, ctx, reduce=False, fold_into=mlp_out)
            mlp = self.mlp_sconv.from_mailbox(mlp_out, tokens, ctx)
        elif share is not None:
            attention, mlp = self._shared_sublayers(
                state, normalized, ctx, log_scaling_tau, share
            )
        else:
            attention, mlp = self._reduced_sublayers(
                state, normalized, ctx, log_scaling_tau
            )
        if ctx.attn_backend.enable_layerwise_cache_ready:
            mlp = ctx.attn_backend.record_layer_cache_ready(mlp, ctx.forward_mode)
        state.finalize_layer(self.layer_id, attention, mlp)

    def _shared_sublayers(
        self,
        state: AttentionResidualState,
        normalized: tuple[torch.Tensor, ...],
        ctx: ForwardContext,
        log_scaling_tau: torch.Tensor | None,
        share: SequenceShare,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The attention and MoE outputs on this rank's share of the tokens:
        each partial is reduced on the share's rows only, and the output
        convolution computes those rows (other rows are unspecified)."""
        # The partials land in the symmetric buffers the reductions read.
        tokens = normalized[0].shape[0]
        attention = self.attn(
            normalized,
            ctx,
            log_scaling_tau,
            reduce_output=False,
            publish_into=None,
            partial_out=share.attention_partials.comm_buff[:tokens],
        )
        attention = self.attn_sconv(
            share.reduce(attention, share.attention_partials),
            ctx,
            all_reduce_group=None,
            token_range=share.token_range,
        )
        mlp_input = state.mlp_input(self.layer_id, attention, self.mlp_norm)
        mlp = self.mlp(
            mlp_input,
            ctx,
            reduce=False,
            fold_into=share.mlp_partials.comm_buff[:tokens],
        )
        mlp = self.mlp_sconv(
            share.reduce(mlp, share.mlp_partials),
            ctx,
            all_reduce_group=None,
            token_range=share.token_range,
        )
        return attention, mlp

    def _reduced_sublayers(
        self,
        state: AttentionResidualState,
        normalized: tuple[torch.Tensor, ...],
        ctx: ForwardContext,
        log_scaling_tau: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The attention and MoE outputs, each summed over its group by an
        all-reduce (with the output convolution where ``all_reduce_sconv``
        admits it)."""
        attn_group = self.attn.tp_group
        conv_sums_attn = attn_group is not None and self.attn_sconv.sums_partial(
            normalized[0], ctx, attn_group
        )
        attention = self.attn(
            normalized,
            ctx,
            log_scaling_tau,
            reduce_output=not conv_sums_attn,
            publish_into=None,
            partial_out=None,
        )
        attention = self.attn_sconv(
            attention,
            ctx,
            all_reduce_group=attn_group if conv_sums_attn else None,
            token_range=None,
        )
        mlp_input = state.mlp_input(self.layer_id, attention, self.mlp_norm)
        mlp_group = self.mlp.mapping.moe.ep_group
        conv_sums_mlp = self.mlp.all_reduces(
            mlp_input.shape[0]
        ) and self.mlp_sconv.sums_partial(mlp_input, ctx, mlp_group)
        # The convolution folds the MoE's route rows as it loads them.
        mlp = self.mlp_sconv(
            self.mlp(mlp_input, ctx, reduce=not conv_sums_mlp, fold_into=None),
            ctx,
            all_reduce_group=mlp_group if conv_sums_mlp else None,
            token_range=None,
        )
        return attention, mlp


class Inkling2TextModel(nn.Module):
    """The target decoder. Depth history is local to one forward invocation."""

    def __init__(
        self,
        config: Inkling2ModelConfig,
        mapping: Mapping,
        quant_config: Inkling2QuantConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layers_to_capture: list[int] = []
        # The decode chain's mailbox (prepare_communication_runtime); None keeps the
        # sublayers' all-reduces.
        self.decode_mailbox: Mailbox | None = None
        # The prefill's sequence-parallel buffers (prepare_communication_runtime);
        # None keeps the sublayers' all-reduces.
        self.sequence_parallel: Inkling2SequenceParallel | None = None
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.padded_vocab_size,
            embedding_dim=config.hidden_size,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix="model.llm.embed",
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.embed_norm = PrecisionRMSNorm(
            config.hidden_size, config.rms_norm_eps, False, torch.bfloat16, True, False
        )
        self.layers = nn.ModuleList(
            [
                Inkling2DecoderLayer(config, mapping, i, quant_config)
                for i in range(config.num_hidden_layers)
            ]
        )
        # Block-set partials run here, alongside the attention ahead of their
        # first mix.
        self.attnres_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self.attn_res = AttentionResidual(
            config.num_hidden_layers,
            config.hidden_size,
            config.n_layers_per_attn_res_block,
            config.local_layer_ids,
            config.attn_res_normalize_values,
        )
        self.norm = PrecisionRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            config.final_norm_pre_affine,
            torch.bfloat16,
            True,
            False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        input_embeds: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if ctx.query_shard is not None:
            raise NotImplementedError(
                "Inkling2 query context parallelism is not supported"
            )
        hidden = (
            self.embed_norm(self.embed_tokens(input_ids))
            if input_embeds is None
            else input_embeds
        )
        if hidden.shape[0] == 0:
            auxiliary = (
                [hidden.to(torch.bfloat16)] * (len(self.layers_to_capture) + 1)
                if self.layers_to_capture
                else None
            )
            return hidden, auxiliary
        tau = None
        floor, alpha = self.config.log_scaling_n_floor, self.config.log_scaling_alpha
        if floor is not None and positions.is_cuda:
            # One launch, byte for byte the statement below.
            tau = log_scaling_tau(positions, floor, alpha, torch_order=True)
        elif floor is not None:
            ratio = ((positions + 1).float() / floor).clamp_min(1)
            tau = 1.0 + alpha * ratio.log()
        # The aux stream joins back one sublayer later: prefill graphs break
        # at attention, so only decode-family forwards fork it.
        share = None
        if self.sequence_parallel is not None and self.sequence_parallel.serves(
            ctx, hidden.shape[0]
        ):
            share = self.sequence_parallel.share(ctx, hidden.shape[0])
        state = AttentionResidualState(
            self.attn_res,
            hidden,
            None if ctx.forward_mode.is_extend_or_mixed() else self.attnres_stream,
            None if share is None else share.residual,
        )
        auxiliary: list[torch.Tensor] = []
        mailbox = None if ctx.forward_mode.is_extend_or_mixed() else self.decode_mailbox
        for layer_id, layer in enumerate(self.layers):
            if layer_id in self.layers_to_capture:
                auxiliary.append(state.full_partial().to(torch.bfloat16))
                if ctx.target_capture_sink is not None:
                    ctx.target_capture_sink.on_target_capture(
                        len(auxiliary) - 1, auxiliary[-1]
                    )
            layer(state, ctx, tau, mailbox, share)
        hidden = state.compute_output(self.norm)
        if auxiliary:
            auxiliary.append(hidden)
            if ctx.target_capture_sink is not None:
                ctx.target_capture_sink.on_target_capture(len(auxiliary) - 1, hidden)
        return hidden, auxiliary or None


def _decode_mailbox(config: Inkling2ModelConfig, mapping: Mapping) -> Mailbox | None:
    """A mailbox over the attention TP group when it is also the MoE EP group
    and the platform builds one (collective over the group)."""
    if mapping.attn.tp_size == 1 or mapping.attn.tp_group != mapping.moe.ep_group:
        return None
    device = torch.device("cuda", torch.cuda.current_device())
    if not mailbox_supported(device):
        return None
    group = pg_manager.get_device_process_group(mapping.attn.tp_group)
    try:
        return Mailbox(group, width=config.hidden_size, device=device)
    except RuntimeError:
        # No multicast memory over this group.
        return None


class Inkling2ForConditionalGeneration(BaseCausalLM):
    """Text and precomputed-image target on the common runtime path."""

    fall_back_to_pt_during_load = False
    model_cls = Inkling2TextModel
    # MoE slots follow a static expert placement; --enable-eplb is not supported.
    supports_expert_placement = True

    def __init__(
        self,
        config: Inkling2MMConfig,
        mapping: Mapping,
        quant_config: Inkling2QuantConfig | None,
        is_multimodal_active: bool,
        mm_attention_backend: str | None,
    ) -> None:
        if not isinstance(config, Inkling2MMConfig):
            raise TypeError("Inkling2 requires Inkling2MMConfig")
        if not isinstance(quant_config, Inkling2QuantConfig):
            raise ValueError("Inkling2 requires its checkpoint precision metadata")
        if is_multimodal_active and config.vision_config is None:
            raise ValueError("Inkling2 PE vision requires vision_config")
        if is_multimodal_active and mapping.vision.tp_size != 1:
            raise NotImplementedError(
                "Inkling2 vision requires TP1; use --mm-encoder-tp-mode data"
            )
        if not is_multimodal_active and mm_attention_backend is not None:
            raise ValueError(
                "Inkling2 text execution does not select a vision attention backend"
            )
        if mapping.pp_size != 1 or mapping.attn.has_dp or mapping.moe.has_dp:
            raise NotImplementedError(
                "Inkling2 pipeline and data parallelism are not supported"
            )
        if mapping.attn.qcp_size != 1:
            raise NotImplementedError(
                "Inkling2 query context parallelism is not supported"
            )
        if global_server_args_dict["enable_eplb"]:
            raise NotImplementedError(
                "Inkling2 supports static expert placements only, not --enable-eplb"
            )
        super().__init__(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            prefix="",
            encoder_only=False,
        )
        self.text_config = config.text_config
        self.is_multimodal_active = is_multimodal_active
        self.visual = (
            Inkling2VisionEncoder(
                config.vision_config, self.text_config.hidden_size, mm_attention_backend
            )
            if is_multimodal_active
            else None
        )
        self.vision_embedder = (
            VisionEmbedder(encoder_mapping=mapping.vision)
            if is_multimodal_active
            else None
        )
        if self.visual is not None:
            for name, parameter in self.visual.named_parameters():
                self.quant_config.validate_tensor(f"model.visual.{name}", parameter)

    @classmethod
    def get_model_config_for_expert_location(
        cls, config: Inkling2MMConfig
    ) -> ModelConfigForExpertLocation:
        return ModelConfigForExpertLocation(
            num_layers=config.text_config.num_hidden_layers,
            num_logical_experts=config.text_config.n_routed_experts,
            num_groups=None,
        )

    def resolve_model(
        self,
        config: Inkling2MMConfig,
        mapping: Mapping,
        quant_config: Inkling2QuantConfig,
        prefix: str,
    ) -> Inkling2TextModel:
        return Inkling2TextModel(config.text_config, mapping, quant_config)

    def prepare_communication_runtime(self, max_num_tokens: int) -> bool:
        """Build the decode chain's mailbox and the prefill's sequence-parallel
        buffers (eager, before capture)."""
        if self.model.decode_mailbox is None:
            self.model.decode_mailbox = _decode_mailbox(self.model.config, self.mapping)
        if (
            self.model.decode_mailbox is not None
            and self.model.sequence_parallel is None
        ):
            # The same group and platform as the mailbox.
            self.model.sequence_parallel = Inkling2SequenceParallel(
                pg_manager.get_device_process_group(self.mapping.attn.tp_group),
                max_tokens=max_num_tokens,
                hidden=self.model.config.hidden_size,
                conv_window=self.model.config.sconv_kernel_size,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
        return self.model.decode_mailbox is not None

    def post_load_weights(self) -> None:
        """Derive the fused-kernel weights from the loaded queries, norms and
        convolution taps."""
        self.model.attn_res.derive_score_weights()
        for layer in self.model.layers:
            layer.derive_input_norm_gains()
            layer.attn.derive_projected_taps()

    def resolve_lm_head(
        self,
        config: Inkling2MMConfig,
        quant_config: Inkling2QuantConfig,
        prefix: str,
    ) -> ParallelLMHead:
        text = config.text_config
        head = ParallelLMHead(
            num_embeddings=text.padded_vocab_size,
            embedding_dim=text.hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix="model.llm.unembed",
            tp_rank=self.mapping.lm_head.tp_rank,
            tp_size=self.mapping.lm_head.tp_size,
            tp_group=self.mapping.lm_head.tp_group,
        )
        head.quant_method = Inkling2LMHeadMethod()
        return head

    def resolve_logits_processor(
        self, config: Inkling2MMConfig
    ) -> Inkling2LogitsProcessor:
        return Inkling2LogitsProcessor(config.text_config, self.mapping)

    def get_decoder_input_embedding(self) -> DecoderInputEmbedding:
        """Keep learned embedding normalization outside the supplied branch."""
        return DecoderInputEmbedding(
            lookup=self.model.embed_tokens, transform=self.model.embed_norm
        )

    def prepare_model_kwargs(
        self, ctx: ForwardContext, input_ids: torch.Tensor, kwargs: dict
    ) -> dict:
        unsupported = kwargs.keys() - {"input_embeds", "multimodal_context"}
        if unsupported:
            raise TypeError(
                f"Unsupported Inkling2 forward arguments: {sorted(unsupported)}"
            )
        input_embeds = kwargs.get("input_embeds")
        multimodal_context = kwargs.get("multimodal_context")
        if multimodal_context is not None and multimodal_context.has_inputs():
            if input_embeds is not None:
                raise ValueError(
                    "Inkling2 image inputs cannot be combined with input_embeds"
                )
            input_embeds = self.multimodal_input_embeds(
                input_ids, ctx, multimodal_context
            )
        return {"input_embeds": input_embeds}

    def _image_grid_counts(self, item: MultimodalDataItem) -> list[int]:
        """Validate the item's modality and vision grid; return each image's token count."""
        if item.modality != Modality.IMAGE:
            raise ValueError("Inkling2 supports only image multimodal inputs")
        grid = item.model_specific_data.get("vision_grid_thw")
        if (
            grid is None
            or grid.device.type != "cpu"
            or grid.dtype not in (torch.int32, torch.int64)
            or grid.ndim != 2
            or grid.shape[1] != 3
            or grid.shape[0] == 0
        ):
            raise ValueError(
                "vision_grid_thw must be a nonempty CPU integer [images, 3] tensor"
            )
        rows = grid.tolist()
        if any(t != 1 or h <= 0 or w <= 0 for t, h, w in rows):
            raise ValueError(
                "Inkling2 image grids require temporal size 1 and positive spatial dimensions"
            )
        return [h * w for _, h, w in rows]

    def _validate_image_item(self, item: MultimodalDataItem) -> None:
        counts = self._image_grid_counts(item)
        if item.offsets is None or len(item.offsets) != len(counts):
            raise ValueError("Image offsets must contain one range per vision grid")
        previous_end = -1
        for (start, end), count in zip(item.offsets, counts, strict=True):
            if (
                type(start) is not int
                or type(end) is not int
                or start <= previous_end
                or end - start + 1 != count
            ):
                raise ValueError(
                    "Image offsets must be ordered, nonoverlapping ranges matching "
                    "grid token counts"
                )
            previous_end = end
        if item.encoded is not None:
            if item.encoded.dtype != torch.bfloat16 or item.encoded.shape != (
                sum(counts),
                self.text_config.hidden_size,
            ):
                raise ValueError(
                    "Encoded image rows must be BF16 and match grid token count and text width"
                )

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        """Encode native patches in item/grid order with one output row per patch."""
        vision = self.config.vision_config
        side = vision.native_patch_size * vision.adapter_merge_size
        for item in items:
            # EPD encode calls this directly; shared-memory patches materialize just before it.
            count = sum(self._image_grid_counts(item))
            if item.feature is None:
                raise ValueError("Image encoder requires materialized patches")
            if item.feature.dtype != torch.bfloat16 or item.feature.shape != (
                count,
                vision.temporal_patch_size,
                side,
                side,
                3,
            ):
                raise ValueError(
                    "Image patches must be BF16 [N, temporal, side, side, 3] matching "
                    "vision_grid_thw"
                )
        grid = torch.cat(
            [item.model_specific_data["vision_grid_thw"] for item in items]
        )
        patches = torch.cat([item.feature for item in items]).to(
            device=self.visual.proj_out.weight.device
        )
        return self.visual(patches, grid)

    def get_multimodal_encoder_specs(self) -> dict[Modality, EncoderSpec]:
        if self.visual is None:
            return {}
        return {
            Modality.IMAGE: EncoderSpec(
                fn=self.get_image_feature,
                deepstack=False,
                make_warmup_items=self.visual.make_warmup_items,
            )
        }

    @torch.no_grad()
    def multimodal_input_embeds(
        self,
        input_ids: torch.Tensor,
        ctx: ForwardContext,
        multimodal_context: MultimodalForwardContext | None,
    ) -> torch.Tensor | None:
        """Normalize text lookup, then scatter image rows through the shared embedder."""
        if multimodal_context is None or not multimodal_context.has_extend_inputs():
            return None
        if ctx.forward_mode.is_decode_or_idle():
            return None
        if self.vision_embedder is None:
            raise ValueError("Inkling2 image inputs require active vision execution")
        for mm_input in multimodal_context.mm_inputs:
            if mm_input is not None:
                ranges = []
                for item in mm_input.mm_items:
                    self._validate_image_item(item)
                    ranges.extend(item.offsets)
                ordered = sorted(ranges)
                if any(
                    right[0] <= left[1] for left, right in zip(ordered, ordered[1:])
                ):
                    raise ValueError(
                        "Image offset ranges must not overlap across items"
                    )
        input_embeds, _ = self.vision_embedder.apply(
            input_ids=input_ids,
            text_embedding=self.get_decoder_input_embedding(),
            ctx=multimodal_context,
            encoders=self.get_multimodal_encoder_specs(),
            multimodal_model=self,
        )
        return input_embeds

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        """Capture trained raw layer entries followed by the final-norm output.

        Args:
            layer_ids: Exact checkpoint taps, including the final-layer sentinel.
        """
        trained = Inkling2MTPConfig.from_target(self.config).target_layer_ids
        if layer_ids != trained:
            raise ValueError(
                "Native draft capture must match checkpoint target_layer_ids"
            )
        self.model.layers_to_capture = list(layer_ids[:-1])
        self.capture_aux_hidden_states = True

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        raise NotImplementedError(
            "Inkling2 requires a separate DSpark draft implementation"
        )

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None) -> None:
        raise NotImplementedError("Inkling2 draft feature capture is not supported")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load exact checkpoint dtypes and all target tensors without conversion.

        Args:
            weights: The checkpoint name/tensor stream. Inactive vision and
                separate MTP tensors are excluded from target loading.

        Returns:
            The runtime parameter names that received checkpoint data.
        """
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        packed_shards: dict[str, set[int]] = {}
        projections = (
            "wq_du",
            "wk_dv",
            "wv_dv",
            "wr_du",
            "gate_attn",
            "attn_sink_proj",
        )
        for checkpoint_name, weight in weights:
            is_vision = checkpoint_name.startswith("model.visual.")
            if checkpoint_name.startswith("model.mtp.") or (
                self.visual is None
                and checkpoint_name.startswith(("model.visual.", "model.vision."))
            ):
                continue
            self.quant_config.validate_tensor(checkpoint_name, weight)
            name = (
                checkpoint_name.removeprefix("model.")
                if is_vision
                else checkpoint_name.replace("model.llm.", "model.", 1)
            )
            name = name.replace("model.embed.weight", "model.embed_tokens.weight")
            name = name.replace("model.unembed.weight", "lm_head.weight")
            shard = None
            for index, projection in enumerate(projections):
                if f".attn.{projection}." in name:
                    name = name.replace(f".{projection}.", ".qkvr.")
                    shard = index
                    break
            for suffix in ("scale", "amax", "outer_scale_denominator"):
                name = name.replace(f"_weight_{suffix}", f"_{suffix}")
            if name not in params:
                raise ValueError(f"Unrecognized checkpoint tensor: {checkpoint_name}")
            param = params[name]
            if is_vision and param.shape != weight.shape:
                raise ValueError(
                    f"Vision checkpoint tensor shape mismatch: {checkpoint_name}"
                )
            if param.dtype != weight.dtype:
                raise ValueError(
                    f"{checkpoint_name}: parameter dtype {param.dtype} "
                    f"would change checkpoint precision {weight.dtype}"
                )
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard is None:
                if name in loaded:
                    raise ValueError(
                        f"Repeated or mixed packed checkpoint tensor: {checkpoint_name}"
                    )
                loader(param, weight)
            else:
                if name in loaded and name not in packed_shards:
                    raise ValueError(
                        f"Mixed packed and unpacked checkpoint tensors: {checkpoint_name}"
                    )
                if shard in packed_shards.get(name, set()):
                    raise ValueError(f"Repeated checkpoint tensor: {checkpoint_name}")
                loader(param, weight, shard)
                packed_shards.setdefault(name, set()).add(shard)
            loaded.add(name)
        missing = set(params) - loaded
        for name, shards in packed_shards.items():
            expected = set(range(len(projections))) if name.endswith(".weight") else {0}
            if not expected.issubset(shards):
                missing.add(f"{name} shards {sorted(expected - shards)}")
        if missing:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        self.post_load_weights()
        return loaded


EntryClass = Inkling2ForConditionalGeneration
