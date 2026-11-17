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

"""DFlash2 draft monokernel seam (Muse Spark's draft on TP8).

With ``--decode-monokernel``, the draft model builds its layers from
``DFlash2MonokernelDecoderLayer`` (:class:`DFlash2MonokernelLayer` before
the regular layer class), and a draft block forward of one or two requests
runs each of the draft's layers as one launch of
``tokenspeed_kernel.ops.model.muse_spark.monokernel.muse_spark_draft_layer_decode``
on every rank of the TP8 group (``docs/design/monokernel.md``):

* after loading, :func:`enable_draft_monokernels` packs every layer and the
  context write and gives the layers and the model one
  :class:`DFlash2MonokernelState`;
* ``prepare_communication_runtime`` calls
  :meth:`DFlash2MonokernelState.prepare`, which allocates the mailbox
  scratch on the Iris symmetric heap;
* the model's ``run_decoder`` calls ``start_step()`` before the layers and
  takes the final norm's output, which the last launch writes, from
  ``finish_step()``;
* after a verify of one or two requests, the drafter runs
  :meth:`DFlash2MonokernelState.write_context` in place of its context write
  (``fc`` and ``hidden_norm`` of the target's taps, then every layer's K and
  V of the verify rows into the draft cache);
* the draft's ``configure_target`` calls
  :meth:`DFlash2MonokernelState.pack_head`, which packs the draft head (the
  vocabulary-parallel top 16 of each draft step and the candidate selector's
  scores) from the target's LM head. It keeps a tiled copy of the rank's
  LM-head shard (414 MB a GPU): the tiles stream faster than the row-major
  weight the target's verify reads. The copy is made before the KV-cache
  budget is profiled, so the budget counts it;
* once the drafter is wired to the target,
  :meth:`DFlash2MonokernelState.enable_head` makes the packed head one more
  launch, :class:`DraftHeadMonokernel`, which the drafter runs in place of
  its top-k and selector for one or two requests.

The draft follows the target's rule (``docs/design/monokernel.md``): the
switch never serves a piece without its kernel. When the layers, the context
write or the head cannot be packed, startup fails with
:func:`unserved_draft_error`, which names each piece and why.

The launches hand each other this rank's partial of the layer output and its
owned residual columns (``1024 rank ..``), so the layers are served all
together or not at all. The last launch also runs the draft's final norm and
stores its output, the whole rows on every rank.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.model.muse_spark.monokernel import (
    FusedDecodeLayer,
    muse_spark_draft_context_decode,
    muse_spark_draft_head_decode,
    muse_spark_draft_head_scratch_pairs,
    muse_spark_draft_layer_decode,
    prepare_muse_spark_draft_context,
    prepare_muse_spark_draft_head,
    prepare_muse_spark_draft_layer,
)

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.configs.mha import mha_kv_cache_dtype
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.logits_processor import (
    should_apply_lm_head_quant_method,
)
from tokenspeed.runtime.models.monokernel import MonokernelLayer, MonokernelState

# The draft serves exactly the target's steps, so it takes the target's
# request count.
from tokenspeed.runtime.models.muse_spark_monokernel import MAX_REQUESTS
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)

WORLD = 8
HIDDEN = 8192
OWNED = HIDDEN // WORLD
# One request's draft block.
BLOCK_ROWS = 8
PAGE_SIZE = 64
# The layers' window_left: every block row attends the request's last 2048
# keys (the block's 8 among them).
WINDOW = 2047

# The draft architecture the kernel bakes in; another one keeps the regular
# path.
KERNEL_SHAPE = {
    "hidden_size": HIDDEN,
    "intermediate_size": 28672,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_hidden_layers": 5,
}
KERNEL_DFLASH = {
    "block_size": BLOCK_ROWS,
    "conv_kernel_size": 2,
    "conv_group_size": 16,
}
RMS_NORM_EPS = 1e-5

NOT_TP8 = "attention and the dense layers are not one TP8 group"
OTHER_SHAPES = "draft dimensions other than the kernel's"
OTHER_ATTENTION = (
    "an attention other than the kernel's (non-causal GQA, NeoX RoPE, "
    f"window_left {WINDOW}, RMS eps {RMS_NORM_EPS})"
)
NOT_BF16 = "a projection is quantized or a weight is not BF16"
# The draft head the kernel bakes in.
VOCAB = 202048
SHARD = VOCAB // WORLD
SELECTOR_RANK = 256
TOPK = 16
STEPS = BLOCK_ROWS - 1  # the block's rows 1 .. 7 draft a token each
CONTEXT_ROWS = 8  # one request's verify rows
# MAX_REQUESTS (imported from the target): the requests a launch serves,
# their rows one request after another.
# The rows each launch is packed for, which also size its Iris scratch (and
# so the draft's heap reservation): up to MAX_REQUESTS requests' blocks,
# draft steps and verify rows.
LAYER_ROWS = MAX_REQUESTS * BLOCK_ROWS
HEAD_ROWS = MAX_REQUESTS * STEPS
CONTEXT_WRITE_ROWS = MAX_REQUESTS * CONTEXT_ROWS
NO_KERNEL = "no draft monokernel for this platform"
MLA_DRAFT = "an MLA draft: the kernels serve GQA attention"
NOT_BF16_KV = "the draft's KV cache is not BF16: the kernels read BF16 keys"


def _config_value(config, key: str):
    nested = getattr(config, "dflash_config", None) or {}
    return nested.get(key, getattr(config, key, None))


def config_reasons(config) -> list[str]:
    """Why a draft of ``config`` is not the one the kernels bake in (Muse
    Spark's DFlash2 draft: its dimensions and GQA attention); empty when it
    is (part of :func:`model_reasons`)."""
    reasons = []
    if any(getattr(config, k, None) != v for k, v in KERNEL_SHAPE.items()) or any(
        _config_value(config, k) != v for k, v in KERNEL_DFLASH.items()
    ):
        reasons.append(OTHER_SHAPES)
    if str(_config_value(config, "attention_mode") or "gqa").lower() != "gqa":
        reasons.append(MLA_DRAFT)
    return reasons


def model_reasons(model: torch.nn.Module) -> list[str]:
    """Why the layers of ``model`` (a ``DFlash2DraftModel``) cannot have
    draft monokernels; empty when they can."""
    if model._uses_mla:
        return [MLA_DRAFT]
    config, mapping = model.config, model.mapping
    reasons = []
    if not (
        mapping.attn.tp_size == WORLD
        and mapping.dense.tp_size == WORLD
        and mapping.attn.tp_group == mapping.dense.tp_group
        and mapping.pp_size == 1
    ):
        reasons.append(NOT_TP8)
    # An MLA draft returned above, so only the dimensions can differ here.
    reasons += config_reasons(config)
    if (
        any(not _attention_matches(layer) for layer in model.layers)
        # The last launch runs the final norm.
        or abs(model.norm.variance_epsilon - RMS_NORM_EPS) >= 1e-12
    ):
        reasons.append(OTHER_ATTENTION)
    if any(not _bf16(layer) for layer in model.layers):
        reasons.append(NOT_BF16)
    # The draft's own cache ("auto" keeps the model's BF16): a block drafter
    # keeps BF16 under an FP8 --kv-cache-dtype unless it attends with trtllm.
    draft_kv = mha_kv_cache_dtype(
        global_server_args_dict["kv_cache_dtype"],
        global_server_args_dict["speculative_algorithm"],
        global_server_args_dict["drafter_attention_backend"],
        is_draft=True,
    )
    if draft_kv not in ("auto", "bfloat16", "bf16"):
        reasons.append(NOT_BF16_KV)
    return reasons


def _attention_matches(layer: torch.nn.Module) -> bool:
    attn = layer.self_attn
    norms = (
        layer.input_layernorm,
        layer.post_attention_layernorm,
        attn.q_norm,
        attn.k_norm,
    )
    return (
        attn.attn.sliding_window_size == WINDOW
        and attn.rotary_emb.is_neox_style
        and attn.rotary_emb.cos_sin_cache.dtype == torch.float32
        and all(abs(n.variance_epsilon - RMS_NORM_EPS) < 1e-12 for n in norms)
    )


def _bf16(layer: torch.nn.Module) -> bool:
    linears = (
        layer.self_attn.qkv_proj,
        layer.self_attn.o_proj,
        layer.mlp.gate_up_proj,
        layer.mlp.down_proj,
    )
    return all(
        isinstance(linear.quant_method, UnquantizedLinearMethod)
        and linear.weight.dtype == torch.bfloat16
        for linear in linears
    )


def layer_tensors(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    """A ``DFlash2DecoderLayer``'s weights by the roles of
    :func:`prepare_muse_spark_draft_layer` (this rank's shards of the
    sharded projections, as loaded)."""
    attn, mlp = layer.self_attn, layer.mlp
    return {
        "input_norm": layer.input_layernorm.weight.data,
        "post_norm": layer.post_attention_layernorm.weight.data,
        "attn_coef": layer.attention_conv.kernel_projection.weight.data,
        "mlp_coef": layer.mlp_conv.kernel_projection.weight.data,
        "attn_base": layer.attention_conv.base_kernel.data,
        "mlp_base": layer.mlp_conv.base_kernel.data,
        "qkv": attn.qkv_proj.weight.data,
        "q_norm": attn.q_norm.weight.data,
        "k_norm": attn.k_norm.weight.data,
        "o_proj": attn.o_proj.weight.data,
        "gate_up": mlp.gate_up_proj.weight.data,
        "down": mlp.down_proj.weight.data,
    }


def takes_monokernels(rows: int, ctx: ForwardContext) -> bool:
    """Whether a draft forward of ``rows`` token rows takes the monokernels:
    the blocks of up to ``MAX_REQUESTS`` requests. More requests keep the
    regular path."""
    return (
        ctx.forward_mode.is_decode()
        and 1 <= ctx.bs <= MAX_REQUESTS
        and rows == ctx.bs * BLOCK_ROWS
    )


def request_seq_lens(leaf, bs: int) -> torch.Tensor:
    """Each request's length after its block, int32 ``[bs]``, from a block
    decode leaf's cache-seqlens buffer. Under block decode the buffer holds
    ``block_decode_expansion`` entries a request (every block position, all
    the block-end length; ``fill_block_decode_seq_lens``), so request ``q``'s
    length is entry ``q * expansion``, not entry ``q``. One request's is a
    one-element view (no copy); more requests' are gathered into a
    contiguous tensor, as the launch reads ``seq_len + q``."""
    expansion = leaf.block_decode_expansion
    lengths = leaf.decode_seq_lens_buffer[: bs * expansion].view(bs, expansion)
    return lengths[:, 0].contiguous()


def mailbox_scratches(
    group: dist.ProcessGroup, rank: int, scratch_pairs: tuple[int, ...]
) -> tuple[list[torch.Tensor | None], torch.Tensor]:
    """A mailbox scratch of each of ``scratch_pairs`` pairs (None for 0) on
    the Iris symmetric heap of ``group`` (``rank`` this process's rank in
    it), and the group's heap bases: one group preparation for all of them.
    Nothing clears the scratch: the tags make old pairs stale."""
    from tokenspeed_kernel.ops.communication.iris import iris_symmetric_heap

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    context, iris_group = iris_symmetric_heap(group, rank, device)
    scratches = [
        context.zeros((pairs,), dtype=torch.int64) if pairs else None
        for pairs in scratch_pairs
    ]
    bases = iris_group.bases.to(device)
    iris_group.finish_prepare()
    return scratches, bases


class DFlash2MonokernelState(MonokernelState):
    """The state that a DFlash2 draft's launches share: the layers' mailbox
    scratch on the Iris symmetric heap and its heap bases, the context
    write and the draft head with their own scratch, and the ordering marks
    between them. Made when the layers are packed after loading;
    :meth:`prepare` allocates the scratch.

    The draft keeps a scratch of its own: its tags (``step * 64 + layer +
    1``) would repeat the target's, whose step counter moves at a rate of
    its own.

    Attributes:
        layers: The draft's decoder layers.
        context: The packed context write.
        context_scratch: Its mailbox scratch, from :meth:`prepare`.
        context_pending: Whether a context write ran whose block forward
            has not run yet (the write reads the layers' step counter, which
            the block forward advances).
        context_compiled: The context write's row counts every rank
            compiled.
        final_gain: FP32 ``[1024]``: the final norm's gain of the rank's
            owned columns (the last launch runs the final norm).
        bases: The TP8 group's heap bases, from :meth:`prepare` (None
            before it; a launch then raises: a startup ordering error).
        cpu_group: The TP8 group's CPU (gloo) process group, from
            :meth:`prepare` (None before it).
        device_group: The TP8 device group, from :meth:`prepare` (None
            before it).
        rank: This process's rank in the TP8 group, which every piece is
            packed for.
        compiled: Which binaries every rank compiled, by ``(last, rows)``:
            the last layer's (True) and the others' (False), at each row
            count.
        block_pending: Whether the layers' launches ran a block whose head
            has not run yet (the head reads their step counter).
        packed_head: The packed draft head, from :meth:`pack_head` (None
            before it).
        head_scale, head_softcap: The logits' multiplier and final softcap
            that :meth:`pack_head` packed the head with (None before it).
        head: The draft head's launch, from :meth:`enable_head` (None
            before it).
        head_pairs: The mailbox pairs reserved on the Iris heap for the
            head's scratch (0 when no head monokernel serves here);
            :meth:`pack_head` refuses a head that needs more.
        head_scratch: The head's scratch, from :meth:`prepare` (None before
            it, or when no head monokernel serves here).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scratch_pairs: int,
        rank: int,
        context: FusedDecodeLayer,
    ) -> None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        # prepare allocates the scratch on the Iris heap, which does not exist
        # yet; until then it is empty.
        super().__init__(torch.empty(0, dtype=torch.int64, device=device))
        self.layers = model.layers
        self.scratch_pairs = scratch_pairs
        self.context = context
        self.context_scratch: torch.Tensor | None = None
        self.context_pending = False
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        self.final_gain = model.norm.weight.data[own].float().contiguous()
        self.bases: torch.Tensor | None = None
        self.device_group: dist.ProcessGroup | None = None
        self.cpu_group: dist.ProcessGroup | None = None
        self.rank = rank
        self.compiled: set[tuple[bool, int]] = set()
        self.block_pending = False
        self.context_compiled: set[int] = set()
        # The current block's positions as int32 (start_step resets it).
        self.positions: torch.Tensor | None = None
        self.packed_head: FusedDecodeLayer | None = None
        self.head_scale: float | None = None
        self.head_softcap: float | None = None
        self.head: DraftHeadMonokernel | None = None
        self.head_scratch: torch.Tensor | None = None
        # What prepare() allocates on the Iris heap (the head's scratch too),
        # reserved before the heap exists.
        from tokenspeed_kernel.ops.communication.iris import reserve_iris_heap

        self.head_pairs = muse_spark_draft_head_scratch_pairs(max_rows=HEAD_ROWS) or 0
        reserve_iris_heap("muse_spark_dflash2_monokernel", self.heap_bytes())

    def heap_bytes(self) -> int:
        """The Iris heap bytes the draft's monokernels allocate, at the heap
        allocator's alignment: the int64 mailbox scratch of the layers
        (:meth:`prepare`), the context write (:meth:`prepare`) and the head
        (:meth:`prepare`; none when no head monokernel serves here)."""
        from tokenspeed_kernel.ops.communication.iris import iris_heap_bytes

        pairs = (self.scratch_pairs, self.context.scratch_pairs, self.head_pairs)
        return sum(iris_heap_bytes((n,), torch.int64) for n in pairs if n)

    def prepare(
        self, group: dist.ProcessGroup, cpu_group: dist.ProcessGroup, rank: int
    ) -> None:
        """Allocate the layers', the context write's and the head's scratch on
        the Iris heap of ``group`` (the TP8 device
        process group; ``cpu_group`` its CPU twin, ``rank`` this process's
        rank in it, the rank every piece was packed for). Call once, before
        graph capture."""
        if rank != self.rank:
            raise RuntimeError(
                f"the draft monokernels were packed for TP8 rank {self.rank}, "
                f"but prepared on rank {rank}"
            )
        if self.bases is None:
            (self.scratch, self.context_scratch, self.head_scratch), self.bases = (
                mailbox_scratches(
                    group,
                    rank,
                    (self.scratch_pairs, self.context.scratch_pairs, self.head_pairs),
                )
            )
            self.device_group = group
            self.cpu_group = cpu_group

    def start_step(self) -> None:
        super().start_step()
        self.positions = None

    def finish_step(self) -> torch.Tensor | None:
        """Give the next block's launches a new tag after a block that
        launched, and return the final norm's output that the last launch
        wrote (``lm_head_input``). The draft launches never move the step
        counter themselves, so this moves it after every block that
        launched (graph safe)."""
        if self.launched:
            self.step[:1].add_(1)
            self.block_pending = True
            self.context_pending = False
        lm_head_input, self.lm_head_input = self.lm_head_input, None
        return lm_head_input

    def serves_context(self, rows: int) -> bool:
        """Whether a context write of ``rows`` verify rows (decode rows only)
        takes :meth:`write_context`: those of up to ``MAX_REQUESTS``
        requests."""
        return (
            self.context_scratch is not None
            and rows % CONTEXT_ROWS == 0
            and 1 <= rows // CONTEXT_ROWS <= MAX_REQUESTS
        )

    def write_context(
        self,
        taps: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        token_to_kv_pool,
    ) -> None:
        """The context write of the verify rows of up to ``MAX_REQUESTS``
        requests (8 each) as one launch.

        Args:
            taps: BF16 ``[rows, 40960]``: the target's tapped hidden states.
            positions: ``[rows]``: the rows' positions.
            slots: ``[rows]``: the rows' draft cache slots.
            token_to_kv_pool: The draft's KV pool.
        """
        if self.context_pending:
            raise RuntimeError(
                "a second draft context write before the block forward would "
                "repeat its launch's tag"
            )
        caches = [
            token_to_kv_pool.get_kv_buffer(layer.self_attn.attn.layer_id)
            for layer in self.layers
        ]
        operands = dict(
            taps=taps.contiguous(),
            positions=positions,
            slots=slots,
            k_caches=tuple(k.view(-1, 1, k.shape[-1]) for k, _ in caches),
            v_caches=tuple(v.view(-1, 1, v.shape[-1]) for _, v in caches),
            cos_sin=self.layers[0].self_attn.rotary_emb.cos_sin_cache,
            scratch=self.context_scratch,
            step=self.step[:1],
            peer_bases=self.bases,
        )
        rows = taps.shape[0]
        if rows not in self.context_compiled:
            muse_spark_draft_context_decode(self.context, compile_only=True, **operands)
            dist.barrier(group=self.cpu_group)
            self.context_compiled.add(rows)
        muse_spark_draft_context_decode(self.context, compile_only=False, **operands)
        self.context_pending = True

    def run_head(
        self, hidden_states: torch.Tensor, first_token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The draft head's launch (:class:`DraftHeadMonokernel`) after the
        layers' launches of the same forward."""
        if not self.block_pending:
            raise RuntimeError(
                "the draft head monokernel runs after the draft layers' launches"
            )
        self.block_pending = False
        return self.head(hidden_states, first_token)

    def pack_head(
        self,
        lm_head: torch.nn.Module,
        selector: torch.nn.Module,
        *,
        scale: float,
        softcap: float,
    ) -> None:
        """Pack the draft head from the target's LM head (``lm_head``,
        vocabulary-parallel) and the draft's candidate ``selector``, with the
        logits' multiplier ``scale`` and final ``softcap``. Called by the
        draft's ``configure_target``, before the KV-cache budget is profiled,
        so the budget counts the head's tiled LM-head copy.

        Raises:
            ValueError: The head cannot be packed (:func:`unserved_draft_error`).
        """
        reasons = head_reasons(lm_head, selector, softcap)
        if not reasons:
            fused = prepare_muse_spark_draft_head(
                tensors={
                    "lm_head": lm_head.weight.data[:SHARD],
                    "hidden_proj": selector.hidden_projection.weight.data,
                    "pred_codebook": selector.predecessor_codebook.data,
                    "succ_codebook": selector.successor_codebook.data,
                },
                rank=self.rank,
                max_rows=HEAD_ROWS,
                scale=scale,
                softcap=softcap,
            )
            if fused is None:
                reasons.append(NO_KERNEL)
        if reasons:
            raise ValueError(unserved_draft_error("DFlash2", {"head": reasons}))
        if fused.scratch_pairs > self.head_pairs:
            raise RuntimeError(
                f"the draft head needs {fused.scratch_pairs} mailbox pairs but "
                f"{self.head_pairs} were reserved on the Iris heap"
            )
        self.packed_head = fused
        self.head_scale = scale
        self.head_softcap = softcap

    def enable_head(self, *, top_k_exact: bool, scale: float, softcap: float) -> None:
        """Make the packed head (:meth:`pack_head`) a launch once the drafter
        is wired to the target, after :meth:`prepare`. ``top_k_exact``:
        whether the shard-local top-k is exact for the head's shard geometry
        (``VocabParallelTopK.enabled``); ``scale`` and ``softcap``: the
        drafter's, which must be the ones the head was packed with.

        Raises:
            RuntimeError: The head was not packed or its scratch not
                allocated before the drafter was wired, or the drafter's
                scale or softcap differ from the packed head's (internal
                ordering or wiring errors).
            ValueError: The head's top-k is not exact for its shard geometry
                (:func:`unserved_draft_error`).
        """
        if self.packed_head is None or self.bases is None:
            raise RuntimeError(
                "the DFlash2 drafter was wired before the draft's "
                "configure_target packed its head monokernel and "
                "prepare_communication_runtime allocated its scratch"
            )
        if (scale, softcap) != (self.head_scale, self.head_softcap):
            raise RuntimeError(
                f"the DFlash2 drafter's logit scale {scale} and softcap {softcap} "
                f"differ from the head monokernel's ({self.head_scale}, "
                f"{self.head_softcap})"
            )
        if not top_k_exact:
            raise ValueError(
                unserved_draft_error("DFlash2", {"head": [HEAD_NOT_SHARDED]})
            )
        self.head = DraftHeadMonokernel(
            self.packed_head,
            self.head_scratch,
            self.bases,
            self.cpu_group,
            self.step[:1],
        )
        logger.info("Decode monokernel: the DFlash2 draft head (top 16 and selector)")


class DFlash2MonokernelLayer(MonokernelLayer):
    """The monokernel half of a DFlash2 draft layer whose block forwards run
    as one launch per TP8 rank (``--decode-monokernel``). The draft model's
    ``DFlash2MonokernelDecoderLayer`` puts it before the regular layer
    class. Every layer of a packed draft launches, and the last launch also
    runs the final norm (``lm_head_input``)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # This layer's packed launch, set after loading by
        # enable_draft_monokernels. None keeps the regular code (only a draft
        # built without packing: --decode-monokernel packs every layer or
        # startup fails).
        self.fused: FusedDecodeLayer | None = None
        # Whether the layer is the draft's last (its launch runs the final
        # norm).
        self.last = False

    def launch_chain(self, rows: int, ctx: ForwardContext) -> int | None:
        if self.fused is None or not takes_monokernels(rows, ctx):
            return None
        if self.monokernel_state.bases is None:
            # Packed, but the draft's collectives never prepared the scratch:
            # a wiring error, not a reason to serve the regular code.
            raise RuntimeError(
                f"DFlash2 draft layer {self.layer_id}: a block forward before "
                "prepare_communication_runtime allocated the draft monokernels' "
                "scratch"
            )
        return BLOCK_ROWS

    def _decode(
        self,
        chain: int,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the layer's launch on the block. Takes this rank's partial of
        the layer's input and the residual entering it, and returns the next
        layer's and the residual. The last launch writes the final norm's
        output to ``lm_head_input`` instead."""
        state = self.monokernel_state
        backend = ctx.attn_backend
        attn = self.self_attn.attn
        view = backend.group_view(attn.group_id, ctx.bs)
        if view.kernel_page_size != PAGE_SIZE:
            raise RuntimeError(
                f"DFlash2 draft monokernel: KV page size "
                f"{view.kernel_page_size}, the kernel reads {PAGE_SIZE}"
            )
        k_cache, v_cache = ctx.token_to_kv_pool.get_kv_buffer(attn.layer_id)
        leaf = backend.leaves[attn.group_id]
        rows = hidden_states.shape[0]
        if state.positions is None:
            # One cast a block forward, not one a layer launch.
            state.positions = positions.to(torch.int32)
        out = torch.empty_like(hidden_states)
        res_out = torch.empty_like(residual)
        last = self.last
        operands = dict(
            x=hidden_states,
            residual=residual,
            residual_out=res_out,
            out=None if last else out,
            final_gain=state.final_gain if last else None,
            normed=out if last else None,
            seq_len=request_seq_lens(leaf, ctx.bs),
            positions=state.positions,
            slots=backend.write_locations(attn, ctx.forward_mode)[:rows],
            page_table=view.page_table[: ctx.bs],
            k_cache=k_cache.view(-1, 1, k_cache.shape[-1]),
            v_cache=v_cache.view(-1, 1, v_cache.shape[-1]),
            cos_sin=self.self_attn.rotary_emb.cos_sin_cache,
            scratch=state.scratch,
            step=state.step[:1],
            peer_bases=state.bases,
            layer_index=self.layer_id,
        )
        if (last, rows) not in state.compiled:
            # A launch traps when its peers' values do not come within about
            # a minute, and a first launch compiles for longer on some ranks
            # than on others: every rank compiles, then the ranks meet. The
            # layers before the last share one binary.
            muse_spark_draft_layer_decode(self.fused, compile_only=True, **operands)
            dist.barrier(group=state.cpu_group)
            state.compiled.add((last, rows))
        muse_spark_draft_layer_decode(self.fused, compile_only=False, **operands)
        if last:
            state.lm_head_input = out
        return out, res_out


HEAD_NOT_SHARDED = "the LM head is quantized or not an exact 8-way vocabulary shard"
HEAD_OTHER_SHAPES = "an LM head or selector of other dimensions than the kernel's"
HEAD_NO_SOFTCAP = "no final logit softcap"


def head_reasons(
    lm_head: torch.nn.Module,
    selector: torch.nn.Module,
    softcap: float,
) -> list[str]:
    """Why the draft head (``lm_head``, ``selector``) cannot have its
    monokernel; empty when it can. Whether its shard-local top-k is exact is
    checked once the drafter is wired (:meth:`DFlash2MonokernelState.enable_head`)."""
    reasons = []
    weight = lm_head.weight
    if (
        should_apply_lm_head_quant_method(lm_head, lm_head.quant_method)
        or weight.dtype != torch.bfloat16
        or weight.shape[0] < SHARD
        or weight.shape[1] != HIDDEN
    ):
        reasons.append(HEAD_NOT_SHARDED)
    if (
        selector.top_k != TOPK
        or tuple(selector.predecessor_codebook.shape) != (VOCAB, SELECTOR_RANK)
        or tuple(selector.hidden_projection.weight.shape) != (SELECTOR_RANK, HIDDEN)
        or not isinstance(
            selector.hidden_projection.quant_method, UnquantizedLinearMethod
        )
    ):
        reasons.append(HEAD_OTHER_SHAPES)
    if not softcap > 0:
        reasons.append(HEAD_NO_SOFTCAP)
    return reasons


def takes_head(rows: int) -> bool:
    """Whether a draft head over ``rows`` draft rows (``bs * 7``) takes the
    monokernel: those of up to ``MAX_REQUESTS`` requests. More requests keep
    the regular path."""
    return rows % STEPS == 0 and 1 <= rows // STEPS <= MAX_REQUESTS


class DraftHeadMonokernel:
    """The draft head's launch: its packed weights, its own mailbox scratch
    (:meth:`DFlash2MonokernelState.prepare` allocates it), and the TP8 group's
    CPU group for the compile-ahead.

    It keeps no step counter of its own: it reads the draft layers', which
    their launches advance once a block forward, before the head runs (the
    head has its own scratch, so its tags never meet the layers'). The head
    therefore runs only after the layers' launches of the same forward
    (:meth:`DFlash2MonokernelState.finish_step` marks the block, the head
    takes the mark); a head without them would repeat the last launch's tag.

    Attributes:
        fused: The packed head.
        scratch, bases: Its mailbox scratch and the group's heap bases.
        step: The draft layers' step counter.
        cpu_group: The TP8 group's CPU (gloo) process group.
        compiled: The row counts every rank compiled the launch for.
    """

    def __init__(
        self,
        fused: FusedDecodeLayer,
        scratch: torch.Tensor,
        bases: torch.Tensor,
        cpu_group: dist.ProcessGroup,
        step: torch.Tensor,
    ) -> None:
        self.fused = fused
        self.scratch = scratch
        self.bases = bases
        self.step = step
        self.cpu_group = cpu_group
        self.compiled: set[int] = set()

    def __call__(
        self, hidden_states: torch.Tensor, first_token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The candidates and edge scores of the draft steps of up to
        ``MAX_REQUESTS`` requests.

        Args:
            hidden_states: BF16 ``[requests * 7, 8192]``: the draft's
                final-norm output of each block's rows 1 .. 7, request by
                request.
            first_token: int32 ``[requests]``: each block's first token.

        Returns:
            ``(candidate_ids, scores, anchor)``: int64 ``[rows, 16]`` and
            FP32 ``[rows, 16, 16]``, as ``VocabParallelTopK`` and
            ``CandidateSelector`` give them, and the first tokens clamped to
            the vocabulary (int64 ``[requests]``).
        """
        device = hidden_states.device
        rows = hidden_states.shape[0]
        operands = dict(
            x=hidden_states.contiguous(),
            first_token=first_token,
            anchor=torch.empty(first_token.shape[0], dtype=torch.int64, device=device),
            cand_ids=torch.empty(rows, TOPK, dtype=torch.int64, device=device),
            unary=torch.empty(rows, TOPK, dtype=torch.float32, device=device),
            scores=torch.empty(rows, TOPK, TOPK, dtype=torch.float32, device=device),
            scratch=self.scratch,
            step=self.step,
            peer_bases=self.bases,
        )
        if rows not in self.compiled:
            muse_spark_draft_head_decode(self.fused, compile_only=True, **operands)
            dist.barrier(group=self.cpu_group)
            self.compiled.add(rows)
        muse_spark_draft_head_decode(self.fused, compile_only=False, **operands)
        return operands["cand_ids"], operands["scores"], operands["anchor"]


def unserved_draft_error(name: str, unserved: dict[str, list[str]]) -> str:
    """The startup error of a draft ``name`` whose pieces ``unserved``
    (``"layers"``, ``"context write"`` or ``"head"`` -> reasons) have no
    monokernel."""
    report = "; ".join(f"{piece}: {', '.join(r)}" for piece, r in unserved.items())
    return (
        f"--decode-monokernel: no monokernel for the {name} draft's "
        f"{' and '.join(unserved)} ({report}). The draft follows the target's rule: the "
        "switch never serves a piece without its kernel. Serve without "
        "--decode-monokernel."
    )


def enable_draft_monokernels(model: torch.nn.Module) -> None:
    """Pack each layer and the context write of ``model`` (a
    ``DFlash2DraftModel`` built with the monokernel layer class) after
    loading, and give the layers and the model one
    :class:`DFlash2MonokernelState`. A draft launches every layer or none.

    Raises:
        ValueError: The layers or the context write cannot be packed
            (:func:`unserved_draft_error` names both pieces' reasons).
    """
    name = type(model).__name__
    unserved: dict[str, list[str]] = {}
    reasons = model_reasons(model)
    fused: list[FusedDecodeLayer] = []
    rank = model.mapping.attn.tp_rank
    if not reasons:
        for layer in model.layers:
            prepared = prepare_muse_spark_draft_layer(
                tensors=layer_tensors(layer), rank=rank, max_rows=LAYER_ROWS
            )
            if prepared is None:
                reasons.append(NO_KERNEL)
                break
            fused.append(prepared)
    if reasons:
        unserved["layers"] = reasons
    # An MLA draft has no fc and hidden_norm of the kernel's kind to check.
    reasons = [] if model._uses_mla else context_reasons(model)
    if not reasons and not unserved:
        context = prepare_muse_spark_draft_context(
            tensors=context_tensors(model), rank=rank, max_rows=CONTEXT_WRITE_ROWS
        )
        if context is None:
            reasons.append(NO_KERNEL)
    if reasons:
        unserved["context write"] = reasons
    if unserved:
        raise ValueError(unserved_draft_error(name, unserved))
    state = DFlash2MonokernelState(
        model, max(f.scratch_pairs for f in fused), rank, context
    )
    for layer, prepared in zip(model.layers, fused):
        layer.fused = prepared
        layer.monokernel_state = state
    model.layers[-1].last = True
    model.monokernel_state = state
    logger.info(
        f"Decode monokernel: {len(fused)} of {len(model.layers)} {name} layers, and the "
        "context write"
    )


CONTEXT_OTHER = "an fc or hidden_norm other than the kernel's (5 taps, BF16)"


def context_reasons(model: torch.nn.Module) -> list[str]:
    """Why the context write of ``model`` (whose layers have monokernels)
    cannot have its monokernel; empty when it can."""
    fc, norm = model.fc, model.hidden_norm
    if (
        not isinstance(fc.quant_method, UnquantizedLinearMethod)
        or fc.weight.dtype != torch.bfloat16
        # fc is column-parallel: each rank holds the rows of its output columns.
        or fc.tp_size != 8
        or tuple(fc.weight.shape) != (HIDDEN // 8, 5 * HIDDEN)
        or norm.weight.dtype != torch.bfloat16
        or abs(norm.variance_epsilon - RMS_NORM_EPS) >= 1e-12
    ):
        return [CONTEXT_OTHER]
    return []


def context_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The context write's weights by the roles of
    :func:`prepare_muse_spark_draft_context`."""
    tensors = {"fc": model.fc.weight.data, "hidden_norm": model.hidden_norm.weight.data}
    for i, layer in enumerate(model.layers):
        tensors[f"qkv_{i}"] = layer.self_attn.qkv_proj.weight.data
        tensors[f"k_norm_{i}"] = layer.self_attn.k_norm.weight.data
    return tensors


__all__ = [
    "BLOCK_ROWS",
    "DFlash2MonokernelLayer",
    "DFlash2MonokernelState",
    "MAX_REQUESTS",
    "DraftHeadMonokernel",
    "enable_draft_monokernels",
    "head_reasons",
    "mailbox_scratches",
    "layer_tensors",
    "config_reasons",
    "model_reasons",
    "takes_head",
    "takes_monokernels",
    "unserved_draft_error",
]
