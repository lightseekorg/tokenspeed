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

"""Muse Spark decode monokernel seam (``docs/design/monokernel.md``).

With ``--decode-monokernel``, the model builds every decoder layer from
``MuseSparkMonokernelDecoderLayer`` (:class:`MuseSparkMonokernelLayer`
before the regular layer class). A decode step that the kernels serve runs
layers 1 to 60 as one persistent launch each per TP8 rank, from
``tokenspeed_kernel.ops.model.muse_spark.monokernel``:

* after loading, :func:`enable_decode_monokernels` packs each layer's launch;
  every layer but the first and the last must have one;
* when the Iris collectives are prepared, before cache sizing and graph
  capture, :func:`prepare_monokernel_state` gives the layers and the model
  one :class:`MuseSparkMonokernelState` (the mailbox scratch on the Iris
  symmetric heap);
* the model calls ``start_step()`` before its layers and ``finish_step()``
  after them, and calls :meth:`MuseSparkMonokernelState.claim_taps`, which
  decides whether the launches store a draft's taps.

A step serves one or two requests of equal rows: a token row each, or the
rows of a speculative verify each (a request's draft tokens, one row each;
row ``t`` of a request of ``chain`` rows has length ``seq - chain + t + 1``
for its entry of the step's ``seq_lens``), one request's rows after the
other's. A draft model that taps the residual entering a layer
(``layers_to_capture``) gets the whole row there: the fused run before that
layer gathers it, as before a regular layer.

The served MoE split is MoE TP8 / EP1: each rank holds its slice of every
expert, which the kernel reads in place. Other splits keep the regular path.

The launch boundary: rank ``r`` keeps hidden columns ``1024 r ..`` of the
FP32 residual and the whole BF16 attention input of the next layer (normed by
that layer's input norm). The regular layers hand over the whole residual
row. So the first launch of a run of fused layers reads its slice of that
row, and the last one gathers the 8 slices back into a whole row for the
regular layer after it. Layers 0 and 61 keep the regular code: a forward
mixes them with the launches, which the design doc allows for a model's
boundary layers. They are the first and last layers (they hand over to the
embedding and the LM head) and this checkpoint's layers with BF16 routed
experts. Fusing them changes their summation order, a numerics change that
needs accuracy evidence first, and a fused build gave no clear served gain.
"""

from __future__ import annotations

import atexit
import json
import logging
import os

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.model.muse_spark.monokernel import (
    MUSE_SPARK_PROJ_WORDS,
    FusedDecodeLayer,
    muse_spark_layer_decode,
    muse_spark_stuck_record,
    prepare_muse_spark_layer,
)

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.models.monokernel import (
    NO_KERNEL,
    MonokernelLayer,
    MonokernelState,
    unserved_error,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)

# Token rows of one launch: a decode row a request, or a verify of up to 7
# draft tokens (the DFlash2 drafters' block of 8) a request, for up to
# MAX_REQUESTS requests, their rows one request after another (16 at most).
# A step of more requests keeps the regular path.
MAX_REQUESTS = 2
CHAIN_ROWS = 8  # a request's rows at most
MAX_ROWS = MAX_REQUESTS * CHAIN_ROWS
WORLD = 8
# CTAs of a layer launch (layer.layer_plan) and waves of each (its num_warps):
# one stuck-poll record per wave.
GRID = 256
WAVES = 8
HIDDEN = 8192
OWNED = HIDDEN // WORLD
PAGE_SIZE = 64
WINDOW = 2048

# Values the kernel bakes in; a checkpoint with other values keeps the
# regular path.
KERNEL_CONSTANTS = {
    "qk_scale_factor": 5.473006248474121,
    "output_multiplier": 0.17677669529663687,
    "routed_scaling_factor": 1.0,
    "rms_norm_eps": 1e-5,
    "post_norm_eps": 1e-8,
}

# Why a layer has no monokernel. The startup report groups layers by these.
BOUNDARY_LAYER = "the first and last layers hand over to the embedding and LM head"
NOT_TP8 = "attention and MoE are not one TP8 group with MoE TP8 / EP1"
OTHER_SHAPES = "model dimensions other than the kernel's"
OTHER_CONSTANTS = "model constants other than the kernel's"
NOT_BF16 = "a dense projection or norm gain is not BF16"
NOT_SUPPORTED_EXPERTS = "the routed experts are not MXFP4"
PACK_REFUSED = "the kernel's packer refused a tensor"
OTHER_KV = (
    "the KV cache is neither BF16 nor FP8 e4m3 (--kv-cache-dtype): the kernels "
    "read those"
)
SCALED_KV = "the layer has FP8 KV-cache scales: the kernels read unit-scale FP8"


def _gain(norm: torch.nn.Module) -> torch.Tensor | None:
    """The FP32 multiplier ``bf16(w + offset)`` of a weighted Muse RMS norm
    (the regular path adds the offset to the BF16 weight), else None."""
    weight = norm.weight
    if weight is None or weight.dtype != torch.bfloat16:
        return None
    return (weight.data + norm.weight_offset).to(torch.bfloat16).float()


def _bf16(proj: torch.nn.Module) -> torch.Tensor | None:
    """The weight of an unquantized BF16 projection, else None."""
    if not isinstance(proj.quant_method, UnquantizedLinearMethod):
        return None
    weight = proj.weight.data
    return weight if weight.dtype == torch.bfloat16 else None


def _experts(experts: torch.nn.Module) -> tuple[object, ...]:
    """``(w13, w13 scales, w2, w2 scales)`` as the regular MXFP4 MoE prepared
    them, which the kernel reads in place. The kernel boundary takes the raw
    storage out of them and refuses a layout it does not read."""
    return (
        experts.w13_weight_triton_tensor,
        experts.w13_precision_config.b_mx_scale,
        experts.w2_weight_triton_tensor,
        experts.w2_precision_config.b_mx_scale,
    )


def model_reasons(config, mapping) -> list[str]:
    """Why no layer of a Muse Spark backbone with ``config`` (its
    ``MuseSparkConfig``) and ``mapping`` can have a monokernel."""
    reasons = []
    if not (
        mapping.attn.tp_size == WORLD
        and (mapping.moe.tp_size, mapping.moe.ep_size) == (WORLD, 1)
        and mapping.attn.tp_group == mapping.moe.tp_ep_group
        and mapping.pp_size == 1
    ):
        reasons.append(NOT_TP8)
    if (
        config.hidden_size,
        config.head_dim,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.num_local_experts,
        config.num_experts_per_tok,
        config.moe_hidden_size,
    ) != (HIDDEN, 64, 128, 16, 256, 8, 4096):
        reasons.append(OTHER_SHAPES)
    if any(
        abs(float(getattr(config, name)) - value) > 1e-6 * max(1.0, abs(value))
        for name, value in KERNEL_CONSTANTS.items()
    ):
        reasons.append(OTHER_CONSTANTS)
    # "auto" keeps the model's BF16; FP8 e4m3 is the regular path's
    # unit-scale cast (per-layer scales are refused in prepare_layer).
    if global_server_args_dict["kv_cache_dtype"] not in (
        "auto",
        "bfloat16",
        "bf16",
        "fp8",
        "fp8_e4m3",
    ):
        reasons.append(OTHER_KV)
    return reasons


def kv_reasons(attention: torch.nn.Module) -> list[str]:
    """Why the cache of a layer's ``PagedAttention`` (after loading) keeps the
    regular path: loaded per-tensor scales apply to an FP8 cache, and the
    kernels apply none."""
    fp8 = global_server_args_dict["kv_cache_dtype"] in ("fp8", "fp8_e4m3")
    return [SCALED_KV] if fp8 and attention.fp8_kv_scales is not None else []


def prepare_layer(model: torch.nn.Module, index: int) -> FusedDecodeLayer | list[str]:
    """Pack decoder layer ``index`` of ``model`` for the monokernel, or say
    why not (a list of reasons)."""
    layers = model.layers
    # TODO(bangtianliu): layers 0 and 61 (the first and last; they hand over
    # to the embedding and the LM head, and are this checkpoint's layers with
    # BF16 routed experts) keep the regular path for now. Fusing them changes
    # their summation order, a numerics change that needs accuracy evidence
    # first, and a fused build gave no clear served gain. Revisit with
    # accuracy evidence.
    if index == 0 or index == len(layers) - 1:
        return [BOUNDARY_LAYER]
    layer = layers[index]
    attn, mlp = layer.self_attn, layer.mlp
    window = model.config.sliding_window[index]
    if window not in (None, WINDOW):
        return [OTHER_SHAPES]
    dense = [
        _bf16(p)
        for p in (attn.qkv_proj, attn.o_proj, mlp.pre_expert_proj, mlp.post_expert_proj)
    ]
    gains = [
        _gain(n)
        for n in (
            layer.pre_feedforward_layernorm,
            layer.post_feedforward_layernorm,
            layers[index + 1].input_layernorm,
            mlp.pre_expert_norm,
            mlp.post_expert_norm,
        )
    ]
    if any(t is None for t in dense + gains):
        return [NOT_BF16]
    if kv_reasons(attn.attn):
        return kv_reasons(attn.attn)
    if not mlp.quantized_experts:
        return [NOT_SUPPORTED_EXPERTS]
    experts = _experts(mlp.experts)
    _, o, pool, post = dense
    # The fused projection may hold every rank's rows (rank major, for large
    # batches); the kernel reads this rank's q | k | v | gate rows.
    qkvg = attn.qkv_proj._local_weight()
    try:
        fused = _pack(model, index, window, qkvg, o, pool, post, gains, experts)
    except ValueError as error:
        # The packer names every tensor (or expert layout) the kernel does
        # not expect.
        return [f"{PACK_REFUSED} ({error})"]
    return [NO_KERNEL] if fused is None else fused


def _pack(model, index, window, qkvg, o, pool, post, gains, experts):
    """:func:`prepare_muse_spark_layer` of layer ``index``'s tensors."""
    layer = model.layers[index]
    mlp = layer.mlp
    pre_gain, post_gain, in_gain, pre_exp_gain, pen_gain = gains
    alpha1, beta1 = layer.post_attention_residual_gate._gate_coefficients()
    alpha2, beta2 = layer.post_feedforward_residual_gate._gate_coefficients()
    return prepare_muse_spark_layer(
        tensors={
            "rank": model.mapping.attn.tp_rank,
            "full": window is None,
            "qkvg": qkvg,
            "o": o,
            "alpha1": alpha1,
            "beta1": beta1,
            "pre_gain": pre_gain,
            "router": mlp.gate.weight.data,
            "router_bias": mlp.gate.e_score_correction_bias.data,
            "pool": pool,
            "experts": experts,
            "experts_kernel": mlp.experts.plan["apply_kernel_name"],
            "pre_exp_gain": pre_exp_gain,
            "pen_gain": pen_gain,
            "post": post,
            "post_gain": post_gain,
            "alpha2": alpha2,
            "beta2": beta2,
            "in_gain": in_gain,
        },
        max_rows=MAX_ROWS,
    )


class CompiledLaunches:
    """The layer launches every rank has compiled. A launch's binary is set by
    its layer, its rows, the rows of each request (chain), whether it hands
    whole rows on and whether it stores a draft's tap (both compile-time
    arguments of the kernel); one not compiled yet must compile on every rank before
    any rank runs it (a first launch that compiles inside the real launch
    leaves the faster ranks polling for it)."""

    def __init__(self) -> None:
        self._done: set[tuple[int, int, int, bool, bool]] = set()

    def first(
        self, index: int, rows: int, chain: int, whole_rows: bool, tap: bool
    ) -> bool:
        """Whether this is the first launch of its binary, which the caller
        then compiles on every rank. Records it as compiled."""
        key = (index, rows, chain, whole_rows, tap)
        if key in self._done:
            return False
        self._done.add(key)
        return True


def takes_monokernels(rows: int, ctx: ForwardContext) -> bool:
    """Whether a forward of ``rows`` token rows takes the monokernels: a
    decode step of 1 to ``MAX_REQUESTS`` requests of equal rows (one token
    each, or a speculative verify), at most ``CHAIN_ROWS`` rows a request and
    ``MAX_ROWS`` rows in all. Prefill, more requests and longer chains keep
    the regular path: the kernels compile per chain up to ``CHAIN_ROWS``."""
    return (
        ctx.forward_mode.is_decode()
        and 1 <= ctx.bs <= MAX_REQUESTS
        and 1 <= rows <= MAX_ROWS
        and rows % ctx.bs == 0
        and rows // ctx.bs <= CHAIN_ROWS
    )


def projection_hand_off(
    index: int, rows: int, fused, proj: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """``(proj_in, proj_out)`` of layer ``index``'s launch in a forward of
    ``rows`` token rows: past one row, a launch whose next layer is fused
    (``fused``: the fused layer indices) ends with that layer's q | k | v
    projection, into copy ``(index + 1) & 1`` of ``proj`` (int32 ``[2, rows+,
    words]``), and a launch whose previous layer is fused reads its own from
    copy ``index & 1`` instead of projecting it."""
    if rows == 1:
        return None, None
    return (
        proj[index & 1] if index - 1 in fused else None,
        proj[(index + 1) & 1] if index + 1 in fused else None,
    )


def hands_whole_rows(
    index: int, served, tapped: frozenset[int], store_taps: bool
) -> bool:
    """Whether layer ``index``'s launch stores whole residual rows on every
    rank (no gather after it), rather than this rank's owned columns for the
    next launch: when the next layer keeps the regular code (``served``: the
    launching layers), or when the model's regular capture of a draft's tap
    (``tapped``: the layers whose entering residual a draft reads) reads the
    whole rows entering it, which it does unless the launches store the
    taps (``store_taps``)."""
    return index + 1 not in served or (index + 1 in tapped and not store_taps)


def row_lengths(
    seq_lens: torch.Tensor, row_offsets: torch.Tensor, requests: int, chain: int
) -> torch.Tensor:
    """Each row's length from a step's ``seq_lens`` (each request's length
    after the step): ``seq - chain + t + 1`` for row ``t`` of a request of
    ``chain`` rows. ``row_offsets`` is ``arange(1 - MAX_ROWS, 1)``."""
    if chain == 1:
        return seq_lens[:requests]
    offsets = row_offsets[MAX_ROWS - chain :]
    return (seq_lens[:requests, None] + offsets[None, :]).reshape(-1)


class MuseSparkMonokernelState(MonokernelState):
    """The state that all Muse Spark layer launches of a backbone share.

    Beyond :class:`MonokernelState`, the mailbox scratch lives on the Iris
    symmetric heap (the ranks write each other's), and the state holds the
    heap bases of the TP8 group, the q | k | v hand-off between launches, the
    draft's taps, and the stuck-poll record. Within a step it holds the owned
    residual columns that the last launch wrote, while the next layer
    launches too.

    Args:
        fused: The packed launch of each served layer, by layer index.
        group: The TP8 device process group.
        cpu_group: Its CPU (gloo) twin.
        rank: This process's rank in ``group``.
        taps: The layers whose entering residual a draft reads (the model's
            ``layers_to_capture``). When the layer before each launches, the
            launches store them (:meth:`claim_taps`).
    """

    def __init__(
        self,
        fused: dict[int, FusedDecodeLayer],
        group: dist.ProcessGroup,
        cpu_group: dist.ProcessGroup,
        rank: int,
        taps: list[int],
    ) -> None:
        from tokenspeed_kernel.ops.communication.iris import iris_symmetric_heap

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        context, iris_group = iris_symmetric_heap(group, rank, device)
        # The scratch is on the Iris heap. Nothing clears it: the tags make old
        # pairs stale.
        scratch_shape, taps_shape = self.heap_shapes(fused, taps)
        super().__init__(context.zeros(scratch_shape, dtype=torch.int64))
        self.fused = fused
        self.cpu_group = cpu_group
        # The layers whose entering residual a draft reads.
        self.tapped = frozenset(taps)
        self.tap_slots: dict[int, int] = self.store_tap_slots(fused, taps)
        # Where a launch's poll that gives up writes what it waited for
        # before the trap: a file per rank with a record directory (it
        # outlives the process), else pinned memory.
        record_dir = global_server_args_dict["monokernel_stuck_record_dir"]
        path = (
            None
            if record_dir is None
            else os.path.join(record_dir, f"stuck-rank{rank}-pid{os.getpid()}.bin")
        )
        if path is not None:
            os.makedirs(record_dir, exist_ok=True)
        self.stuck = muse_spark_stuck_record(
            grid=GRID, waves=WAVES, path=path, scratch=self.scratch, rank=rank
        )
        # A draft's taps: the residual entering each captured layer, BF16, in
        # the layout of the concatenated captures (row, tap, column). The
        # launch before a captured layer stores it on every rank.
        self.taps: torch.Tensor | None = (
            None
            if taps_shape is None
            else context.zeros(taps_shape, dtype=torch.bfloat16)
        )
        self.bases = iris_group.bases.to(device)
        # The q | k | v projection a launch hands to the next one (past one
        # row, between two launching layers): launch ``i`` writes copy
        # ``(i + 1) & 1`` at its end, launch ``i + 1`` reads it at its start.
        self.proj = torch.zeros(
            (2, MAX_ROWS, MUSE_SPARK_PROJ_WORDS), dtype=torch.int32, device=device
        )
        # Row t of a request of c rows is c - 1 - t tokens short of the
        # request's length: the last c values (row_lengths), made once
        # instead of an arange every step.
        self.row_offsets = torch.arange(
            1 - MAX_ROWS, 1, dtype=torch.int32, device=device
        )
        iris_group.finish_prepare()
        # The launches compiled on every rank.
        self.compiled = CompiledLaunches()
        # The (rows, chain) of the launches compiled, for decoding stuck-poll
        # records.
        self.launches: set[tuple[int, int]] = set()
        # The current step's values; start_step resets them.
        # Whether the launches store this step's taps.
        self.store_taps = False
        # The owned residual columns that the last launch wrote, while the
        # next layer launches too.
        self.owned: torch.Tensor | None = None
        # Each row's length, by cache group (made once per step).
        self.seq_lens: dict[int, torch.Tensor] = {}
        self.write_stuck_sidecar()
        atexit.register(self.report_stuck)

    @staticmethod
    def store_tap_slots(
        fused: dict[int, FusedDecodeLayer], taps: list[int]
    ) -> dict[int, int]:
        """The draft taps the launches store, by captured layer: its slot in
        ``taps``. Empty when some captured layer follows a layer without a
        launch (``fused``: the launching layers; the regular capture then
        serves them all)."""
        ordered = sorted(taps)
        if ordered and all(t - 1 in fused for t in ordered):
            return {t: i for i, t in enumerate(ordered)}
        return {}

    @staticmethod
    def heap_shapes(
        fused: dict[int, FusedDecodeLayer], taps: list[int]
    ) -> tuple[tuple[int], tuple[int, int, int] | None]:
        """The shapes the state allocates on the Iris heap: the int64 mailbox
        scratch, and the BF16 draft taps (None when the launches store none).
        Every layer's plan lays the regions out alike (``layer.layer_plan``),
        so the largest plan's scratch serves both kinds of layer."""
        scratch = (max(f.scratch_pairs for f in fused.values()),)
        slots = MuseSparkMonokernelState.store_tap_slots(fused, taps)
        return scratch, (MAX_ROWS, len(slots), HIDDEN) if slots else None

    @staticmethod
    def heap_bytes(fused: dict[int, FusedDecodeLayer], taps: list[int]) -> int:
        """The Iris heap bytes the state allocates (:meth:`heap_shapes`), at
        the heap allocator's alignment."""
        from tokenspeed_kernel.ops.communication.iris import iris_heap_bytes

        scratch, taps_shape = MuseSparkMonokernelState.heap_shapes(fused, taps)
        total = iris_heap_bytes(scratch, torch.int64)
        if taps_shape is not None:
            total += iris_heap_bytes(taps_shape, torch.bfloat16)
        return total

    def start_step(self) -> None:
        super().start_step()
        self.store_taps = False
        self.owned = None
        self.seq_lens = {}

    def stores_taps(self, rows: int, ctx: ForwardContext) -> bool:
        """Whether a step of ``rows`` rows would store every tap in its
        launches. The draft asks before the forward, and then attaches no
        capture sink: a sink reads each tap during the forward, before the
        collectives that order the launches' stores."""
        return bool(self.tap_slots) and takes_monokernels(rows, ctx)

    def claim_taps(self, rows: int, ctx: ForwardContext) -> bool:
        """Claim the taps for this step's launches when they can store every
        tap (sets ``store_taps``), and return whether they do: the model then
        skips its captures and hands the draft :meth:`tap_rows`. The model
        calls it once a step, before its first layer. A step with a capture
        sink keeps the regular capture."""
        self.store_taps = ctx.target_capture_sink is None and self.stores_taps(
            rows, ctx
        )
        return self.store_taps

    def tap_rows(self, rows: int) -> torch.Tensor:
        """The taps the launches stored: BF16 ``[rows, taps * 8192]``, the
        concatenated captures."""
        return self.taps[:rows].view(rows, -1)

    def row_lengths(
        self, group_id: int, seq_lens: torch.Tensor, requests: int, chain: int
    ) -> torch.Tensor:
        """Each row's length from the step's ``seq_lens`` (each request's
        length after the step): ``seq - chain + t + 1`` for row ``t`` of a
        request."""
        lengths = self.seq_lens.get(group_id)
        if lengths is None:
            lengths = row_lengths(seq_lens, self.row_offsets, requests, chain)
            self.seq_lens[group_id] = lengths
        return lengths

    def write_stuck_sidecar(self) -> None:
        """Describe the launches next to a file-backed stuck-poll record (its
        ``.json``), for decoding the file after the process died."""
        path = self.stuck.path
        if path is None:
            return
        side = {
            "layers": self.stuck.layer_kinds(self.fused),
            "launches": sorted(self.launches),
            # Serving's launches do not check the peer parity.
            "check_parity": False,
        }
        with open(f"{path}.json.tmp", "w") as f:
            json.dump(side, f)
        os.replace(f"{path}.json.tmp", f"{path}.json")

    def report_stuck(self) -> None:
        """At exit: log the stuck polls the launches recorded, if any; else
        remove a record file and its sidecar."""
        record = self.stuck
        if record.any():
            for line in record.describe(self.fused, sorted(self.launches)):
                logger.error(f"decode monokernel stuck poll: {line}")
            return
        path = record.path
        record.close(remove=True)
        if path is not None and os.path.exists(f"{path}.json"):
            os.remove(f"{path}.json")


class MuseSparkMonokernelLayer(MonokernelLayer):
    """The monokernel half of a Muse Spark decoder layer whose decode steps
    run as one launch per TP8 rank (``--decode-monokernel``). The model's
    ``MuseSparkMonokernelDecoderLayer`` puts it before the regular layer
    class.

    Every layer of the backbone is of that class. The first and the last
    layer keep the regular code (:data:`BOUNDARY_LAYER`): they hand over to
    the embedding and to the LM head, and their experts are BF16. A launch
    takes and returns the regular layer's boundary, the whole FP32 residual
    rows and the next layer's BF16 attention input, except between two
    launches: there the residual stays as each rank's owned columns in the
    state (:attr:`MuseSparkMonokernelState.owned`), and the residual rows
    the layer returns are its input, unchanged.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_id: int = self.comm_manager.layer_id
        # This layer's packed launch, set after loading by
        # enable_decode_monokernels. None for a boundary layer.
        self.fused: FusedDecodeLayer | None = None

    def launch_chain(self, rows: int, ctx: ForwardContext) -> int | None:
        if self.fused is None or not takes_monokernels(rows, ctx):
            return None
        return rows // ctx.bs

    def _decode(
        self,
        chain: int,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        attention_input: torch.Tensor,
        next_input_layernorm: torch.nn.Module | None,
        owned_global_rows: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the layer's launch. Returns ``(residual, next attention
        input)`` as the regular layer does: the whole residual rows when the
        next layer keeps the regular code or a draft's regular capture taps
        it, else the input rows unchanged (the next launch reads the owned
        columns this one wrote)."""
        if owned_global_rows is not None:
            # The owned-row path starts at MUSE_OWNED_ROWS_MIN_TOKENS rows, far past
            # a launch's MAX_ROWS.
            raise RuntimeError("Muse Spark monokernel: an owned-row forward")
        state = self.monokernel_state
        index, fused = self.layer_id, self.fused
        served = state.fused
        rank = fused.weights.rank
        rows = hidden_states.shape[0]
        requests = rows // chain
        if state.owned is None:
            h = hidden_states[:rows, rank * OWNED : (rank + 1) * OWNED].contiguous()
        else:
            h = state.owned
        attn = self.self_attn
        group_id = attn.attn.group_id
        view = ctx.attn_backend.group_view(group_id, ctx.bs)
        if view.kernel_page_size != PAGE_SIZE:
            raise RuntimeError(
                f"Muse Spark monokernel: KV page size {view.kernel_page_size}, "
                f"the kernel reads {PAGE_SIZE}"
            )
        k_cache, v_cache = ctx.token_to_kv_pool.get_kv_buffer(
            self.comm_manager.layer_id
        )
        leaf = ctx.attn_backend.leaves[group_id]
        slot = ctx.attn_backend.write_locations(attn.attn, ctx.forward_mode)
        cos_sin = (
            attn.rotary_emb.cos_sin_cache
            if attn.rotary_emb is not None
            else attn.nope_cos_sin_cache
        )
        whole_rows = hands_whole_rows(index, served, state.tapped, state.store_taps)
        h_out = h.new_empty(rows, HIDDEN if whole_rows else OWNED)
        a_out = attention_input.new_empty(rows, HIDDEN)
        operands = dict(
            h=h,
            a=attention_input[:rows],
            h_out=h_out,
            a_out=a_out,
            seq_len=state.row_lengths(
                group_id, leaf.forward_decode_metadata.seq_lens, requests, chain
            ),
            slot=slot[:rows],
            page_table=view.page_table[:requests],
            k_cache=k_cache.view(-1, 2, 64),
            v_cache=v_cache.view(-1, 2, 64),
            cos_sin=cos_sin,
            scratch=state.scratch,
            step=state.step[:1],
            peer_bases=state.bases,
            stuck_rec=state.stuck.tensor,
            layer_index=index,
            chain=chain,
            whole_rows=whole_rows,
            tap=None,
            tap_slot=None,
            # The next launching layer: past one row this launch ends with its
            # q | k | v projection (and that launch starts at its attention);
            # for one row it warms its QKVG weights into the MALL.
            warm_next=served.get(index + 1),
        )
        operands["proj_in"], operands["proj_out"] = projection_hand_off(
            index, rows, served, state.proj
        )
        slot_of_next = state.tap_slots.get(index + 1)
        if state.store_taps and slot_of_next is not None:
            operands["tap"] = state.taps
            operands["tap_slot"] = slot_of_next
        # A forward with a capture sink and one without differ in whole_rows
        # before a tapped layer and in whether the launch stores the tap, and
        # the same rows can be requests of different chains (2 rows: two
        # decodes or one 2-token verify): each variant compiles on every rank
        # first.
        tap = operands["tap"] is not None
        if state.compiled.first(index, rows, chain, whole_rows, tap):
            # A launch traps when its peers' values do not come within about
            # a minute, and a first launch compiles for longer on some ranks
            # than on others: every rank compiles, then the ranks meet.
            muse_spark_layer_decode(fused, compile_only=True, **operands)
            dist.barrier(group=state.cpu_group)
            if (rows, chain) not in state.launches:
                state.launches.add((rows, chain))
                state.write_stuck_sidecar()
        muse_spark_layer_decode(fused, compile_only=False, **operands)
        if whole_rows:
            state.owned = None
            return h_out, a_out
        state.owned = h_out
        return hidden_states, a_out


def enable_decode_monokernels(model: torch.nn.Module) -> None:
    """Pack each launching layer of ``model`` (a ``MuseSparkModel`` built
    with the monokernel layer class) after loading.

    Every layer but the first and the last must have a launch: the switch
    never serves a layer without its kernel.

    Raises:
        ValueError: A layer other than a boundary layer cannot be packed.
    """
    layers = model.layers
    unserved: dict[int, list[str]] = {}
    common = model_reasons(model.config, model.mapping)
    for index, layer in enumerate(layers):
        prepared = common or prepare_layer(model, index)
        if isinstance(prepared, FusedDecodeLayer):
            layer.fused = prepared
            continue
        if prepared != [BOUNDARY_LAYER]:
            unserved[index] = prepared
    name = type(model).__name__
    if unserved:
        raise ValueError(unserved_error(name, unserved))
    reserve_monokernel_heap(model)
    served = sum(layer.fused is not None for layer in layers)
    logger.info(
        f"Decode monokernel: {served} of {len(layers)} {name} layers (the first "
        "and the last keep the regular code)"
    )


def reserve_monokernel_heap(model: torch.nn.Module) -> None:
    """Reserve what :class:`MuseSparkMonokernelState` allocates on the Iris
    heap, before the heap exists (:meth:`MuseSparkMonokernelState.heap_bytes`).
    :func:`enable_decode_monokernels` calls it after packing, and the model
    again when its captured layers change."""
    fused = {i: layer.fused for i, layer in enumerate(model.layers) if layer.fused}
    if not fused:
        return
    from tokenspeed_kernel.ops.communication.iris import reserve_iris_heap

    reserve_iris_heap(
        "muse_spark_monokernel",
        MuseSparkMonokernelState.heap_bytes(fused, list(model.layers_to_capture)),
    )


def prepare_monokernel_state(
    model: torch.nn.Module,
    group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
    rank: int,
) -> None:
    """Give ``model``'s launching layers and the model one
    :class:`MuseSparkMonokernelState` on the Iris heap of ``group`` (the TP8
    device process group; ``cpu_group`` its CPU twin, ``rank`` this
    process's rank in it). Call once, after the Iris collectives are
    prepared and before graph capture."""
    if model.monokernel_state is not None:
        return
    fused = {i: layer.fused for i, layer in enumerate(model.layers) if layer.fused}
    state = MuseSparkMonokernelState(
        fused, group, cpu_group, rank, list(model.layers_to_capture)
    )
    for index in fused:
        model.layers[index].monokernel_state = state
    model.monokernel_state = state


__all__ = [
    "MAX_ROWS",
    "MuseSparkMonokernelLayer",
    "MuseSparkMonokernelState",
    "enable_decode_monokernels",
    "hands_whole_rows",
    "model_reasons",
    "prepare_layer",
    "prepare_monokernel_state",
    "reserve_monokernel_heap",
    "projection_hand_off",
    "row_lengths",
    "takes_monokernels",
]
