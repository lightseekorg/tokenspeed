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

"""The decode monokernel seam of an Inkling2 target (``docs/design/monokernel.md``).

With ``--decode-monokernel``, the target's layers (the mid local ones
``6 k + 1 .. 6 k + 4``, the first of each attention-residual block, ``6 k``,
layer 0 included, and the full attention ones, ``6 k + 5``, with NVFP4 or
MXFP8 experts) run a verify step of 1, 2, 4 or 8
requests (``BATCHES``: the decode graphs' batch sizes up to 8; a graph batch
past the live requests pads them) as one persistent launch a rank: ``8 bs``
rows, each request's verify block of 8 after the previous one's. Every
other forward runs the regular code. A launch takes the regular layer's
boundary (the residual state's pending terms and the set partials) and
returns it, so launched and regular layers follow each other in one
forward.

A looped core (``loop_config``) visits its layers once a pass. Each visit
addresses its pass's caches (the visit's cache layer: its KV pages and conv
ring) and has a launch tag of its own (its visit index). The first pass's
visits launch as any layer's. A later pass gates the previous pass's
message into the layer's Q, K and V inputs and into its MoE input, which
mixes the attention output: the regular path computes those gated,
normalized inputs, and the visit's attention and MoE parts are a launch
each, around them (the kernel ops' ``PARTS``). The two parts write disjoint mailbox
regions, so they share the visit's tag.

After loading, :meth:`Inkling2Monokernel.pack` packs the layers;
``prepare_communication_runtime`` allocates the launches' mailbox on the
attention TP group's Iris heap (:meth:`Inkling2Monokernel.prepare`).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.model.inkling2.monokernel import (
    LocalLayer,
    inkling2_full_layer_splits,
    inkling2_local_layer_decode,
    prepare_inkling2_full_layer,
    prepare_inkling2_local_layer,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention_residual import AttentionResidualState
from tokenspeed.runtime.models.monokernel import MonokernelState

# Iris uses Gluon dtypes unavailable on Ascend. Model discovery imports this
# module even when the serving model does not use decode monokernels.
if not current_platform().is_npu:
    from tokenspeed_kernel.ops.communication.iris import (
        create_iris_heap,
        iris_heap_bytes,
        iris_symmetric_heap,
        reserve_iris_heap,
    )

# The verify widths the launches serve: one request's verify block of
# ``chain`` rows (DSPARK's draft tokens and the token they extend), chained.
CHAINS = (4, 8)
# The verify batch sizes the launches serve. Larger batches keep the regular
# path.
BATCHES = (1, 2, 4, 8)
ATTENTION_PAGE = 128  # KV page (the regular decode's KV block) of bit identity
# Query rows of a block set's partials: the 6 layers' MLP queries and the
# attention queries of the layers after them (one local, three full).
SET_QUERIES = 14


def launch_rows(chain: int) -> tuple[int, ...]:
    """The rows of the launches of verify blocks of ``chain`` rows."""
    return tuple(chain * bs for bs in BATCHES)


def launch_requests(ctx: ForwardContext, spec_tokens: int, chain: int) -> int | None:
    """The requests of a forward that the launches serve (a decode verify
    batch of ``BATCHES``, ``spec_tokens`` = ``chain`` tokens a request), or
    None. It depends only on the forward, so every rank decides the same."""
    if ctx.forward_mode is None or not ctx.forward_mode.is_decode():
        return None
    if spec_tokens != chain or ctx.bs not in BATCHES:
        return None
    return ctx.bs if ctx.input_num_tokens == chain * ctx.bs else None


def verify_positions(
    seq_lens: torch.Tensor, offsets: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """The rows' positions, int64 into ``out[: chain * bs]``: row ``j`` of
    request ``i`` at ``seq_lens[i] - chain + j`` (``offsets`` is
    ``arange(-chain, 0)``). ``seq_lens`` are the regular attention's decode
    lengths, at least the verify floor ``chain``: a padded request of a graph
    batch replays at length 1, which the attention leaf raises to the floor
    (``docs/design/unified_path.md``), so its rows sit at ``0 .. chain - 1``
    as in the regular attention."""
    bs, chain = seq_lens.shape[0], offsets.shape[0]
    rows = out[: chain * bs]
    torch.add(seq_lens.view(bs, 1), offsets.view(1, chain), out=rows.view(bs, chain))
    return rows


class Inkling2Monokernel:
    """The packed layers of one Inkling2 target and the state their launches
    share.

    Args:
        model: The ``Inkling2TextModel``.
        mapping: The parallel mapping; attention TP must be the MoE EP group.
        chain: The server's verify width (DSPARK's draft tokens a step), one
            of ``CHAINS``.
    """

    def __init__(self, model, mapping, chain: int) -> None:
        config = model.config
        if mapping.attn.tp_group != mapping.moe.ep_group:
            raise ValueError("the Inkling2 monokernel needs attention TP == MoE EP")
        if chain not in CHAINS:
            raise ValueError(
                f"the Inkling2 monokernel serves verify blocks of {CHAINS} tokens, "
                f"not {chain}"
            )
        self.chain = chain
        self.rows = launch_rows(chain)
        self.model = model
        self.config = config
        # Each visit's launch tag: its index in the forward's visit order.
        self.visit_index = {
            visit: index for index, visit in enumerate(config.layer_visits)
        }
        self.mapping = mapping
        self.rank = mapping.attn.tp_rank
        block = config.n_layers_per_attn_res_block
        local = set(config.local_layer_ids)
        # Layers with one expert format for both projections: the mid local
        # ones, the boundary ones (the first of an attention-residual block,
        # layer 0 too: its block starts from the embedding) and the full
        # attention ones (the last of a block).
        self.block = block
        self.local = local
        self.eligible = [
            i
            for i, layer in enumerate(model.layers)
            if layer.mlp.experts.formats in (("nvfp4", "nvfp4"), ("mxfp8", "mxfp8"))
        ]
        self.layers: dict[int, LocalLayer] = {}
        self.state: MonokernelState | None = None
        self.bases: torch.Tensor | None = None
        # This step's values: its requests and row positions when it may
        # launch, and the launches so far (their count alternates the
        # peer-region parity).
        self.requests: int | None = None
        self.positions: torch.Tensor | None = None
        self.row_offsets: torch.Tensor | None = None
        self.row_positions: torch.Tensor | None = None
        self.launches = 0
        # Each layer's (P, A, M) outputs: the residual state may hold a
        # layer's terms past the next launch.
        self.outputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        # A boundary layer's stored-block count, and the buffers of the set
        # partials its sweep writes: a boundary launch reads the previous
        # set's while it writes its own, so consecutive sets alternate
        # between two buffers. A looped core's later passes read the set
        # its entry holds again after the core's boundaries wrote theirs, so
        # with a loop every set has its own buffer.
        self.block_counts: dict[int, torch.Tensor] = {}
        self.set_buffers: tuple[tuple[torch.Tensor, ...], ...] = ()

    def pack(self) -> None:
        """Pack every eligible layer (after loading; again after a weight
        update). Raises when no monokernel serves a layer."""
        for i in self.eligible:
            layer = self.model.layers[i]
            attn, ex = layer.attn, layer.mlp.experts
            experts = ex.formats[0]
            if experts == "nvfp4":
                w13 = (
                    ex.w13_weight,
                    ex.w13_scale,
                    ex.w13_amax,
                    ex.w13_outer_scale_denominator,
                )
                w2 = (
                    ex.w2_weight,
                    ex.w2_scale,
                    ex.w2_amax,
                    ex.w2_outer_scale_denominator,
                )
            else:
                w13 = (ex.w13_weight, ex.w13_scale)
                w2 = (ex.w2_weight, ex.w2_scale)
            tensors = {
                "qkvr": attn.qkvr.weight,
                "wo": attn.wo_ud.weight,
                "router": layer.mlp.gate.weight,
                "router_bias": layer.mlp.gate.bias,
                "head": (
                    attn.q_l2_clip.scale,
                    attn.rel_l2_clip.scale,
                    attn.rel_logits_proj.proj,
                    attn.k_sconv.taps().contiguous(),
                    attn.v_sconv.taps().contiguous(),
                ),
                "attn_taps": layer.attn_sconv.taps().contiguous(),
                "mlp_taps": layer.mlp_sconv.taps().contiguous(),
                "w13": w13,
                "w2": w2,
            }
            if i in self.local:
                packed = prepare_inkling2_local_layer(
                    rows=self.rows,
                    experts=experts,
                    boundary=i % self.block == 0,
                    first=i == 0,
                    later_passes=self.config.layer_passes(i) > 1,
                    tensors=tensors,
                )
            else:
                packed = prepare_inkling2_full_layer(
                    rows=self.rows,
                    experts=experts,
                    later_passes=self.config.layer_passes(i) > 1,
                    tensors=tensors,
                )
            if packed is None:
                raise RuntimeError(
                    f"--decode-monokernel: no Inkling2 monokernel for layer {i} here"
                )
            self.layers[i] = packed
            layer.monokernel = self

    def reserve(self) -> None:
        """Reserve the mailbox on the Iris heap (before the heap exists)."""
        pairs = max(packed.scratch_pairs for packed in self.layers.values())
        reserve_iris_heap("inkling2-monokernel", iris_heap_bytes((pairs,), torch.int64))

    def prepare(self, device: torch.device) -> None:
        """Create the heap, allocate the mailbox and the outputs (eager, every
        rank of the TP group, before graph capture)."""
        create_iris_heap()
        group = pg_manager.get_process_group("nccl", self.mapping.attn.tp_group)
        context, iris_group = iris_symmetric_heap(group, self.rank, device)
        pairs = max(packed.scratch_pairs for packed in self.layers.values())
        self.state = MonokernelState(context.zeros((pairs,), dtype=torch.int64))
        iris_group.finish_prepare()
        self.bases = iris_group.bases
        # Row j of a verify block sits at seq_len - chain + j.
        self.row_offsets = torch.arange(
            -self.chain, 0, dtype=torch.int64, device=device
        )
        rows = max(self.rows)
        self.row_positions = torch.zeros(rows, dtype=torch.int64, device=device)
        hidden = self.model.config.hidden_size
        # A launch of r rows writes the first r rows.
        self.outputs = {
            i: tuple(
                torch.zeros(rows, hidden, dtype=torch.bfloat16, device=device)
                for _ in range(3)
            )
            for i in self.layers
        }
        self.block_counts = {
            i: torch.tensor([i // self.block], dtype=torch.int32, device=device)
            for i in self.layers
            if i % self.block == 0
        }
        queries = SET_QUERIES
        sets = self.model.attn_res.num_blocks if self.config.loop_config else 2
        self.set_buffers = tuple(
            (
                torch.zeros(queries * rows, dtype=torch.float32, device=device),
                torch.zeros(queries * rows, dtype=torch.float32, device=device),
                torch.zeros(
                    queries * rows * hidden, dtype=torch.float32, device=device
                ),
            )
            for _ in range(sets)
        )

    def start_step(self, ctx: ForwardContext) -> None:
        """Before the layers: this step's row positions, for a forward that
        may launch. They come from the verify metadata that the regular
        layers read (:func:`verify_positions`), into a buffer of the seam:
        the scheduler may refill the model's input positions for the next
        step while this step's launches still run."""
        self.launches = 0
        self.requests = None
        self.positions = None
        if self.state is None:
            return
        self.state.start_step()
        backend = ctx.attn_backend
        requests = launch_requests(ctx, backend.conv_spec_num_tokens, self.chain)
        if requests is not None:
            # The rows' lengths, from the first packed layer's attention leaf
            # (every leaf of a forward holds the same sequence lengths).
            attn = self.model.layers[next(iter(self.layers))].attn.attn
            leaf = backend.inner._leaf_for(attn)
            seq_lens = leaf.forward_decode_metadata.seq_lens[:requests]
            self.positions = verify_positions(
                seq_lens, self.row_offsets, self.row_positions
            )
            self.requests = requests

    def finish_step(self) -> None:
        """After the layers: a new tag for the next step's launches."""
        if self.state is not None:
            self.state.finish_step()
        self.requests = None
        self.positions = None

    def serves(
        self,
        layer_id: int,
        pass_idx: int,
        state: AttentionResidualState,
        ctx: ForwardContext,
        mailbox,
        share,
    ) -> bool:
        """Whether this forward launches the visit of layer ``layer_id`` in
        pass ``pass_idx``. It depends only on the forward, so every rank
        decides the same."""
        if pass_idx > 0:
            # The regular path mixes a later pass's inputs (its own set
            # partials included).
            return (
                self.state is not None
                and layer_id in self.layers
                and self.requests is not None
                and launch_requests(
                    ctx, ctx.attn_backend.conv_spec_num_tokens, self.chain
                )
                == self.requests
                and state.sweep_sets
                and state.share is None
                and share is None
                and mailbox is None
                and self.positions is not None
            )
        return (
            self.state is not None
            and layer_id in self.layers
            and self.requests is not None
            and launch_requests(ctx, ctx.attn_backend.conv_spec_num_tokens, self.chain)
            == self.requests
            and state.sweep_sets
            and state.share is None
            and share is None
            and mailbox is None
            and self.positions is not None
            # Layer 0 starts the first block: no set partials yet.
            and (
                state.set_partials is None
                if layer_id == 0
                else state.set_partials is not None
                and state.set_blocks == state.num_blocks
            )
        )

    def run(
        self,
        layer,
        state: AttentionResidualState,
        ctx: ForwardContext,
        log_scaling_tau: torch.Tensor | None,
        pass_idx: int,
        messages,
    ) -> None:
        """Launch the visit of ``layer`` in pass ``pass_idx`` and leave the
        residual state as its regular forward does (``log_scaling_tau``: the
        forward's FP32 attention logit scales, which a full layer applies).
        A later pass of a looped core (``messages``, the previous pass's
        ``Inkling2LoopMessages``) runs the regular path's gated input mixes
        around its two part launches."""
        if pass_idx == 0:
            self._run_layer(layer, state, ctx, log_scaling_tau)
        else:
            self._run_later_pass(layer, state, ctx, log_scaling_tau, pass_idx, messages)

    def _visit(self, layer, ctx: ForwardContext, pass_idx: int, rows: int) -> dict:
        """The operands of a visit's launches: its launch tag, its pass's
        caches and the forward's attention metadata. Registers the visit's
        conv streams for the ring commit after acceptance, as the regular
        convolutions do: a local layer's K and V as one stream, a full
        layer's each its own."""
        attn = layer.attn
        full = layer.layer_id not in self.local
        # The pass's KV cache layer (shared by every pass under a shared
        # cache) and its own conv state layer.
        paged = attn.pass_attention(pass_idx)
        cache_layer = paged.layer_id
        conv_layer = attn.conv_layer_ids[pass_idx]
        backend = ctx.attn_backend
        leaf = backend.inner._leaf_for(paged)
        # The attention stages equal the regular split-KV decode bit for bit
        # only at its 128-token KV block, which is the cache page. The page is
        # fixed per server: this fails at graph capture, before serving.
        if leaf.kernel_page_size != ATTENTION_PAGE:
            raise RuntimeError(
                f"the Inkling2 monokernel needs a {ATTENTION_PAGE}-token KV page, "
                f"not {leaf.kernel_page_size}"
            )
        pool = ctx.token_to_kv_pool
        heads, head_dim = paged.tp_k_head_num, paged.qk_head_dim
        kv = (attn.k_sconv, attn.v_sconv) if full else (attn.k_sconv,)
        for conv in (*kv, layer.attn_sconv, layer.mlp_sconv):
            _register_stream(ctx, conv_layer, conv, full)
        return {
            "tag": self.visit_index[(layer.layer_id, pass_idx)],
            "full_splits": (
                inkling2_full_layer_splits(leaf.max_context_len) if full else 0
            ),
            # The cache rows, one a token slot: [slots, heads, head].
            "k_cache": pool.get_key_buffer(cache_layer).view(-1, heads, head_dim),
            "v_cache": pool.get_value_buffer(cache_layer).view(-1, heads, head_dim),
            "locations": backend.padded_write_locations(paged, ctx.forward_mode, rows),
            "positions": self.positions,
            "page_table": leaf.forward_decode_metadata.page_table,
            "page_size": leaf.kernel_page_size,
            "ring": backend.conv_pool.layer_state_wd(conv_layer),
            "ring_slots": backend.conv_decode_metadata.cache_indices,
        }

    def _launch(
        self,
        layer_id: int,
        part: str,
        inputs: tuple[torch.Tensor, ...],
        visit: dict,
        terms: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        queries: tuple[tuple, tuple],
        sweep: tuple,
        tau: torch.Tensor | None,
    ) -> None:
        inkling2_local_layer_decode(
            self.layers[layer_id],
            part=part,
            inputs=inputs,
            rank=self.rank,
            rows=self.chain * self.requests,
            chain=self.chain,
            scratch=self.state.scratch,
            step=self.state.step,
            layer_index=visit["tag"],
            # Launches alternate the peer copies. Across steps (an odd count
            # of launches a step) the regular layers between the last launch
            # and the next step's first, whose collectives every rank
            # finishes, keep the ranks within one launch.
            parity=self.launches % 2,
            bases=self.bases,
            terms=terms,
            outputs=outputs,
            queries=queries,
            sweep=sweep,
            full_splits=visit["full_splits"],
            tau=tau,
            k_cache=visit["k_cache"],
            v_cache=visit["v_cache"],
            locations=visit["locations"],
            positions=visit["positions"],
            page_table=visit["page_table"],
            page_size=visit["page_size"],
            ring=visit["ring"],
            ring_slots=visit["ring_slots"],
        )
        self.launches += 1
        self.state.launched = True

    def _tau(self, visit: dict, log_scaling_tau: torch.Tensor | None, rows: int):
        """A full layer's rows' logit scales; None for a local layer."""
        if not visit["full_splits"]:
            return None
        if log_scaling_tau is None:
            raise RuntimeError("a full layer's launch needs the logit scales")
        return log_scaling_tau[:rows]

    def _run_later_pass(
        self,
        layer,
        state: AttentionResidualState,
        ctx: ForwardContext,
        log_scaling_tau: torch.Tensor | None,
        pass_idx: int,
        messages,
    ) -> None:
        """A later pass's visit: the regular path gates the messages into
        the layer's normalized inputs, the attention part launches on the Q,
        K and V inputs, the regular path mixes the MoE input over its output,
        and the MoE part launches on it."""
        L = layer.layer_id
        rows = self.chain * self.requests
        visit = self._visit(layer, ctx, pass_idx, rows)
        outputs = tuple(t[:rows] for t in self.outputs[L])
        sources = layer.attention_input(state, messages)
        self._launch(
            L,
            "attention",
            tuple(sources),
            visit,
            (),
            outputs,
            ((), ()),
            (),
            self._tau(visit, log_scaling_tau, rows),
        )
        attention = outputs[1]
        mlp_input = layer.mlp_input(state, attention, messages)
        self._launch(L, "mlp", (mlp_input,), visit, (), outputs, ((), ()), (), None)
        state.finalize_layer(L, attention, outputs[2])

    def _run_layer(
        self,
        layer,
        state: AttentionResidualState,
        ctx: ForwardContext,
        log_scaling_tau: torch.Tensor | None,
    ) -> None:
        """A first pass's visit (every visit of a layer outside a looped
        core): one launch from the residual state's boundary."""
        L = layer.layer_id
        full = L not in self.local
        res = state.attn_res
        boundary = L % self.block == 0
        if state.partials_ready is not None:
            torch.cuda.current_stream().wait_event(state.partials_ready)
            state.partials_ready = None
        attn_row = res.attn_rows[L][0]
        mlp_row = res.mlp_rows[L]

        def query(row: int, gain: torch.Tensor) -> tuple:
            first, _ = res.set_rows[state.num_blocks - 1]
            m, s, acc = state.set_partials
            i = row - first
            return (res.score_weights[row], gain, acc[i], m[i], s[i])

        rows = self.chain * self.requests
        visit = self._visit(layer, ctx, 0, rows)
        outputs = tuple(t[:rows] for t in self.outputs[L])
        sweep = ()
        if full:
            # Three queries (the Q, K and V sources) with the stacked gains
            # the regular mix takes.
            gains = layer.input_norm_gains
            queries = (
                tuple(query(attn_row + j, gains[j]) for j in range(3)),
                query(mlp_row, layer.mlp_norm.weight),
            )
        elif not boundary:
            queries = (
                query(attn_row, layer.attn_norm.weight),
                query(mlp_row, layer.mlp_norm.weight),
            )
        else:
            # The layer stores its block, sweeps the new set's partials over
            # the stored blocks and its own, and mixes its MLP input over
            # them (the set's first query).
            b = state.num_blocks
            if state.blocks.shape[1] != rows:
                raise RuntimeError(
                    "the residual state's blocks are not the launch's rows"
                )
            set_first, set_last = res.set_rows[b]
            if set_last - set_first != SET_QUERIES:
                raise RuntimeError(f"block set {b} has {set_last - set_first} queries")
            m_buf, s_buf, acc_buf = self.set_buffers[b % len(self.set_buffers)]
            hidden = acc_buf.shape[0] // (SET_QUERIES * max(self.rows))
            new_set = (
                m_buf[: SET_QUERIES * rows].view(SET_QUERIES, rows),
                s_buf[: SET_QUERIES * rows].view(SET_QUERIES, rows),
                acc_buf[: SET_QUERIES * rows * hidden].view(SET_QUERIES, rows, hidden),
            )
            sweep = (
                state.blocks,
                self.block_counts[L],
                res.score_weights[set_first:set_last],
                *new_set,
            )
            new_m, new_s, new_acc = new_set
            mlp_query = (
                res.score_weights[mlp_row],
                layer.mlp_norm.weight,
                new_acc[0],
                new_m[0],
                new_s[0],
            )
            # Layer 0's attention input is the embedding's RMSNorm, not a
            # mix: its query reads only the gain (the MLP query's other
            # fields keep the operand's shapes; no set partials exist yet).
            attn_query = (
                (res.score_weights[attn_row], layer.attn_norm.weight) + mlp_query[2:]
                if L == 0
                else query(attn_row, layer.attn_norm.weight)
            )
            queries = (attn_query, mlp_query)
            outputs = (state.blocks[b],) + outputs[1:]
        self._launch(
            L,
            "layer",
            (),
            visit,
            tuple(state.terms),
            outputs,
            queries,
            sweep,
            self._tau(visit, log_scaling_tau, rows),
        )
        p_term, a_term, m_term = outputs
        # As state.attention_input leaves the layer: its candidate is the one
        # pending term; a boundary layer stored it as its block and started
        # the new set's partials.
        state.terms = (p_term,)
        state.block_stored = boundary
        if boundary:
            state.set_partials, state.set_blocks = new_set, state.num_blocks + 1
            state.partials_ready = None
        state.finalize_layer(L, a_term, m_term)


def _register_stream(ctx: ForwardContext, layer_id: int, conv, full: bool) -> None:
    from tokenspeed.runtime.models.inkling import _sconv_stream

    # A local layer's adjacent K and V streams register as one.
    dim = (
        2 * conv.dim
        if not full and conv.channel_offset == 0 and conv._group() == "kvconv"
        else conv.dim
    )
    _sconv_stream(ctx, layer_id, conv.channel_offset, dim, conv._group())
