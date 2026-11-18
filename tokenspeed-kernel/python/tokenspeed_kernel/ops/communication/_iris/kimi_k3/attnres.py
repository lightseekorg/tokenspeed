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

"""K3 attn all-reduce with an AttnRes epilogue using precomputed history partials."""

import torch
from tokenspeed_kernel._triton import gl, gluon, triton
from tokenspeed_kernel.ops.communication._iris import IrisGroup, allocation_bytes
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_heap_base,
    _iris_lamport_push,
    _iris_sanitize_lamport_bf16,
    _iris_subgroup_any,
    _iris_sync_rank_token,
    _iris_wait_lamport_slots,
    iris_arch_policy,
)
from tokenspeed_kernel.platform import current_platform

if current_platform().is_cdna4:
    # The gated o-projection keeps the registered gated GEMV's tiles and bits.
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.gemv import (
        gluon_gated_norm_gemv_a16w16_tiling_gfx950,
        gluon_gated_norm_gemv_tile_gfx950,
        supports_gluon_gated_norm_gemv_a16w16_gfx950,
    )

ATTNRES_WORLD_SIZE = 8
ATTNRES_HIDDEN_SIZE = 7168
ATTNRES_MAX_ROWS = 16
# Rows the reduced-row epilogue serves after a Lamport all-reduce.
ATTNRES_EPILOGUE_MAX_ROWS = 32
# Rows the fused Lamport two-shot serves.
ATTNRES_LAMPORT_MAX_ROWS = 32
_LAMPORT_STAGES = 3
# Columns of a row block that one exchange program carries.
_LAMPORT_TILE = 128
_NUM_SUBGROUPS = 16
_ELEMENTS_PER_THREAD = 8


def attnres_supported(
    partial: torch.Tensor,
    residual: torch.Tensor,
    score_weight: torch.Tensor,
    output_weight: torch.Tensor,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    world_size: int,
    device: torch.device,
    max_rows: int = ATTNRES_MAX_ROWS,
) -> bool:
    """Check the fused tensor contract without allocating a state."""
    m, s, acc = scratch
    rows = partial.shape[0] if partial.ndim == 2 else 0
    hidden = ATTNRES_HIDDEN_SIZE
    return (
        current_platform().is_cdna4
        and world_size == ATTNRES_WORLD_SIZE
        and partial.is_cuda
        and 0 < rows <= max_rows
        and partial.shape == residual.shape == acc.shape == (rows, hidden)
        and score_weight.shape == output_weight.shape == (hidden,)
        and m.shape == s.shape == (rows,)
        and partial.dtype
        == residual.dtype
        == score_weight.dtype
        == output_weight.dtype
        == torch.bfloat16
        and m.dtype == s.dtype == acc.dtype == torch.float32
        and all(
            tensor.device == device and tensor.is_contiguous()
            for tensor in (partial, residual, score_weight, output_weight, m, s, acc)
        )
    )


def _lamport_mailbox_shape(max_rows: int, world_size: int) -> tuple[int, ...]:
    """[stage][reduce-scatter, all-gather][source][row column blocks]."""
    return (
        _LAMPORT_STAGES,
        2,
        world_size,
        max_rows * (ATTNRES_HIDDEN_SIZE // world_size),
    )


class IrisAttnResState:
    """Own the push inbox, Lamport mailboxes, per-row generations, and AttnRes
    launch contract."""

    @staticmethod
    def heap_bytes(
        max_numel: int, max_rows: int, world_size: int, dtype: torch.dtype
    ) -> int:
        return allocation_bytes((2, world_size, max_numel), dtype) + allocation_bytes(
            (2, max_rows, world_size), torch.int32
        )

    @staticmethod
    def lamport_heap_bytes(max_rows: int, world_size: int, dtype: torch.dtype) -> int:
        return allocation_bytes(_lamport_mailbox_shape(max_rows, world_size), dtype)

    def __init__(
        self,
        peers: IrisGroup,
        max_numel: int,
        max_rows: int,
        dtype: torch.dtype,
        lamport_max_rows: int = 0,
    ) -> None:
        self._arch = iris_arch_policy()
        self.peers = peers
        self.world_size = peers.world_size
        self.rank = peers.rank
        self.device = peers.device
        self.heap_bases = peers.addresses
        self.max_numel = max_numel
        self.max_rows = max_rows
        self.inbox = peers.ctx.zeros((2, self.world_size, max_numel), dtype=dtype)
        self.epochs = torch.zeros((max_rows,), dtype=torch.int32, device=self.device)
        self.ready_flags = peers.ctx.zeros(
            (2, max_rows, self.world_size), dtype=torch.int32
        )
        offset = self.inbox.data_ptr() - self.heap_bases[self.rank]
        self.peer_inboxes = tuple(base + offset for base in self.heap_bases)
        assert lamport_max_rows <= ATTNRES_LAMPORT_MAX_ROWS
        self.lamport_max_rows = lamport_max_rows
        self.lamport_mailbox: torch.Tensor | None = None
        self.lamport_epochs: torch.Tensor | None = None
        self.peer_lamport_mailboxes: tuple[int, ...] = ()
        if lamport_max_rows:
            self.lamport_mailbox = peers.ctx.empty(
                _lamport_mailbox_shape(lamport_max_rows, self.world_size), dtype=dtype
            )
            # Negative zero marks an empty word.
            self.lamport_mailbox.view(torch.int32).fill_(-2147483648)
            self.lamport_epochs = torch.zeros(
                (lamport_max_rows,), dtype=torch.int32, device=self.device
            )
            offset = self.lamport_mailbox.data_ptr() - self.heap_bases[self.rank]
            self.peer_lamport_mailboxes = tuple(
                base + offset for base in self.heap_bases
            )

    def run(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        score_weight: torch.Tensor,
        output_weight: torch.Tensor,
        scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce a Kimi-K3 attention partial and finish its AttnRes mix."""
        assert attnres_supported(
            partial,
            residual,
            score_weight,
            output_weight,
            scratch,
            world_size=self.world_size,
            device=self.device,
        )
        assert partial.shape[0] <= self.max_rows
        assert partial.numel() <= self.max_numel
        assert eps > 0.0
        num_tokens = partial.shape[0]
        m, s_, acc = scratch

        hidden = torch.empty_like(partial)
        residual_out = torch.empty_like(residual)
        iris_k3attn_push_oneshot[(num_tokens,)](
            partial,
            residual,
            self.inbox,
            score_weight,
            output_weight,
            m,
            s_,
            acc,
            hidden,
            residual_out,
            self.epochs,
            self.ready_flags,
            *self.peer_inboxes,
            *self.heap_bases,
            RANK=self.rank,
            WORLD_SIZE=self.world_size,
            HIDDEN=ATTNRES_HIDDEN_SIZE,
            BLOCK=triton.next_power_of_2(ATTNRES_HIDDEN_SIZE),
            MAX_ELEMENTS=self.max_numel,
            READY_SLOT_STRIDE=self.max_rows * self.world_size,
            EPS=eps,
            ELEMENTS_PER_THREAD=_ELEMENTS_PER_THREAD,
            NUM_WARPS=_NUM_SUBGROUPS,
            SUBGROUP_SIZE=64,
            num_warps=_NUM_SUBGROUPS,
            WAIT_STORES=self._arch.wait_stores,
            WAIT_VMEM=self._arch.wait_vmem,
        )
        return hidden, residual_out

    def lamport(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        score_weight: torch.Tensor,
        output_weight: torch.Tensor,
        scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``run`` through a Lamport two-shot, bit for bit, in one launch."""
        assert attnres_supported(
            partial,
            residual,
            score_weight,
            output_weight,
            scratch,
            world_size=self.world_size,
            device=self.device,
            max_rows=self.lamport_max_rows,
        )
        assert eps > 0.0
        rows = partial.shape[0]
        m, s_, acc = scratch
        hidden = torch.empty_like(partial)
        residual_out = torch.empty_like(residual)
        exchangers = rows * (ATTNRES_HIDDEN_SIZE // self.world_size // _LAMPORT_TILE)
        iris_k3attn_lamport_twoshot[(exchangers + rows,)](
            partial,
            residual,
            self.lamport_mailbox,
            score_weight,
            output_weight,
            m,
            s_,
            acc,
            hidden,
            residual_out,
            self.lamport_epochs,
            *self.peer_lamport_mailboxes,
            exchangers,
            RANK=self.rank,
            WORLD_SIZE=self.world_size,
            HIDDEN=ATTNRES_HIDDEN_SIZE,
            BLOCK=triton.next_power_of_2(ATTNRES_HIDDEN_SIZE),
            TILE=_LAMPORT_TILE,
            MAX_SLOT_ELEMENTS=self.lamport_mailbox.shape[-1],
            NUM_STAGES=_LAMPORT_STAGES,
            EPS=eps,
            ELEMENTS_PER_THREAD=_ELEMENTS_PER_THREAD,
            NUM_WARPS=_NUM_SUBGROUPS,
            SUBGROUP_SIZE=64,
            PEER_SIGNAL_STORE_CACHE=self._arch.peer_signal_store,
            num_warps=_NUM_SUBGROUPS,
        )
        return hidden, residual_out

    def gated_projection_supported(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
        norm_weight: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        score_weight: torch.Tensor,
        output_weight: torch.Tensor,
        scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> bool:
        """Whether ``gated_projection_lamport`` serves these rows."""
        if not (
            current_platform().is_cdna4
            and x.ndim == 2
            and weight.ndim == 2
            and norm_weight.ndim == 1
            and gate.shape == x.shape
            and 0 < x.shape[0] <= self.lamport_max_rows
        ):
            return False
        rows, k = x.shape
        head_dim = norm_weight.numel()
        if not (
            weight.shape == (ATTNRES_HIDDEN_SIZE, k)
            and x.dtype == gate.dtype == weight.dtype == torch.bfloat16
            and x.is_contiguous()
            and gate.stride(1) == 1
            and weight.is_contiguous()
            and norm_weight.is_contiguous()
            and all(
                tensor.device == self.device
                for tensor in (x, gate, norm_weight, weight)
            )
            and head_dim > 0
            and supports_gluon_gated_norm_gemv_a16w16_gfx950(
                rows, ATTNRES_HIDDEN_SIZE, k, k // head_dim, head_dim
            )
        ):
            return False
        columns, num_warps, tiling = gluon_gated_norm_gemv_a16w16_tiling_gfx950(
            rows, ATTNRES_HIDDEN_SIZE, k, self.device
        )
        lanes = 64 * num_warps
        block = triton.next_power_of_2(ATTNRES_HIDDEN_SIZE)
        block_columns = ATTNRES_HIDDEN_SIZE // self.world_size
        # Each tile lies in one owner's block and holds one word a lane; the
        # finisher holds one pack of its row a lane.
        return (
            block_columns % columns == 0
            and columns % 2 == 0
            and tiling["BLOCK_M"] * tiling["BLOCK_N"] == 2 * lanes
            and block % lanes == 0
            and block_columns % (block // lanes) == 0
            and attnres_supported(
                residual,
                residual,
                score_weight,
                output_weight,
                scratch,
                world_size=self.world_size,
                device=self.device,
                max_rows=self.lamport_max_rows,
            )
        )

    def gated_projection_lamport(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
        norm_weight: torch.Tensor,
        weight: torch.Tensor,
        norm_eps: float,
        residual: torch.Tensor,
        score_weight: torch.Tensor,
        output_weight: torch.Tensor,
        scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``lamport`` of ``launch_gluon_gated_norm_gemv_a16w16_gfx950``'s
        partial, in one launch.

        The residual matches bit for bit. The finishers run on the GEMV's four
        warps rather than the two-shot's sixteen, which reorders the hidden
        rows' sums; where that moves the mix across a BF16 rounding boundary,
        a hidden element moves by about a BF16 step of the mix.
        """
        assert self.gated_projection_supported(
            x, gate, norm_weight, weight, residual, score_weight, output_weight, scratch
        )
        assert eps > 0.0
        rows, k = x.shape
        columns, num_warps, tiling = gluon_gated_norm_gemv_a16w16_tiling_gfx950(
            rows, ATTNRES_HIDDEN_SIZE, k, self.device
        )
        block = triton.next_power_of_2(ATTNRES_HIDDEN_SIZE)
        tiles = ATTNRES_HIDDEN_SIZE // columns
        m, s_, acc = scratch
        hidden = torch.empty_like(residual)
        residual_out = torch.empty_like(residual)
        iris_k3attn_gated_projection_twoshot[(tiles + rows,)](
            x,
            gate,
            norm_weight,
            weight,
            residual,
            self.lamport_mailbox,
            score_weight,
            output_weight,
            m,
            s_,
            acc,
            hidden,
            residual_out,
            self.lamport_epochs,
            *self.peer_lamport_mailboxes,
            rows,
            x.stride(0),
            gate.stride(0),
            norm_eps,
            TILES=tiles,
            COLUMNS=columns,
            **tiling,
            RANK=self.rank,
            WORLD_SIZE=self.world_size,
            HIDDEN=ATTNRES_HIDDEN_SIZE,
            BLOCK=block,
            MAX_SLOT_ELEMENTS=self.lamport_mailbox.shape[-1],
            NUM_STAGES=_LAMPORT_STAGES,
            EPS=eps,
            ELEMENTS_PER_THREAD=block // (64 * num_warps),
            PEER_SIGNAL_STORE_CACHE=self._arch.peer_signal_store,
            num_warps=num_warps,
        )
        return hidden, residual_out

    def epilogue(
        self,
        reduced: torch.Tensor,
        residual: torch.Tensor,
        score_weight: torch.Tensor,
        output_weight: torch.Tensor,
        scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Finish ``run`` from an already reduced BF16 partial, bit for bit."""
        assert attnres_supported(
            reduced,
            residual,
            score_weight,
            output_weight,
            scratch,
            world_size=self.world_size,
            device=self.device,
            max_rows=ATTNRES_EPILOGUE_MAX_ROWS,
        )
        assert eps > 0.0
        m, s_, acc = scratch
        hidden = torch.empty_like(reduced)
        residual_out = torch.empty_like(residual)
        iris_k3attn_attnres_rows[(reduced.shape[0],)](
            reduced,
            residual,
            score_weight,
            output_weight,
            m,
            s_,
            acc,
            hidden,
            residual_out,
            HIDDEN=ATTNRES_HIDDEN_SIZE,
            BLOCK=triton.next_power_of_2(ATTNRES_HIDDEN_SIZE),
            EPS=eps,
            ELEMENTS_PER_THREAD=_ELEMENTS_PER_THREAD,
            NUM_WARPS=_NUM_SUBGROUPS,
            SUBGROUP_SIZE=64,
            num_warps=_NUM_SUBGROUPS,
        )
        return hidden, residual_out


@gluon.jit
def iris_k3attn_attnres_rows(
    reduced_ptr,
    residual_ptr,
    score_weight_ptr,
    output_weight_ptr,
    scratch_m_ptr,
    scratch_s_ptr,
    scratch_acc_ptr,
    hidden_ptr,
    residual_out_ptr,
    HIDDEN: gl.constexpr,
    BLOCK: gl.constexpr,
    EPS: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
):
    """``iris_k3attn_push_oneshot``'s epilogue for one row of a BF16 reduction.

    The same operations as ``_iris_attnres_epilogue``, with every operand
    loaded up front: none depends on the reduction.
    """
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(
        [ELEMENTS_PER_THREAD], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    element = gl.arange(0, BLOCK, layout=layout)
    mask = element < HIDDEN
    row_offsets = (row * HIDDEN + element).to(gl.int32)
    weight_offsets = element.to(gl.int32)
    reduced = gl.amd.cdna4.buffer_load(reduced_ptr, row_offsets, mask=mask, other=0.0)
    residual = gl.amd.cdna4.buffer_load(residual_ptr, row_offsets, mask=mask, other=0.0)
    score_weight = gl.amd.cdna4.buffer_load(
        score_weight_ptr, weight_offsets, mask=mask, other=0.0
    )
    block_acc = gl.amd.cdna4.buffer_load(
        scratch_acc_ptr, row_offsets, mask=mask, other=0.0
    )
    output_weight = gl.amd.cdna4.buffer_load(
        output_weight_ptr, weight_offsets, mask=mask, other=0.0
    )
    block_m = gl.load(scratch_m_ptr + row)
    block_s = gl.load(scratch_s_ptr + row)
    _iris_attnres_mix_row(
        reduced,
        residual,
        score_weight,
        block_acc,
        output_weight,
        block_m,
        block_s,
        hidden_ptr,
        residual_out_ptr,
        row_offsets,
        mask,
        HIDDEN,
        EPS,
    )


@gluon.jit
def _iris_attnres_mix_row(
    reduced,
    residual,
    score_weight,
    block_acc,
    output_weight,
    block_m,
    block_s,
    hidden_ptr,
    residual_out_ptr,
    row_offsets,
    mask,
    HIDDEN: gl.constexpr,
    EPS: gl.constexpr,
):
    """``_iris_attnres_epilogue`` on one BF16-reduced row with loaded operands."""
    prefix = (reduced.to(gl.float32) + residual.to(gl.float32)).to(gl.bfloat16)
    gl.amd.cdna4.buffer_store(
        prefix.to(residual_out_ptr.dtype.element_ty),
        residual_out_ptr,
        row_offsets,
        mask=mask,
    )
    prefix = prefix.to(gl.float32)
    square_sum = gl.sum(gl.where(mask, prefix * prefix, 0.0), axis=0)
    dot = gl.sum(gl.where(mask, prefix * score_weight.to(gl.float32), 0.0), axis=0)
    prefix_logit = dot * gl.rsqrt(square_sum / HIDDEN + EPS)
    maximum = gl.maximum(block_m, prefix_logit)
    block_correction = gl.exp(block_m - maximum)
    prefix_weight = gl.exp(prefix_logit - maximum)
    inverse_sum = 1.0 / (block_s * block_correction + prefix_weight)
    mixed = (
        (
            (block_acc.to(gl.float32) * block_correction + prefix_weight * prefix)
            * inverse_sum
        )
        .to(gl.bfloat16)
        .to(gl.float32)
    )
    output_square_sum = gl.sum(gl.where(mask, mixed * mixed, 0.0), axis=0)
    inverse_rms = gl.rsqrt(output_square_sum / HIDDEN + EPS)
    gl.amd.cdna4.buffer_store(
        (mixed * inverse_rms * output_weight.to(gl.float32)).to(
            hidden_ptr.dtype.element_ty
        ),
        hidden_ptr,
        row_offsets,
        mask=mask,
    )


@gluon.jit
def _iris_wait_lamport_sources(slots, offsets, valid, LAYOUT: gl.constexpr):
    """Wait for the BF16 packs at offsets from slots to replace the empty markers.

    ``_iris_wait_lamport_slots`` for packs that each come from one source:
    offsets already select each pack's source slot, so a lane polls one pack
    rather than one per peer. Returns them as BF16 in LAYOUT.
    """
    ROWS: gl.constexpr = offsets.shape[0]
    WORDS: gl.constexpr = offsets.shape[1] // 2
    gl.static_assert(offsets.shape[1] % 2 == 0)
    words_ptr = gl.multiple_of(slots.to(gl.pointer_type(gl.uint32), bitcast=True), 16)
    first, _ = gl.split(gl.reshape(offsets, [ROWS, WORDS, 2]))
    word_offsets = gl.max_contiguous(gl.multiple_of(first // 2, [1, WORDS]), [1, WORDS])
    word_valid = gl.convert_layout(valid, gl.SliceLayout(1, word_offsets.type.layout))
    loaded = gl.zeros_like(word_offsets).to(gl.uint32)
    waiting = _iris_subgroup_any(valid.to(gl.int32))
    while waiting:
        loaded = gl.amd.cdna4.buffer_load(
            words_ptr, word_offsets, mask=word_valid[:, None], cache=".cv"
        )
        # Payloads never hold negative zero, so a word is still empty exactly
        # when its high half is 0x8000.
        empty = (loaded >> 16) == 0x8000
        missing = word_valid & (gl.max(empty.to(gl.int32), 1) != 0)
        waiting = _iris_subgroup_any(missing.to(gl.int32))
    low = loaded.to(gl.uint16).to(gl.bfloat16, bitcast=True)
    high = (loaded >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    return gl.convert_layout(gl.reshape(gl.join(low, high), [ROWS, 2 * WORDS]), LAYOUT)


@gluon.jit(do_not_specialize=["NUM_EXCHANGERS"])
def iris_k3attn_lamport_twoshot(
    partial_ptr,
    residual_ptr,
    region_sym_ptr,
    score_weight_ptr,
    output_weight_ptr,
    scratch_m_ptr,
    scratch_s_ptr,
    scratch_acc_ptr,
    hidden_ptr,
    residual_out_ptr,
    epochs,
    region_0,
    region_1,
    region_2,
    region_3,
    region_4,
    region_5,
    region_6,
    region_7,
    NUM_EXCHANGERS,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    HIDDEN: gl.constexpr,
    BLOCK: gl.constexpr,
    TILE: gl.constexpr,
    MAX_SLOT_ELEMENTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    EPS: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """``iris_k3attn_push_oneshot`` as a Lamport two-shot over row column blocks.

    Rank k owns column block k of every row. Each of the NUM_EXCHANGERS
    programs exchanges TILE columns of one row's blocks: it sends this rank's
    tiles of the other blocks to their owners, sums the owned tile in FP32
    rank order with one BF16 rounding (``iris_lamport_twoshot_rs_wait_reduce``'s
    result), and publishes the sum to every rank's all-gather slot, this
    rank's included. Program NUM_EXCHANGERS + r then waits for row r's
    gathered blocks and finishes the row as ``iris_k3attn_attnres_rows`` does.
    Mailboxes hold [stage][reduce-scatter, all-gather][source][row block].

    A row's exchangers and finisher share its counter. The finisher advances
    it only after every local exchanger of the row, having read it, has
    published, so no program of the launch reads the advanced stage.
    """
    gl.static_assert(WORLD_SIZE == 8)
    gl.static_assert(NUM_STAGES >= 3)
    gl.static_assert(MAX_SLOT_ELEMENTS % 8 == 0)
    program = gl.program_id(0)
    regions = (
        region_0,
        region_1,
        region_2,
        region_3,
        region_4,
        region_5,
        region_6,
        region_7,
    )
    if program < NUM_EXCHANGERS:
        _iris_attnres_twoshot_exchange(
            partial_ptr,
            region_sym_ptr,
            epochs,
            regions,
            program,
            RANK,
            WORLD_SIZE,
            HIDDEN,
            TILE,
            MAX_SLOT_ELEMENTS,
            NUM_STAGES,
            NUM_WARPS,
            SUBGROUP_SIZE,
            PEER_SIGNAL_STORE_CACHE,
        )
    else:
        _iris_attnres_twoshot_finish(
            residual_ptr,
            region_sym_ptr,
            score_weight_ptr,
            output_weight_ptr,
            scratch_m_ptr,
            scratch_s_ptr,
            scratch_acc_ptr,
            hidden_ptr,
            residual_out_ptr,
            epochs,
            program - NUM_EXCHANGERS,
            WORLD_SIZE,
            HIDDEN,
            BLOCK,
            MAX_SLOT_ELEMENTS,
            NUM_STAGES,
            EPS,
            ELEMENTS_PER_THREAD,
            NUM_WARPS,
            SUBGROUP_SIZE,
        )


@gluon.jit
def _iris_attnres_twoshot_exchange(
    partial_ptr,
    region_sym_ptr,
    epochs,
    regions,
    program,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    HIDDEN: gl.constexpr,
    TILE: gl.constexpr,
    MAX_SLOT_ELEMENTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Exchange tile ``program`` of a row's column blocks.

    One subgroup carries the tile, as in ``iris_lamport_twoshot_bf16_allreduce``.
    """
    COLUMNS: gl.constexpr = HIDDEN // WORLD_SIZE
    VECTOR: gl.constexpr = TILE // SUBGROUP_SIZE
    TILES: gl.constexpr = COLUMNS // TILE
    gl.static_assert(TILE == SUBGROUP_SIZE * VECTOR and VECTOR % 2 == 0)
    gl.static_assert(COLUMNS % TILE == 0)
    PHASE: gl.constexpr = WORLD_SIZE * MAX_SLOT_ELEMENTS
    row = program // TILES
    stage = gl.load(epochs + row).to(gl.uint32).to(gl.uint64) % NUM_STAGES
    stage_base = stage * (2 * PHASE)
    # The other subgroups have no lanes.
    lanes: gl.constexpr = gl.BlockedLayout(
        [1, VECTOR], [SUBGROUP_SIZE, 1], [NUM_WARPS, 1], [0, 1]
    )
    lane = gl.arange(0, NUM_WARPS * SUBGROUP_SIZE, layout=gl.SliceLayout(1, lanes))
    element = gl.arange(0, VECTOR, layout=gl.SliceLayout(0, lanes))
    valid = lane < SUBGROUP_SIZE
    mask = valid[:, None] & (element[None, :] < VECTOR)
    column = (program % TILES) * TILE + lane[:, None] * VECTOR + element[None, :]
    slot_offsets = row * COLUMNS + column
    # chunks[delta] is this rank's tile of block (RANK + delta) % WORLD_SIZE.
    chunks = ()
    for delta in gl.static_range(WORLD_SIZE):
        chunks += (
            _iris_sanitize_lamport_bf16(
                gl.amd.cdna4.buffer_load(
                    partial_ptr
                    + row * HIDDEN
                    + ((RANK + delta) % WORLD_SIZE) * COLUMNS,
                    column,
                    mask=mask,
                    other=0.0,
                )
            ),
        )
    _iris_lamport_push(
        chunks[1:],
        regions,
        stage_base + RANK * MAX_SLOT_ELEMENTS,
        slot_offsets,
        mask,
        program,
        RANK,
        WORLD_SIZE,
        PEER_SIGNAL_STORE_CACHE,
    )
    scattered = _iris_wait_lamport_slots(
        region_sym_ptr + stage_base,
        slot_offsets,
        valid,
        RANK,
        WORLD_SIZE,
        MAX_SLOT_ELEMENTS,
        lanes,
    )
    for source in gl.static_range(0, WORLD_SIZE):
        if source == RANK:
            term = chunks[0]
        else:
            term = scattered[(source - RANK + WORLD_SIZE) % WORLD_SIZE - 1]
        if source == 0:
            accumulator = term.to(gl.float32)
        else:
            accumulator += term.to(gl.float32)
    owned = _iris_sanitize_lamport_bf16(accumulator.to(gl.bfloat16))
    published = ()
    for _ in gl.static_range(1, WORLD_SIZE):
        published += (owned,)
    gathered_base = stage_base + PHASE + RANK * MAX_SLOT_ELEMENTS
    _iris_lamport_push(
        published,
        regions,
        gathered_base,
        slot_offsets,
        mask,
        program,
        RANK,
        WORLD_SIZE,
        PEER_SIGNAL_STORE_CACHE,
    )
    gl.amd.cdna4.buffer_store(
        owned, region_sym_ptr + gathered_base, slot_offsets, mask=mask, cache=".wt"
    )
    sentinel = (
        gl.where(slot_offsets % 2 == 0, 0, 0x8000)
        .to(gl.uint16)
        .to(gl.bfloat16, bitcast=True)
    )
    for delta in gl.static_range(1, WORLD_SIZE):
        source = (RANK + delta) % WORLD_SIZE
        gl.amd.cdna4.buffer_store(
            sentinel,
            region_sym_ptr + stage_base + source * MAX_SLOT_ELEMENTS,
            slot_offsets,
            mask=mask,
            cache=".wt",
        )


@gluon.jit
def _iris_attnres_twoshot_finish(
    residual_ptr,
    region_sym_ptr,
    score_weight_ptr,
    output_weight_ptr,
    scratch_m_ptr,
    scratch_s_ptr,
    scratch_acc_ptr,
    hidden_ptr,
    residual_out_ptr,
    epochs,
    row,
    WORLD_SIZE: gl.constexpr,
    HIDDEN: gl.constexpr,
    BLOCK: gl.constexpr,
    MAX_SLOT_ELEMENTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    EPS: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
):
    """Wait for ``row``'s gathered blocks, clear them, and finish the row."""
    COLUMNS: gl.constexpr = HIDDEN // WORLD_SIZE
    BLOCK_PACKS: gl.constexpr = COLUMNS // ELEMENTS_PER_THREAD
    gl.static_assert(COLUMNS % ELEMENTS_PER_THREAD == 0)
    PHASE: gl.constexpr = WORLD_SIZE * MAX_SLOT_ELEMENTS
    stage = gl.load(epochs + row).to(gl.uint32).to(gl.uint64) % NUM_STAGES
    gathered_base = stage * (2 * PHASE) + PHASE
    packed: gl.constexpr = gl.BlockedLayout(
        [1, ELEMENTS_PER_THREAD], [SUBGROUP_SIZE, 1], [NUM_WARPS, 1], [0, 1]
    )
    layout: gl.constexpr = gl.BlockedLayout(
        [ELEMENTS_PER_THREAD], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    pack = gl.arange(0, BLOCK // ELEMENTS_PER_THREAD, layout=gl.SliceLayout(1, packed))
    pack_element = gl.arange(0, ELEMENTS_PER_THREAD, layout=gl.SliceLayout(0, packed))
    valid = pack < HIDDEN // ELEMENTS_PER_THREAD
    pack_mask = valid[:, None] & (pack_element[None, :] < ELEMENTS_PER_THREAD)
    source_offsets = (
        (pack // BLOCK_PACKS)[:, None] * MAX_SLOT_ELEMENTS
        + row * COLUMNS
        + (pack % BLOCK_PACKS)[:, None] * ELEMENTS_PER_THREAD
        + pack_element[None, :]
    ).to(gl.int32)

    element = gl.arange(0, BLOCK, layout=layout)
    mask = element < HIDDEN
    row_offsets = (row * HIDDEN + element).to(gl.int32)
    weight_offsets = element.to(gl.int32)
    # None of the epilogue's operands depends on the reduction.
    residual = gl.amd.cdna4.buffer_load(residual_ptr, row_offsets, mask=mask, other=0.0)
    score_weight = gl.amd.cdna4.buffer_load(
        score_weight_ptr, weight_offsets, mask=mask, other=0.0
    )
    block_acc = gl.amd.cdna4.buffer_load(
        scratch_acc_ptr, row_offsets, mask=mask, other=0.0
    )
    output_weight = gl.amd.cdna4.buffer_load(
        output_weight_ptr, weight_offsets, mask=mask, other=0.0
    )
    block_m = gl.load(scratch_m_ptr + row)
    block_s = gl.load(scratch_s_ptr + row)

    gathered = _iris_wait_lamport_sources(
        region_sym_ptr + gathered_base, source_offsets, valid, packed
    )
    sentinel = (
        gl.where(source_offsets % 2 == 0, 0, 0x8000)
        .to(gl.uint16)
        .to(gl.bfloat16, bitcast=True)
    )
    gl.amd.cdna4.buffer_store(
        sentinel,
        region_sym_ptr + gathered_base,
        source_offsets,
        mask=pack_mask,
        cache=".wt",
    )
    reduced = gl.convert_layout(gl.reshape(gathered, [BLOCK]), layout)
    _iris_attnres_mix_row(
        reduced,
        residual,
        score_weight,
        block_acc,
        output_weight,
        block_m,
        block_s,
        hidden_ptr,
        residual_out_ptr,
        row_offsets,
        mask,
        HIDDEN,
        EPS,
    )
    # Every subgroup read this row's stage before any advances it.
    gl.barrier()
    gl.store(epochs + row, ((stage + 1) % NUM_STAGES).to(gl.int32))


@gluon.jit(do_not_specialize=["M"])
def iris_k3attn_gated_projection_twoshot(
    x_ptr,
    gate_ptr,
    norm_weight_ptr,
    weight_ptr,
    residual_ptr,
    region_sym_ptr,
    score_weight_ptr,
    output_weight_ptr,
    scratch_m_ptr,
    scratch_s_ptr,
    scratch_acc_ptr,
    hidden_ptr,
    residual_out_ptr,
    epochs,
    region_0,
    region_1,
    region_2,
    region_3,
    region_4,
    region_5,
    region_6,
    region_7,
    M,
    stride_xm,
    stride_gm,
    norm_eps,
    TILES: gl.constexpr,
    COLUMNS: gl.constexpr,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NORM_M: gl.constexpr,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    HIDDEN: gl.constexpr,
    BLOCK: gl.constexpr,
    MAX_SLOT_ELEMENTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    EPS: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """``gluon_gated_norm_gemv_a16w16_gfx950`` into ``iris_k3attn_lamport_twoshot``.

    Program t < TILES computes the gated GEMV's tile t, COLUMNS columns of
    every row inside one rank's column block, and exchanges it in place of
    the two-shot's exchangers: a tile of another rank's block goes straight
    to that owner's reduce-scatter slots, and a tile of this rank's block
    waits for the other ranks' copies, sums all eight in FP32 rank order with
    one BF16 rounding, and publishes the sum to every all-gather slot.
    Program TILES + r then finishes row r as the two-shot does, on this
    launch's warps. A row's finisher advances its counter only after every
    rank's sum of the row arrived, so every program that read the counter
    has sent or published by then.
    """
    gl.static_assert(WORLD_SIZE == 8)
    gl.static_assert(NUM_STAGES >= 3)
    gl.static_assert(MAX_SLOT_ELEMENTS % 8 == 0)
    gl.static_assert((HIDDEN // WORLD_SIZE) % COLUMNS == 0 and COLUMNS % 2 == 0)
    program = gl.program_id(0)
    regions = (
        region_0,
        region_1,
        region_2,
        region_3,
        region_4,
        region_5,
        region_6,
        region_7,
    )
    if program < TILES:
        tile = gluon_gated_norm_gemv_tile_gfx950(
            x_ptr,
            gate_ptr,
            norm_weight_ptr,
            weight_ptr,
            program,
            M,
            HIDDEN,
            COLUMNS,
            stride_xm,
            stride_gm,
            norm_eps,
            K,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            NORM_M,
        )
        _iris_attnres_projection_exchange(
            tile.to(gl.bfloat16),
            program,
            M,
            region_sym_ptr,
            epochs,
            regions,
            COLUMNS,
            BLOCK_M,
            BLOCK_N,
            RANK,
            WORLD_SIZE,
            HIDDEN,
            MAX_SLOT_ELEMENTS,
            NUM_STAGES,
            PEER_SIGNAL_STORE_CACHE,
        )
    else:
        _iris_attnres_twoshot_finish(
            residual_ptr,
            region_sym_ptr,
            score_weight_ptr,
            output_weight_ptr,
            scratch_m_ptr,
            scratch_s_ptr,
            scratch_acc_ptr,
            hidden_ptr,
            residual_out_ptr,
            epochs,
            program - TILES,
            WORLD_SIZE,
            HIDDEN,
            BLOCK,
            MAX_SLOT_ELEMENTS,
            NUM_STAGES,
            EPS,
            ELEMENTS_PER_THREAD,
            gl.num_warps(),
            64,
        )


@gluon.jit
def _iris_lamport_words(values):
    """[packs, 2] BF16 as one 32-bit word a pack, its first element low."""
    low, high = gl.split(values.to(gl.uint16, bitcast=True).to(gl.uint32))
    return low | (high << 16)


@gluon.jit
def _iris_attnres_projection_exchange(
    tile_values,
    tile,
    M,
    region_sym_ptr,
    epochs,
    regions,
    COLUMNS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    HIDDEN: gl.constexpr,
    MAX_SLOT_ELEMENTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Send or reduce GEMV tile ``tile``, BF16 ``[BLOCK_M, BLOCK_N]``.

    Peers poll whole 32-bit words, so the tile moves as packs of two
    columns of a row, one a lane, each pack in its row's stage.
    """
    NUM_WARPS: gl.constexpr = gl.num_warps()
    BLOCK_COLUMNS: gl.constexpr = HIDDEN // WORLD_SIZE
    PHASE: gl.constexpr = WORLD_SIZE * MAX_SLOT_ELEMENTS
    PAIRS: gl.constexpr = BLOCK_N // 2
    PACKS: gl.constexpr = BLOCK_M * PAIRS
    pair_layout: gl.constexpr = gl.BlockedLayout(
        [1, 2], [64 // PAIRS, PAIRS], [NUM_WARPS, 1], [1, 0]
    )
    packs: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [NUM_WARPS, 1], [1, 0])
    values = _iris_sanitize_lamport_bf16(
        gl.convert_layout(
            gl.reshape(gl.convert_layout(tile_values, pair_layout), [PACKS, 2]), packs
        )
    )
    pack = gl.arange(0, PACKS, layout=gl.SliceLayout(1, packs))
    element = gl.arange(0, 2, layout=gl.SliceLayout(0, packs))
    row = pack // PAIRS
    pair = pack % PAIRS
    valid = (row < M) & (2 * pair < COLUMNS)
    stage = gl.load(epochs + row, mask=row < M, other=0) % NUM_STAGES
    owner = tile * COLUMNS // BLOCK_COLUMNS
    # Each pack's element offset in a source's reduce-scatter slot.
    slot = (
        stage * (2 * PHASE)
        + row * BLOCK_COLUMNS
        + (tile * COLUMNS - owner * BLOCK_COLUMNS)
        + 2 * pair
    )
    if owner != RANK:
        words = _iris_lamport_words(values)
        word_layout: gl.constexpr = words.type.layout
        for peer in gl.static_range(WORLD_SIZE):
            if peer != RANK:
                if owner == peer:
                    destination = gl.multiple_of(
                        regions[peer].to(gl.pointer_type(gl.uint32)), 16
                    )
                    gl.amd.cdna4.buffer_store(
                        words,
                        destination,
                        gl.convert_layout(
                            (slot + RANK * MAX_SLOT_ELEMENTS) // 2, word_layout
                        ),
                        mask=gl.convert_layout(valid, word_layout),
                        cache=PEER_SIGNAL_STORE_CACHE,
                    )
    else:
        scattered = _iris_wait_lamport_slots(
            region_sym_ptr,
            slot[:, None] + element[None, :],
            valid,
            RANK,
            WORLD_SIZE,
            MAX_SLOT_ELEMENTS,
            packs,
        )
        for source in gl.static_range(0, WORLD_SIZE):
            if source == RANK:
                term = values
            else:
                term = scattered[(source - RANK + WORLD_SIZE) % WORLD_SIZE - 1]
            if source == 0:
                accumulator = term.to(gl.float32)
            else:
                accumulator += term.to(gl.float32)
        words = _iris_lamport_words(
            _iris_sanitize_lamport_bf16(accumulator.to(gl.bfloat16))
        )
        word_layout: gl.constexpr = words.type.layout
        word_valid = gl.convert_layout(valid, word_layout)
        word_slot = gl.convert_layout(slot, word_layout) // 2
        gathered = word_slot + (PHASE + RANK * MAX_SLOT_ELEMENTS) // 2
        rotation = tile % (WORLD_SIZE - 1)
        for first in gl.static_range(WORLD_SIZE - 1):
            if rotation == first:
                for step in gl.static_range(WORLD_SIZE - 1):
                    destination = gl.multiple_of(
                        regions[
                            (RANK + 1 + (first + step) % (WORLD_SIZE - 1)) % WORLD_SIZE
                        ].to(gl.pointer_type(gl.uint32)),
                        16,
                    )
                    gl.amd.cdna4.buffer_store(
                        words,
                        destination,
                        gathered,
                        mask=word_valid,
                        cache=PEER_SIGNAL_STORE_CACHE,
                    )
        local = gl.multiple_of(region_sym_ptr.to(gl.pointer_type(gl.uint32)), 16)
        gl.amd.cdna4.buffer_store(words, local, gathered, mask=word_valid, cache=".wt")
        empty = gl.full(words.shape, -2147483648, gl.int32, word_layout).to(
            gl.uint32, bitcast=True
        )
        for delta in gl.static_range(1, WORLD_SIZE):
            source = (RANK + delta) % WORLD_SIZE
            gl.amd.cdna4.buffer_store(
                empty,
                local,
                word_slot + source * MAX_SLOT_ELEMENTS // 2,
                mask=word_valid,
                cache=".wt",
            )


@gluon.jit
def _iris_attnres_epilogue(
    reduced,
    residual_ptr,
    score_weight_ptr,
    output_weight_ptr,
    scratch_m_ptr,
    scratch_s_ptr,
    scratch_acc_ptr,
    hidden_ptr,
    residual_out_ptr,
    row_offsets,
    weight_offsets,
    mask,
    row,
    HIDDEN: gl.constexpr,
    EPS: gl.constexpr,
):
    """Add the residual, merge history softmax partials, and RMS-normalize."""
    reduced = reduced.to(gl.bfloat16).to(gl.float32)
    residual = gl.amd.cdna4.buffer_load(
        residual_ptr,
        row_offsets,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    prefix = (reduced + residual).to(gl.bfloat16).to(gl.float32)
    gl.amd.cdna4.buffer_store(
        prefix.to(residual_out_ptr.dtype.element_ty),
        residual_out_ptr,
        row_offsets,
        mask=mask,
    )
    score_weight = gl.amd.cdna4.buffer_load(
        score_weight_ptr,
        weight_offsets,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    square_sum = gl.sum(gl.where(mask, prefix * prefix, 0.0), axis=0)
    dot = gl.sum(gl.where(mask, prefix * score_weight, 0.0), axis=0)
    prefix_logit = dot * gl.rsqrt(square_sum / HIDDEN + EPS)
    block_m = gl.load(scratch_m_ptr + row)
    block_s = gl.load(scratch_s_ptr + row)
    maximum = gl.maximum(block_m, prefix_logit)
    block_correction = gl.exp(block_m - maximum)
    prefix_weight = gl.exp(prefix_logit - maximum)
    inverse_sum = 1.0 / (block_s * block_correction + prefix_weight)
    block_acc = gl.amd.cdna4.buffer_load(
        scratch_acc_ptr,
        row_offsets,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    mixed = (
        ((block_acc * block_correction + prefix_weight * prefix) * inverse_sum)
        .to(gl.bfloat16)
        .to(gl.float32)
    )
    output_square_sum = gl.sum(gl.where(mask, mixed * mixed, 0.0), axis=0)
    inverse_rms = gl.rsqrt(output_square_sum / HIDDEN + EPS)
    output_weight = gl.amd.cdna4.buffer_load(
        output_weight_ptr,
        weight_offsets,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    gl.amd.cdna4.buffer_store(
        (mixed * inverse_rms * output_weight).to(hidden_ptr.dtype.element_ty),
        hidden_ptr,
        row_offsets,
        mask=mask,
    )


@gluon.jit
def iris_k3attn_push_oneshot(
    partial_ptr,
    residual_ptr,
    inbox_sym_ptr,
    score_weight_ptr,
    output_weight_ptr,
    scratch_m_ptr,
    scratch_s_ptr,
    scratch_acc_ptr,
    hidden_ptr,
    residual_out_ptr,
    generations,
    ready_flags,
    inbox_0: gl.pointer_type(gl.bfloat16),
    inbox_1: gl.pointer_type(gl.bfloat16),
    inbox_2: gl.pointer_type(gl.bfloat16),
    inbox_3: gl.pointer_type(gl.bfloat16),
    inbox_4: gl.pointer_type(gl.bfloat16),
    inbox_5: gl.pointer_type(gl.bfloat16),
    inbox_6: gl.pointer_type(gl.bfloat16),
    inbox_7: gl.pointer_type(gl.bfloat16),
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    HIDDEN: gl.constexpr,
    BLOCK: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    READY_SLOT_STRIDE: gl.constexpr,
    EPS: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
    WAIT_STORES: gl.constexpr,
    WAIT_VMEM: gl.constexpr,
):
    """Push K3 attn rows into two alternating inboxes and reduce in rank order.

    Add the residual to the BF16 reduction, merge historical AttnRes partials,
    and RMS-normalize the BF16 mixture. Per-row generations guard slot reuse.
    """
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(
        [ELEMENTS_PER_THREAD], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    element = gl.arange(0, BLOCK, layout=layout)
    mask = element < HIDDEN
    row_offsets = (row * HIDDEN + element).to(gl.int32)
    weight_offsets = element.to(gl.int32)
    local = gl.amd.cdna4.buffer_load(
        partial_ptr,
        row_offsets,
        mask=mask,
        other=0.0,
    )
    generation = gl.load(generations + row).to(gl.int32) + 1
    slot = generation & 1
    inbox_slot_offset = slot * WORLD_SIZE * MAX_ELEMENTS
    sync_ready_flags = ready_flags + slot * READY_SLOT_STRIDE
    local_heap = _iris_heap_base(
        RANK,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    for peer_delta in gl.static_range(1, WORLD_SIZE):
        destination = (RANK + peer_delta) % WORLD_SIZE
        destination_inbox = _iris_heap_base(
            destination,
            inbox_0,
            inbox_1,
            inbox_2,
            inbox_3,
            inbox_4,
            inbox_5,
            inbox_6,
            inbox_7,
        )
        gl.amd.cdna4.buffer_store(
            local,
            destination_inbox + inbox_slot_offset + RANK * MAX_ELEMENTS,
            row_offsets,
            mask=mask,
            cache=".wt",
        )
    WAIT_STORES()
    # Wait for every subgroup's stores before signaling readiness to peers.
    gl.barrier()

    _iris_sync_rank_token(
        sync_ready_flags,
        row,
        generation,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        SUBGROUP_SIZE,
        WAIT_VMEM=WAIT_VMEM,
    )
    local_inbox = inbox_sym_ptr + inbox_slot_offset

    if RANK == 0:
        reduced = local.to(gl.float32)
    else:
        reduced = gl.amd.cdna4.buffer_load(
            local_inbox,
            row_offsets,
            mask=mask,
            other=0.0,
            cache=".cg",
        ).to(gl.float32)
    for source in gl.static_range(1, WORLD_SIZE):
        if source == RANK:
            reduced += local.to(gl.float32)
        else:
            reduced += gl.amd.cdna4.buffer_load(
                local_inbox + source * MAX_ELEMENTS,
                row_offsets,
                mask=mask,
                other=0.0,
                cache=".cg",
            ).to(gl.float32)

    gl.store(generations + row, generation)
    _iris_attnres_epilogue(
        reduced,
        residual_ptr,
        score_weight_ptr,
        output_weight_ptr,
        scratch_m_ptr,
        scratch_s_ptr,
        scratch_acc_ptr,
        hidden_ptr,
        residual_out_ptr,
        row_offsets,
        weight_offsets,
        mask,
        row,
        HIDDEN,
        EPS,
    )
