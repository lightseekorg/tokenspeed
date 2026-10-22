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

"""Gluon helpers for current-window KDA correction records."""

from tokenspeed_kernel_amd._triton import gl, gluon, triton


def kda_records_launch_metadata(grid, kernel, args):
    return {"name": kernel.name}


@gluon.jit
def kda_record_interval(
    metadata,
    stamps,
    batch_row,
    layer,
    steps,
    SLOTS,
    CAPACITY: gl.constexpr,
    stamp_layer_stride,
    MAX_STEPS: gl.constexpr,
    WAVE: gl.constexpr,
):
    slot = batch_row.to(gl.int64)
    start = slot * 0
    generation = start
    if metadata is not None:
        slot = gl.load(metadata + batch_row * 3)
        generation = gl.load(metadata + batch_row * 3 + 1)
        start = gl.load(metadata + batch_row * 3 + 2)
    valid = (
        (slot >= 0)
        & (slot < SLOTS)
        & (start >= 0)
        & (generation >= 0)
        & (steps >= 0)
        & (steps <= MAX_STEPS)
        & (steps <= CAPACITY)
    )
    gl.device_assert(valid, "invalid replay interval")
    if stamps is not None:
        # Mask every access when metadata is invalid, before reducing stamps.
        offsets = gl.arange(
            0,
            triton.next_power_of_2(CAPACITY),
            layout=gl.BlockedLayout([1], [WAVE], [gl.num_warps()], [0]),
        )
        positions = start + offsets
        ptr = (
            stamps
            + layer * stamp_layer_stride
            + (slot * CAPACITY + positions % CAPACITY) * 2
        )
        live = valid & (offsets < steps)
        old_pos = gl.load(ptr, mask=live, other=-1)
        old_gen = gl.load(ptr + 1, mask=live, other=-1)
        fresh = (
            gl.sum(
                (live & ((old_pos != positions) | (old_gen != generation))).to(
                    gl.int32
                ),
                0,
            )
            == 0
        )
        gl.device_assert(fresh, "stale replay record")
        valid = valid & fresh
    return slot, start, valid


def validate_kda_records_verify(
    conv_qkv,
    original_qkv,
    raw_g,
    beta,
    state,
    reads,
    correction,
    key_gate,
    metadata,
    stamps,
    query_start_loc,
    a_log,
    bias,
    heads,
    dim,
    window,
    *,
    f_a_out,
    f_b_weight,
):
    """Check the shared records layout before a pointer-based verify launch."""
    import torch

    batch, tokens = reads.numel(), conv_qkv.shape[0]
    if raw_g is None:
        if f_a_out is None or f_b_weight is None:
            raise ValueError(
                "KDA verify requires either a raw gate or both projection inputs"
            )
        if (
            f_a_out.shape != (tokens, dim)
            or f_b_weight.shape != (heads * dim, dim)
            or not f_b_weight.is_contiguous()
        ):
            raise ValueError(
                "KDA gate projection requires [tokens, D] and [heads * D, D]"
            )
        gate_inputs = (f_a_out, f_b_weight)
    else:
        if f_a_out is not None or f_b_weight is not None:
            raise ValueError("A supplied raw gate excludes projection inputs")
        if raw_g.shape != (tokens, heads * dim):
            raise ValueError("KDA raw gate shape does not match the head geometry")
        gate_inputs = (raw_g,)

    if original_qkv is not None:
        if (
            original_qkv.shape != conv_qkv.shape
            or original_qkv.dtype != conv_qkv.dtype
            or original_qkv.device != conv_qkv.device
            or original_qkv.stride(-1) != 1
            or original_qkv.stride(0) < original_qkv.shape[-1]
        ):
            raise ValueError("Original QKV destination must match the input rows")
        destination = original_qkv.untyped_storage()
        begin, end = (
            destination.data_ptr(),
            destination.data_ptr() + destination.nbytes(),
        )
        for source in (conv_qkv, beta, *gate_inputs):
            storage = source.untyped_storage()
            if (
                begin < storage.data_ptr() + storage.nbytes()
                and storage.data_ptr() < end
            ):
                raise ValueError("Original QKV destination must not overlap producers")

    if dim != 128 or window <= 0 or tokens > batch * window:
        raise ValueError("KDA verify needs D=128 and a positive bounded window")
    if conv_qkv.shape != (tokens, 3 * heads * dim) or beta.shape != (tokens, heads):
        raise ValueError("KDA verify producer shapes do not match the head geometry")
    if (
        state.ndim != 4
        or state.shape[1] < heads
        or state.shape[2:] != (dim, dim)
        or state.dtype != torch.float32
    ):
        raise ValueError("KDA checkpoint must be FP32 [page, head, value, key]")
    if (
        correction.ndim != 4
        or correction.shape[1] != heads
        or correction.shape[3] != dim
        or correction.dtype != torch.float32
    ):
        raise ValueError(
            "KDA correction buffer must be FP32 [slot, head, capacity, value]"
        )
    slots, _, capacity, _ = correction.shape
    if (
        slots < batch
        or capacity < window
        or key_gate.shape != (slots, heads, capacity, 2 * dim)
    ):
        raise ValueError(
            "KDA records must cover the runtime batch and candidate window"
        )
    if conv_qkv.dtype not in (torch.bfloat16, torch.float16) or any(
        x.dtype != conv_qkv.dtype for x in (*gate_inputs, beta, key_gate)
    ):
        raise ValueError(
            "KDA producers and K/raw-g records must use the activation dtype"
        )
    if any(
        x.stride(-1) != 1 for x in (conv_qkv, *gate_inputs, beta, correction, key_gate)
    ):
        raise ValueError("KDA producers and records need contiguous channels")
    if query_start_loc.shape != (batch + 1,) or reads.shape != (batch,):
        raise ValueError(
            "KDA query boundaries and checkpoint indices must match the batch"
        )
    if metadata.shape != (batch, 3) or stamps.shape != (slots, capacity, 2):
        raise ValueError("KDA record metadata and stamps have incompatible shapes")
    if any(x.dtype != torch.int64 or not x.is_contiguous() for x in (metadata, stamps)):
        raise ValueError("KDA record metadata and stamps must be contiguous INT64")
    if (
        a_log.shape != (heads,)
        or bias.numel() != heads * dim
        or any(x.dtype != torch.float32 or not x.is_contiguous() for x in (a_log, bias))
    ):
        raise ValueError(
            "KDA gate parameters must be contiguous FP32 with matching heads"
        )
    if any(
        x.device != conv_qkv.device
        for x in (
            *gate_inputs,
            beta,
            state,
            reads,
            correction,
            key_gate,
            metadata,
            stamps,
            query_start_loc,
            a_log,
            bias,
        )
    ):
        raise ValueError("KDA verify inputs must be on the same device")


def validate_kda_records_commit(
    descriptors, groups, reads, writes, accepted, metadata, stamps, window
):
    """Check shared descriptor and record metadata shapes before recovery."""
    import torch

    if (
        descriptors.ndim != 2
        or descriptors.shape[1] != 10
        or descriptors.dtype != torch.uint64
        or not descriptors.is_contiguous()
    ):
        raise ValueError("KDA descriptors must be contiguous uint64 [layers, 10]")
    layers, batch = descriptors.shape[0], accepted.numel()
    if (
        groups.shape != (layers,)
        or accepted.shape != (batch,)
        or reads.ndim != 2
        or reads.shape[1] != batch
        or reads.shape != writes.shape
    ):
        raise ValueError(
            "KDA layer groups, page tables and accepted counts have incompatible shapes"
        )
    if any(
        x.dtype not in (torch.int32, torch.int64) or not x.is_contiguous()
        for x in (groups, reads, writes, accepted)
    ):
        raise ValueError(
            "KDA group/page indices and accepted counts must be contiguous integers"
        )
    if (metadata is None) != (stamps is None):
        raise ValueError("KDA record metadata and stamps must be supplied together")
    if window <= 0:
        raise ValueError("KDA candidate window must be positive")
    tensors = (groups, reads, writes, accepted)
    if metadata is not None:
        if (
            stamps.ndim != 4
            or stamps.shape[0] != layers
            or stamps.shape[-1] != 2
            or stamps.shape[1] < batch
            or stamps.shape[2] < window
            or metadata.shape != (batch, 3)
        ):
            raise ValueError(
                "KDA record metadata must cover all layers, rows and candidate positions"
            )
        if any(
            x.dtype != torch.int64 or not x.is_contiguous() for x in (metadata, stamps)
        ):
            raise ValueError("KDA record metadata and stamps must be contiguous INT64")
        tensors += (metadata, stamps)
    if any(x.device != descriptors.device for x in tensors):
        raise ValueError("KDA recovery metadata must be on the descriptor device")
