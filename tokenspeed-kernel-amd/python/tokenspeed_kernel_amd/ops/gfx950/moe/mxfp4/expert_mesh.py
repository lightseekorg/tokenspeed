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

"""Token/expert mesh sorting with output initialization in the scatter launch."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit(do_not_specialize=("SIZE",))
def _clear_mesh(mesh, SIZE):
    x = gl.arange(0, 4096, layout=gl.BlockedLayout([4], [64], [16], [0]))
    for start in range(gl.program_id(0) * 4096, SIZE, gl.num_programs(0) * 4096):
        gl.store(mesh + start + x, 0, mask=start + x < SIZE)


@gluon.jit(do_not_specialize=("M", "PITCH"))
def _build_small_mesh(
    ids,
    mesh,
    counts,
    M,
    PITCH,
    TOPK: gl.constexpr,
    START: gl.constexpr,
    LOCAL_E: gl.constexpr,
    ID_M: gl.constexpr,
    ID_K: gl.constexpr,
):
    expert = gl.program_id(0)
    if expert >= START and expert < START + LOCAL_E:
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [4, 1], [1, 0])
        token = gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
        slot = gl.arange(
            0, triton.next_power_of_2(TOPK), layout=gl.SliceLayout(0, layout)
        )
        route = gl.load(
            ids + token[:, None].to(gl.int64) * ID_M + slot[None, :] * ID_K,
            mask=(token[:, None] < M) & (slot[None, :] < TOPK),
            other=-1,
        )
        # Each CTA owns one expert. Distinct slot bits preserve repeated IDs
        # without atomics or a preceding clear, including the sign bit at slot31.
        matches = (route == expert) & (token[:, None] < M) & (slot[None, :] < TOPK)
        bits = gl.sum(gl.where(matches, 1 << slot[None, :], 0).to(gl.uint32), 1)
        gl.store(mesh + expert.to(gl.int64) * PITCH + token, bits, mask=token < PITCH)
        count = gl.sum(gl.sum(matches.to(gl.int32), 1), 0)
        gl.store(counts + expert, count)
    else:
        # The scatter phase never reads the mesh for an empty remote expert.
        gl.store(counts + expert, 0)


@gluon.jit(do_not_specialize=("M", "PITCH"))
def _fill_mesh(
    ids,
    mesh,
    M,
    TOPK: gl.constexpr,
    E: gl.constexpr,
    PITCH,
    WORDS: gl.constexpr,
    ID_M: gl.constexpr,
    ID_K: gl.constexpr,
):
    x = gl.arange(0, 1024, layout=gl.BlockedLayout([4], [64], [4], [0]))
    for start in range(gl.program_id(0) * 1024, M * TOPK, gl.num_programs(0) * 1024):
        route = start + x
        expert = gl.load(
            ids + (route // TOPK).to(gl.int64) * ID_M + (route % TOPK) * ID_K,
            mask=route < M * TOPK,
            other=-1,
        )
        valid = (route < M * TOPK) & (expert >= 0) & (expert < E)
        slot = route % TOPK
        offset = (expert.to(gl.int64) * PITCH + route // TOPK) * WORDS + slot // 32
        gl.atomic_or(
            mesh + offset, (1 << (slot % 32)).to(gl.uint32), mask=valid, sem="relaxed"
        )


@gluon.jit(do_not_specialize=("M", "PITCH"))
def _count_mesh(
    mesh,
    counts,
    M,
    PITCH,
    START: gl.constexpr,
    LOCAL_E: gl.constexpr,
    WORDS: gl.constexpr,
):
    expert = gl.program_id(0)
    if expert >= START and expert < START + LOCAL_E:
        x = gl.arange(0, 1024, layout=gl.BlockedLayout([4], [64], [4], [0]))
        count = 0
        for start in range(0, M * WORDS, 1024):
            bits = gl.load(
                mesh + expert.to(gl.int64) * PITCH * WORDS + start + x,
                mask=start + x < M * WORDS,
                other=0,
            )
            count += gl.sum(gl.extra.libdevice.popc(bits.to(gl.int32)), 0)
        gl.store(counts + expert, count)
    else:
        gl.store(counts + expert, 0)


@gluon.jit(do_not_specialize=("M", "PITCH", "OUT_SIZE"))
def _scatter_mesh(
    mesh,
    counts,
    weights,
    sorted_ids,
    sorted_weights,
    expert_ids,
    valid_ids,
    out,
    M,
    OUT_SIZE,
    TOPK: gl.constexpr,
    E: gl.constexpr,
    LOCAL_E: gl.constexpr,
    START: gl.constexpr,
    D: gl.constexpr,
    OUT_STRIDE: gl.constexpr,
    PITCH,
    E_PAD: gl.constexpr,
    BM: gl.constexpr,
    WORDS: gl.constexpr,
    WEIGHT_M: gl.constexpr,
    WEIGHT_K: gl.constexpr,
):
    pid = gl.program_id(0)
    if pid >= E:
        layout: gl.constexpr = gl.BlockedLayout([8], [64], [4], [0])
        x = gl.arange(0, 2048, layout=layout)
        for start in range((pid - E) * 2048, OUT_SIZE, (gl.num_programs(0) - E) * 2048):
            index = start + x
            gl.store(
                out + (index // D).to(gl.int64) * OUT_STRIDE + index % D,
                0.0,
                mask=index < OUT_SIZE,
            )
    else:
        layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
        expert = gl.arange(0, E_PAD, layout=layout)
        count = gl.load(counts + expert, mask=expert < E, other=0)
        padded = gl.cdiv(count, BM) * BM
        prefix = gl.associative_scan(padded, 0, _add)
        start_slot = gl.sum(gl.where(expert == pid, prefix - padded, 0), 0)
        total = gl.sum(padded, 0)
        own_count = gl.sum(gl.where(expert == pid, count, 0), 0)
        if pid == 0:
            gl.store(valid_ids, total)
            gl.store(valid_ids + 1, M)
        if pid >= START and pid < START + LOCAL_E and own_count > 0:
            for block in range(0, gl.cdiv(own_count, BM)):
                gl.store(expert_ids + start_slot // BM + block, pid - START)
            x = gl.arange(0, 1024, layout=gl.BlockedLayout([4], [64], [4], [0]))
            written = 0
            for start in range(0, M * WORDS, 1024):
                word = start + x
                token = word // WORDS
                bits = gl.load(
                    mesh + pid.to(gl.int64) * PITCH * WORDS + word,
                    mask=word < M * WORDS,
                    other=0,
                )
                routes = gl.extra.libdevice.popc(bits.to(gl.int32))
                ranks = gl.associative_scan(routes, 0, _add) - routes
                for repeat in range(gl.max(routes, 0)):
                    bit = gl.inline_asm_elementwise(
                        "v_ffbl_b32 $0, $1",
                        constraints="=v,v",
                        args=[bits],
                        dtype=gl.int32,
                        is_pure=True,
                        pack=1,
                    )
                    slot = word % WORDS * 32 + bit
                    valid = bits != 0
                    weight = gl.load(
                        weights + token.to(gl.int64) * WEIGHT_M + slot * WEIGHT_K,
                        mask=valid,
                        other=0.0,
                    ).to(gl.float32)
                    destination = start_slot + written + ranks + repeat
                    gl.store(sorted_ids + destination, (slot << 24) | token, mask=valid)
                    gl.store(sorted_weights + destination, weight, mask=valid)
                    bits &= bits - 1
                written += gl.sum(routes, 0)
            pad = own_count + gl.arange(0, BM, layout=layout)
            gl.store(
                sorted_ids + start_slot + pad,
                (TOPK << 24) | M,
                mask=pad < gl.cdiv(own_count, BM) * BM,
            )


def sort_expert_slots(
    ids: torch.Tensor,
    weights: torch.Tensor,
    out: torch.Tensor,
    *,
    global_experts: int,
    local_experts: int,
    expert_start: int,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort route slots, preserving repeated experts, and initialize output."""
    if block_m not in (32, 128):
        raise ValueError("expert mesh blocks must contain 32 or 128 rows")
    m, topk = ids.shape
    pitch = triton.cdiv(m, 32) * 32
    capacity = m * topk + global_experts * block_m - min(topk, global_experts)
    if capacity >= 2**31:
        raise ValueError("expert route capacity exceeds the int32 prefix range")
    words = triton.cdiv(topk, 32)
    mesh = torch.empty(
        (global_experts, pitch, words), dtype=torch.uint32, device=ids.device
    )
    counts = torch.empty((global_experts,), dtype=torch.int32, device=ids.device)
    sorted_ids = torch.empty((capacity,), dtype=torch.int32, device=ids.device)
    sorted_weights = torch.empty((capacity,), dtype=torch.float32, device=ids.device)
    expert_ids = torch.empty(
        (triton.cdiv(capacity, block_m),), dtype=torch.int32, device=ids.device
    )
    valid_ids = torch.empty((2,), dtype=torch.int32, device=ids.device)
    cus = torch.cuda.get_device_properties(ids.device).multi_processor_count
    if m <= 64 and topk <= 32:
        _build_small_mesh[(global_experts,)](
            ids,
            mesh,
            counts,
            m,
            pitch,
            topk,
            expert_start,
            local_experts,
            ids.stride(0),
            ids.stride(1),
            num_warps=4,
        )
    else:
        _clear_mesh[(cus,)](mesh, mesh.numel(), num_warps=16)
        _fill_mesh[(2 * cus,)](
            ids,
            mesh,
            m,
            topk,
            global_experts,
            pitch,
            words,
            ids.stride(0),
            ids.stride(1),
            num_warps=4,
        )
        _count_mesh[(global_experts,)](
            mesh, counts, m, pitch, expert_start, local_experts, words, num_warps=4
        )
    _scatter_mesh[(global_experts + 2 * cus,)](
        mesh,
        counts,
        weights,
        sorted_ids,
        sorted_weights,
        expert_ids,
        valid_ids,
        out,
        m,
        # Keep wide flat counts in host arithmetic before runtime type binding.
        out.numel(),
        topk,
        global_experts,
        local_experts,
        expert_start,
        out.shape[1],
        out.stride(0),
        pitch,
        triton.next_power_of_2(global_experts),
        block_m,
        words,
        weights.stride(0),
        weights.stride(1),
        num_warps=4,
    )
    return sorted_ids, sorted_weights, expert_ids, valid_ids
