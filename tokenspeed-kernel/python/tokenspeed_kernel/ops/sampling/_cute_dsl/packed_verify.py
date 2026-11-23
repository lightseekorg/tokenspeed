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

"""CTA-parallel draws and p/q verification over CSR target rows."""

import math

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import Boolean, Float32, Int32, Int64, Uint32, Uint64, cute, utils
from tokenspeed_kernel.ops.sampling.packed_verify import (
    DRAFT_POS_HASH_MUL,
    RNG_TAG_ACCEPT,
    RNG_TAG_BONUS,
    RNG_TAG_REJECT,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_THREADS = 256
_ITEMS = 4
_TILE = _THREADS * _ITEMS
_RESIDUAL_SCALE = math.exp(-5e-5)


@cute.jit
def uniform(seed: Int64, position: Int64, tag: Uint32):
    """Philox4x32-10, counter=(position.lo, position.hi, tag, 0)."""
    bits = Uint64(seed)
    pos = Uint64(position)
    k0, k1 = Uint32(bits), Uint32(bits >> 32)
    x0, x1, x2, x3 = Uint32(pos), Uint32(pos >> 32), tag, Uint32(0)
    for _ in cutlass.range_constexpr(10):
        p0 = Uint64(x0) * Uint64(0xD2511F53)
        p1 = Uint64(x2) * Uint64(0xCD9E8D57)
        x0, x1, x2, x3 = (
            Uint32(p1 >> 32) ^ x1 ^ k0,
            Uint32(p1),
            Uint32(p0 >> 32) ^ x3 ^ k1,
            Uint32(p0),
        )
        k0 += Uint32(0x9E3779B9)
        k1 += Uint32(0xBB67AE85)
    return Float32(x0 >> 8) * Float32(1.0 / 16777216.0)


@cute.jit
def block_prefix(value: Float32, scratch: cute.Tensor):
    """Exclusive prefix and total for the CTA's 256 scalar values."""
    tid = cute.arch.thread_idx()[0]
    lane, warp = tid % 32, tid // 32
    prefix = value
    # CuTe's default clamp is 31; shfl.up needs a zero lower boundary.
    for shift in cutlass.range_constexpr(5):
        previous = cute.arch.shuffle_sync_up(prefix, 1 << shift, mask_and_clamp=0)
        if lane >= (1 << shift):
            prefix += previous
    exclusive = cute.arch.shuffle_sync_up(prefix, 1, mask_and_clamp=0)
    if lane == 0:
        exclusive = Float32(0)
    if lane == 31:
        scratch[warp] = prefix
    cute.arch.sync_threads()
    if warp == 0:
        warp_sum = Float32(0)
        if lane < 8:
            warp_sum = scratch[lane]
        for shift in cutlass.range_constexpr(3):
            previous = cute.arch.shuffle_sync_up(warp_sum, 1 << shift, mask_and_clamp=0)
            if lane >= (1 << shift):
                warp_sum += previous
        if lane < 8:
            scratch[lane] = warp_sum
    cute.arch.sync_threads()
    if warp > 0:
        exclusive += scratch[warp - 1]
    total = scratch[7]
    cute.arch.sync_threads()
    return exclusive, total


@cute.jit
def block_min(value: Int32, scratch: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    lane, warp = tid % 32, tid // 32
    result = value
    for shift in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_bfly(result, 1 << shift)
        result = min(result, other)
    if lane == 0:
        scratch[warp] = result
    cute.arch.sync_threads()
    if warp == 0:
        result = Int32(2147483647)
        if lane < 8:
            result = scratch[lane]
        for shift in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_bfly(result, 1 << shift)
            result = min(result, other)
        if lane == 0:
            scratch[0] = result
    cute.arch.sync_threads()
    result = scratch[0]
    cute.arch.sync_threads()
    return result


@cute.jit
def row_argmax(
    p: cute.Tensor,
    q: cute.Tensor,
    residual: Boolean,
    sums: cute.Tensor,
    ids: cute.Tensor,
    vocab: cutlass.Constexpr,
):
    tid = cute.arch.thread_idx()[0]
    lane, warp = tid % 32, tid // 32
    best = Float32(-1)
    best_id = Int32(vocab)
    for token in range(tid, vocab, _THREADS):
        value = probability(p, q, Int32(token), residual)
        if value > best:
            best, best_id = value, Int32(token)
    maximum = best
    for shift in cutlass.range_constexpr(5):
        maximum = max(maximum, cute.arch.shuffle_sync_bfly(maximum, 1 << shift))
    if lane == 0:
        sums[warp] = maximum
    cute.arch.sync_threads()
    if warp == 0:
        maximum = Float32(-1)
        if lane < 8:
            maximum = sums[lane]
        for shift in cutlass.range_constexpr(5):
            maximum = max(maximum, cute.arch.shuffle_sync_bfly(maximum, 1 << shift))
        if lane == 0:
            sums[0] = maximum
    cute.arch.sync_threads()
    if best != sums[0]:
        best_id = Int32(vocab)
    cute.arch.sync_threads()
    return block_min(best_id, ids)


@cute.jit
def probability(p: cute.Tensor, q: cute.Tensor, token: Int32, residual: Boolean):
    value = Float32(p[token])
    if residual:
        value = max(value - Float32(_RESIDUAL_SCALE) * Float32(q[token]), Float32(0))
    return value


@cute.jit
def draw(
    p: cute.Tensor,
    q: cute.Tensor,
    residual: Boolean,
    coin: Float32,
    sums: cute.Tensor,
    ids: cute.Tensor,
    vocab: cutlass.Constexpr,
):
    """Two coalesced vocabulary passes; no lane scans the whole vocabulary."""
    tid = cute.arch.thread_idx()[0]
    mass = Float32(0)
    for tile in range(cute.ceil_div(vocab, _TILE)):
        partial = Float32(0)
        for item in cutlass.range_constexpr(_ITEMS):
            token = tile * _TILE + tid * _ITEMS + item
            if token < vocab:
                partial += probability(p, q, token, residual)
        _, tile_mass = block_prefix(partial, sums)
        # Use the same tile reductions and addition order as the CDF pass.
        mass += tile_mass
    threshold = coin * mass
    sampled = Int32(vocab)
    cumulative = Float32(0)
    tile = Int32(0)
    found = Boolean(False)
    while tile < cute.ceil_div(vocab, _TILE) and not found and mass > 0:
        values = cute.make_rmem_tensor(cute.make_layout(_ITEMS), Float32)
        local_sum = Float32(0)
        for item in cutlass.range_constexpr(_ITEMS):
            token = tile * _TILE + tid * _ITEMS + item
            value = Float32(0)
            if token < vocab:
                value = probability(p, q, token, residual)
            values[item] = value
            local_sum += value
        before, tile_sum = block_prefix(local_sum, sums)
        # All lanes take the same branch, keeping CTA barriers converged.
        if cumulative + tile_sum > threshold:
            prefix = cumulative + before
            candidate = Int32(vocab)
            for item in cutlass.range_constexpr(_ITEMS):
                prefix += values[item]
                token = tile * _TILE + tid * _ITEMS + item
                if token < vocab and values[item] > 0 and prefix > threshold:
                    candidate = min(candidate, token)
            sampled = block_min(candidate, ids)
            found = sampled < vocab
        cumulative += tile_sum
        tile += 1
    if not found:
        # Rounded CDF boundaries must stay inside the sampled law's support.
        # A genuinely zero mass uses the target argmax, as in DeepInfer.
        sampled = row_argmax(p, q, residual and mass > 0, sums, ids, vocab)
    return sampled


class SampleProbs:
    def __init__(self, vocab):
        self.vocab = vocab

    @cute.jit
    def __call__(
        self,
        probs: cute.Tensor,
        seeds: cute.Tensor,
        positions: cute.Tensor,
        output: cute.Tensor,
        tag: Uint32,
        stream: cuda.CUstream,
    ):
        self.sample(probs, seeds, positions, output, tag).launch(
            grid=(probs.shape[0], 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
            use_pdl=False,
        )

    @cute.kernel
    def sample(
        self,
        probs: cute.Tensor,
        seeds: cute.Tensor,
        positions: cute.Tensor,
        output: cute.Tensor,
        tag: Uint32,
    ):
        row = Int64(cute.arch.block_idx()[0])
        tid = cute.arch.thread_idx()[0]
        smem = utils.SmemAllocator()
        sums = smem.allocate_tensor(Float32, cute.make_layout(8), byte_alignment=16)
        ids = smem.allocate_tensor(Int32, cute.make_layout(8), byte_alignment=16)
        p = cute.make_tensor(
            probs.iterator + row * Int64(probs.stride[0]),
            cute.make_layout(self.vocab),
        )
        coin = uniform(seeds[row], positions[row], tag)
        token = draw(p, p, Boolean(False), coin, sums, ids, self.vocab)
        if tid == 0:
            output[row] = token


class PackedVerify:
    def __init__(self, vocab, width):
        self.vocab = vocab
        self.width = width

    @cute.jit
    def __call__(
        self,
        predicts: cute.Tensor,
        accept_index: cute.Tensor,
        accept_length: cute.Tensor,
        candidates: cute.Tensor,
        target_probs: cute.Tensor,
        draft_probs: cute.Tensor,
        query_start_loc: cute.Tensor,
        seeds: cute.Tensor,
        positions: cute.Tensor,
        draft_pool_indices: cute.Tensor,
        reject_threshold: Float32,
        stream: cuda.CUstream,
    ):
        self.verify(
            predicts,
            accept_index,
            accept_length,
            candidates,
            target_probs,
            draft_probs,
            query_start_loc,
            seeds,
            positions,
            draft_pool_indices,
            reject_threshold,
        ).launch(
            grid=(candidates.shape[0], 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
            use_pdl=False,
        )

    @cute.kernel
    def verify(
        self,
        predicts: cute.Tensor,
        accept_index: cute.Tensor,
        accept_length: cute.Tensor,
        candidates: cute.Tensor,
        target_probs: cute.Tensor,
        draft_probs: cute.Tensor,
        query_start_loc: cute.Tensor,
        seeds: cute.Tensor,
        positions: cute.Tensor,
        draft_pool_indices: cute.Tensor,
        reject_threshold: Float32,
    ):
        b = Int64(cute.arch.block_idx()[0])
        tid = cute.arch.thread_idx()[0]
        smem = utils.SmemAllocator()
        sums = smem.allocate_tensor(Float32, cute.make_layout(8), byte_alignment=16)
        ids = smem.allocate_tensor(Int32, cute.make_layout(8), byte_alignment=16)
        output_base = b * Int64(self.width)
        for col in range(tid, self.width, _THREADS):
            predicts[output_base + col] = 0
            accept_index[b, col] = -1
        # Output initialization must retire before thread zero publishes tokens.
        cute.arch.sync_threads()
        start = query_start_loc[b]
        count = query_start_loc[b + 1] - start
        if tid == 0:
            accept_length[b] = 0
        if count > 0:
            draft_slot = Int64(draft_pool_indices[b])
            target_base = target_probs.iterator + Int64(start) * Int64(
                target_probs.stride[0]
            )
            draft_base = draft_probs.iterator + draft_slot * Int64(
                draft_probs.stride[0]
            )
            step = Int32(0)
            rejected = Boolean(False)
            sentinel = Boolean(False)
            seed = seeds[b]
            position = positions[b]
            while step < count - 1 and not rejected:
                token = candidates[b, step + 1]
                p_row = cute.make_tensor(
                    target_base + Int64(step) * Int64(target_probs.stride[0]),
                    cute.make_layout(self.vocab),
                )
                q_row = cute.make_tensor(
                    draft_base + Int64(step) * Int64(draft_probs.stride[1]),
                    cute.make_layout(self.vocab),
                )
                p = p_row[token]
                q = q_row[token]
                sentinel = q > reject_threshold
                ratio = Float32(0)
                if q > 0 and not sentinel:
                    ratio = min(Float32(1), p / q)
                elif q == 0 and p > 0:
                    ratio = Float32(1)
                encoded = Int64(
                    Uint64(position) + Uint64(step) * Uint64(DRAFT_POS_HASH_MUL)
                )
                coin = uniform(seed, encoded, Uint32(RNG_TAG_ACCEPT))
                if not sentinel and coin < ratio:
                    if tid == 0:
                        predicts[output_base + step] = token
                    step += 1
                else:
                    rejected = True
            p_row = cute.make_tensor(
                target_base + Int64(step) * Int64(target_probs.stride[0]),
                cute.make_layout(self.vocab),
            )
            # The draw reads this row only for a genuine rejected proposal.
            q_row = cute.make_tensor(
                draft_base + Int64(step) * Int64(draft_probs.stride[1]),
                cute.make_layout(self.vocab),
            )
            residual = rejected and not sentinel
            encoded = Int64(
                Uint64(position) + Uint64(step) * Uint64(DRAFT_POS_HASH_MUL)
            )
            tag = Uint32(RNG_TAG_BONUS)
            if rejected:
                tag = Uint32(RNG_TAG_REJECT)
            coin = uniform(seed, encoded, tag)
            token = draw(p_row, q_row, residual, coin, sums, ids, self.vocab)
            if tid == 0:
                predicts[output_base + step] = token
                accept_length[b] = step + 1
            for col in range(tid, step + 1, _THREADS):
                accept_index[b, col] = Int32(output_base + col)


_COMPILED = {}


class _GraphSafeDLPack:
    def __init__(self, tensor):
        self._tensor = tensor

    def __dlpack__(self, stream=None):
        return self._tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


def _launch(mode, vocab, width, tensors, device, scalar):
    from cutlass.cute.runtime import from_dlpack

    # All extents and non-unit strides are runtime values, even for one row.
    args = tuple(
        from_dlpack(
            _GraphSafeDLPack(tensor.detach()), assumed_align=tensor.element_size()
        ).mark_layout_dynamic(leading_dim=tensor.ndim - 1)
        for tensor in tensors
    )
    args += (scalar,)
    with torch.cuda.device(device):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        key = (device.index, mode, vocab, width)
        compiled = _COMPILED.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm packed sampling before CUDA graph capture")
            factory = (
                SampleProbs(vocab) if mode == "sample" else PackedVerify(vocab, width)
            )
            compiled = cute.compile(factory, *args, stream)
            _COMPILED[key] = compiled
        compiled(*args, stream)


@register_kernel(
    "sampling",
    "sample_probs",
    name="cute_dsl_sample_probs",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0), vendors=frozenset({"nvidia"})
    ),
    priority=Priority.PERFORMANT,
    signatures=format_signatures("probs", "dense", {torch.float32}),
)
def cute_dsl_sample_probs(probs, seeds, positions, output, rng_tag):
    _launch(
        "sample",
        probs.shape[1],
        0,
        (probs, seeds, positions, output),
        probs.device,
        Uint32(rng_tag),
    )


@register_kernel(
    "sampling",
    "verify_packed_chain",
    name="cute_dsl_verify_packed_chain",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0), vendors=frozenset({"nvidia"})
    ),
    priority=Priority.PERFORMANT,
    signatures=format_signatures("target_probs", "dense", {torch.float32}),
)
def cute_dsl_verify_packed_chain(
    predicts,
    accept_index,
    accept_length,
    candidates,
    target_probs,
    draft_probs,
    query_start_loc,
    seeds,
    positions,
    draft_pool_indices,
    reject_draft_prob_threshold,
):
    _launch(
        "verify",
        target_probs.shape[1],
        candidates.shape[1],
        (
            predicts,
            accept_index,
            accept_length,
            candidates,
            target_probs,
            draft_probs,
            query_start_loc,
            seeds,
            positions,
            draft_pool_indices,
        ),
        target_probs.device,
        Float32(reject_draft_prob_threshold),
    )
