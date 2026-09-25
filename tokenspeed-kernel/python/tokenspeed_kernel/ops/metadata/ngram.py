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

from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import tl, triton


@dataclass(frozen=True)
class NGramHashParams:
    """Borrowed immutable hash constants; no request state or owned buffers."""

    token_map: torch.Tensor
    primes: torch.Tensor
    reciprocals: torch.Tensor
    offsets: torch.Tensor
    multipliers: torch.Tensor
    pad_id: int

    def hash(self, ids: torch.Tensor, previous: torch.Tensor, mask: torch.Tensor):
        """Reference hash for initialization and CPU execution."""
        raw = torch.cat((ids.unsqueeze(-1), previous), dim=-1).long()
        dead = raw == -1
        dead[..., 0] |= ~mask
        mapped = self.token_map[raw.masked_fill(dead, 0)]
        tokens = mapped.masked_fill(dead.long().cumsum(-1) > 0, self.pad_id)
        products = tokens.unsqueeze(-2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for shift in range(1, 4):
            rolling = torch.bitwise_xor(rolling, products[..., shift])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, shift - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


def engram_hash_reciprocals(primes: torch.Tensor) -> torch.Tensor:
    """Build exact uint64 division reciprocals once at initialization.

    Stored as int64 bit patterns for ordinary model-buffer/device handling.
    This initialization helper reads constants on the host; never call it in
    a forward or a captured graph.
    """
    values = primes.flatten().tolist()
    if any(p < 2 for p in values):
        raise ValueError("Engram moduli must be at least two")
    reciprocals = [(1 << 64) // p for p in values]
    return torch.tensor(
        [v if v < (1 << 63) else v - (1 << 64) for v in reciprocals],
        dtype=torch.int64,
        device=primes.device,
    ).reshape(primes.shape)


@triton.jit
def _engram_remainder(value, prime, reciprocal):
    # reciprocal = floor(2**64 / prime). The high product underestimates the
    # quotient by at most one, so one subtraction corrects the remainder.
    # This is exact over uint64; no floating-point reciprocal or approximation.
    quotient = tl.umulhi(value.to(tl.uint64), reciprocal.to(tl.uint64))
    remainder = value - quotient * prime
    return tl.where(remainder >= prime, remainder - prime, remainder)


@triton.jit
def _engram_hash_values(
    current,
    previous,
    live,
    token_map,
    primes,
    reciprocals,
    offsets,
    multipliers,
    layer,
    heads: tl.constexpr,
    pad_id: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    head = tl.arange(0, COLS)
    blocked = (current == -1) | ~live
    token = tl.load(token_map + current, ~blocked, other=pad_id).to(tl.uint64)
    rolling = token * tl.load(multipliers + layer * 4).to(tl.uint64)
    value = tl.full((ROWS, COLS), 0, tl.uint64)
    for shift in tl.static_range(1, 4):
        raw = previous[shift - 1]
        blocked = blocked | (raw == -1)
        token = tl.load(token_map + raw, ~blocked, other=pad_id).to(tl.uint64)
        rolling = rolling ^ (
            token * tl.load(multipliers + layer * 4 + shift).to(tl.uint64)
        )
        value = tl.where((head[None, :] // heads) == shift - 1, rolling[:, None], value)
    column = layer * 3 * heads + head
    prime = tl.load(primes + column, head < 3 * heads, other=1).to(tl.uint64)
    offset = tl.load(offsets + column, head < 3 * heads, other=0).to(tl.uint64)
    reciprocal = tl.load(reciprocals + column, head < 3 * heads, other=0).to(tl.uint64)
    return (
        _engram_remainder(value, prime[None, :], reciprocal[None, :]) + offset[None, :]
    )


@triton.jit
def _ngram_prefix(
    snapshots,
    positions,
    reset,
    slots,
    cache_lengths,
    tail,
    needs_seed,
    request,
    column,
    live,
    vocab: tl.constexpr,
):
    slot = tl.load(slots + request, live, other=0)
    seed = tl.load(reset + request, live, other=0).to(tl.int1) | tl.load(
        needs_seed + slot, live, other=False
    )
    delta = tl.load(cache_lengths + slot, live, other=0).to(tl.int64) - tl.load(
        positions + request, live, other=0
    )
    tl.device_assert(
        ~live | ~seed | ((delta >= 0) & (delta <= 1)),
        "Engram seed snapshot does not cover the accepted input frontier",
    )
    source_col = tl.minimum(tl.maximum(column + 1 - delta, 0), 3)
    initial = tl.load(snapshots + request * 4 + source_col, live & seed, other=-1)
    accepted = tl.load(tail + slot * 3 + column, live & ~seed, other=-1)
    value = tl.where(seed, initial, accepted)
    return tl.where(live & (value >= 0) & (value < vocab), value, -1)


@triton.jit
def _ngram_predecessor(
    ids,
    snapshots,
    positions,
    reset,
    slots,
    cache_lengths,
    tail,
    needs_seed,
    rows,
    request,
    local_row,
    live,
    distance: tl.constexpr,
    vocab: tl.constexpr,
    CHECK_HISTORY: tl.constexpr,
):
    from_batch = live & (local_row >= distance)
    current = tl.load(ids + rows - distance, from_batch, other=-1).to(tl.int64)
    # Most programs in a long packed run need no request history. Skip its
    # pointer chasing and seed checks when every live row has this predecessor.
    if not CHECK_HISTORY or tl.sum((live & ~from_batch).to(tl.int32), 0) > 0:
        prefix = _ngram_prefix(
            snapshots,
            positions,
            reset,
            slots,
            cache_lengths,
            tail,
            needs_seed,
            request,
            tl.minimum(tl.maximum(distance - local_row - 1, 0), 2),
            live & ~from_batch,
            vocab,
        )
        value = tl.where(from_batch, current, prefix)
    else:
        value = current
    return tl.where(live & (value >= 0) & (value < vocab), value, -1)


@triton.jit(do_not_specialize=["capacity", "bs"])
def _prepare_ngram_hash(
    snapshots,
    positions,
    reset,
    slots,
    cache_lengths,
    ends,
    tail,
    needs_seed,
    ids,
    previous,
    token_mask,
    prefix,
    hashes,
    token_map,
    primes,
    reciprocals,
    offsets,
    multipliers,
    total,
    capacity,
    bs,
    uniform_length,
    vocab: tl.constexpr,
    layers: tl.constexpr,
    heads: tl.constexpr,
    pad_id: tl.constexpr,
    UNIFORM: tl.constexpr,
    CHECK_HISTORY: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    layer = tl.program_id(1)
    live = rows < total
    if UNIFORM:
        request = rows // uniform_length
        start = request * uniform_length
    else:
        lo = tl.full((ROWS,), 0, tl.int32)
        hi = tl.full((ROWS,), bs, tl.int32)
        while tl.sum((lo < hi).to(tl.int32), 0) > 0:
            mid = (lo + hi) // 2
            end = tl.load(ends + mid, (lo < hi) & live, other=0)
            right = live & (rows >= end)
            lo = tl.where((lo < hi) & right, mid + 1, lo)
            hi = tl.where((lo < hi) & ~right, mid, hi)
        request = lo
        start = tl.load(ends + request - 1, live & (request > 0), other=0)
    local_row = rows - start
    raw = tl.load(ids + rows, live, other=-1).to(tl.int64)
    valid = live & (raw >= 0) & (raw < vocab)
    lookbacks = ()
    for distance in tl.static_range(1, 4):
        value = _ngram_predecessor(
            ids,
            snapshots,
            positions,
            reset,
            slots,
            cache_lengths,
            tail,
            needs_seed,
            rows,
            request,
            local_row,
            live,
            distance,
            vocab,
            CHECK_HISTORY,
        )
        lookbacks += (value,)
        if layer == 0:
            tl.store(previous + rows * 3 + distance - 1, value, rows < capacity)
            tl.store(
                prefix + request * 3 + distance - 1, value, live & (local_row == 0)
            )
    result = _engram_hash_values(
        raw,
        lookbacks,
        valid,
        token_map,
        primes,
        reciprocals,
        offsets,
        multipliers,
        layer,
        heads,
        pad_id,
        ROWS,
        COLS,
    )
    head = tl.arange(0, COLS)
    tl.store(
        hashes + (rows[:, None] * layers + layer) * 3 * heads + head[None, :],
        result,
        (rows[:, None] < capacity) & (head[None, :] < 3 * heads),
    )
    if layer == 0:
        tl.store(token_mask + rows, valid, rows < capacity)
        # Nonempty requests reuse the first token's lookbacks above. Only empty
        # requests need a separate prefix read for a later zero-accept commit.
        if tl.program_id(0) * ROWS < bs:
            end = tl.load(ends + rows, rows < bs, other=0)
            request_start = tl.load(ends + rows - 1, (rows < bs) & (rows > 0), other=0)
            empty = (rows < bs) & (end == request_start)
            if tl.sum(empty.to(tl.int32), 0) > 0:
                for col in tl.static_range(3):
                    value = _ngram_prefix(
                        snapshots,
                        positions,
                        reset,
                        slots,
                        cache_lengths,
                        tail,
                        needs_seed,
                        rows,
                        col,
                        empty,
                        vocab,
                    )
                    tl.store(prefix + rows * 3 + col, value, empty)


def prepare_ngram_inputs(
    snapshots: torch.Tensor,
    positions: torch.Tensor,
    reset: torch.Tensor,
    slots: torch.Tensor,
    cache_lengths: torch.Tensor,
    request_ends: torch.Tensor,
    tail: torch.Tensor,
    needs_seed: torch.Tensor,
    ids: torch.Tensor,
    previous: torch.Tensor,
    token_mask: torch.Tensor,
    prefix: torch.Tensor,
    hashes: torch.Tensor,
    total: int,
    vocab_size: int,
    uniform_length: int,
    hash_params: NGramHashParams,
) -> None:
    """Resolve raw lookbacks, token masks and hashes in one read-only state pass.

    ``snapshots`` is [requests,4], current token then newest-first history.
    ``positions`` and ``reset`` describe those host snapshots; only reseeded
    slots require a zero/one-token frontier delta. ``request_ends`` contains
    cumulative packed input lengths, including zero-length requests.
    ``tail`` and ``needs_seed`` are runtime-owned and never mutated here.
    ``prefix`` [requests,3] retains the resolved starting history for commit.
    ``previous`` [capacity,3], ``token_mask`` [capacity] and ``hashes``
    [capacity,layers,3*heads] are caller-owned stable outputs. Rows after
    ``total`` are dead padding. Uniform positive request length selects direct
    indexing; zero selects ragged indexing. All modes use this same operation.
    """
    bs = slots.numel()
    if not ids.is_cuda:
        seed = reset.bool() | needs_seed[slots]
        delta = cache_lengths[slots] - positions
        torch._assert_async(
            (~seed | ((delta >= 0) & (delta <= 1))).all(),
            "Engram seed snapshot does not cover the accepted input frontier",
        )
        distances = torch.arange(1, 4, device=ids.device)
        columns = (distances - delta[:, None]).clamp(0, 3)
        prefix.copy_(
            torch.where(seed[:, None], snapshots.gather(1, columns), tail[slots])
        )
        prefix.masked_fill_((prefix < 0) | (prefix >= vocab_size), -1)
        rows = torch.arange(total, device=ids.device)
        requests = torch.searchsorted(request_ends, rows, right=True)
        starts = torch.cat((request_ends.new_zeros(1), request_ends[:-1]))
        local = rows - starts[requests]
        columns = (distances - local[:, None] - 1).clamp(0, 2)
        values = torch.where(
            local[:, None] >= distances,
            ids[(rows[:, None] - distances).clamp_min(0)],
            prefix[requests].gather(1, columns),
        )
        values.masked_fill_((values < 0) | (values >= vocab_size), -1)
        previous[:total].copy_(values)
        token_mask[:total].copy_((ids[:total] >= 0) & (ids[:total] < vocab_size))
        previous[total:].fill_(-1)
        token_mask[total:].zero_()
        hashes.copy_(hash_params.hash(ids[: previous.shape[0]], previous, token_mask))
        return
    capacity = previous.shape[0]
    if max(capacity, bs) == 0:
        return
    layers, _, heads = hash_params.primes.shape
    # Small tiles minimize latency; larger runs amortize programs and skip
    # history reads once all predecessors are inside the packed input.
    rows = 4 if capacity <= 512 else 8 if capacity <= 2048 else 16
    _prepare_ngram_hash[(triton.cdiv(max(capacity, bs), rows), layers)](
        snapshots,
        positions,
        reset,
        slots,
        cache_lengths,
        request_ends,
        tail,
        needs_seed,
        ids,
        previous,
        token_mask,
        prefix,
        hashes,
        hash_params.token_map,
        hash_params.primes,
        hash_params.reciprocals,
        hash_params.offsets,
        hash_params.multipliers,
        total,
        capacity,
        bs,
        uniform_length,
        vocab_size,
        layers,
        heads,
        hash_params.pad_id,
        uniform_length > 0,
        capacity > 2048,
        rows,
        triton.next_power_of_2(3 * heads),
        num_warps=4,
        debug=True,
    )


@triton.jit
def _commit_ngram(
    slots,
    lengths,
    accepted,
    ids,
    previous,
    token_mask,
    prefix,
    tail,
    needs_seed,
    cache_lengths,
    bs: tl.constexpr,
    num_extends: tl.constexpr,
    padding_slot: tl.constexpr,
    capacity: tl.constexpr,
    ROWS: tl.constexpr,
):
    first = tl.program_id(0) * ROWS
    rows = first + tl.arange(0, ROWS)
    cols = tl.arange(0, 4)
    valid = rows < bs
    slot = tl.load(slots + rows, valid, other=padding_slot)
    live = valid & (slot != padding_slot)
    sizes = tl.load(lengths + rows, valid, other=0)
    count = tl.load(accepted + rows, valid, other=0)
    delta = tl.where(rows < num_extends, sizes, count)
    delta = tl.where(live, delta, 0)
    start = tl.full((), 0, tl.int32)
    for chunk in range(0, first, ROWS):
        start += tl.sum(tl.load(lengths + chunk + tl.arange(0, ROWS)))
    last = tl.minimum(
        tl.maximum(start + tl.cumsum(sizes) - sizes + delta - 1, 0), capacity - 1
    )
    advancing = live & (delta > 0)
    raw = tl.load(ids + last, advancing, other=-1).to(tl.int64)
    mask = tl.load(token_mask + last, advancing, other=False)
    current = tl.where(mask, raw, -1)
    past = tl.load(
        previous + last[:, None] * 3 + cols[None, :] - 1,
        advancing[:, None] & (cols[None, :] > 0) & (cols[None, :] < 3),
        other=-1,
    )
    initial = tl.load(
        prefix + rows[:, None] * 3 + cols[None, :],
        live[:, None] & (delta[:, None] == 0) & (cols[None, :] < 3),
        other=-1,
    )
    value = tl.where(
        delta[:, None] > 0,
        tl.where(cols[None, :] == 0, current[:, None], past),
        initial,
    )
    tl.store(
        tail + slot[:, None] * 3 + cols[None, :],
        value,
        live[:, None] & (cols[None, :] < 3),
    )
    tl.store(needs_seed + slot, False, live)
    frontier = tl.load(cache_lengths + slot, live, other=0)
    tl.store(cache_lengths + slot, frontier + delta, live)


def commit_ngram_inputs(
    slots: torch.Tensor,
    lengths: torch.Tensor,
    accepted: torch.Tensor,
    ids: torch.Tensor,
    previous: torch.Tensor,
    token_mask: torch.Tensor,
    prefix: torch.Tensor,
    tail: torch.Tensor,
    needs_seed: torch.Tensor,
    cache_lengths: torch.Tensor,
    num_extends: int,
    padding_slot: int,
) -> None:
    """Commit accepted input history and cache frontier together after verification.

    Extend rows commit their input length; other rows commit accepted counts,
    excluding the sampled bonus. Zero acceptance preserves the prepared starting
    prefix. Padding writes nothing. Preparation's ``prefix``/``previous``/mask
    must remain alive through commit, including after embedding IDs are clamped.
    """
    bs = slots.numel()
    if not ids.is_cuda:
        rows = torch.arange(bs, device=slots.device)
        delta = torch.where(rows < num_extends, lengths, accepted)
        live = slots != padding_slot
        delta = torch.where(live, delta, 0)
        advancing = live & (delta > 0)
        last = (lengths.cumsum(0) - lengths + delta - 1)[advancing]
        current = torch.where(token_mask[last], ids[last], -1)
        value = prefix.clone()
        value[advancing] = torch.cat((current[:, None], previous[last, :2]), dim=1)
        tail[slots[live]] = value[live]
        needs_seed[slots[live]] = False
        cache_lengths[slots[live]] += delta[live]
        return
    if bs:
        rows = min(1024, triton.next_power_of_2(bs))
        _commit_ngram[(triton.cdiv(bs, rows),)](
            slots,
            lengths,
            accepted,
            ids,
            previous,
            token_mask,
            prefix,
            tail,
            needs_seed,
            cache_lengths,
            bs,
            num_extends,
            padding_slot,
            previous.shape[0],
            rows,
            num_warps=4,
        )
