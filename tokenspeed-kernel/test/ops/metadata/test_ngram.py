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

import pytest
import torch
from tokenspeed_kernel.ops.metadata.ngram import NGramHashParams, prepare_ngram_inputs


def hash_params(device):
    from tokenspeed_kernel.ops.metadata.ngram import engram_hash_reciprocals

    primes = torch.tensor([101, 103, 107, 109, 113, 127] * 2, device=device).reshape(
        2, 3, 2
    )
    return NGramHashParams(
        torch.arange(128, device=device).flip(0),
        primes,
        engram_hash_reciprocals(primes),
        primes.flatten(1).cumsum(-1) - primes.flatten(1),
        torch.tensor([[13, 17, 23, 29], [31, 37, 41, 43]], device=device),
        2,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "bs,max_length", [(1, 1), (16, 1), (16, 4), (16, 256), (128, 3), (3, 2731)]
)
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("uniform", [False, True])
def test_ngram_preparation(bs, max_length, graph, device, uniform):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if graph and device == "cpu":
        pytest.skip("CUDA graph")
    generator = torch.Generator().manual_seed(1043317292)
    width, vocab, pool = 3, 128, bs + 3
    slots = torch.randperm(pool, generator=generator)[:bs]
    lengths = torch.randint(
        0, max_length + 1, (bs,), generator=generator, dtype=torch.int32
    )
    lengths[0] = max_length
    if uniform:
        lengths.fill_(max_length)
    total = int(lengths.sum())
    capacity = total + 137
    snapshots = torch.randint(-3, vocab + 3, (bs, width + 1), generator=generator)
    positions = torch.full((bs,), 100, dtype=torch.int64)
    reset = torch.arange(bs) % 3 == 0
    needs_seed = torch.arange(pool) % 2 == 0
    cache = torch.full((pool,), 200, dtype=torch.int32)
    for row, slot in enumerate(slots):
        if reset[row] or needs_seed[slot]:
            cache[slot] = 100 + row % 2
    tail = torch.randint(-3, vocab + 3, (pool, width), generator=generator)
    ids = torch.randint(
        -3, vocab + 3, (capacity,), generator=generator, dtype=torch.int32
    )
    previous = torch.full((capacity, width), 777, dtype=torch.int64)
    mask = torch.ones(capacity, dtype=torch.bool)
    inputs = [
        snapshots,
        positions,
        reset,
        slots,
        cache,
        lengths.cumsum(0),
        tail,
        needs_seed,
        ids,
        previous,
        mask,
    ]
    gpu = [t.to(device).clone() for t in inputs]
    prefix = torch.empty((bs, 3), dtype=torch.int64, device=device)
    hashes = torch.empty((capacity, 2, 6), dtype=torch.int64, device=device)
    expected_prefix = torch.empty((bs, 3), dtype=torch.int64)
    params = hash_params(device)
    expected_previous = torch.full_like(previous, -1)
    expected_mask = torch.zeros_like(mask)
    start = 0
    for row, slot in enumerate(slots.tolist()):
        seed = bool(reset[row] or needs_seed[slot])
        delta = int(cache[slot] - positions[row])
        initial = [
            (
                int(snapshots[row, min(max(d - delta, 0), width)])
                if seed
                else int(tail[slot, d - 1])
            )
            for d in range(1, width + 1)
        ]
        initial = [v if 0 <= v < vocab else -1 for v in initial]
        expected_prefix[row] = torch.tensor(initial)
        for local in range(int(lengths[row])):
            values = [
                int(ids[start + local - d]) if local >= d else initial[d - local - 1]
                for d in range(1, width + 1)
            ]
            expected_previous[start + local] = torch.tensor(
                [v if 0 <= v < vocab else -1 for v in values]
            )
            expected_mask[start + local] = 0 <= ids[start + local] < vocab
        start += int(lengths[row])
    uniform_length = max_length if uniform else 0
    prepare_ngram_inputs(*gpu, prefix, hashes, total, vocab, uniform_length, params)
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            prepare_ngram_inputs(
                *gpu, prefix, hashes, total, vocab, uniform_length, params
            )
        for target, source in zip(gpu, inputs):
            target.copy_(source)
        capture.replay()
    for actual, expected in [
        (gpu[6], tail),
        (gpu[7], needs_seed),
        (prefix, expected_prefix),
        (hashes, hash_params("cpu").hash(ids, expected_previous, expected_mask)),
        (gpu[9], expected_previous),
        (gpu[10], expected_mask),
    ]:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("num_extends", [0, 2, 4])
def test_commit_ngram_independent_reference(graph, num_extends, device):
    from tokenspeed_kernel.ops.metadata.ngram import commit_ngram_inputs

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if graph and device == "cpu":
        pytest.skip("CUDA graph")
    lengths = torch.tensor([3, 2, 4, 1], dtype=torch.int32, device=device)
    slots = torch.tensor([3, 0, 1, 4], device=device)
    accepted = torch.tensor([0, 1, 2, 0], dtype=torch.int32, device=device)
    ids = torch.arange(10, dtype=torch.int32, device=device)
    previous = torch.arange(30, dtype=torch.int64, device=device).reshape(10, 3)
    mask = ids % 2 == 0
    tail = torch.full((5, 3), 91, device=device, dtype=torch.int64)
    prefix = torch.arange(12, dtype=torch.int64, device=device).reshape(4, 3)
    needs_seed = torch.ones(5, dtype=torch.bool, device=device)
    cache = torch.arange(5, dtype=torch.int32, device=device)

    def invoke():
        commit_ngram_inputs(
            slots,
            lengths,
            accepted,
            ids,
            previous,
            mask,
            prefix,
            tail,
            needs_seed,
            cache,
            num_extends,
            4,
        )

    invoke()
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            invoke()
    for shift in [0, 100]:
        ids.copy_(torch.arange(10, device=device) + shift)
        tail.fill_(91)
        cache.copy_(torch.arange(5, dtype=torch.int32, device=device))
        ref_tail, ref_cache = tail.cpu().clone(), cache.cpu().clone()
        pos = 0
        for i, (slot, length, count) in enumerate(
            zip(slots.tolist(), lengths.tolist(), accepted.tolist())
        ):
            delta = length if i < num_extends else count
            if slot != 4:
                if delta:
                    last = pos + delta - 1
                    ref_tail[slot, 0] = int(ids[last]) if bool(mask[last]) else -1
                    ref_tail[slot, 1:] = previous[last, :2].cpu()
                else:
                    ref_tail[slot] = prefix[i].cpu()
                ref_cache[slot] += delta
            pos += length
        capture.replay() if graph else invoke()
        torch.testing.assert_close(tail.cpu(), ref_tail, rtol=0, atol=0)
        torch.testing.assert_close(cache.cpu(), ref_cache, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("bs", [0, 2])
def test_empty_input_seeds_only_on_commit(device, bs):
    from tokenspeed_kernel.ops.metadata.ngram import commit_ngram_inputs

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    slots = torch.arange(bs, device=device)
    lengths = torch.zeros(bs, dtype=torch.int32, device=device)
    snapshots = torch.arange(bs * 4, device=device).reshape(bs, 4)
    positions = torch.zeros(bs, dtype=torch.int64, device=device)
    reset = torch.zeros_like(positions)
    cache = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    tail = torch.full((bs + 1, 3), 91, dtype=torch.int64, device=device)
    needs_seed = torch.ones(bs + 1, dtype=torch.bool, device=device)
    ids = torch.empty(0, dtype=torch.int32, device=device)
    previous = torch.empty((0, 3), dtype=torch.int64, device=device)
    mask = torch.empty(0, dtype=torch.bool, device=device)
    prefix = torch.empty((bs, 3), dtype=torch.int64, device=device)
    hashes = torch.empty((0, 2, 6), dtype=torch.int64, device=device)
    prepare_ngram_inputs(
        snapshots,
        positions,
        reset,
        slots,
        cache,
        lengths.cumsum(0),
        tail,
        needs_seed,
        ids,
        previous,
        mask,
        prefix,
        hashes,
        0,
        128,
        0,
        hash_params(device),
    )
    assert (tail == 91).all() and needs_seed.all()
    commit_ngram_inputs(
        slots,
        lengths,
        lengths,
        ids,
        previous,
        mask,
        prefix,
        tail,
        needs_seed,
        cache,
        0,
        bs,
    )
    torch.testing.assert_close(tail[:bs], snapshots[:, 1:])
    assert not needs_seed[:bs].any()
    assert needs_seed[bs] and (tail[bs] == 91).all()
    assert (cache == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_exact_remainder_at_uint64_boundaries():
    import random

    from tokenspeed_kernel._triton import tl, triton
    from tokenspeed_kernel.ops.metadata.ngram import (
        _engram_remainder,
        engram_hash_reciprocals,
    )

    @triton.jit
    def run(values, primes, reciprocals, out, size: tl.constexpr):
        row = tl.program_id(0) * 128 + tl.arange(0, 128)
        value = tl.load(values + row, row < size, other=0).to(tl.uint64)
        prime = tl.load(primes + row, row < size, other=2).to(tl.uint64)
        reciprocal = tl.load(reciprocals + row, row < size, other=0).to(tl.uint64)
        tl.store(out + row, _engram_remainder(value, prime, reciprocal), row < size)

    rng = random.Random(1043317292)
    values, moduli = [], []
    for prime in [2, 3, 5, 101, 16000057, 2147483647, 9223372036854775783]:
        largest_multiple = ((1 << 64) - 1) // prime * prime
        points = [
            0,
            1,
            prime - 1,
            prime,
            prime + 1,
            (1 << 63) - 1,
            1 << 63,
            (1 << 64) - 1,
            largest_multiple - 1,
            largest_multiple,
        ]
        points += [rng.getrandbits(64) for _ in range(2048)]
        values.extend(points)
        moduli.extend([prime] * len(points))
    # Signed storage preserves every uint64 bit pattern on the host and GPU.
    inputs = torch.tensor(
        [v if v < (1 << 63) else v - (1 << 64) for v in values],
        dtype=torch.int64,
        device="cuda",
    )
    primes = torch.tensor(moduli, dtype=torch.int64, device="cuda")
    reciprocals = engram_hash_reciprocals(primes)
    output = torch.empty_like(inputs)
    run[(triton.cdiv(len(values), 128),)](
        inputs, primes, reciprocals, output, len(values)
    )
    assert output.cpu().tolist() == [v % p for v, p in zip(values, moduli)]
