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
from tokenspeed_kernel.ops.ple import ple_host_gather
from tokenspeed_kernel.ops.ple._host_cache_triton import (
    _ple_fill_staged_kernel,
    _ple_gather_staged_tail_kernel,
)
from tokenspeed_kernel.ops.ple.host_cache import PLEHostCache
from utils import assert_no_triton_compile


@pytest.mark.parametrize("vocab_start", [0, 113, (1 << 33) + 113])
@pytest.mark.parametrize("dim", [160, 512])
@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_ple_host_cache_fill_and_gather(vocab_start, dim, table_dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device)[0] not in (9, 10):
        pytest.skip("requires Hopper or Blackwell")

    slots = 32
    vocab = 509
    vocab_end = vocab_start + vocab
    scale = 0.125 if table_dtype == torch.float8_e4m3fn else None
    table = (
        ((torch.arange(512 * dim).reshape(512, dim) % 127 - 63) / 7)
        .to(table_dtype)
        .pin_memory()
    )
    cache = PLEHostCache.create(
        slot_count=slots,
        table=table,
        ngram_heads=PLEHostCache.HEADS,
        output_dtype=torch.bfloat16,
        vocab_start=vocab_start,
        vocab_end=vocab_end,
        device=device,
    )
    assert cache is not None
    tags = torch.full((slots,), -1, dtype=torch.int32)
    table_bytes = table.view(torch.uint8)
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    ready = torch.cuda.Event()
    for step, tokens in enumerate((256, 261, 277, 512)):
        ids = vocab_start + torch.arange(tokens * 16).reshape(tokens, 16) % (
            83 + step * 7
        )
        ids[31:34] = vocab_start
        ids[-1, 0] = vocab_end - 1
        ids[-2, 0] = vocab_start + 12
        if step % 2:
            ids[1] = -1
            ids[2] = vocab_start + (1 << 32)
            ids[:, 3] = vocab_start - 1
            ids[:, 4] = vocab_end
        flat = ids.flatten()
        local = flat - vocab_start
        valid = (local >= 0) & (local < vocab)
        slot = ((local * 2654435761) & 0xFFFFFFFF) >> (32 - 5)
        miss = valid & (tags[slot] != local)
        winners = torch.full((slots,), -1, dtype=torch.int64)
        winners.scatter_reduce_(
            0,
            slot[miss],
            torch.arange(flat.numel())[miss],
            reduce="amax",
            include_self=True,
        )
        filled = winners >= 0
        tags[filled] = local[winners[filled]].to(torch.int32)
        expected_tags = tags.clone()
        ids = ids.to(device)
        out = torch.empty((flat.numel(), dim), dtype=torch.bfloat16, device=device)
        ref = torch.empty_like(out)
        stream = streams[step % 2]
        stream.wait_stream(torch.cuda.current_stream())
        if step:
            stream.wait_event(ready)
        with torch.cuda.stream(stream):
            if step < 2:
                cache.gather(table, ids, out, scale=scale)
            else:
                with assert_no_triton_compile(
                    _ple_gather_staged_tail_kernel, _ple_fill_staged_kernel
                ):
                    cache.gather(table, ids, out, scale=scale)
            ple_host_gather(
                table,
                ids,
                ref,
                vocab_start,
                vocab_end,
                scale,
                None,
                table_shards=1 if vocab_start == 0 else 4,
            )
            ready.record(stream)
        stream.synchronize()
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
        cache._fill_done.synchronize()
        assert torch.equal(cache._tags.cpu(), expected_tags)
        populated = expected_tags >= 0
        assert torch.equal(
            cache._hot.view(torch.uint8).cpu()[populated],
            table_bytes[expected_tags[populated].to(torch.int64)],
        )

    cache.invalidate()
    table.view(torch.uint8).fill_(0)
    out = torch.empty_like(out)
    cache.gather(table, ids, out, scale=scale)
    assert torch.count_nonzero(out).item() == 0
