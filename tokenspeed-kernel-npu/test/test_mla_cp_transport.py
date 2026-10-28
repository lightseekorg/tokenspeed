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

"""Run with torchrun: reuse the DSA transport with only the MLA KV plane."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from tokenspeed_kernel_npu.ops.dsa import AscendDSAPrefill


def bits(rows, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        -32768, 32768, (rows, 576), generator=generator, dtype=torch.int16
    ).view(torch.bfloat16)


def exact(actual, expected):
    torch.testing.assert_close(
        actual.cpu().contiguous().view(torch.int16),
        expected.contiguous().view(torch.int16),
        rtol=0,
        atol=0,
    )


def main():
    rank = int(os.environ["RANK"])
    degree = int(os.environ["WORLD_SIZE"])
    torch.set_num_threads(1)
    torch.npu.set_device(rank)
    dist.init_process_group("gloo")
    page, local_rows = 128, 2176
    blocks = degree * 16 + 1
    cache = torch.empty((local_rows, 576), dtype=torch.bfloat16, device="npu")
    runtime = AscendDSAPrefill(
        cache,
        None,
        ranks=tuple(range(degree)),
        bootstrap_group=dist.group.WORLD,
        max_current_rows=16384,
    )
    assert runtime.index.numel() == runtime.send_index.numel() == 0
    pointer = runtime.kv.data_ptr()
    results = []
    for case, (rows, alias) in enumerate(
        [(n, False) for n in (0, 1, 17, 63, 64, 65, 127, 128, 129, 2048, 16384)]
        + [(16384, True)]
    ):
        history = [bits(local_rows, 7100 + case * 31 + peer) for peer in range(degree)]
        cache.copy_(history[rank])
        values = bits(rows, 10000 + case)
        slots = torch.arange(rows, dtype=torch.int64) + page
        if rows >= 3 and not alias:
            slots[:3] = torch.tensor([-1, 0, blocks * page])
        counts = [rows // degree + (peer < rows % degree) for peer in range(degree)]
        begin, count = sum(counts[:rank]), counts[rank]
        current = values[begin : begin + count].npu()
        flags = torch.full((rows,), int(alias), dtype=torch.uint8, device="npu")
        runtime.prefetch(case, cache, None, has_history=True)
        runtime.gather_current(case, current, width=576, total_rows=rows)
        # Ordinary router slots are int32; standalone callers may retain int64.
        slot_dtype = torch.int32 if case % 2 == 0 else torch.int64
        runtime.patch(
            case,
            slots.to(dtype=slot_dtype, device="npu"),
            flags,
            width=576,
            block_rows=page,
            virtual_blocks=blocks,
        )
        runtime.finish(case)
        expected = torch.cat(history)
        owner_cache = history[rank].clone()
        for token, slot in enumerate(slots.tolist()):
            block, offset = divmod(slot, page)
            if not 0 < block < blocks:
                continue
            owner = (block - 1) % degree
            local = ((block - 1) // degree + 1) * page + offset
            if not alias:
                expected[owner * local_rows + local] = values[token]
            if owner == rank:
                owner_cache[local] = values[token]
        exact(runtime.kv[: runtime.history_rows], expected)
        exact(cache, owner_cache)
        for peer, peer_count in enumerate(counts):
            start = runtime.history_rows + peer * runtime._send_rows
            offset = sum(counts[:peer])
            exact(
                runtime.kv[start : start + peer_count],
                values[offset : offset + peer_count],
            )
        assert runtime.kv.data_ptr() == pointer
        assert (
            runtime.workspace_bytes
            == (runtime.history_rows + (degree + 1) * runtime.send_capacity) * 576 * 2
        )
        results.append(
            dict(
                rows=rows,
                slot_dtype=str(slot_dtype),
                direct_alias=alias,
                byteexact=True,
            )
        )
    runtime.close()
    from tokenspeed_kernel_npu.ops.mla import mla_prefill

    causal_cases = []
    for query_rows, prefix in ((1, 0), (3, 0), (2, 5), (3, 129)):
        length = query_rows + prefix
        query = torch.zeros((query_rows, 8, 192), dtype=torch.bfloat16, device="npu")
        key = torch.zeros((length, 8, 192), dtype=torch.bfloat16, device="npu")
        values = torch.arange(1, length + 1, dtype=torch.float32)[:, None, None]
        values = values.expand(-1, 8, 128).bfloat16().contiguous()
        actual = (
            mla_prefill(
                query,
                key,
                values.npu(),
                torch.tensor([0, query_rows]),
                torch.tensor([0, length]),
                query_rows,
                length,
                192**-0.5,
                None,
                True,
                0.0,
                False,
                None,
            )
            .cpu()
            .double()
        )
        golden = torch.stack(
            [values[: prefix + row + 1].double().mean(0) for row in range(query_rows)]
        )
        delta = (actual - golden).abs()
        assert bool((delta <= 2**-6 + 2**-6 * golden.abs()).all())
        causal_cases.append(
            dict(
                query_rows=query_rows,
                prefix=prefix,
                max_abs=float(delta.max()),
                all_within=True,
            )
        )
    output = Path(os.environ["TEST_OUTPUT"])
    (output / f"result-rank{rank}.json").write_text(
        json.dumps(
            dict(
                rank=rank,
                pid=os.getpid(),
                complete=True,
                passed=True,
                cases=results,
                workspace_pointer_stable=True,
                no_index_plane=True,
                causal_cases=causal_cases,
            ),
            indent=2,
        )
        + "\n"
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
