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

"""CPU/Gloo tests; no CUDA kernel packages are needed for the query layout."""

import importlib.util
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.layers import ple_lookup


def _mapping_class():
    # Avoid distributed/__init__.py, which imports GPU communication kernels.
    path = Path(ple_lookup.__file__).parents[1] / "distributed/mapping.py"
    spec = importlib.util.spec_from_file_location("ple_test_mapping", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Mapping


def _lookup(mapping):
    return ple_lookup.PLELookup(
        mapping,
        vocab_size=256,
        ngram_heads=2,
        head_dim=3,
        storage_dtype=None,
        output_dtype=torch.float32,
        offload=False,
        prefix="",
    )


def _worker(rank, rendezvous, tp_size):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    try:
        dp_size = 4 // tp_size
        groups = {}
        for lane in range(tp_size):
            ranks = tuple(range(lane, 4, tp_size))
            groups[ranks] = dist.new_group(ranks, backend="gloo")

        def gather(output, ids, group):
            dist.all_gather_single(output, ids, group=groups[group])

        ple_lookup._all_gather_ids = gather
        mapping = _mapping_class()(
            rank=rank,
            world_size=4,
            attn_tp_size=tp_size,
            attn_dp_size=dp_size,
        )
        lookup = _lookup(mapping)
        assert lookup.table_group == (0, 1, 2, 3)
        assert lookup.table_rank == rank
        full_table = torch.arange(256 * 3).reshape(256, 3).float()
        lookup.ngram_embedding.weight.data.zero_()
        lookup.load_shard(full_table, 0, 1)

        def reduce(values, group):
            assert values.ndim == 2
            assert group == (0, 1, 2, 3)
            dist.all_reduce(values)
            return values

        ple_lookup._all_reduce = reduce
        patterns = (
            [3] + [1] * (dp_size - 1),
            [0] + [2] * (dp_size - 1),
            [2] * dp_size,
            [0] * dp_size,
        )
        for sizes in patterns:
            for shift in (0, 63, 127, 191):
                dp = mapping.attn.dp_rank
                n = sizes[dp]
                # Last pattern with live rows includes one physical padding row.
                live = max(0, n - 1) if sizes == [2] * dp_size else n
                ids = (torch.arange(live * 2).reshape(live, 2) + dp * 63 + shift) % 256
                counts = [sizes[r // tp_size] for r in range(4)]
                layout = lookup.make_layout(counts, ids.shape[0])
                pending = lookup.start(ids, layout)
                values = lookup.finish(pending)
                expected = ids.unsqueeze(-1) * 3 + torch.arange(3)
                torch.testing.assert_close(values, expected.float().reshape(live, 6))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_global_table_matches_unsharded_lookup(tmp_path, tp_size):
    mp.spawn(
        _worker,
        args=((tmp_path / "rendezvous").as_uri(), tp_size),
        nprocs=4,
        join=True,
    )


def test_pure_tp_has_no_ids_collective(monkeypatch):
    mapping = _mapping_class()(rank=1, world_size=4)
    lookup = _lookup(mapping)
    monkeypatch.setattr(
        ple_lookup, "_all_gather_ids", lambda *a: pytest.fail("TP must not gather IDs")
    )
    ids = torch.tensor([[1, 20], [8, 31]])
    gathered, local_slice = lookup._gather_ids(
        ids, lookup.make_layout(None, ids.shape[0])
    )
    assert gathered is ids
    torch.testing.assert_close(gathered[local_slice], ids)


def test_pipeline_stage_uses_global_rank_counts(monkeypatch):
    mapping = _mapping_class()(
        rank=6, world_size=8, pp_size=2, attn_tp_size=2, attn_dp_size=2
    )
    lookup = _lookup(mapping)
    assert lookup.table_group == (4, 5, 6, 7)
    assert lookup.table_rank == 2
    assert lookup.dp_group == (4, 6)

    def gather(output, ids, group):
        assert group == (4, 6)
        assert ids.shape == (3, 2)
        output[:3].fill_(11)
        output[3:].copy_(ids)

    monkeypatch.setattr(ple_lookup, "_all_gather_ids", gather)
    ids = torch.tensor([[25, 26]])
    output, local_slice = lookup._gather_ids(
        ids, lookup.make_layout([9, 9, 9, 9, 3, 3, 1, 1], 1)
    )
    assert output.shape == (4, 2)
    torch.testing.assert_close(output[local_slice], ids)


def test_invalid_dp_layout_fails_before_collective(monkeypatch):
    mapping = _mapping_class()(rank=0, world_size=2, attn_tp_size=1, attn_dp_size=2)
    lookup = _lookup(mapping)
    monkeypatch.setattr(
        ple_lookup, "_all_gather_ids", lambda *a: pytest.fail("unexpected collective")
    )
    for counts in (None, [-1, 1], [0, 1]):
        with pytest.raises(ValueError):
            lookup.start(
                torch.ones((1, 2), dtype=torch.long), lookup.make_layout(counts, 1)
            )


def test_dp_cp_rejected():
    mapping = _mapping_class()(
        rank=0, world_size=4, attn_tp_size=1, attn_cp_size=2, attn_dp_size=2
    )
    with pytest.raises(ValueError, match="CP=1"):
        _lookup(mapping)


def test_lookup_handle_lifecycle():
    mapping = _mapping_class()(rank=0, world_size=1)
    lookup, other = _lookup(mapping), _lookup(mapping)
    ids = torch.tensor([[1, 2]], dtype=torch.int64)
    layout = lookup.make_layout(None, 1)
    pending = lookup.start(ids, layout)
    with pytest.raises(RuntimeError, match="in-flight"):
        lookup.start(ids, layout)
    with pytest.raises(ValueError, match="another lookup"):
        other.finish(pending)
    assert lookup.finish(pending).shape == (1, 6)
    with pytest.raises(RuntimeError, match="consumed"):
        lookup.finish(pending)
    lookup.finish(lookup.start(ids, layout))


@pytest.mark.parametrize(
    "ids",
    [
        torch.ones((1, 2), dtype=torch.int32),
        torch.ones((1, 3), dtype=torch.int64),
        torch.ones((2, 4), dtype=torch.int64)[:, ::2],
    ],
)
def test_lookup_rejects_invalid_ids(ids):
    lookup = _lookup(_mapping_class()(rank=0, world_size=1))
    with pytest.raises(ValueError):
        lookup.start(ids, lookup.make_layout(None, ids.shape[0]))


@pytest.mark.parametrize("sizes", [(-1,), (1.5,), (True,), [1]])
def test_lookup_rejects_invalid_physical_extents(sizes):
    with pytest.raises(ValueError):
        ple_lookup.LookupTokenLayout(sizes)
