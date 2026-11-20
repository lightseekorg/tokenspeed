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

"""Lookup layout and CUDA stream handoff tests."""

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


def _lookup(mapping, *, use_attn_tp=False):
    return ple_lookup.PLELookup(
        mapping,
        vocab_size=256,
        ngram_heads=2,
        head_dim=3,
        storage_dtype=None,
        output_dtype=torch.float32,
        offload=False,
        table_layout="attn_tp_sharded" if use_attn_tp else "global_sharded",
        enable_hot_cache=False,
        prefix="",
    )


def test_hot_cache_requires_offloading():
    with pytest.raises(ValueError, match="requires ple_offload_embedding=true"):
        ple_lookup.PLELookup(
            _mapping_class()(rank=0, world_size=1),
            vocab_size=256,
            ngram_heads=16,
            head_dim=160,
            storage_dtype=None,
            output_dtype=torch.bfloat16,
            offload=False,
            table_layout="global_sharded",
            enable_hot_cache=True,
            prefix="",
        )


def _worker(rank, rendezvous, tp_size, use_attn_tp):
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

        for start in range(0, 4, tp_size):
            ranks = tuple(range(start, start + tp_size))
            groups[ranks] = dist.new_group(ranks, backend="gloo")
        groups[(0, 1, 2, 3)] = dist.group.WORLD

        def gather(output, ids, group):
            assert not use_attn_tp
            dist.all_gather_single(output, ids, group=groups[group])

        ple_lookup._all_gather_ids = gather
        mapping = _mapping_class()(
            rank=rank,
            world_size=4,
            attn_tp_size=tp_size,
            attn_dp_size=dp_size,
        )
        lookup = _lookup(mapping, use_attn_tp=use_attn_tp)
        table_group = mapping.attn.tp_group if use_attn_tp else (0, 1, 2, 3)
        assert lookup.table_group == table_group
        assert lookup.table_rank == table_group.index(rank)
        full_table = torch.arange(256 * 3).reshape(256, 3).float()
        lookup.ngram_embedding.weight.data.zero_()
        lookup.load_shard(full_table, 0, 1, prescanned=False)

        def reduce(values, group):
            assert values.ndim == 2
            assert group == table_group
            dist.all_reduce(values, group=groups[group])
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
                layout = lookup.make_layout(
                    None if use_attn_tp else counts, ids.shape[0]
                )
                pending = lookup.start(ids, layout)
                values = lookup.finish(pending)
                expected = ids.unsqueeze(-1) * 3 + torch.arange(3)
                torch.testing.assert_close(values, expected.float().reshape(live, 6))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("use_attn_tp", [False, True])
def test_sharded_table_matches_unsharded_lookup(tmp_path, tp_size, use_attn_tp):
    mp.spawn(
        _worker,
        args=((tmp_path / "rendezvous").as_uri(), tp_size, use_attn_tp),
        nprocs=4,
        join=True,
    )


def _cached_lookup_worker(rank, rendezvous, tp_size):
    from tokenspeed.runtime.distributed.comm_backend import set_global_backend
    from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    mapping = Mapping(
        rank=rank, world_size=4, attn_tp_size=tp_size, attn_dp_size=4 // tp_size
    )
    process_group_manager.init_distributed(
        mapping, rendezvous, timeout=60, device_id=device
    )
    try:
        process_group_manager.init_process_group(mapping.attn.world_group)
        set_global_backend(NcclBackend())
        vocab, heads, dim = 1021, 16, 160
        table_dtype = torch.float8_e4m3fn
        lookup = ple_lookup.PLELookup(
            mapping,
            vocab_size=vocab,
            ngram_heads=heads,
            head_dim=dim,
            storage_dtype=table_dtype,
            output_dtype=torch.bfloat16,
            offload=True,
            table_layout="global_sharded",
            enable_hot_cache=True,
            prefix="",
        )
        full_table = (
            (torch.arange(vocab * dim).reshape(vocab, dim) % 383 - 191) / 7
        ).to(table_dtype)
        lookup.load_shard(full_table, 0, 1, prescanned=False)
        scale = 0.125
        lookup.load_scale(torch.tensor(scale))

        counts = [99, 0, 107, 131] if mapping.attn.has_dp else None
        tokens = counts[rank] if counts is not None else 325
        offset = rank * 53 if counts is not None else 0
        ids = (
            torch.arange(tokens * heads, device=device).reshape(tokens, heads) + offset
        ) % vocab
        layout = lookup.make_layout(counts, tokens)
        expected = (full_table.float()[ids.cpu()] * scale).reshape(tokens, heads * dim)
        for _ in range(2):
            actual = lookup.finish(lookup.start(ids, layout))
            torch.testing.assert_close(
                actual.cpu(), expected.bfloat16(), rtol=0, atol=0
            )
        assert lookup._host_cache is not None
        lookup._host_cache.invalidate()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size", [1, 4], ids=["dp4", "tp4"])
def test_sharded_host_cache_matches_unsharded_lookup(tmp_path, monkeypatch, tp_size):
    from tokenspeed_kernel.platform import current_platform

    if (
        not current_platform().is_nvidia
        or torch.cuda.device_count() < 4
        or any(
            torch.cuda.get_device_capability(rank)[0] not in (9, 10)
            for rank in range(4)
        )
    ):
        pytest.skip("requires four Hopper or Blackwell GPUs")
    monkeypatch.setenv("TOKENSPEED_PLE_HOST_CACHE_SLOTS", "512")
    mp.spawn(
        _cached_lookup_worker,
        args=((tmp_path / "cached-rendezvous").as_uri(), tp_size),
        nprocs=4,
        join=True,
    )


def test_online_fp8_host_lookup_uses_hot_cache(monkeypatch):
    from tokenspeed_kernel.platform import current_platform

    if not torch.cuda.is_available() or not (
        current_platform().is_hopper or current_platform().is_blackwell
    ):
        pytest.skip("requires a Hopper or Blackwell GPU")

    monkeypatch.setenv("TOKENSPEED_PLE_HOST_CACHE_SLOTS", "512")
    tokens, heads, dim, vocab = 257, 16, 160, 4
    lookup = ple_lookup.PLELookup(
        _mapping_class()(rank=0, world_size=1),
        vocab_size=vocab,
        ngram_heads=heads,
        head_dim=dim,
        storage_dtype=torch.float8_e4m3fn,
        output_dtype=torch.bfloat16,
        offload=True,
        table_layout="global_sharded",
        enable_hot_cache=True,
        prefix="",
    )
    weights = torch.tensor([0.125, 1.0, -8.0, 32.0], dtype=torch.bfloat16)
    table = weights[:, None].expand(vocab, dim).contiguous()
    lookup.load_shard(table, 0, 1, prescanned=False)
    monkeypatch.setattr(
        lookup, "gather_host", lambda *_: pytest.fail("expected hot cache")
    )

    ids = torch.arange(tokens * heads, device="cuda").reshape(tokens, heads) % vocab
    layout = lookup.make_layout(None, tokens)
    scale = (table.float().abs().max() / 448).bfloat16().item()
    quantized = (table.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    expected = (quantized.float() * scale).bfloat16()[ids.cpu()].flatten(1)
    for _ in range(2):
        actual = lookup.finish(lookup.start(ids, layout))
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


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


def _mapped_lookup_worker(rank, rendezvous):
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )

    torch.cuda.set_device(rank)
    mapping = _mapping_class()(rank=rank, world_size=2, attn_tp_size=1, attn_dp_size=2)
    process_group_manager.init_distributed(
        mapping, rendezvous, device_id=torch.device("cuda", rank)
    )
    process_group_manager.init_process_group(mapping.attn.world_group)
    try:
        for mode in ("shared", "peer_sharded"):
            lookup = ple_lookup.PLELookup(
                mapping,
                vocab_size=250,
                ngram_heads=2,
                head_dim=16,
                storage_dtype=torch.float8_e4m3fn,
                output_dtype=torch.bfloat16,
                offload=True,
                table_layout=mode,
                enable_hot_cache=False,
                prefix="",
            )
            src = (torch.arange(250 * 16).reshape(250, 16) % 101).to(
                torch.float8_e4m3fn
            )
            lookup.load_shard(src, 0, 1, prescanned=False)
            lookup.load_scale(torch.tensor(0.125))
            lookup.shared_table.publish()
            ids = torch.tensor(
                [[1 + rank, 139 + rank], [249, 127]], device="cuda"
            ).repeat(65, 1)
            layout = lookup.make_layout(None, ids.shape[0])
            result = lookup.finish(lookup.start(ids, layout))
            expected = (
                (src.float()[ids.cpu()] * 0.125).bfloat16().view(ids.shape[0], 32)
            )
            torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two GPUs for mapped PLE lookup"
)
def test_mapped_tables_lookup_local_ids(tmp_path):
    mp.spawn(_mapped_lookup_worker, args=((tmp_path / "mapped").as_uri(),), nprocs=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_host_lookup_stream_handoff(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_PLE_HOST_CACHE_SLOTS", "512")
    tokens, heads, dim, vocab = 4, 16, 160, 256
    lookup = ple_lookup.PLELookup(
        _mapping_class()(rank=0, world_size=1),
        vocab_size=vocab,
        ngram_heads=heads,
        head_dim=dim,
        storage_dtype=None,
        output_dtype=torch.bfloat16,
        offload=True,
        table_layout="global_sharded",
        enable_hot_cache=False,
        prefix="",
    )
    assert lookup._host_cache is None
    table = torch.randn(vocab, dim, dtype=torch.bfloat16)
    lookup.ngram_embedding.weight.data.copy_(table)
    ids = torch.arange(tokens * heads, device="cuda").reshape(tokens, heads)
    layout = lookup.make_layout(None, tokens)
    consumer = torch.cuda.Stream()

    def run():
        pending = lookup.start(ids, layout)
        with torch.cuda.stream(consumer):
            result = lookup.finish(pending).clone()
        torch.cuda.current_stream().wait_stream(consumer)
        return result

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    ids.add_(7)
    graph.replay()
    expected = table[ids.cpu()].reshape(tokens, -1)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
