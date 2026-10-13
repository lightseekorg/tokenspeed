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

"""DSA token parallelism: a sparse launch spread over the group's ranks returns
every rank the output of its own heads."""

import argparse
import os
import sys
import threading
from types import SimpleNamespace
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import pytest
import torch

import tokenspeed.runtime.distributed.comm_ops as comm_ops
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.attention.backends.paged import dsa
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.utils.server_args import ServerArgs

GROUP = 4
GROUP_RANKS = tuple(range(GROUP))
HEADS = 2
DIM = 6
DIM_V = 5
WIDTH = 8
# SM counts under which every sparse launch of these tests spreads, or none.
SPREADING_SMS = 1
LOCAL_SMS = 10**6


class _Exchange:
    """``comm_ops.all_to_all_single`` across one thread per group rank."""

    def __init__(self, size: int):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.sent: list = [None] * size
        self.local = threading.local()
        self.calls = 0

    def rank(self, group=None) -> int:
        return self.local.rank

    def __call__(
        self,
        output,
        input,
        group,
        backend=None,
        output_split_sizes=None,
        input_split_sizes=None,
    ):
        assert tuple(group) == GROUP_RANKS
        rank = self.local.rank
        even = [input.shape[0] // self.size] * self.size
        self.sent[rank] = input.clone().split(input_split_sizes or even)
        if rank == 0:
            self.calls += 1
        self.barrier.wait()
        received = [self.sent[source][rank] for source in range(self.size)]
        if output_split_sizes is not None:
            assert [chunk.shape[0] for chunk in received] == output_split_sizes
        output.copy_(torch.cat(received))
        self.barrier.wait()


def _attend(q, slots, lens, kv_lens, sink):
    # Stands in for the sparse kernel: every (row, head) output depends only on
    # that row's query and metadata and that head's sink.
    rows, heads, _ = q.shape
    meta = slots.float().sum(-1)
    if lens is not None:
        meta = meta + lens.float() * 3
    if kv_lens is not None:
        meta = meta + kv_lens.float() * 7
    out = q[..., :DIM_V].float() * 2 + meta.view(rows, 1, 1)
    if sink is not None:
        out = out + sink.view(1, heads, 1) * 11
    return out.to(q.dtype)


def _dsa_spec(
    *,
    backend_name: str = "dsa",
    num_attention_heads: int = GROUP * HEADS,
    attn_tp_size: int = GROUP,
    kv_lora_rank: int = DIM_V,
    qk_nope_head_dim: int = 4,
    qk_rope_head_dim: int = DIM - DIM_V,
    index_topk: int = WIDTH,
) -> DSAConfig:
    return DSAConfig(
        backend_name=backend_name,
        num_attention_heads=num_attention_heads,
        num_kv_heads=1,
        head_dim=kv_lora_rank + qk_rope_head_dim,
        attn_tp_size=attn_tp_size,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=kv_lora_rank,
        scaling=1.0,
        kv_cache_dim=kv_lora_rank + qk_rope_head_dim,
        index_topk=index_topk,
        index_head_dim=128,
        index_n_heads=1,
        index_k_format="fp8_scaled",
    )


def _mha_spec() -> MHAConfig:
    return MHAConfig(
        num_attention_heads=GROUP * HEADS,
        num_kv_heads=GROUP,
        head_dim=64,
        attn_tp_size=GROUP,
    )


def _attn_config(
    spec,
    group: tuple[int, ...],
    token_parallel_rank: int,
    *,
    device: str = "cuda",
    is_draft: bool = False,
) -> AttnConfig:
    return AttnConfig(
        device=device,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        prefix_granularity=64,
        context_len=300,
        max_bs=8,
        is_draft=is_draft,
        token_parallel_rank=token_parallel_rank,
        token_parallel_group=group,
        components=(spec,),
    )


def _backend(rank: int, group: tuple[int, ...], **spec_fields) -> dsa.DSABackend:
    """The DSA backend of attention-TP rank ``rank``, a member of ``group``."""
    spec = _dsa_spec(**spec_fields)
    config = _attn_config(spec, group, group.index(rank))
    return dsa.DSABackend(config, spec, kernel_page_size=64)


def _patch_sm_count(monkeypatch, sms: int) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=sms),
    )


def _patch_device(monkeypatch, *, sms: int | None, decode_metadata=None) -> None:
    """Stand in for what the DSA backend constructor reads off the device.

    The dense MLA leaf serves the sparse paths only the decode metadata.
    ``sms`` replaces the device's SM count; None keeps the real one.
    """
    monkeypatch.setattr(dsa, "current_platform", lambda: None)
    monkeypatch.setattr(
        dsa,
        "_make_dense_leaf",
        lambda config, spec, platform, kernel_page_size: SimpleNamespace(
            forward_decode_metadata=decode_metadata
        ),
    )
    if sms is not None:
        _patch_sm_count(monkeypatch, sms)


def _layer(
    sink: torch.Tensor | None,
    *,
    head_dim: int = DIM,
    v_head_dim: int = DIM_V,
    scaling: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        attn_sink=sink,
        tp_q_head_num=HEADS,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        scaling=scaling,
        logit_cap=0.0,
        layer_id=0,
    )


def _inputs(tokens: int):
    torch.manual_seed(tokens)
    q = torch.randn(tokens, GROUP * HEADS, DIM).bfloat16()
    slots = torch.randint(-1, 100, (tokens, WIDTH), dtype=torch.int32)
    lens = torch.randint(0, WIDTH + 1, (tokens,), dtype=torch.int32)
    kv_lens = torch.randint(1, 300, (tokens,), dtype=torch.int32)
    sink = torch.randn(GROUP * HEADS)
    return q, slots, lens, kv_lens, sink


def _run_group(monkeypatch, fn):
    exchange = _Exchange(GROUP)
    monkeypatch.setattr(comm_ops, "all_to_all_single", exchange)
    monkeypatch.setattr(torch.distributed, "get_rank", exchange.rank)
    results: list = [None] * GROUP
    errors: list[BaseException] = []

    def run(rank):
        exchange.local.rank = rank
        try:
            results[rank] = fn(rank)
        except BaseException as error:  # surfaced below, after every rank joins
            errors.append(error)
            exchange.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(GROUP)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results, exchange


def _own_heads(x: torch.Tensor, rank: int) -> torch.Tensor:
    return x[:, rank * HEADS : (rank + 1) * HEADS].contiguous()


@pytest.mark.parametrize("tokens", [8, 7, 3])
@pytest.mark.parametrize("with_kv_lens", [True, False])
def test_spread_rows_return_each_rank_its_own_heads(monkeypatch, tokens, with_kv_lens):
    q, slots, lens, kv_lens, sink = _inputs(tokens)
    kv_lens = kv_lens if with_kv_lens else None
    layer = _layer(sink)
    _patch_device(monkeypatch, sms=SPREADING_SMS)
    backends = [_backend(rank, GROUP_RANKS) for rank in GROUP_RANKS]

    def spread(rank):
        backend = backends[rank]
        return backend._attend_spread_rows(
            _own_heads(q, rank),
            slots,
            lens,
            kv_lens,
            backend._group_sink(layer),
            _attend,
        )

    results, exchange = _run_group(monkeypatch, spread)
    assert exchange.calls == 2
    for rank, out in enumerate(results):
        sink_heads = sink[rank * HEADS : (rank + 1) * HEADS]
        expected = _attend(_own_heads(q, rank), slots, lens, kv_lens, sink_heads)
        assert out.shape == (tokens, HEADS, DIM_V)
        assert torch.equal(out, expected)


def test_spread_launches_require_the_group_sinks(monkeypatch):
    q, slots, lens, kv_lens, sink = _inputs(8)
    _patch_device(monkeypatch, sms=SPREADING_SMS)
    backend = _backend(1, GROUP_RANKS)
    pool = SimpleNamespace(get_key_buffer=lambda layer_id: None)
    with pytest.raises(
        RuntimeError, match="sink logits of the token-parallel group's 8 heads, got 2"
    ):
        backend.forward_sparse_prefill(
            q=_own_heads(q, 1).reshape(8, HEADS * DIM),
            layer=_layer(sink[:HEADS]),
            token_to_kv_pool=pool,
            kv_seq_lens=kv_lens,
            topk_slots=slots,
            topk_lens=lens,
            max_seq_len=300,
        )


def test_local_sink_is_the_rank_slice_of_the_group_sinks(monkeypatch):
    _patch_device(monkeypatch, sms=SPREADING_SMS)
    sink = torch.arange(GROUP * HEADS, dtype=torch.float32)
    member = _backend(2, GROUP_RANKS)
    assert member._local_sink(_layer(sink)).tolist() == [4.0, 5.0]
    assert member._local_sink(_layer(None)) is None
    # A launch kept on its rank checks the sinks as a spread launch does.
    with pytest.raises(RuntimeError, match="sink logits of the token-parallel"):
        member._local_sink(_layer(sink[:HEADS]))
    alone = _backend(2, (2,))
    assert alone._local_sink(_layer(sink[:HEADS])).tolist() == [0.0, 1.0]


def test_launches_spread_from_a_quarter_of_the_sms_full_rows(monkeypatch):
    # 78 SMs and 2048-slot rows: up to 19 full rows the kernel still splits
    # every row four ways; from 20 rows on it can split them only twice.
    _patch_device(monkeypatch, sms=78)
    backend = _backend(0, GROUP_RANKS, index_topk=2048)
    assert not backend._spreads_rows(19, 2048)
    assert backend._spreads_rows(20, 2048)
    assert not backend._spreads_rows(1024, 38)
    assert backend._spreads_rows(8192, 1024)
    # Every rank of the group gets a row.
    assert not _backend(0, GROUP_RANKS, index_topk=2**20)._spreads_rows(3, 2**20)
    assert not _backend(0, (0,), index_topk=2048)._spreads_rows(8192, 2048)


@pytest.mark.parametrize("spread", [True, False])
def test_sparse_prefill_matches_each_rank_alone(monkeypatch, spread):
    tokens = 9
    q, slots, lens, kv_lens, sink = _inputs(tokens)
    layer = _layer(sink)
    calls = []

    def fake_prefill(*, q, topk_slots, topk_lens, kv_seq_lens, attn_sink, **kwargs):
        assert not kwargs["return_lse"]
        calls.append(q.shape[1])
        return _attend(q, topk_slots, topk_lens, kv_seq_lens, attn_sink)

    monkeypatch.setattr(dsa, "dsa_prefill", fake_prefill)
    _patch_device(monkeypatch, sms=SPREADING_SMS if spread else LOCAL_SMS)
    backends = [_backend(rank, GROUP_RANKS) for rank in GROUP_RANKS]
    pool = SimpleNamespace(get_key_buffer=lambda layer_id: None)

    def prefill(rank):
        return backends[rank].forward_sparse_prefill(
            q=_own_heads(q, rank).reshape(tokens, HEADS * DIM),
            layer=layer,
            token_to_kv_pool=pool,
            kv_seq_lens=kv_lens,
            topk_slots=slots,
            topk_lens=lens,
            max_seq_len=300,
        )

    results, exchange = _run_group(monkeypatch, prefill)
    assert exchange.calls == (2 if spread else 0)
    assert set(calls) == {GROUP * HEADS if spread else HEADS}
    for rank, out in enumerate(results):
        sink_heads = sink[rank * HEADS : (rank + 1) * HEADS]
        expected = _attend(_own_heads(q, rank), slots, lens, kv_lens, sink_heads)
        assert torch.equal(out, expected.reshape(tokens, HEADS * DIM_V))


@pytest.mark.parametrize("spread", [True, False])
@pytest.mark.parametrize("q_len_per_req", [1, 2])
@pytest.mark.parametrize("with_lens", [True, False])
def test_sparse_decode_matches_each_rank_alone(
    monkeypatch, spread, q_len_per_req, with_lens
):
    reqs = 5
    tokens = reqs * q_len_per_req
    q, slots, lens, _, sink = _inputs(tokens)
    lens = lens if with_lens else None
    seq_lens = torch.randint(q_len_per_req, 300, (reqs,), dtype=torch.int32)
    # Verify row t of a request sees its prefix up to and including token t.
    offsets = torch.arange(q_len_per_req, dtype=torch.int32) - (q_len_per_req - 1)
    kv_lens = (seq_lens.unsqueeze(1) + offsets).reshape(-1)
    layer = _layer(sink)
    launches = []

    def fake_decode(*, q, topk_slots, topk_lens, kv_seq_lens, attn_sink, **kwargs):
        launches.append(("decode", q.shape[1]))
        return _attend(q, topk_slots, topk_lens, kv_seq_lens, attn_sink)

    def fake_prefill(*, q, topk_slots, topk_lens, kv_seq_lens, attn_sink, **kwargs):
        launches.append(("rows", q.shape[1]))
        return _attend(q, topk_slots, topk_lens, kv_seq_lens, attn_sink)

    monkeypatch.setattr(dsa, "dsa_decode", fake_decode)
    monkeypatch.setattr(dsa, "dsa_prefill", fake_prefill)
    _patch_device(
        monkeypatch,
        sms=SPREADING_SMS if spread else LOCAL_SMS,
        decode_metadata=SimpleNamespace(
            seq_lens_k=seq_lens, num_extends=0, max_seq_len_k=300
        ),
    )
    backends = [_backend(rank, GROUP_RANKS) for rank in GROUP_RANKS]
    pool = SimpleNamespace(get_key_buffer=lambda layer_id: None)

    def decode(rank):
        return backends[rank].forward_sparse_decode(
            q=_own_heads(q, rank).reshape(tokens, HEADS * DIM),
            layer=layer,
            token_to_kv_pool=pool,
            bs=reqs,
            topk_indices=slots,
            topk_lens=lens,
        )

    results, exchange = _run_group(monkeypatch, decode)
    assert exchange.calls == (2 if spread else 0)
    expected_launch = ("rows", GROUP * HEADS) if spread else ("decode", HEADS)
    assert set(launches) == {expected_launch}
    for rank, out in enumerate(results):
        sink_heads = sink[rank * HEADS : (rank + 1) * HEADS]
        expected = _attend(_own_heads(q, rank), slots, lens, kv_lens, sink_heads)
        assert torch.equal(out, expected.reshape(tokens, HEADS * DIM_V))


@pytest.mark.parametrize(
    "spec, device, message",
    [
        (_dsa_spec(backend_name="flashmla"), "cuda", "requires the GPU DSA backend"),
        (_dsa_spec(), "cpu", "requires CUDA"),
        (_mha_spec(), "cuda", "requires DSA attention"),
    ],
    ids=["flashmla", "cpu", "mha"],
)
def test_config_rejects_token_parallelism_off_the_gpu_dsa_backend(
    spec, device, message
):
    with pytest.raises(ValueError, match=message):
        _attn_config(spec, GROUP_RANKS, 1, device=device)


def test_config_accepts_a_non_dsa_draft_of_a_token_parallel_target():
    # A DFLASH or DSPARK MHA draft keeps attending every row on its rank.
    config = _attn_config(_mha_spec(), GROUP_RANKS, 1, is_draft=True)
    assert config.token_parallel_group == GROUP_RANKS
    with pytest.raises(ValueError, match="requires the GPU DSA backend"):
        _attn_config(_dsa_spec(backend_name="flashmla"), GROUP_RANKS, 1, is_draft=True)


@pytest.mark.parametrize(
    "parallelism", [dict(attn_kvp_size=2), dict(attn_qcp_size=8)], ids=["kvp", "qcp"]
)
def test_mapping_rejects_token_parallelism_beside_sharded_rows(parallelism):
    with pytest.raises(ValueError, match="does not combine with KV parallelism"):
        Mapping(
            rank=0,
            world_size=8,
            attn_tp_size=8,
            attn_token_parallel_size=2,
            nprocs_per_node=8,
            **parallelism,
        )


def _mapping_from_cli(*argv: str) -> Mapping:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        ["--model", "test/model", "--tensor-parallel-size", "32", "--nnodes", "4"]
        + list(argv)
    )
    with mock.patch.object(ServerArgs, "__post_init__"):
        server_args = ServerArgs.from_cli_args(args)
    server_args.resolve_basic_defaults()
    server_args.resolve_parallelism()
    return server_args.mapping


@pytest.mark.parametrize("size", [1, 8])
def test_server_args_carry_the_token_parallel_size_into_the_mapping(size):
    argv = [] if size == 1 else ["--attn-token-parallel-size", str(size)]
    mapping = _mapping_from_cli(*argv)
    mapping.rank = 13
    first = 13 // size * size
    assert mapping.nprocs_per_node == 8
    assert mapping.attn.token_parallel_size == size
    assert mapping.attn.token_parallel_group == tuple(range(first, first + size))
    assert mapping.attn.token_parallel_rank == 13 - first


def test_server_args_reject_a_token_parallel_group_across_nodes():
    with pytest.raises(
        ValueError, match="token-parallel size 16 must divide the 8 ranks per node"
    ):
        _mapping_from_cli("--attn-token-parallel-size", "16")


def test_hy_v4_loads_the_sink_heads_the_backend_attends(monkeypatch):
    from tokenspeed.runtime.models import hy_v4

    class _MLAHeads:
        """The part of the MLA constructor the sink loader reads: the head split."""

        def __init__(self, *, mapping, num_heads, **kwargs):
            torch.nn.Module.__init__(self)
            self.num_local_heads = num_heads // mapping.attn.tp_size

    monkeypatch.setattr(hy_v4, "DeepseekV3AttentionMLA", _MLAHeads)
    _patch_device(monkeypatch, sms=132)
    hf_config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=64,
        qk_nope_head_dim=4,
        qk_rope_head_dim=DIM - DIM_V,
        v_head_dim=DIM_V,
        q_lora_rank=8,
        kv_lora_rank=DIM_V,
        rope_theta=10000.0,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        index_topk=WIDTH,
        # Layer 1 reuses layer 0's top-k, so it builds no indexer.
        indexer_types=("full", "shared"),
        gated_mla=False,
        learnable_sink=True,
    )
    checkpoint_sink = torch.arange(64, dtype=torch.float32)
    spec = _dsa_spec(num_attention_heads=64, attn_tp_size=32)
    for rank in range(32):
        mapping = Mapping(
            rank=rank,
            world_size=32,
            attn_tp_size=32,
            attn_token_parallel_size=8,
            nprocs_per_node=8,
        )
        attn = hy_v4.HYV4Attention(hf_config, mapping, layer_id=1, is_nextn=False)
        sink = attn.learnable_sink_param
        sink.weight_loader(sink, checkpoint_sink)
        layer = SimpleNamespace(
            attn_sink=sink.detach(), tp_q_head_num=attn.num_local_heads
        )
        config = _attn_config(
            spec, mapping.attn.token_parallel_group, mapping.attn.token_parallel_rank
        )
        backend = dsa.DSABackend(config, spec, kernel_page_size=64)
        first = 2 * mapping.attn.token_parallel_group[0]
        assert backend._group_sink(layer).tolist() == list(range(first, first + 16))
        assert backend._local_sink(layer).tolist() == [2 * rank, 2 * rank + 1]


def _initialize(
    monkeypatch, sm_counts: list[int], *, emulate_rank_zero: bool = False
) -> list:
    import tokenspeed_kernel.ops.communication.fabric as fabric

    import tokenspeed.runtime.execution.distributed_initializer as initializer

    events = []
    mapping = Mapping(
        rank=0, world_size=8, attn_tp_size=8, attn_token_parallel_size=GROUP
    )
    config = SimpleNamespace(
        device="cuda",
        gpu_id=0,
        dist_init_addr=None,
        nccl_port=1234,
        distributed_timeout_seconds=10,
        emulate_rank_zero=emulate_rank_zero,
        mapping=mapping,
        hidden_size=0,
        world_size=1,
    )
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda device: SimpleNamespace(set_device=lambda gpu: None),
    )
    monkeypatch.setattr(
        initializer, "get_available_gpu_memory", lambda *args, **kwargs: 1.0
    )
    monkeypatch.setattr(
        initializer, "maybe_set_numa_aware_cpu_affinity", lambda gpu: None
    )
    monkeypatch.setattr(
        initializer.pg_manager, "init_distributed", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        initializer.pg_manager,
        "init_emulated_rank_zero",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(initializer, "set_global_backend", lambda backend: None)
    monkeypatch.setattr(initializer.pg_manager, "init_process_group", events.append)
    monkeypatch.setattr(
        initializer.pg_manager,
        "get_process_group",
        lambda backend, group: (backend, group),
    )

    def all_gather_object(counts, count, group):
        assert group == ("gloo", GROUP_RANKS)
        assert count == sm_counts[0]
        events.append("sm counts")
        counts[:] = sm_counts

    def all_to_all_single(output, input, group):
        assert group == GROUP_RANKS
        assert input.is_cuda and input.numel() == output.numel() == GROUP
        events.append("connect")

    monkeypatch.setattr(initializer.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(comm_ops, "all_to_all_single", all_to_all_single)
    _patch_sm_count(monkeypatch, sm_counts[0])
    monkeypatch.setattr(fabric, "gather_fabric_map", lambda: events.append("fabric"))
    initializer.DistributedInitializer.initialize(config)
    return events


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@requires_cuda
def test_initializer_compares_sm_counts_once_the_token_parallel_group_exists(
    monkeypatch,
):
    events = _initialize(monkeypatch, [132] * GROUP)
    assert events.count("sm counts") == 1
    assert events[events.index(GROUP_RANKS) + 1] == "sm counts"


@requires_cuda
def test_initializer_connects_the_token_parallel_group_before_cache_sizing(
    monkeypatch,
):
    events = _initialize(monkeypatch, [132] * GROUP)
    assert events.count("connect") == 1
    assert events.index("connect") > events.index("fabric")


def test_initializer_rejects_token_parallel_ranks_with_different_sm_counts(
    monkeypatch,
):
    with pytest.raises(
        ValueError, match=r"same SM count, got \{0: 132, 1: 132, 2: 114, 3: 132\}"
    ):
        _initialize(monkeypatch, [132, 132, 114, 132])


def test_initializer_leaves_an_emulated_rank_without_peers(monkeypatch):
    events = _initialize(monkeypatch, [132] * GROUP, emulate_rank_zero=True)
    assert GROUP_RANKS in events
    assert "sm counts" not in events and "connect" not in events


SLOTS = 4096
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
ROW = KV_LORA_RANK + QK_ROPE_HEAD_DIM
TOPK = 512
SCALE = ROW**-0.5


def _flashmla_inputs(tokens: int, with_sink: bool):
    generator = torch.Generator(device="cuda").manual_seed(tokens)
    kv = (torch.randn(SLOTS, ROW, device="cuda", generator=generator) * 0.5).to(
        torch.bfloat16
    )
    # Peaked logits keep the outputs and the sink's share above BF16 rounding.
    q = (
        torch.randn(tokens, GROUP * HEADS, ROW, device="cuda", generator=generator) * 3
    ).to(torch.bfloat16)
    slots = torch.stack(
        [
            torch.randperm(SLOTS, device="cuda", generator=generator)[:TOPK]
            for _ in range(tokens)
        ]
    ).to(torch.int32)
    invalid = torch.rand(slots.shape, device="cuda", generator=generator) < 0.1
    invalid[:, 0] = False
    slots[invalid] = -1
    lens = torch.randint(
        1, TOPK + 1, (tokens,), device="cuda", generator=generator, dtype=torch.int32
    )
    sink = None
    if with_sink:
        sink = torch.randn(GROUP * HEADS, device="cuda", generator=generator) * 2 + 4
    return q, kv, slots, lens, sink


def _reference(q, kv, slots, lens, sink):
    """FP32 attention over every row's live slots; the sink is one more logit."""
    keys = kv.float()[slots.clamp_min(0).long()]
    columns = torch.arange(slots.shape[1], device=slots.device)
    live = (slots >= 0) & (columns < lens[:, None])
    logits = torch.einsum("thd,tkd->thk", q.float(), keys) * SCALE
    logits = logits.masked_fill(~live[:, None, :], float("-inf"))
    if sink is not None:
        sink_logits = sink.view(1, -1, 1).expand(q.shape[0], -1, 1)
        logits = torch.cat([logits, sink_logits], dim=-1)
    probs = torch.softmax(logits, dim=-1)[..., : slots.shape[1]]
    return torch.einsum("thk,tkd->thd", probs, keys[..., :KV_LORA_RANK])


@requires_cuda
@pytest.mark.parametrize("tokens", [72, 70], ids=["whole_rows", "padded_rows"])
@pytest.mark.parametrize("with_sink", [True, False], ids=["sink", "no_sink"])
def test_spread_flashmla_prefill_matches_each_rank_alone(
    monkeypatch, tokens, with_sink
):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("FlashMLA sparse prefill is checked on SM90")
    q, kv, slots, lens, sink = _flashmla_inputs(tokens, with_sink)
    pool = SimpleNamespace(quant_method=None, get_key_buffer=lambda layer_id: kv)
    spec_fields = dict(
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        index_topk=TOPK,
    )
    _patch_device(monkeypatch, sms=None)
    prefill = dsa.dsa_prefill
    lock = threading.Lock()

    def one_launch_at_a_time(**kwargs):
        # The group's ranks share this GPU and its kernel selection state.
        with lock:
            return prefill(**kwargs)

    monkeypatch.setattr(dsa, "dsa_prefill", one_launch_at_a_time)

    def attend(backend, rank, layer):
        out = backend.forward_sparse_prefill(
            q=_own_heads(q, rank).reshape(tokens, HEADS * ROW),
            layer=layer,
            token_to_kv_pool=pool,
            kv_seq_lens=None,
            topk_slots=slots,
            topk_lens=lens,
            max_seq_len=SLOTS,
        )
        return out.view(tokens, HEADS, KV_LORA_RANK)

    def flashmla_layer(sink_heads):
        return _layer(sink_heads, head_dim=ROW, v_head_dim=KV_LORA_RANK, scaling=SCALE)

    backends = [_backend(rank, GROUP_RANKS, **spec_fields) for rank in GROUP_RANKS]
    results, exchange = _run_group(
        monkeypatch,
        lambda rank: attend(backends[rank], rank, flashmla_layer(sink)),
    )
    assert exchange.calls == 2
    reference = _reference(q, kv, slots, lens, sink)
    for rank, spread in enumerate(results):
        sink_heads = None if sink is None else sink[rank * HEADS : (rank + 1) * HEADS]
        alone = attend(
            _backend(rank, (rank,), **spec_fields), rank, flashmla_layer(sink_heads)
        )
        # With a quarter of the rows the kernel splits their columns and merges
        # BF16 partials, one rounding more than the launch alone.
        torch.testing.assert_close(
            spread.float(), alone.float(), rtol=1.6e-2, atol=4e-3
        )
        expected = _own_heads(reference, rank)
        torch.testing.assert_close(spread.float(), expected, rtol=1.6e-2, atol=4e-3)
        torch.testing.assert_close(alone.float(), expected, rtol=1.6e-2, atol=4e-3)
