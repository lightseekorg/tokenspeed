from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tokenspeed.runtime.distributed.comm_backend import hccl as hccl_module
from tokenspeed.runtime.distributed.comm_backend import registry
from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend


def test_registry_selects_hccl_on_npu(monkeypatch):
    monkeypatch.setattr(registry, "_global_backend", None)
    monkeypatch.setattr(
        registry,
        "current_platform",
        lambda: SimpleNamespace(is_npu=True),
    )

    backend = registry.initialize_comm_backend()

    assert isinstance(backend, HcclBackend)


def test_hccl_all_reduce_uses_hccl_process_group(monkeypatch):
    process_group = Mock()
    get_process_group = Mock(return_value=process_group)
    all_reduce = Mock()
    monkeypatch.setattr(hccl_module.pg_manager, "get_process_group", get_process_group)
    monkeypatch.setattr(hccl_module.dist, "all_reduce", all_reduce)
    tensor = torch.ones(4)

    output = HcclBackend().all_reduce(tensor, (0, 1))

    assert output is tensor
    get_process_group.assert_called_once_with("hccl", (0, 1))
    all_reduce.assert_called_once_with(
        tensor,
        op=torch.distributed.ReduceOp.SUM,
        group=process_group,
    )


@pytest.mark.parametrize("shape", [(), (1,), (7,), (3, 5), (512,)])
@pytest.mark.parametrize("ranks", [2, 3, 8])
@pytest.mark.parametrize("contiguous", [True, False])
def test_small_int64_max_preserves_every_bit_and_output_buffer(
    monkeypatch, shape, ranks, contiguous
):
    count = torch.empty(shape).numel()
    generator = torch.Generator().manual_seed(43)
    values = torch.randint(
        -(1 << 63), (1 << 63) - 1, (ranks, count), generator=generator
    )
    # Signed extremes and equal high words exercise unsigned low-word order.
    values[:, 0] = -(1 << 63)
    if count > 1:
        values[:, 1] = (-3 << 32) | torch.linspace(
            0, (1 << 32) - 1, ranks, dtype=torch.float64
        ).to(torch.int64)
    if count > 2:
        values[0, 2] = (1 << 63) - 1
    if contiguous:
        tensor = values[ranks // 2].reshape(shape).clone()
    elif len(shape) == 2:
        tensor = values[ranks // 2].reshape(shape).t().contiguous().t()
    else:
        tensor = torch.empty((*shape, 2), dtype=torch.int64)[..., 0]
        tensor.copy_(values[ranks // 2].reshape(shape))
    original = values[ranks // 2].reshape(shape).clone()
    backend = HcclBackend()
    group = tuple(range(ranks))

    def all_gather_single(output, input, actual_group):
        assert actual_group == group
        # The wire type must be AIV-compatible without converting the key.
        assert input.dtype == output.dtype == torch.int32
        torch.testing.assert_close(
            input, original.view(-1).view(torch.int32), rtol=0, atol=0
        )
        output.view(torch.int64).copy_(values.view(-1))

    monkeypatch.setattr(backend, "all_gather_single", all_gather_single)
    result = backend.all_reduce(tensor, group, op=torch.distributed.ReduceOp.MAX)

    assert result is tensor
    assert result.shape == shape
    torch.testing.assert_close(
        result, values.amax(dim=0).reshape(shape), rtol=0, atol=0
    )


@pytest.mark.parametrize("counts", [[8] * 8, [0] * 8, [8], [2, 1, 3], [0, 2, 1]])
def test_token_all_gather_preserves_rank_order_without_repacking_equal_shards(
    monkeypatch, counts
):
    backend = HcclBackend()
    width = 4
    shards = [
        torch.arange(count * width, dtype=torch.float32).reshape(count, width)
        + rank * 100
        for rank, count in enumerate(counts)
    ]
    expected = torch.cat(shards)
    gathered_buffers = []

    def all_gather_single(output, tensor, group):
        assert group == tuple(range(len(counts)))
        assert tensor.is_contiguous()
        torch.testing.assert_close(tensor[: counts[0]], shards[0])
        assert torch.count_nonzero(tensor[counts[0] :]) == 0
        # Padding must never leak into the returned token rows.
        output.fill_(-999)
        for rank, shard in enumerate(shards):
            start = rank * max(counts)
            output[start : start + counts[rank]].copy_(shard)
        gathered_buffers.append(output)

    monkeypatch.setattr(backend, "all_gather_single", all_gather_single)
    # Also exercise the input-contiguity handling.
    tensor = shards[0].t().contiguous().t()
    output = backend.token_all_gather(tensor, tuple(range(len(counts))), counts)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.is_contiguous()
    if len(set(counts)) == 1:
        assert output is gathered_buffers[0]


@pytest.mark.parametrize("counts", [[8] * 8, [16] * 8, [0] * 8, [2, 1, 3], [0, 2, 1]])
@pytest.mark.parametrize("contiguous", [True, False])
def test_token_reduce_scatter_preserves_rows_and_skips_equal_shard_packing(
    monkeypatch, counts, contiguous
):
    backend = HcclBackend()
    group = tuple(2 * rank for rank in range(len(counts)))
    tensor = torch.arange(sum(counts) * 8, dtype=torch.float32).reshape(-1, 8)
    if not contiguous:
        tensor = tensor[:, ::2]
    process_group = Mock()
    monkeypatch.setattr(backend, "_process_group", Mock(return_value=process_group))
    current_rank = 0

    def reduce_scatter(output, input, *, group):
        assert group is process_group
        assert input.is_contiguous()
        if len(set(counts)) == 1:
            torch.testing.assert_close(input, tensor, rtol=0, atol=0)
            if tensor.is_contiguous():
                assert input.data_ptr() == tensor.data_ptr()
        else:
            offset = 0
            for index, count in enumerate(counts):
                chunk = input[index * max(counts) : (index + 1) * max(counts)]
                torch.testing.assert_close(
                    chunk[:count], tensor[offset : offset + count]
                )
                assert torch.count_nonzero(chunk[count:]) == 0
                offset += count
        # Simulate reduction of identical peer inputs, selecting this rank's chunk.
        output.copy_(input.chunk(len(counts))[current_rank] * len(counts))

    # Older torch-npu deployments install this collective alias at startup.
    monkeypatch.setattr(
        hccl_module.dist, "reduce_scatter_single", reduce_scatter, raising=False
    )
    monkeypatch.setattr(hccl_module.dist, "get_rank", lambda: group[current_rank])
    for current_rank in range(len(counts)):
        output = backend.token_reduce_scatter(tensor, group, counts)
        offset = sum(counts[:current_rank])
        expected = tensor[offset : offset + counts[current_rank]] * len(counts)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert output.is_contiguous()


def test_single_rank_token_reduce_scatter_needs_no_collective(monkeypatch):
    reduce_scatter = Mock()
    monkeypatch.setattr(
        hccl_module.dist, "reduce_scatter_single", reduce_scatter, raising=False
    )
    tensor = torch.ones(8, 4)
    output = HcclBackend().token_reduce_scatter(tensor, (2,), [8])
    torch.testing.assert_close(output, tensor)
    reduce_scatter.assert_not_called()
