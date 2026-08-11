from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.host_transfer import (
    _triton_is_unavailable,
    transfer_cache_ranges,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def test_unrelated_triton_runtime_error_does_not_fall_back_to_dma():
    assert not _triton_is_unavailable(
        RuntimeError("requested kernel specialization is not available")
    )


@requires_cuda
@pytest.mark.parametrize("backend", ["dma", "auto", "triton"])
def test_cache_ranges_round_trip_across_multiple_device_buffers(backend):
    first = torch.arange(64, dtype=torch.uint8, device="cuda")
    second = torch.arange(48, dtype=torch.bfloat16, device="cuda")
    second_bytes = second.view(torch.uint8)
    host = torch.zeros(96, dtype=torch.uint8, pin_memory=True)
    ranges = ((0, 8, 0, 24), (1, 16, 48, 32))
    stream = torch.cuda.Stream()

    try:
        transfer_cache_ranges(
            "d2h", (first, second), host, ranges, stream, backend=backend
        )
    except RuntimeError as error:
        message = str(error).lower()
        if backend == "triton" and (
            "unavailable" in message or "not device-mapped" in message
        ):
            pytest.skip(str(error))
        raise
    stream.synchronize()
    assert torch.equal(host[0:24], first[8:32].cpu())
    assert torch.equal(host[48:80], second_bytes[16:48].cpu())

    host[0:24].fill_(7)
    host[48:80].fill_(9)
    transfer_cache_ranges("h2d", (first, second), host, ranges, stream, backend=backend)
    stream.synchronize()
    assert torch.equal(first[8:32].cpu(), torch.full((24,), 7, dtype=torch.uint8))
    assert torch.equal(
        second_bytes[16:48].cpu(), torch.full((32,), 9, dtype=torch.uint8)
    )
