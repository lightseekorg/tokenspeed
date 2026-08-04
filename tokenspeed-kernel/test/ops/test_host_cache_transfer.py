from __future__ import annotations

import pytest
import torch

from tokenspeed_kernel.ops.kvcache.host_transfer import transfer_cache_segments


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


@pytest.mark.parametrize("backend", ["dma", "auto", "kernel"])
def test_cache_segments_round_trip_across_multiple_device_buffers(backend):
    first = torch.arange(64, dtype=torch.uint8, device="cuda")
    second = torch.arange(96, dtype=torch.uint8, device="cuda") + 64
    host = torch.zeros(96, dtype=torch.uint8, pin_memory=True)
    segments = ((0, 8, 0, 24), (1, 16, 48, 32))
    stream = torch.cuda.Stream()

    try:
        transfer_cache_segments(
            "d2h", (first, second), host, segments, stream, backend=backend
        )
    except RuntimeError as error:
        message = str(error).lower()
        if backend == "kernel" and (
            "unavailable" in message or "not device-mapped" in message
        ):
            pytest.skip(str(error))
        raise
    stream.synchronize()
    assert torch.equal(host[0:24], first[8:32].cpu())
    assert torch.equal(host[48:80], second[16:48].cpu())

    host[0:24].fill_(7)
    host[48:80].fill_(9)
    transfer_cache_segments(
        "h2d", (first, second), host, segments, stream, backend=backend
    )
    stream.synchronize()
    assert torch.equal(first[8:32].cpu(), torch.full((24,), 7, dtype=torch.uint8))
    assert torch.equal(second[16:48].cpu(), torch.full((32,), 9, dtype=torch.uint8))
