from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion

import tokenspeed.runtime.distributed.comm_backend.triton_rsag as triton_rsag


class _Fallback:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def all_gather(self, tensor, group, dim=0):
        self.calls.append("all_gather")
        return tensor + 1

    def token_all_gather(self, tensor, group, scattered_num_tokens):
        self.calls.append("token_all_gather")
        return tensor + 2

    def token_reduce_scatter(self, tensor, group, scattered_num_tokens):
        self.calls.append("token_reduce_scatter")
        return tensor + 3


def _platform(major: int) -> SimpleNamespace:
    return SimpleNamespace(is_nvidia=True, arch_version=ArchVersion(major, 0))


def test_sm120_uses_fallback_for_all_collectives(monkeypatch) -> None:
    fallback = _Fallback()
    backend = triton_rsag.TritonRSAGBackend(fallback=fallback)
    monkeypatch.setattr(triton_rsag, "current_platform", lambda: _platform(12))
    monkeypatch.setattr(
        backend,
        "_get_or_create",
        lambda *_args: pytest.fail("SM120 must not create Triton RSAG state"),
    )
    tensor = torch.zeros(1, 8, dtype=torch.bfloat16)

    torch.testing.assert_close(
        backend.all_gather(tensor, group=(0, 1), dim=-1), tensor + 1
    )
    torch.testing.assert_close(
        backend.token_all_gather(tensor, group=(0, 1), scattered_num_tokens=[1, 1]),
        tensor + 2,
    )
    torch.testing.assert_close(
        backend.token_reduce_scatter(
            tensor, group=(0, 1), scattered_num_tokens=[1, 1]
        ),
        tensor + 3,
    )
    assert fallback.calls == [
        "all_gather",
        "token_all_gather",
        "token_reduce_scatter",
    ]


@pytest.mark.parametrize("major", [9, 10])
def test_qualified_nvidia_architectures_keep_triton_path(major, monkeypatch) -> None:
    fallback = _Fallback()
    backend = triton_rsag.TritonRSAGBackend(fallback=fallback)
    monkeypatch.setattr(triton_rsag, "current_platform", lambda: _platform(major))
    state = object()
    monkeypatch.setattr(backend, "_get_or_create", lambda *_args: state)
    expected = torch.ones(2, 8, dtype=torch.bfloat16)
    monkeypatch.setattr(triton_rsag, "all_gather", lambda *_args, **_kwargs: expected)

    assert backend.token_all_gather(
        torch.zeros(1, 8), group=(0, 1), scattered_num_tokens=[1, 1]
    ) is expected
    assert fallback.calls == []
