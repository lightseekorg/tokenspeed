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

"""Guards for the trainer weight-update NCCL group in
``tokenspeed.runtime.execution.weight_update_group``.

CPU-only, no real NCCL: ``model_runner`` itself cannot be imported on a host
without the compiled ``tokenspeed_kernel`` / ``tokenspeed_triton``
extensions, so these two helpers live in their own dependency-free module and
are exercised here with fakes instead of a real process group.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.weight_update_group import (  # noqa: E402
    _assert_not_split,
    _no_default_group_split,
)


def test_noop_when_not_initialized(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    # Any access to _get_default_group would raise, since there is none;
    # the context manager must not even look at it when not initialized.
    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_get_default_group",
        lambda: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    with _no_default_group_split():
        pass


def test_clears_and_restores_bound_device(monkeypatch):
    fake_default_pg = SimpleNamespace(bound_device_id=torch.device("cuda", 0))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_get_default_group",
        lambda: fake_default_pg,
    )

    with _no_default_group_split():
        assert fake_default_pg.bound_device_id is None

    assert fake_default_pg.bound_device_id == torch.device("cuda", 0)


def test_restores_bound_device_when_body_raises(monkeypatch):
    fake_default_pg = SimpleNamespace(bound_device_id=torch.device("cuda", 0))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_get_default_group",
        lambda: fake_default_pg,
    )

    with pytest.raises(RuntimeError, match="boom"):
        with _no_default_group_split():
            assert fake_default_pg.bound_device_id is None
            raise RuntimeError("boom")

    assert fake_default_pg.bound_device_id == torch.device("cuda", 0)


def test_assert_not_split_raises_when_split_from_set():
    fake_backend = SimpleNamespace(options=SimpleNamespace(split_from=object()))
    fake_pg = SimpleNamespace(_get_backend=lambda device: fake_backend)

    with pytest.raises(RuntimeError, match="split"):
        _assert_not_split(fake_pg, torch.device("cuda", 0))


def test_assert_not_split_passes_when_clean():
    fake_backend = SimpleNamespace(options=SimpleNamespace(split_from=None))
    fake_pg = SimpleNamespace(_get_backend=lambda device: fake_backend)

    _assert_not_split(fake_pg, torch.device("cuda", 0))  # no raise


def test_assert_not_split_passes_when_backend_unavailable():
    def _raise_no_backend(device):
        raise RuntimeError("no backend for device")

    fake_pg = SimpleNamespace(_get_backend=_raise_no_backend)

    _assert_not_split(fake_pg, torch.device("cuda", 0))  # no raise


def test_rejected_group_is_destroyed_before_retry(monkeypatch):
    from tokenspeed.runtime.execution.model_runner import ModelRunner

    c10d = torch.distributed.distributed_c10d
    rank_maps = {}
    registered = set()
    split_backend = SimpleNamespace(options=SimpleNamespace(split_from=object()))

    # Use an identity-hashable group, like torch's ProcessGroup.
    class Group:
        def _get_backend(self, device):
            return split_backend

    pg = Group()
    store = SimpleNamespace(set_timeout=lambda timeout: None)
    runner = SimpleNamespace(
        global_rank=0, gpu_id=0, _weight_update_pg=None, _weight_update_device=None
    )
    request = SimpleNamespace(
        rank_offset=1,
        world_size=2,
        group_name="test-group",
        backend="nccl",
        master_address="localhost",
        master_port=12345,
    )

    def create_group(*args, group_name, **kwargs):
        assert group_name not in registered
        registered.add(group_name)
        return pg, None

    def destroy_group(group):
        assert group is pg
        assert rank_maps.pop(group) == {0: 0, 1: 1}
        registered.remove(request.group_name)

    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        c10d, "rendezvous", lambda *args, **kwargs: iter([(store, 1, 2)])
    )
    monkeypatch.setattr(c10d, "PrefixStore", lambda name, store: store)
    monkeypatch.setattr(c10d, "_world", SimpleNamespace(pg_group_ranks=rank_maps))
    monkeypatch.setattr(c10d, "_new_process_group_helper", create_group)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroy_group)

    ok, message = ModelRunner.init_weights_update_group(runner, request)
    assert not ok
    assert "split" in message
    assert not registered
    assert not rank_maps
    assert runner._weight_update_pg is None

    split_backend.options.split_from = None
    ok, _ = ModelRunner.init_weights_update_group(runner, request)
    assert ok
    assert runner._weight_update_pg is pg
