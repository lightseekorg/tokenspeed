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

"""``/update_weights_from_disk`` from the scheduler request to parameter values.

The route's existing coverage replaces the engine with a fake, so nothing
below it was exercised. These tests drive the real
``RequestHandler.process_requests`` and assert on parameter values loaded from
a real safetensors checkpoint, including a round trip (load B, then A again),
because a load that writes the wrong data also changes the model.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
)
from tokenspeed.runtime.engine.request_handler import RequestHandler  # noqa: E402
from tokenspeed.runtime.execution.device import DeviceHandle  # noqa: E402
from tokenspeed.runtime.execution.model_runner import ModelRunner  # noqa: E402


class _TinyModel(torch.nn.Module):
    """A real module with a real ``load_weights``, small enough for CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(4, 3))
        self.b = torch.nn.Parameter(torch.zeros(4))
        self.seen: list[str] = []

    def load_weights(self, weights) -> None:
        params = dict(self.named_parameters())
        for name, tensor in weights:
            self.seen.append(name)
            params[name].data.copy_(tensor)


def _write_checkpoint(directory: Path, scale: float) -> dict[str, torch.Tensor]:
    """A one-shard safetensors checkpoint, the shape a trainer writes."""
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        "w": torch.arange(12, dtype=torch.float32).reshape(4, 3) * scale,
        "b": torch.arange(4, dtype=torch.float32) * scale,
    }
    save_file(tensors, str(directory / "model.safetensors"))
    return tensors


def _runner(model: torch.nn.Module, booted_from: Path) -> ModelRunner:
    """A ModelRunner with only what the disk reload touches (no GPU load)."""
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner.device = "cpu"
    runner.gpu_id = 0
    runner.model_config = SimpleNamespace(model_path=str(booted_from))
    runner.server_args = SimpleNamespace(
        load_format="auto",
        download_dir=None,
        ext_yaml=None,
        weight_loader_prefetch_checkpoints=False,
        weight_loader_prefetch_num_threads=1,
    )
    return runner


def _device_handle(runner, drafter=None) -> DeviceHandle:
    handle = DeviceHandle.__new__(DeviceHandle)
    handle._executor = SimpleNamespace(model_runner=runner, drafter=drafter)
    handle._l2 = None
    # The real forward thread orders the call against forwards; here it only
    # has to run it.
    handle._thread = SimpleNamespace(run=lambda fn: fn())
    return handle


def _handler(device) -> RequestHandler:
    handler = RequestHandler.__new__(RequestHandler)
    handler.send_func = mock.Mock()
    handler.server_args = mock.Mock(weight_version="v1", kvstore_storage_backend=None)
    handler._device = device
    handler._replica_tp_size = 1
    handler._replica_tp_cpu_group = None
    handler.attn_cp_size = 1
    handler.attn_cp_cpu_group = None
    handler.pp_size = 1
    handler.pp_cpu_group = None
    handler.attn_dp_size = 1
    handler.attn_dp_cpu_group = None
    handler._replica_decision_buf = torch.zeros(1, dtype=torch.int32)
    handler._replica_flush_want_buf = torch.zeros(1, dtype=torch.int32)
    handler.can_clear_cache_fn = mock.Mock(return_value=True)
    handler.clear_cache_fn = mock.Mock(return_value=True)
    return handler


def _mock_device(ok: bool = True, message: str = "loaded"):
    device = mock.Mock()
    device.update_weights = mock.Mock(return_value=(ok, message))
    device.delete_l3_namespace.return_value = True
    return device


class TestSchedulerDispatch(unittest.TestCase):
    def test_disk_update_is_dispatched_and_answered(self):
        # Without a branch the request reaches the NotImplementedError
        # fallthrough and the scheduler process exits.
        handler = _handler(_mock_device())
        req = UpdateWeightFromDiskReqInput(model_path="/ckpt/step-42")

        handler.process_requests([req])

        handler._device.update_weights.assert_called_once_with(req)
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertIsInstance(output, UpdateWeightFromDiskReqOutput)
        self.assertTrue(output.success)
        self.assertEqual(output.message, "loaded")

    def test_failed_disk_update_is_answered(self):
        handler = _handler(_mock_device(ok=False, message="no such checkpoint"))

        handler.process_requests([UpdateWeightFromDiskReqInput(model_path="/nope")])

        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertIsInstance(output, UpdateWeightFromDiskReqOutput)
        self.assertFalse(output.success)
        self.assertEqual(output.message, "no such checkpoint")

    def test_prefix_cache_is_flushed_before_the_load(self):
        # Cached KV was computed under the old weights; prefix caching is on by
        # default, so reusing it would mix two sets of weights in one response.
        order = []
        device = _mock_device()
        device.update_weights.side_effect = lambda req: order.append("load") or (
            True,
            "ok",
        )
        handler = _handler(device)
        handler.clear_cache_fn = mock.Mock(
            side_effect=lambda: order.append("flush") or True
        )

        handler.process_requests([UpdateWeightFromDiskReqInput(model_path="/ckpt")])

        self.assertEqual(order, ["flush", "load"])

    def test_failed_flush_skips_the_load(self):
        handler = _handler(_mock_device())
        handler.can_clear_cache_fn = mock.Mock(return_value=False)

        handler.process_requests([UpdateWeightFromDiskReqInput(model_path="/ckpt")])

        handler._device.update_weights.assert_not_called()
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertIsInstance(output, UpdateWeightFromDiskReqOutput)
        self.assertFalse(output.success)
        self.assertIn("cache flush failed", output.message)

    def test_weight_version_is_committed_after_a_successful_load(self):
        handler = _handler(_mock_device())

        handler.process_requests(
            [UpdateWeightFromDiskReqInput(model_path="/ckpt", weight_version="v2")]
        )

        self.assertEqual(handler.server_args.weight_version, "v2")
        handler._device.set_l3_weight_version.assert_called_once_with("v2")

    def test_l3_flush_without_weight_version_is_rejected_before_the_load(self):
        handler = _handler(_mock_device())
        handler.server_args.kvstore_storage_backend = "memory"

        handler.process_requests([UpdateWeightFromDiskReqInput(model_path="/ckpt")])

        handler._device.update_weights.assert_not_called()
        output = handler.send_func.send_pyobj.call_args.args[0]
        self.assertIsInstance(output, UpdateWeightFromDiskReqOutput)
        self.assertFalse(output.success)


class TestDeviceRouting(unittest.TestCase):
    def test_disk_request_reaches_the_model_runner(self):
        runner = mock.Mock()
        runner.update_weights_from_disk = mock.Mock(return_value=(True, "ok"))
        req = UpdateWeightFromDiskReqInput(model_path="/ckpt")

        self.assertEqual(_device_handle(runner).update_weights(req), (True, "ok"))
        runner.update_weights_from_disk.assert_called_once_with(req)

    def test_drafter_is_told_the_target_weights_changed(self):
        runner = mock.Mock()
        runner.update_weights_from_disk = mock.Mock(return_value=(True, "ok"))
        drafter = mock.Mock()

        _device_handle(runner, drafter).update_weights(
            UpdateWeightFromDiskReqInput(model_path="/ckpt")
        )

        drafter.on_target_weights_updated.assert_called_once_with()


class TestRealCheckpoint(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _load(self, runner, path, **kwargs):
        return runner.update_weights_from_disk(
            UpdateWeightFromDiskReqInput(model_path=str(path), **kwargs)
        )

    def test_weights_arrive_from_the_named_checkpoint(self):
        original = _write_checkpoint(self.root / "step-0", scale=1.0)
        updated = _write_checkpoint(self.root / "step-1", scale=7.0)
        model = _TinyModel()
        runner = _runner(model, self.root / "step-0")

        ok, message = self._load(runner, self.root / "step-1")

        self.assertTrue(ok, message)
        torch.testing.assert_close(model.w.data, updated["w"])
        torch.testing.assert_close(model.b.data, updated["b"])
        self.assertFalse(torch.equal(model.w.data, original["w"]))

    def test_round_trip_restores_the_original_values(self):
        original = _write_checkpoint(self.root / "step-0", scale=1.0)
        _write_checkpoint(self.root / "step-1", scale=7.0)
        model = _TinyModel()
        runner = _runner(model, self.root / "step-0")

        self._load(runner, self.root / "step-1")
        self.assertFalse(torch.equal(model.w.data, original["w"]))
        ok, message = self._load(runner, self.root / "step-0")

        self.assertTrue(ok, message)
        torch.testing.assert_close(model.w.data, original["w"])
        torch.testing.assert_close(model.b.data, original["b"])

    def test_count_comes_from_the_checkpoint(self):
        _write_checkpoint(self.root / "step-1", scale=2.0)
        model = _TinyModel()
        runner = _runner(model, self.root / "step-1")

        ok, message = self._load(runner, self.root / "step-1")

        self.assertTrue(ok, message)
        self.assertIn("2 checkpoint tensors read", message)
        self.assertEqual(sorted(model.seen), ["b", "w"])

    def test_worker_records_the_new_model_path(self):
        _write_checkpoint(self.root / "step-1", scale=3.0)
        runner = _runner(_TinyModel(), self.root / "step-0")

        self._load(runner, self.root / "step-1")

        self.assertEqual(runner.model_config.model_path, str(self.root / "step-1"))

    def test_checkpoint_with_no_tensors_is_a_failure(self):
        empty = self.root / "empty"
        empty.mkdir()
        torch.save({}, empty / "model.pt")
        model = _TinyModel()
        runner = _runner(model, self.root / "step-0")

        ok, message = self._load(runner, empty, load_format="pt")

        self.assertFalse(ok)
        self.assertIn("no checkpoint tensors", message)
        self.assertEqual(model.seen, [])

    def test_missing_checkpoint_fails_instead_of_raising(self):
        # The scheduler runs this inline; an escaping exception kills it.
        runner = _runner(_TinyModel(), self.root / "step-0")

        ok, message = self._load(runner, self.root / "absent")

        self.assertFalse(ok)
        self.assertTrue(message)

    def test_request_load_format_overrides_the_servers(self):
        directory = self.root / "step-1"
        expected = _write_checkpoint(directory, scale=5.0)
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(
                {"weight_map": {"w": "model.safetensors", "b": "model.safetensors"}}
            )
        )
        model = _TinyModel()
        runner = _runner(model, directory)
        runner.server_args.load_format = "pt"  # would find no *.pt here

        ok, message = self._load(runner, directory, load_format="safetensors")

        self.assertTrue(ok, message)
        torch.testing.assert_close(model.w.data, expected["w"])

    def test_load_format_that_reads_no_files_is_refused(self):
        # ``dummy`` randomises weights instead of reading a checkpoint.
        runner = _runner(_TinyModel(), self.root / "step-0")

        ok, message = self._load(runner, self.root / "step-0", load_format="dummy")

        self.assertFalse(ok)
        self.assertIn("in-place reload", message)

    def test_scheduler_request_reaches_the_parameters(self):
        original = _write_checkpoint(self.root / "step-0", scale=1.0)
        updated = _write_checkpoint(self.root / "step-1", scale=7.0)
        model = _TinyModel()
        handler = _handler(_device_handle(_runner(model, self.root / "step-0")))

        for path, expected in (
            (self.root / "step-1", updated),
            (self.root / "step-0", original),
        ):
            handler.process_requests(
                [UpdateWeightFromDiskReqInput(model_path=str(path))]
            )
            output = handler.send_func.send_pyobj.call_args.args[0]
            self.assertIsInstance(output, UpdateWeightFromDiskReqOutput)
            self.assertTrue(output.success, output.message)
            torch.testing.assert_close(model.w.data, expected["w"])
        self.assertEqual(handler.clear_cache_fn.call_count, 2)


if __name__ == "__main__":
    unittest.main()
