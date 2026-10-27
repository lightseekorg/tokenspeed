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

"""``/update_weights_from_disk`` on the data plane, down to parameter values.

These tests drive the device handle's weight update (what the control plane
asks for once its gate admits the op) and assert on parameter values loaded
from a real safetensors checkpoint, including a round trip (load B, then A
again), because a load that writes the wrong data also changes the model.

A refused reload must leave every parameter as it was, so those tests assert
that ``load_weights`` saw nothing, not only that the request failed.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    UpdateWeightFromDiskReqInput,
)
from tokenspeed.runtime.execution.device import DeviceHandle  # noqa: E402
from tokenspeed.runtime.execution.model_runner import ModelRunner  # noqa: E402
from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod  # noqa: E402
from tokenspeed.runtime.layers.dense.unquant import (  # noqa: E402
    UnquantizedLinearMethod,
)
from tokenspeed.runtime.layers.linear import (  # noqa: E402
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.moe.expert import MoELayer  # noqa: E402
from tokenspeed.runtime.model_loader import loader as loader_module  # noqa: E402


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
    runner.checkpoint_load_group = None
    runner._disk_reload_shapes = None
    runner.model_config = SimpleNamespace(
        model_path=str(booted_from), mapping=Mapping(rank=0)
    )
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
    # No drafter round is pending for ``settle`` to finish.
    handle._executor = SimpleNamespace(
        model_runner=runner, drafter=drafter, complete_pending_drafting=lambda: None
    )
    handle._l2 = None
    return handle


class TestDeviceRouting:
    def test_disk_request_reaches_the_model_runner(self):
        runner = mock.Mock()
        runner.update_weights_from_disk = mock.Mock(return_value=(True, "ok"))
        req = UpdateWeightFromDiskReqInput(model_path="/ckpt")

        assert (_device_handle(runner).update_weights(req)) == ((True, "ok"))
        runner.update_weights_from_disk.assert_called_once_with(req)

    def test_drafter_is_told_the_target_weights_changed(self):
        runner = mock.Mock()
        runner.update_weights_from_disk = mock.Mock(return_value=(True, "ok"))
        drafter = mock.Mock()

        _device_handle(runner, drafter).update_weights(
            UpdateWeightFromDiskReqInput(model_path="/ckpt")
        )

        drafter.on_target_weights_updated.assert_called_once_with()


class TestRealCheckpoint:
    @pytest.fixture(autouse=True)
    def _setup(self, request):
        self._tmp = tempfile.TemporaryDirectory()
        request.addfinalizer(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

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

        _pytest_expr_1 = ok
        _pytest_msg_2 = message
        assert _pytest_expr_1, _pytest_msg_2
        torch.testing.assert_close(model.w.data, updated["w"])
        torch.testing.assert_close(model.b.data, updated["b"])
        assert not (torch.equal(model.w.data, original["w"]))

    def test_round_trip_restores_the_original_values(self):
        original = _write_checkpoint(self.root / "step-0", scale=1.0)
        _write_checkpoint(self.root / "step-1", scale=7.0)
        model = _TinyModel()
        runner = _runner(model, self.root / "step-0")

        self._load(runner, self.root / "step-1")
        assert not (torch.equal(model.w.data, original["w"]))
        ok, message = self._load(runner, self.root / "step-0")

        _pytest_expr_3 = ok
        _pytest_msg_4 = message
        assert _pytest_expr_3, _pytest_msg_4
        torch.testing.assert_close(model.w.data, original["w"])
        torch.testing.assert_close(model.b.data, original["b"])

    def test_count_comes_from_the_checkpoint(self):
        _write_checkpoint(self.root / "step-1", scale=2.0)
        model = _TinyModel()
        runner = _runner(model, self.root / "step-1")

        ok, message = self._load(runner, self.root / "step-1")

        _pytest_expr_5 = ok
        _pytest_msg_6 = message
        assert _pytest_expr_5, _pytest_msg_6
        assert ("2 checkpoint tensors read") in (message)
        assert (sorted(model.seen)) == (["b", "w"])

    def test_worker_records_the_new_model_path(self):
        _write_checkpoint(self.root / "step-1", scale=3.0)
        runner = _runner(_TinyModel(), self.root / "step-0")

        self._load(runner, self.root / "step-1")

        assert (runner.model_config.model_path) == (str(self.root / "step-1"))

    def test_non_unit_kv_scale_fails_the_update(self):
        # A reload keeps the startup KV-cache scales, as a distributed update does.
        directory = self.root / "scaled"
        directory.mkdir()
        save_file(
            {"w": torch.ones(4, 3), "attn.k_scale": torch.tensor([2.0])},
            str(directory / "model.safetensors"),
        )
        model = _TinyModel()
        model.attn = torch.nn.Module()
        model.attn.k_scale = torch.nn.Parameter(torch.ones(1))
        runner = _runner(model, self.root / "step-0")

        ok, message = self._load(runner, directory)

        assert not (ok)
        assert ("only unit KV-cache scales are supported") in (message)
        assert (runner.model_config.model_path) == (str(self.root / "step-0"))

    def test_checkpoint_with_no_tensors_is_a_failure(self):
        empty = self.root / "empty"
        empty.mkdir()
        torch.save({}, empty / "model.pt")
        model = _TinyModel()
        runner = _runner(model, self.root / "step-0")

        ok, message = self._load(runner, empty, load_format="pt")

        assert not (ok)
        assert ("no checkpoint tensors") in (message)
        assert (model.seen) == ([])

    def test_missing_checkpoint_fails_instead_of_raising(self):
        # The scheduler runs this inline; an escaping exception kills it.
        runner = _runner(_TinyModel(), self.root / "step-0")

        ok, message = self._load(runner, self.root / "absent")

        assert not (ok)
        assert message

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

        _pytest_expr_7 = ok
        _pytest_msg_8 = message
        assert _pytest_expr_7, _pytest_msg_8
        torch.testing.assert_close(model.w.data, expected["w"])

    def test_load_format_that_reads_no_files_is_refused(self):
        # ``dummy`` randomises weights instead of reading a checkpoint.
        runner = _runner(_TinyModel(), self.root / "step-0")

        ok, message = self._load(runner, self.root / "step-0", load_format="dummy")

        assert not (ok)
        assert ("in-place reload") in (message)


def _bare(cls: type[torch.nn.Module]) -> torch.nn.Module:
    """An instance of ``cls`` built without its constructor.

    The post-load check reads only a module's class, its ``quant_method`` and
    an MoE layer's ``plan``, so no layer needs real weights or a mapping.
    """
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    return module


def _moe(preprocessor=None) -> MoELayer:
    experts = _bare(MoELayer)
    experts.plan = {"weight_preprocessor": preprocessor}
    return experts


def _quantized(model: torch.nn.Module) -> torch.nn.Module:
    """Add a block-FP8 projection, which the startup loader post-processes."""
    model.proj = torch.nn.Module()
    model.proj.quant_method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    return model


class _SkippingModel(_TinyModel):
    """Skips checkpoint names it has no parameter for, as model loaders do."""

    def load_weights(self, weights) -> None:
        params = dict(self.named_parameters())
        for name, tensor in weights:
            self.seen.append(name)
            if name in params:
                params[name].data.copy_(tensor)


class TestPostLoadTransforms:
    def _transformed(self, **modules) -> list[str]:
        model = torch.nn.Module()
        for name, module in modules.items():
            setattr(model, name, module)
        return loader_module.post_load_transformed_modules(model)

    def test_quantized_projection_is_a_transform(self):
        assert (
            loader_module.post_load_transformed_modules(_quantized(torch.nn.Module()))
        ) == (["proj (Fp8LinearMethod)"])

    def test_moe_layer_whose_kernel_repacks_its_experts_is_a_transform(self):
        def shuffle_experts(plan, w):
            del plan, w

        assert (self._transformed(experts=_moe(shuffle_experts))) == (
            ["experts (shuffle_experts)"]
        )

    def test_moe_layer_whose_kernel_has_no_preprocessor_is_not(self):
        # The unquantized gfx950 MoE kernels plan no weight preprocessor, so
        # their experts keep the checkpoint layout.
        assert (self._transformed(experts=_moe(None))) == ([])

    def test_unknown_hook_is_a_transform(self):
        # A hook nobody has shown to be a no-op is treated as one that is not.
        class Repacked(torch.nn.Module):
            def process_weights_after_loading(self, module) -> None:
                del module

        assert (self._transformed(layer=Repacked())) == (["layer (Repacked)"])

    def test_unquantized_dense_layers_are_not(self):
        layers = {}
        for name, cls in (
            ("qkv_proj", QKVParallelLinear),
            ("gate_up_proj", MergedColumnParallelLinear),
            ("o_proj", RowParallelLinear),
        ):
            layers[name] = _bare(cls)
            layers[name].quant_method = UnquantizedLinearMethod()
        assert (self._transformed(**layers)) == ([])

    def test_no_op_hooks_leave_the_layer_untouched(self, subtests):
        # The allow list is what keeps unquantized models reloadable, so each
        # member must really leave the loaded weights alone.
        for hook in loader_module.no_op_post_load_hooks():
            with subtests.test(hook=hook.__qualname__):
                layer = torch.nn.Module()
                layer.weight = torch.nn.Parameter(
                    torch.arange(6.0).reshape(2, 3), requires_grad=False
                )
                weight, values = layer.weight, layer.weight.detach().clone()
                attributes = set(vars(layer)), set(layer._parameters)
                # The unquantized method's hook reads its own planning state,
                # so it runs on a real method; module hooks run on the layer.
                owner = (
                    UnquantizedLinearMethod()
                    if hook is UnquantizedLinearMethod.process_weights_after_loading
                    else layer
                )

                hook(owner, layer)

                assert (layer.weight) is (weight)
                assert torch.equal(layer.weight.detach(), values)
                assert ((set(vars(layer)), set(layer._parameters))) == (attributes)


class TestReloadRefusals:
    @pytest.fixture(autouse=True)
    def _setup(self, request):
        self._tmp = tempfile.TemporaryDirectory()
        request.addfinalizer(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _load(self, runner, path, **kwargs):
        return runner.update_weights_from_disk(
            UpdateWeightFromDiskReqInput(model_path=str(path), **kwargs)
        )

    def test_post_processed_model_is_refused_before_anything_is_written(self):
        _write_checkpoint(self.root / "step-1", scale=7.0)
        model = _quantized(_TinyModel())
        bound: list[str] = []
        model.bind_checkpoint_dir = bound.append
        runner = _runner(model, self.root / "step-0")

        ok, message = self._load(runner, self.root / "step-1")

        assert not (ok)
        assert ("proj (Fp8LinearMethod)") in (message)
        assert ("Nothing was written") in (message)
        assert ("restart the engine") in (message)
        assert (model.seen) == ([])
        assert (bound) == ([])
        assert torch.equal(model.w.data, torch.zeros(4, 3))
        assert (runner.model_config.model_path) == (str(self.root / "step-0"))

    def test_model_with_only_no_op_hooks_still_reloads(self):
        updated = _write_checkpoint(self.root / "step-1", scale=7.0)
        model = _TinyModel()
        model.o_proj = _bare(RowParallelLinear)
        model.o_proj.quant_method = UnquantizedLinearMethod()
        model.experts = _moe(None)
        runner = _runner(model, self.root / "step-1")

        ok, message = self._load(runner, self.root / "step-1")

        _pytest_expr_9 = ok
        _pytest_msg_10 = message
        assert _pytest_expr_9, _pytest_msg_10
        torch.testing.assert_close(model.w.data, updated["w"])

    def test_shape_mismatch_is_refused_before_anything_is_written(self):
        _write_checkpoint(self.root / "step-0", scale=1.0)
        wrong = self.root / "wrong"
        wrong.mkdir()
        # "b" is read before "w": a load without the check writes "b" and
        # then fails on "w", leaving a half-written model.
        save_file(
            {"b": torch.ones(4), "w": torch.ones(5, 3)},
            str(wrong / "model.safetensors"),
        )
        model = _TinyModel()
        runner = _runner(model, self.root / "step-0")

        ok, message = self._load(runner, wrong)

        assert (model.seen) == ([])
        assert torch.equal(model.b.data, torch.zeros(4))
        assert not (ok)
        assert ("'w' has shape [5, 3], but [4, 3]") in (message)
        assert ("nothing was written") in (message)

    def test_truncated_shard_is_refused_before_anything_is_written(self):
        _write_checkpoint(self.root / "step-0", scale=1.0)
        partial = self.root / "partial"
        partial.mkdir()
        save_file(
            {"b": torch.ones(4)}, str(partial / "model-00001-of-00002.safetensors")
        )
        last = partial / "model-00002-of-00002.safetensors"
        save_file({"w": torch.ones(4, 3)}, str(last))
        last.write_bytes(last.read_bytes()[:-8])  # still being written
        model = _TinyModel()
        runner = _runner(model, self.root / "step-0")
        real_glob = glob.glob

        # Read the shards in name order, so a load without the check writes
        # the first one before it fails on the second.
        with mock.patch.object(
            glob, "glob", side_effect=lambda *a, **k: sorted(real_glob(*a, **k))
        ):
            ok, message = self._load(runner, partial)

        assert (model.seen) == ([])
        assert not (ok)
        assert ("model-00002-of-00002.safetensors") in (message)
        assert ("nothing was written") in (message)

    def test_shapes_of_the_last_reload_survive_its_directory_being_deleted(self):
        # Trainers rotate checkpoint directories; the check must not depend on
        # the previous one still being on disk.
        _write_checkpoint(self.root / "step-0", scale=1.0)
        loaded = _write_checkpoint(self.root / "step-1", scale=2.0)
        model = _TinyModel()
        runner = _runner(model, self.root / "step-0")
        ok, message = self._load(runner, self.root / "step-1")
        _pytest_expr_11 = ok
        _pytest_msg_12 = message
        assert _pytest_expr_11, _pytest_msg_12
        for step in ("step-0", "step-1"):
            for shard in (self.root / step).iterdir():
                shard.unlink()
            (self.root / step).rmdir()
        wrong = self.root / "wrong"
        wrong.mkdir()
        save_file(
            {"b": torch.ones(4), "w": torch.ones(5, 3)},
            str(wrong / "model.safetensors"),
        )

        ok, message = self._load(runner, wrong)

        assert not (ok)
        assert ("'w' has shape [5, 3], but [4, 3]") in (message)
        torch.testing.assert_close(model.b.data, loaded["b"])

    def test_names_the_held_checkpoint_lacks_are_left_to_the_model(self):
        _write_checkpoint(self.root / "step-0", scale=1.0)
        extra = self.root / "extra"
        extra.mkdir()
        save_file(
            {"w": torch.ones(4, 3), "b": torch.ones(4), "mtp.weight": torch.ones(9)},
            str(extra / "model.safetensors"),
        )
        model = _SkippingModel()
        runner = _runner(model, self.root / "step-0")

        ok, message = self._load(runner, extra)

        _pytest_expr_13 = ok
        _pytest_msg_14 = message
        assert _pytest_expr_13, _pytest_msg_14
        assert torch.equal(model.w.data, torch.ones(4, 3))

    def test_one_element_tensors_match_whatever_their_shape(self):
        # The default weight loader fills a one-element parameter from any
        # one-element tensor, so [] and [1] are the same load.
        held = self.root / "step-0"
        held.mkdir()
        save_file({"s": torch.tensor(1.0)}, str(held / "model.safetensors"))
        update = self.root / "step-1"
        update.mkdir()
        save_file({"s": torch.full((1,), 3.0)}, str(update / "model.safetensors"))
        model = _TinyModel()
        model.s = torch.nn.Parameter(torch.zeros(1))
        runner = _runner(model, held)

        ok, message = self._load(runner, update)

        _pytest_expr_15 = ok
        _pytest_msg_16 = message
        assert _pytest_expr_15, _pytest_msg_16
        assert (model.s.item()) == (3.0)

    def test_failure_after_tensors_reached_the_model_reports_a_partial_update(self):
        _write_checkpoint(self.root / "step-1", scale=7.0)
        model = _TinyModel()

        def load_then_fail(weights):
            for name, tensor in weights:
                dict(model.named_parameters())[name].data.copy_(tensor)
                raise RuntimeError("device lost")

        model.load_weights = load_then_fail
        runner = _runner(model, self.root / "step-1")

        ok, message = self._load(runner, self.root / "step-1")

        assert not (ok)
        assert ("device lost") in (message)
        assert ("may now mix the old and new checkpoints") in (message)
        assert ("restart the engine") in (message)

    def test_failure_before_any_tensor_reports_no_partial_update(self):
        runner = _runner(_TinyModel(), self.root / "step-0")

        ok, message = self._load(runner, self.root / "absent")

        assert not (ok)
        assert ("may now mix") not in (message)


class TestReloadMatchesStartupLoad:
    @pytest.fixture(autouse=True)
    def _setup(self, request):
        self._tmp = tempfile.TemporaryDirectory()
        request.addfinalizer(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _load(self, runner, path):
        return runner.update_weights_from_disk(
            UpdateWeightFromDiskReqInput(model_path=str(path))
        )

    def test_checkpoint_dir_is_bound_to_the_new_checkpoint_before_loading(self):
        # DeepSeek-V4.1 reads its Engram tables from the bound directory, so a
        # reload that keeps the old binding loads them from the old checkpoint.
        _write_checkpoint(self.root / "step-0", scale=1.0)
        _write_checkpoint(self.root / "step-1", scale=7.0)
        model = _TinyModel()
        events: list[tuple[str, str]] = []
        model.bind_checkpoint_dir = lambda path: events.append(("bind", path))

        def load(weights):
            for name, tensor in weights:
                events.append(("load", name))
                dict(model.named_parameters())[name].data.copy_(tensor)

        model.load_weights = load
        runner = _runner(model, self.root / "step-0")

        ok, message = self._load(runner, self.root / "step-1")

        _pytest_expr_17 = ok
        _pytest_msg_18 = message
        assert _pytest_expr_17, _pytest_msg_18
        assert (events[0]) == (("bind", str(self.root / "step-1")))
        assert (sorted(name for _, name in events[1:])) == (["b", "w"])

    def _groups_read(self, runner, path) -> list:
        real = loader_module.DefaultModelLoader._get_weights_iterator
        groups = []

        def spy(loader, source, weight_name_filter, checkpoint_load_group, mapping):
            groups.append(checkpoint_load_group)
            return real(
                loader, source, weight_name_filter, checkpoint_load_group, mapping
            )

        with mock.patch.object(
            loader_module.DefaultModelLoader, "_get_weights_iterator", spy
        ):
            ok, message = self._load(runner, path)
        _pytest_expr_19 = ok
        _pytest_msg_20 = message
        assert _pytest_expr_19, _pytest_msg_20
        return groups

    def test_reload_falls_back_to_the_runners_checkpoint_load_group(self):
        # The startup LoadConfig carries the runner's group; a pipeline stage's
        # draft must not join collectives with ranks that never built it.
        _write_checkpoint(self.root / "step-1", scale=7.0)
        runner = _runner(_TinyModel(), self.root / "step-1")
        runner.checkpoint_load_group = (0, 1)

        assert (self._groups_read(runner, self.root / "step-1")) == ([(0, 1)])

    def test_models_own_checkpoint_load_group_wins(self):
        _write_checkpoint(self.root / "step-1", scale=7.0)
        model = _TinyModel()
        # The loading rank (0 here) belongs to every group it reads with; the
        # loader splits shard prefetch among the group's ranks on this node.
        model.checkpoint_load_group = (0, 2)
        runner = _runner(model, self.root / "step-1")
        runner.checkpoint_load_group = (0, 1)

        assert (self._groups_read(runner, self.root / "step-1")) == ([(0, 2)])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
