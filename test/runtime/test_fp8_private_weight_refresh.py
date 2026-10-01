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

"""Keep GLM block-FP8 weights packed without a second resident weight."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from tokenspeed_kernel import PackedFp8WeightCorruptionError
from tokenspeed_kernel.ops.gemm import _PreparedFp8Linear
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp8 import pack_gluon_fp8_blockscale_weight

from tokenspeed.runtime.configs.load_config import LoadConfig, LoadFormat
from tokenspeed.runtime.execution.forward_thread import ForwardThread
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.layers.dense import fp8 as fp8_module
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from tokenspeed.runtime.layers.parameter import ModelWeightParameter
from tokenspeed.runtime.layers.quantization.base_config import (
    finalize_quantized_weights_after_loading,
)
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.model_loader import loader as model_loader
from tokenspeed.runtime.models import glm53_flash, glm53_flash_nextn
from tokenspeed.runtime.utils.env import global_server_args_dict


@pytest.fixture
def packed_model(monkeypatch: pytest.MonkeyPatch) -> torch.nn.Module:
    # A small eligible plan exercises the real CPU pack/unpack and runtime
    # lifecycle without allocating production-sized projection weights.
    def prepare_plan(*args, **kwargs):
        return _PreparedFp8Linear(
            override=None,
            block_size=(128, 128),
            pack_candidate=kwargs["packed_resident"],
        )

    monkeypatch.setattr(fp8_module, "prepare_fp8_linear", prepare_plan)
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
    )
    model = torch.nn.Module()
    model.first = ReplicatedLinear(256, 128, bias=False, quant_config=config)
    model.second = ReplicatedLinear(256, 128, bias=False, quant_config=config)
    for index, layer in enumerate((model.first, model.second)):
        values = ((torch.arange(128 * 256).reshape(128, 256) % 7) + index).to(
            torch.float8_e4m3fn
        )
        layer.weight.data.copy_(values)
        layer.weight_scale_inv.data.fill_(1.0)
        layer.quant_method.packed_resident_requested = True
        layer.quant_method.process_weights_after_loading(layer)
        layer.quant_method.promote_weight_after_loading(layer)
    return model


def test_promotion_replaces_canonical_weight_without_plan_owned_copy(
    packed_model: torch.nn.Module,
) -> None:
    layer = packed_model.first
    plan = layer._prepared_fp8_linear
    assert isinstance(layer.weight, ModelWeightParameter)
    assert layer.quant_method.packed_weight_resident
    assert plan.packed_resident and plan.packed_valid
    assert plan.packed_weight_ptr == layer.weight.data_ptr()
    assert list(plan.parameters()) == []
    assert list(plan.buffers()) == []
    assert next(iter(layer.parameters())).data_ptr() == layer.weight.data_ptr()
    assert torch.equal(
        layer.weight,
        pack_gluon_fp8_blockscale_weight(layer.state_dict()["weight"]),
    )


def test_partial_and_scale_only_reload_leave_untouched_weights_in_place(
    packed_model: torch.nn.Module,
) -> None:
    first, second = packed_model.first, packed_model.second
    first_pointer, second_pointer = first.weight.data_ptr(), second.weight.data_ptr()
    second_before = second.weight.detach().clone()

    assert first.quant_method.packed_weight_resident
    assert second.quant_method.packed_weight_resident

    first.weight_scale_inv.weight_loader(
        first.weight_scale_inv, torch.full_like(first.weight_scale_inv, 2.0)
    )
    finalize_quantized_weights_after_loading(packed_model)
    assert first.weight.data_ptr() == first_pointer
    assert second.weight.data_ptr() == second_pointer
    assert torch.equal(second.weight, second_before)

    updated = torch.full_like(first.state_dict()["weight"], 3.0)
    first.weight.weight_loader(first.weight, updated)
    assert not first.quant_method.packed_weight_resident
    assert not first._prepared_fp8_linear.packed_valid
    assert second.quant_method.packed_weight_resident
    finalize_quantized_weights_after_loading(packed_model)

    assert first.weight.data_ptr() == first_pointer
    assert second.weight.data_ptr() == second_pointer
    assert first._prepared_fp8_linear.packed_valid
    assert torch.equal(first.state_dict()["weight"], updated)
    assert torch.equal(second.weight, second_before)


def test_state_dict_uses_canonical_order_and_load_keeps_packed_pointer(
    packed_model: torch.nn.Module,
) -> None:
    layer = packed_model.first
    pointer = layer.weight.data_ptr()
    canonical = layer.state_dict()["weight"]
    assert not torch.equal(canonical, layer.weight)
    assert torch.equal(pack_gluon_fp8_blockscale_weight(canonical), layer.weight)

    changed = dict(layer.state_dict())
    changed["weight"] = torch.full_like(canonical, 5.0)
    layer.load_state_dict(changed)
    assert layer.weight.data_ptr() == pointer
    assert layer._prepared_fp8_linear.packed_valid
    assert torch.equal(layer.state_dict()["weight"], changed["weight"])

    layer.load_state_dict(
        {"weight_scale_inv": torch.full_like(layer.weight_scale_inv, 4.0)},
        strict=False,
    )
    assert layer.weight.data_ptr() == pointer
    assert layer._prepared_fp8_linear.packed_valid

    with pytest.raises(RuntimeError, match="assign=True"):
        layer.load_state_dict(
            {"weight_scale_inv": torch.full_like(layer.weight_scale_inv, 5.0)},
            strict=False,
            assign=True,
        )


def test_merged_projection_keeps_block_scaled_shard_loaders(monkeypatch) -> None:
    def prepare_plan(*args, **kwargs):
        return _PreparedFp8Linear(
            override=None,
            block_size=(128, 128),
            pack_candidate=kwargs["packed_resident"],
        )

    monkeypatch.setattr(fp8_module, "prepare_fp8_linear", prepare_plan)
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
    )
    layer = MergedColumnParallelLinear(
        4096, [512, 512], bias=False, quant_config=config
    )
    layer.weight.data.zero_()
    layer.weight_scale_inv.data.fill_(1.0)
    layer.quant_method.packed_resident_requested = True
    layer.quant_method.process_weights_after_loading(layer)
    layer.quant_method.promote_weight_after_loading(layer)
    pointer = layer.weight.data_ptr()

    for shard in (0, 1):
        layer.weight.weight_loader(
            layer.weight,
            torch.full((512, 4096), shard + 2, dtype=layer.weight.dtype),
            shard,
        )
        layer.weight_scale_inv.weight_loader(
            layer.weight_scale_inv,
            torch.full((4, 32), shard + 3, dtype=torch.float32),
            shard,
        )
    finalize_quantized_weights_after_loading(layer)

    assert layer.weight.data_ptr() == pointer
    for shard in (0, 1):
        weight = layer.state_dict()["weight"][shard * 512 : (shard + 1) * 512]
        scale = layer.weight_scale_inv[shard * 4 : (shard + 1) * 4]
        assert torch.all(weight == shard + 2)
        assert torch.all(scale == shard + 3)


def test_device_move_rebinds_the_packed_weight_pointer(
    packed_model: torch.nn.Module,
) -> None:
    layer = packed_model.first
    canonical = layer.state_dict()["weight"]
    pointer = layer.weight.data_ptr()
    layer._apply(lambda tensor: tensor.clone())
    assert layer.weight.data_ptr() != pointer
    assert layer._prepared_fp8_linear.packed_weight_ptr == layer.weight.data_ptr()
    assert isinstance(layer.weight, ModelWeightParameter)
    assert torch.equal(layer.state_dict()["weight"], canonical)


def test_dtype_conversion_of_packed_weight_fails_closed(
    packed_model: torch.nn.Module,
) -> None:
    layer = packed_model.first
    with pytest.raises(RuntimeError, match="original dtype and layout"):
        layer.to(torch.bfloat16)
    assert not layer._prepared_fp8_linear.packed_valid
    with pytest.raises(RuntimeError, match="dtype changed"):
        layer.state_dict()


def test_failed_refresh_restores_pointer_but_blocks_execution(
    packed_model: torch.nn.Module, monkeypatch: pytest.MonkeyPatch
) -> None:
    layer = packed_model.first
    pointer = layer.weight.data_ptr()
    updated = torch.full_like(layer.state_dict()["weight"], 6.0)
    layer.weight.weight_loader(layer.weight, updated)

    def fail_refresh(*args):
        raise RuntimeError("injected pack failure")

    monkeypatch.setattr(fp8_module, "refresh_fp8_linear_weight", fail_refresh)
    with pytest.raises(RuntimeError, match="injected pack failure"):
        finalize_quantized_weights_after_loading(layer)

    assert layer.weight.data_ptr() == pointer
    assert layer.quant_method.packed_weight_resident
    assert not layer._prepared_fp8_linear.packed_valid
    with pytest.raises(RuntimeError, match="not ready for execution"):
        fp8_module.fp8_linear(
            layer._prepared_fp8_linear,
            torch.empty((1, 256), dtype=torch.bfloat16),
            layer.weight,
            layer.weight_scale_inv,
        )


def test_failed_weight_loader_keeps_previous_packed_weight(
    packed_model: torch.nn.Module,
) -> None:
    layer = packed_model.first
    canonical = layer.state_dict()["weight"]
    pointer = layer.weight.data_ptr()

    with pytest.raises(AssertionError):
        layer.weight.weight_loader(
            layer.weight, torch.empty((1, 1), dtype=torch.float8_e4m3fn)
        )

    assert layer.weight.data_ptr() == pointer
    assert layer.quant_method.packed_weight_resident
    assert layer._prepared_fp8_linear.packed_valid
    assert torch.equal(layer.state_dict()["weight"], canonical)


def test_failed_model_reload_restores_all_staged_packed_weights(
    packed_model: torch.nn.Module, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = packed_model
    model.config = type("Config", (), {"num_hidden_layers": 2})()
    model.post_load_weights = lambda: None
    old = [module.state_dict()["weight"] for module in (model.first, model.second)]
    old_scales = [
        module.weight_scale_inv.detach().clone()
        for module in (model.first, model.second)
    ]
    pointers = [module.weight.data_ptr() for module in (model.first, model.second)]

    def failed_load(owner, _weights):
        for module in (owner.first, owner.second):
            module.weight.weight_loader(module.weight, torch.full_like(old[0], 4.0))
            module.weight_scale_inv.weight_loader(
                module.weight_scale_inv,
                torch.full_like(module.weight_scale_inv, 3.0),
            )
        raise RuntimeError("checkpoint stream ended")

    monkeypatch.setattr(glm53_flash, "load_glm53_flash_text_weights", failed_load)
    with pytest.raises(RuntimeError, match="checkpoint stream ended"):
        glm53_flash.Glm53FlashForCausalLM.load_weights(model, [])

    for module, previous, old_scale, pointer in zip(
        (model.first, model.second), old, old_scales, pointers, strict=True
    ):
        assert module.weight.data_ptr() == pointer
        assert module._prepared_fp8_linear.packed_valid
        assert torch.equal(module.state_dict()["weight"], previous)
        assert torch.equal(module.weight_scale_inv, old_scale)


def test_failed_scale_restore_is_fatal(
    packed_model: torch.nn.Module, monkeypatch: pytest.MonkeyPatch
) -> None:
    layer = packed_model.first
    layer.weight.weight_loader(
        layer.weight, torch.zeros_like(layer.state_dict()["weight"])
    )
    layer.weight_scale_inv.weight_loader(
        layer.weight_scale_inv, torch.full_like(layer.weight_scale_inv, 3.0)
    )
    previous = layer.quant_method._scale_before_reload
    scale_pointer = layer.weight_scale_inv.data_ptr()
    copy = torch.Tensor.copy_

    def fail_restore(self, source, *args, **kwargs):
        if (
            self.data_ptr() == scale_pointer
            and source.data_ptr() == previous.data_ptr()
        ):
            raise RuntimeError("injected scale restore failure")
        return copy(self, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_restore)
    with pytest.raises(
        PackedFp8WeightCorruptionError, match="scale could not be restored"
    ):
        layer.quant_method.abort_weight_update(layer)
    assert not layer._prepared_fp8_linear.packed_valid


def test_fatal_refresh_error_survives_model_reload_cleanup(
    packed_model: torch.nn.Module, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = packed_model
    model.config = type("Config", (), {"num_hidden_layers": 2})()
    model.post_load_weights = lambda: None
    layer = model.first
    pointer = layer.weight.data_ptr()

    def load_one_weight(owner, _weights):
        owner.first.weight.weight_loader(
            owner.first.weight, torch.zeros_like(owner.first.state_dict()["weight"])
        )

    def fail_refresh(*_args):
        raise PackedFp8WeightCorruptionError("injected irrecoverable restore")

    monkeypatch.setattr(glm53_flash, "load_glm53_flash_text_weights", load_one_weight)
    monkeypatch.setattr(fp8_module, "refresh_fp8_linear_weight", fail_refresh)
    with pytest.raises(PackedFp8WeightCorruptionError, match="irrecoverable"):
        glm53_flash.Glm53FlashForCausalLM.load_weights(model, [])
    assert layer.weight.data_ptr() == pointer
    assert not layer._prepared_fp8_linear.packed_valid


@pytest.mark.parametrize(
    "reload_method",
    (
        glm53_flash.Glm53FlashForCausalLM.load_weights,
        glm53_flash_nextn.Glm53FlashForConditionalGenerationNextN.load_weights,
    ),
    ids=("target", "draft"),
)
@pytest.mark.parametrize("source_is_fatal", (False, True))
def test_failed_abort_is_fatal_without_masking_source_error(
    packed_model: torch.nn.Module,
    monkeypatch: pytest.MonkeyPatch,
    reload_method,
    source_is_fatal: bool,
) -> None:
    model = packed_model
    model.config = type("Config", (), {"num_hidden_layers": 2})()
    model.post_load_weights = lambda: None
    first_weight = model.first.state_dict()["weight"]
    first_pointer = model.first.weight.data_ptr()
    source_error = (
        PackedFp8WeightCorruptionError("original fatal refresh")
        if source_is_fatal
        else RuntimeError("ordinary checkpoint failure")
    )

    def fail_after_staging(owner, _weights):
        for layer in (owner.first, owner.second):
            layer.weight.weight_loader(
                layer.weight, torch.zeros_like(layer.state_dict()["weight"])
            )
        raise source_error

    def fail_second_abort(_layer):
        raise RuntimeError("second-layer abort failed")

    monkeypatch.setattr(
        glm53_flash, "load_glm53_flash_text_weights", fail_after_staging
    )
    monkeypatch.setattr(
        glm53_flash_nextn, "load_glm53_flash_text_weights", fail_after_staging
    )
    monkeypatch.setattr(
        model.second.quant_method, "abort_weight_update", fail_second_abort
    )
    with pytest.raises(PackedFp8WeightCorruptionError) as raised:
        reload_method(model, [])

    if source_is_fatal:
        assert raised.value is source_error
    else:
        assert "could not be restored" in str(raised.value)
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "second-layer abort failed" in str(raised.value.__cause__)
    assert model.first.weight.data_ptr() == first_pointer
    assert torch.equal(model.first.state_dict()["weight"], first_weight)


def test_fatal_packed_weight_error_escapes_online_update(monkeypatch) -> None:
    class FailedModel(torch.nn.Module):
        def load_weights(self, _weights):
            raise PackedFp8WeightCorruptionError("injected irrecoverable restore")

    runner = ModelRunner.__new__(ModelRunner)
    runner.model = FailedModel()
    runner._weight_update_pg = object()
    runner._weight_update_device = torch.device("cpu")
    request = SimpleNamespace(names=[], dtype_names=[], shapes=[])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    with pytest.raises(PackedFp8WeightCorruptionError, match="irrecoverable"):
        runner.update_weights_from_distributed(request)


def test_fatal_packed_weight_error_reaches_scheduler_thread() -> None:
    thread = ForwardThread("cpu")

    def fail_update():
        raise PackedFp8WeightCorruptionError("injected irrecoverable restore")

    try:
        with pytest.raises(PackedFp8WeightCorruptionError, match="irrecoverable"):
            thread.run(fail_update)
    finally:
        thread.shutdown()


def test_sharded_state_save_and_reload_promotes_after_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("TOKENSPEED_EXPERIMENTAL_GLUON_FP8_BLOCKSCALE", "1")
    monkeypatch.setitem(global_server_args_dict, "dense_gemm_backend", "triton")
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
    )

    def make_model():
        model = torch.nn.Module()
        model.proj = ReplicatedLinear(4096, 1024, bias=False, quant_config=config)
        model.proj.quant_method.packed_resident_requested = True
        return model

    source = make_model()
    source.proj.weight.data.copy_(
        (torch.arange(source.proj.weight.numel()).reshape(1024, 4096) % 7).to(
            torch.float8_e4m3fn
        )
    )
    source.proj.weight_scale_inv.data.fill_(2.0)
    expected = {name: value.clone() for name, value in source.state_dict().items()}
    checkpoint = tmp_path / "model-rank-0-part-0.safetensors"
    save_file(expected, checkpoint)

    monkeypatch.setattr(model_loader, "_initialize_model", lambda *_args: make_model())
    loader = model_loader.ShardedStateLoader(
        LoadConfig(load_format=LoadFormat.SHARDED_STATE)
    )
    model_config = SimpleNamespace(
        model_path=str(tmp_path),
        revision=None,
        dtype=torch.bfloat16,
        mapping=SimpleNamespace(rank=0),
    )
    device_config = SimpleNamespace(device="cpu")
    loaded = loader.load_model(model_config=model_config, device_config=device_config)
    assert loaded.proj.quant_method.packed_weight_resident
    assert (
        loaded.proj._prepared_fp8_linear.packed_weight_ptr
        == loaded.proj.weight.data_ptr()
    )
    for name, value in expected.items():
        assert torch.equal(loaded.state_dict()[name], value)

    save_file(
        {name: value.clone() for name, value in loaded.state_dict().items()}, checkpoint
    )
    reloaded = loader.load_model(model_config=model_config, device_config=device_config)
    for name, value in expected.items():
        assert torch.equal(reloaded.state_dict()[name], value)


@pytest.mark.parametrize("load_format", [LoadFormat.AUTO, LoadFormat.DUMMY])
def test_streaming_and_dummy_loaders_promote_after_final_write(
    monkeypatch: pytest.MonkeyPatch, load_format: LoadFormat
) -> None:
    monkeypatch.setenv("TOKENSPEED_EXPERIMENTAL_GLUON_FP8_BLOCKSCALE", "1")
    monkeypatch.setitem(global_server_args_dict, "dense_gemm_backend", "triton")
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
    )

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = ReplicatedLinear(4096, 1024, bias=False, quant_config=config)
            self.proj.quant_method.packed_resident_requested = True

        def load_weights(self, weights):
            params = dict(self.named_parameters())
            for name, value in weights:
                param = params[name]
                param.weight_loader(param, value)

    weight = (torch.arange(1024 * 4096).reshape(1024, 4096) % 7).to(torch.float8_e4m3fn)
    scales = torch.full((8, 32), 1.5)
    expected = {"proj.weight": weight, "proj.weight_scale_inv": scales}
    monkeypatch.setattr(model_loader, "_initialize_model", lambda *_args: ToyModel())
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    device_config = SimpleNamespace(device="cpu")
    if load_format == LoadFormat.AUTO:
        loader = model_loader.DefaultModelLoader(LoadConfig(load_format=load_format))
        monkeypatch.setattr(loader, "_get_all_weights", lambda *_args: expected.items())
    else:
        loader = model_loader.DummyModelLoader(LoadConfig(load_format=load_format))

        def dummy_write(model):
            assert not model.proj.quant_method.packed_weight_resident
            for name, value in expected.items():
                dict(model.named_parameters())[name].data.copy_(value)

        monkeypatch.setattr(model_loader, "initialize_dummy_weights", dummy_write)

    loaded = loader.load_model(model_config=model_config, device_config=device_config)
    assert loaded.proj.quant_method.packed_weight_resident
    assert (
        loaded.proj._prepared_fp8_linear.packed_weight_ptr
        == loaded.proj.weight.data_ptr()
    )
    for name, value in expected.items():
        assert torch.equal(loaded.state_dict()[name], value)
