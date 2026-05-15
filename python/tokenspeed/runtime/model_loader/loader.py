# Adapted from meituan-longcat/SGLang-FluentLLM.
# This file has been modified for this repository.
# Upstream lineage includes ModelTC/lightllm, vllm-project/vllm,
# and sgl-project/sglang. See python/THIRDPARTYNOTICES.
# Licensed under the Apache License, Version 2.0
#
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


# ruff: noqa: SIM117
import collections
import dataclasses
import glob
import os
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, cast

import huggingface_hub
import torch
import yaml
from tokenspeed_kernel.platform import current_platform
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from tokenspeed.runtime.configs.device_config import DeviceConfig
from tokenspeed.runtime.configs.load_config import LoadConfig, LoadFormat
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.model_loader.utils import (
    get_model_architecture,
    set_default_torch_dtype,
)
from tokenspeed.runtime.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    get_quant_config,
    initialize_dummy_weights,
    np_cache_weights_iterator,
    pt_weights_iterator,
    safetensors_weights_iterator,
)
from tokenspeed.runtime.models.extensible import ExtensibleLM
from tokenspeed.runtime.utils import get_colorful_logger, is_pin_memory_available


@contextmanager
def device_loading_context(
    module: torch.nn.Module, target_device: torch.device
) -> Iterator[torch.nn.Module]:
    if target_device.type == "cpu":
        # If target is CPU, no need to move anything
        yield module
        return

    original_device_states: dict[str, torch.device] = {}

    # Store original device states and move parameters to GPU if they're on CPU
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_device_states[name] = p.device
            p.data = p.data.to(target_device)
        # Parameters already on target device are not touched

    try:
        yield module

    finally:
        # Restore parameters to their original devices, ignoring new parameters
        pin_memory = is_pin_memory_available()
        for name, p in module.named_parameters():
            if name in original_device_states:
                original_device: torch.device = original_device_states[name]
                if original_device.type == "cpu":
                    # `torch.empty_like` does not support `pin_memory` argument
                    cpu_data = torch.empty_strided(
                        size=p.data.size(),
                        stride=p.data.stride(),
                        dtype=p.data.dtype,
                        layout=p.data.layout,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                    cpu_data.copy_(p.data)
                    p.data = cpu_data
                else:
                    p.data = p.data.to(original_device)
        # New parameters or parameters already on target device are untouched


logger = get_colorful_logger(__name__)


def _get_quantization_config(
    model_config: ModelConfig, load_config: LoadConfig
) -> QuantizationConfig | None:
    """Get the quantization config."""
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config, load_config)
        platform = current_platform()
        capability = platform.arch_version.major * 10 + platform.arch_version.minor
        if capability < quant_config.get_min_capability():
            raise ValueError(
                f"The quantization method {model_config.quantization} "
                "is not supported for the current GPU. "
                f"Minimum capability: {quant_config.get_min_capability()}. "
                f"Current capability: {capability}."
            )
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}"
            )
        return quant_config
    return None


def _initialize_model(
    model_config: ModelConfig,
    load_config: LoadConfig,
) -> nn.Module:
    """Initialize a model with the given configurations."""
    model_class, _ = get_model_architecture(model_config)
    quant_config = _get_quantization_config(model_config, load_config)
    mapping = model_config.mapping
    return model_class(
        config=model_config.hf_config,
        mapping=mapping,
        quant_config=quant_config,
    )


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig) -> None:
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        """Load a model with the given configurations."""
        raise NotImplementedError


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: str | None
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def _maybe_download_from_modelscope(
        self, model: str, revision: str | None
    ) -> str | None:
        """Download model from ModelScope hub if TOKENSPEED_USE_MODELSCOPE is True.

        Returns the path to the downloaded model, or None if the model is not
        downloaded from ModelScope."""
        from tokenspeed.runtime.utils.env import envs

        if envs.TOKENSPEED_USE_MODELSCOPE.is_set():
            # download model from ModelScope hub,
            # lazy import so that modelscope is not required for normal use.
            from modelscope.hub.snapshot_download import snapshot_download

            if not os.path.exists(model):
                model_path = snapshot_download(
                    model_id=model,
                    cache_dir=self.load_config.download_dir,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    revision=revision,
                    ignore_file_pattern=self.load_config.ignore_patterns,
                )
            else:
                model_path = model
            return model_path
        return None

    def _prepare_weights(
        self, model_name_or_path: str, revision: str | None, fall_back_to_pt: bool
    ) -> tuple[str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = (
            self._maybe_download_from_modelscope(model_name_or_path, revision)
            or model_name_or_path
        )

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME
        # Some quantized models use .pt files for storing the weights.
        if load_format == LoadFormat.AUTO:
            allow_patterns = ["*.safetensors", "*.bin"]
        elif load_format == LoadFormat.SAFETENSORS:
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == LoadFormat.MISTRAL:
            use_safetensors = True
            allow_patterns = ["consolidated*.safetensors"]
            index_file = "consolidated.safetensors.index.json"
        elif load_format == LoadFormat.PT:
            allow_patterns = ["*.pt"]
        elif load_format == LoadFormat.NPCACHE:
            allow_patterns = ["*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    self.load_config.download_dir,
                    revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file
            )
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, source: "Source"
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path, source.revision, source.fall_back_to_pt
        )
        if self.load_config.load_format == LoadFormat.NPCACHE:
            # Currently np_cache only support *.bin checkpoints
            assert use_safetensors is False
            weights_iterator = np_cache_weights_iterator(
                source.model_or_path,
                self.load_config.download_dir,
                hf_folder,
                hf_weights_files,
            )
        elif use_safetensors:
            weights_iterator = safetensors_weights_iterator(
                hf_weights_files,
                prefetch=self.load_config.weight_loader_prefetch_checkpoints,
                prefetch_num_threads=self.load_config.weight_loader_prefetch_num_threads,
            )
        else:
            weights_iterator = pt_weights_iterator(hf_weights_files)

        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    def _get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:

        primary_weights = DefaultModelLoader.Source(
            model_config.model_path,
            model_config.revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", False),
        )
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source], getattr(model, "secondary_weights", ())
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(
            model_config.model_path, model_config.revision, fall_back_to_pt=True
        )

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(
                    model_config,
                    self.load_config,
                )

            model.load_weights(self._get_all_weights(model_config, model))

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    # When quant methods need to process weights after loading
                    # (for repacking, quantizing, etc), they expect parameters
                    # to be on the global target device. This scope is for the
                    # case where cpu offloading is used, where we will move the
                    # parameters onto device for processing and back off after.
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)

                process_method = getattr(module, "process_weights_after_loading", None)
                if process_method is not None:
                    with device_loading_context(module, target_device):
                        module.process_weights_after_loading(module)

        return model.eval()


class ExtensibleModelLoader:
    def __init__(self, load_config: LoadConfig) -> None:
        load_config.load_format = LoadFormat.AUTO
        self.base_lm_loader = DefaultModelLoader(load_config)
        self.ext_yaml = load_config.ext_yaml

    def load_model(
        self,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        with open(self.ext_yaml) as f:
            ext_config = yaml.safe_load(f)
        base_lm = self.base_lm_loader.load_model(
            model_config=model_config, device_config=device_config
        )
        ext_lm = ExtensibleLM(base_lm=base_lm, ext_config=ext_config)
        return ext_lm


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                )
            if getattr(model, "post_load_weights", None):
                model.post_load_weights()

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)
                process_method = getattr(module, "process_weights_after_loading", None)
                if process_method is not None:
                    module.process_weights_after_loading(module)

            #  For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        return model.eval()


class ShardedStateLoader(BaseModelLoader):
    """
    Model loader that directly loads each worker's model state dict, which
    enables a fast load path for large tensor-parallel models where each worker
    only needs to read its own shard rather than the entire checkpoint. See
    `examples/save_sharded_state.py` for creating a sharded checkpoint.
    """

    DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = (
            {}
            if load_config.model_loader_extra_config is None
            else load_config.model_loader_extra_config.copy()
        )
        self.pattern = extra_config.pop("pattern", self.DEFAULT_PATTERN)
        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{load_config.model_loader_extra_config.keys()}"
            )

    @staticmethod
    def _filter_subtensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Filter out all tensors that share the same memory or a subset of the
        memory of another tensor.
        """
        same_storage_groups: dict[Any, list[tuple[str, torch.Tensor]]] = (
            collections.defaultdict(list)
        )
        for key, tensor in tensors.items():
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                same_storage_groups[tensor.device, ptr].append((key, tensor))

        def get_end_ptr(tensor: torch.Tensor) -> int:
            return tensor.view(-1)[-1].data_ptr() + tensor.element_size()

        result: dict[str, torch.Tensor] = {}
        for group in same_storage_groups.values():
            for k, t in group:
                a, b = t.data_ptr(), get_end_ptr(t)
                for k2, t2 in group:
                    if not t2.is_contiguous():
                        continue
                    a2, b2 = t2.data_ptr(), get_end_ptr(t2)
                    if a < a2 or b2 < b:
                        continue
                    if a2 < a or b < b2 or not t.is_contiguous():
                        break  # t2 covers strictly more memory than t.
                    if k2 < k:
                        # Same tensors, keep the one with the smaller key.
                        break
                else:
                    result[k] = t
        return result

    def _prepare_weights(self, model_name_or_path: str, revision: str | None):
        if os.path.isdir(model_name_or_path):
            return model_name_or_path

        allow_patterns = ["*.safetensors"]
        return download_weights_from_hf(
            model_name_or_path,
            self.load_config.download_dir,
            allow_patterns,
            revision,
            ignore_patterns=self.load_config.ignore_patterns,
        )

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model_path, model_config.revision)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        from safetensors.torch import safe_open

        local_model_path = self._prepare_weights(
            model_config.model_path, model_config.revision
        )

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config)
                for _, module in model.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        quant_method.process_weights_after_loading(module)
                    process_method = getattr(
                        module, "process_weights_after_loading", None
                    )
                    if process_method is not None:
                        module.process_weights_after_loading(module)
            rank = model_config.mapping.rank
            pattern = os.path.join(
                local_model_path,
                self.pattern.format(rank=rank, part="*"),
            )
            filepaths = glob.glob(pattern)
            if not filepaths:
                raise ValueError(
                    f"Could not find checkpoint files '{pattern}', only "
                    f"pre-sharded checkpoints are currently supported!"
                )
            state_dict = self._filter_subtensors(model.state_dict())
            for path in filepaths:
                with safe_open(path, framework="pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        tensor = f.get_tensor(key)
                        # If loading with LoRA enabled, additional padding may
                        # be added to certain parameters. We only load into a
                        # narrowed view of the parameter data.
                        param_data = state_dict[key].data
                        param_shape = state_dict[key].shape
                        for dim, size in enumerate(tensor.shape):
                            if size < param_shape[dim]:
                                param_data = param_data.narrow(dim, 0, size)
                        if tensor.shape != param_shape:
                            logger.warning(
                                "loading tensor of shape %s into "
                                "parameter '%s' of shape %s",
                                tensor.shape,
                                key,
                                param_shape,
                            )
                        param_data.copy_(tensor)
                        state_dict.pop(key)
            if state_dict:
                raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")
        return model.eval()


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""

    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.DUMMY:
        return DummyModelLoader(load_config)

    if load_config.load_format == LoadFormat.SHARDED_STATE:
        return ShardedStateLoader(load_config)

    if load_config.load_format == LoadFormat.EXTENSIBLE:
        return ExtensibleModelLoader(load_config)

    return DefaultModelLoader(load_config)
