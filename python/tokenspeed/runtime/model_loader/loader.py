# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from typing import Any, Protocol, cast, runtime_checkable

import huggingface_hub
import torch
import yaml
from tokenspeed_kernel.platform import current_platform
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from tokenspeed.runtime.configs.device_config import DeviceConfig
from tokenspeed.runtime.configs.load_config import LoadConfig, LoadFormat
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from tokenspeed.runtime.model_loader.utils import (
    get_model_architecture,
    set_default_torch_dtype,
)
from tokenspeed.runtime.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    filter_safetensors_files_by_weight_names,
    get_quant_config,
    initialize_dummy_integer_weights,
    initialize_dummy_weights,
    instanttensor_weights_iterator,
    non_unit_kv_scale_message,
    np_cache_weights_iterator,
    pt_weights_iterator,
    runai_streamer_weights_iterator,
    safetensors_filtered_weights_iterator,
    safetensors_weights_iterator,
    split_kv_cache_scales,
)
from tokenspeed.runtime.models.extensible import ExtensibleLM
from tokenspeed.runtime.utils import get_colorful_logger, is_pin_memory_available


@contextmanager
def device_loading_context(
    module: torch.nn.Module, target_device: torch.device
) -> Generator[torch.nn.Module]:
    if target_device.type == "cpu":
        # If target is CPU, no need to move anything
        yield module
        return

    original_device_states: dict[str, torch.device] = {}

    # Store original device states and move parameters to GPU if they're on CPU
    for name, p in module.named_parameters():
        if p.device.type == "cpu" and not getattr(p, "ple_shared_storage", False):
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


from tokenspeed.runtime.utils.startup_timing import startup_phase

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
    if quant_config is not None:
        replacements = getattr(model_class, "quant_module_name_replacements", None)
        if replacements:
            quant_config.apply_checkpoint_name_replacements(replacements)
    mapping = model_config.mapping
    if mapping.moe_weight_sharing_size is not None and not model_config.is_draft_worker:
        from tokenspeed.runtime.layers.moe.peer_weights import PeerWeightOwner

        if not issubclass(model_class, PeerWeightOwner):
            raise ValueError(
                "This model does not implement peer expert-weight ownership"
            )
    # Profiles opt into runtime policy; VLM wrappers also receive encoder settings.
    extra_kwargs: dict = {}
    if model_config.runtime_config is not None:
        extra_kwargs["runtime_config"] = model_config.runtime_config
    if model_config.is_multimodal:
        extra_kwargs["is_multimodal_active"] = model_config.is_multimodal_active
        extra_kwargs["mm_attention_backend"] = model_config.mm_attention_backend
    profile = model_config.model_profile
    if profile is not None and profile.create_model_kwargs is not None:
        model_kwargs = profile.create_model_kwargs(model_config, load_config)
        conflicts = extra_kwargs.keys() & model_kwargs.keys()
        if conflicts:
            raise ValueError(
                f"Model constructor arguments already supplied: {sorted(conflicts)}"
            )
        extra_kwargs.update(model_kwargs)
    return model_class(
        config=model_config.hf_config,
        mapping=mapping,
        quant_config=quant_config,
        **extra_kwargs,
    )


@runtime_checkable
class CheckpointFileSelection(Protocol):
    """A model whose safetensors files are independent of the target index.

    ``checkpoint_weight_files`` lists exact basenames, without glob patterns.
    Missing files are errors, not permission to load a different checkpoint.
    """

    checkpoint_weight_files: tuple[str, ...]


@runtime_checkable
class KVCacheScaleLoader(Protocol):
    """A model that puts its checkpoint's KV-cache scales on its attention layers."""

    def load_kv_cache_scales(self, scales: dict[str, torch.Tensor]) -> None:
        """Hand each scale (keyed by checkpoint name) to its layer, skipping
        only names that belong to another model; raise on any other name."""


def _load_kv_cache_scales(model: nn.Module, scales: dict[str, torch.Tensor]) -> None:
    """Startup is the only time a model takes KV-cache scales other than one;
    weight updates drop them (``record_non_unit_kv_scales``)."""
    if not isinstance(model, KVCacheScaleLoader):
        raise ValueError(
            f"{type(model).__name__}: "
            + non_unit_kv_scale_message(
                [f"{name}={t.flatten()[:4].tolist()}" for name, t in scales.items()]
            )
        )
    model.load_kv_cache_scales(scales)


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

        weight_files: tuple[str, ...] | None
        """Exact safetensors basenames, or normal format/index discovery."""

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
        self,
        model_name_or_path: str,
        revision: str | None,
        fall_back_to_pt: bool,
        weight_files: tuple[str, ...] | None,
    ) -> tuple[str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        from tokenspeed.runtime.model_loader.object_store import is_object_store_uri

        if is_object_store_uri(model_name_or_path):
            raise ValueError(
                f"Object-store URI {model_name_or_path!r} requires "
                f"--load-format runai_streamer, not {self.load_config.load_format!r}"
            )
        model_name_or_path = (
            self._maybe_download_from_modelscope(model_name_or_path, revision)
            or model_name_or_path
        )

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        if weight_files is not None:
            if load_format not in (
                LoadFormat.AUTO,
                LoadFormat.SAFETENSORS,
                LoadFormat.INSTANTTENSOR,
                LoadFormat.RUNAI_STREAMER,
            ):
                raise ValueError(
                    "Explicit checkpoint files require safetensors loading"
                )
            hf_folder = (
                model_name_or_path
                if is_local
                else download_weights_from_hf(
                    model_name_or_path,
                    self.load_config.download_dir,
                    list(weight_files),
                    revision,
                    ignore_patterns=self.load_config.ignore_patterns,
                )
            )
            files = [os.path.join(hf_folder, name) for name in weight_files]
            for path in files:
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"Required checkpoint file is missing: {path}"
                    )
            # These files can intentionally be absent from the target's index.
            return hf_folder, files, True

        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME
        # Some quantized models use .pt files for storing the weights.
        if load_format == LoadFormat.AUTO:
            allow_patterns = ["*.safetensors", "*.bin"]
        elif load_format in (
            LoadFormat.SAFETENSORS,
            LoadFormat.INSTANTTENSOR,
            LoadFormat.RUNAI_STREAMER,
        ):
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
        self,
        source: "Source",
        weight_name_filter: Callable[[str], bool] | None,
        checkpoint_load_group: tuple[int, ...] | None,
        mapping: Mapping,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format.

        When ``weight_name_filter`` is given, safetensors shards whose index
        entries contain no accepted weight name are skipped entirely (not
        read or prefetched). The predicate sees names as yielded to the
        consumer, i.e. with ``source.prefix`` applied. Distributed loaders
        only synchronize ranks in ``checkpoint_load_group`` when one is
        declared: pipeline stages can consume different checkpoint subsets.
        """
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.revision,
            source.fall_back_to_pt,
            source.weight_files,
        )
        hf_weights_files.sort()
        # Filtered models can consume different shard lists on different ranks.
        # Keep their independent prefetch rather than assign reads to absent peers.
        local_rank, local_world_size = 0, 1
        if use_safetensors and weight_name_filter is None:
            node_start = mapping.node_rank * mapping.nprocs_per_node
            local_ranks = tuple(
                rank
                for rank in range(node_start, node_start + mapping.nprocs_per_node)
                if checkpoint_load_group is None or rank in checkpoint_load_group
            )
            local_rank = local_ranks.index(mapping.rank)
            local_world_size = len(local_ranks)
        if use_safetensors and weight_name_filter is not None:
            index_file = (
                "consolidated.safetensors.index.json"
                if self.load_config.load_format == LoadFormat.MISTRAL
                else SAFE_WEIGHTS_INDEX_NAME
            )
            hf_weights_files = filter_safetensors_files_by_weight_names(
                hf_weights_files,
                hf_folder,
                index_file,
                lambda name: weight_name_filter(source.prefix + name),
            )
        if self.load_config.load_format == LoadFormat.NPCACHE:
            # Currently np_cache only support *.bin checkpoints
            if use_safetensors:
                raise ValueError("np_cache only supports PyTorch checkpoint shards.")
            weights_iterator = np_cache_weights_iterator(
                source.model_or_path,
                self.load_config.download_dir,
                hf_folder,
                hf_weights_files,
            )
        elif self.load_config.load_format == LoadFormat.INSTANTTENSOR:
            process_group = None
            if checkpoint_load_group is not None:
                from tokenspeed.runtime.distributed.process_group_manager import (
                    process_group_manager as pg_manager,
                )

                process_group = pg_manager.get_process_group(
                    "nccl", checkpoint_load_group
                )
            elif (
                torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            ):
                process_group = torch.distributed.group.WORLD
            weights_iterator = instanttensor_weights_iterator(
                hf_weights_files, process_group=process_group
            )
        elif use_safetensors and weight_name_filter is not None:
            weights_iterator = safetensors_filtered_weights_iterator(
                hf_weights_files,
                lambda name: weight_name_filter(source.prefix + name),
                prefetch=self.load_config.weight_loader_prefetch_checkpoints,
                prefetch_num_threads=self.load_config.weight_loader_prefetch_num_threads,
                local_rank=local_rank,
                local_world_size=local_world_size,
            )
        elif use_safetensors:
            weights_iterator = safetensors_weights_iterator(
                hf_weights_files,
                prefetch=self.load_config.weight_loader_prefetch_checkpoints,
                prefetch_num_threads=self.load_config.weight_loader_prefetch_num_threads,
                local_rank=local_rank,
                local_world_size=local_world_size,
            )
        else:
            weights_iterator = pt_weights_iterator(hf_weights_files)

        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    def _get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        yield from self.get_checkpoint_weights(
            model_config.model_path, model_config.revision, model, model_config.mapping
        )

    def get_checkpoint_weights(
        self,
        model_path: str,
        revision: str | None,
        model: nn.Module,
        mapping: Mapping,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Stream the checkpoint at ``model_path`` the way ``load_model`` does.

        The in-place reload (``ModelRunner.update_weights_from_disk``) reads
        through here too, so a reloaded model binds the new checkpoint
        directory and picks its load group exactly as at startup.
        """
        # Draft (NextN/MTP) models embedded in the target checkpoint expose
        # ``checkpoint_weight_name_filter`` so only the shards holding their
        # weights are read instead of the whole checkpoint.
        weight_name_filter = getattr(model, "checkpoint_weight_name_filter", None)
        weight_files = (
            model.checkpoint_weight_files
            if isinstance(model, CheckpointFileSelection)
            else None
        )
        bind_checkpoint_dir = getattr(model, "bind_checkpoint_dir", None)
        if callable(bind_checkpoint_dir):
            hf_folder, _, _ = self._prepare_weights(
                model_path,
                revision,
                getattr(model, "fall_back_to_pt_during_load", False),
                weight_files,
            )
            bind_checkpoint_dir(hf_folder)
        # A model that reads a stage-specific checkpoint subset names its own
        # group; otherwise the caller says which ranks load this model at all.
        checkpoint_load_group = getattr(model, "checkpoint_load_group", None)
        if checkpoint_load_group is None:
            checkpoint_load_group = self.load_config.checkpoint_load_group

        primary_weights = DefaultModelLoader.Source(
            model_path,
            revision,
            weight_files=weight_files,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", False),
        )
        yield from self._get_weights_iterator(
            primary_weights,
            weight_name_filter,
            checkpoint_load_group,
            mapping,
        )

        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source], getattr(model, "secondary_weights", ())
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(
                source, weight_name_filter, checkpoint_load_group, mapping
            )

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(
            model_config.model_path,
            model_config.revision,
            fall_back_to_pt=True,
            weight_files=None,
        )

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with startup_phase("weights.allocate"):
                with target_device:
                    model = _initialize_model(
                        model_config,
                        self.load_config,
                    )

            with startup_phase("weights.read_copy"):
                kv_cache_scales: dict[str, torch.Tensor] = {}
                model.load_weights(
                    split_kv_cache_scales(
                        self._get_all_weights(model_config, model), kv_cache_scales
                    )
                )
                if kv_cache_scales:
                    _load_kv_cache_scales(model, kv_cache_scales)

            with startup_phase("weights.postprocess"):
                # post_load_transformed_modules must recognise every hook run here.
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

                    process_method = getattr(
                        module, "process_weights_after_loading", None
                    )
                    if process_method is not None:
                        with device_loading_context(module, target_device):
                            module.process_weights_after_loading(module)

            with startup_phase("weights.post_quant_warmup"):
                post_quant_warmup = getattr(model, "post_quant_warmup", None)
                if callable(post_quant_warmup):
                    post_quant_warmup()

        return model.eval()


def no_op_post_load_hooks() -> frozenset[Callable]:
    """The ``process_weights_after_loading`` hooks that leave weights as loaded."""
    from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
    from tokenspeed.runtime.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )

    return frozenset(
        {
            QuantizeMethodBase.process_weights_after_loading,
            UnquantizedLinearMethod.process_weights_after_loading,
            MergedColumnParallelLinear.process_weights_after_loading,
            QKVParallelLinear.process_weights_after_loading,
            RowParallelLinear.process_weights_after_loading,
        }
    )


def post_load_transformed_modules(model: nn.Module) -> list[str]:
    """Name the modules whose weights the startup post-load step rewrote.

    After ``load_weights``, ``DefaultModelLoader.load_model`` runs each module's
    quant-method hook and the module's own ``process_weights_after_loading``
    once. Those hooks repack, quantize, shuffle or derive copies of what was
    loaded and cannot run a second time, so checkpoint tensors streamed into
    such a module later land in the processed layout. A hook counts as a
    transform unless it is in ``no_op_post_load_hooks``; an MoE layer's hook
    transforms exactly when its planned kernel has a weight preprocessor, and a
    DeepSeek V4 MoE block's exactly when it fused its shared expert.

    Returns:
        ``"<module name> (<hook owner>)"`` for each such module, in
        ``named_modules`` order; empty when every hook is a no-op.
    """
    from tokenspeed.runtime.layers.moe.expert import MoELayer
    from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
    from tokenspeed.runtime.models.deepseek_v41_moe import DeepseekV4MoE

    no_op_hooks: frozenset[Any] = no_op_post_load_hooks()
    if isinstance(model, BaseCausalLM):
        no_op_hooks = no_op_hooks | model.reloadable_post_load_hooks()
    transformed: list[str] = []
    for name, module in model.named_modules():
        owners = [type(module)]
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            owners.insert(0, type(quant_method))
        for owner in owners:
            hook = getattr(owner, "process_weights_after_loading", None)
            if hook is None or hook in no_op_hooks:
                continue
            if hook is MoELayer.process_weights_after_loading:
                preprocessor = (getattr(module, "plan", None) or {}).get(
                    "weight_preprocessor"
                )
                if preprocessor is None or preprocessor in no_op_hooks:
                    continue
                owner_name = getattr(preprocessor, "__name__", repr(preprocessor))
            elif hook is DeepseekV4MoE.process_weights_after_loading:
                # Only fusing rewrites the shared expert's weights in place.
                if not module.shared_expert_fused:
                    continue
                owner_name = owner.__name__
            else:
                owner_name = owner.__name__
            transformed.append(f"{name or type(module).__name__} ({owner_name})")
            break
    return transformed


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
            # Post-processing reads the parameters it derives copies from (such
            # as a prepared GEMV weight) and checks integer weights and scales
            # that start as NaN, so every parameter needs a value before it
            # runs. A parameter that post-processing leaves as it is gets the
            # same values again from the call after it, since its values depend
            # only on the seed, its size and its dtype. A parameter with a
            # dummy initializer takes the initializer's values for
            # post-processing.
            initialize_dummy_weights(model)
            initialize_dummy_integer_weights(model)
            # An EAGLE3 draft cannot share a vocab-sharded target embedding,
            # and its own dummy embedding is as valid as a loaded one.
            mark_initialized = getattr(model, "mark_embedding_initialized", None)
            if callable(mark_initialized):
                mark_initialized()
            if getattr(model, "post_load_weights", None):
                model.post_load_weights()

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)
                process_method = getattr(module, "process_weights_after_loading", None)
                if process_method is not None:
                    module.process_weights_after_loading(module)

            post_quant_warmup = getattr(model, "post_quant_warmup", None)
            if callable(post_quant_warmup):
                post_quant_warmup()

            # Post-processing can replace parameters (new layouts or dtypes);
            # give those random values as well. This also overwrites the
            # values a floating-point parameter's dummy initializer set, so a
            # copy post-processing derived from such a parameter does not
            # match it afterwards. Skip composite peer-weight views: finalize
            # already copied and distributed the initialized owned shards and
            # shared staging pools must not be overwritten across layers.
            from tokenspeed.runtime.layers.moe.peer_weights import (
                peer_weights_controller_for,
            )

            exclude_data_ptrs: set[int] | None = None
            controller = peer_weights_controller_for(model)
            if controller is not None and controller.finalized:
                exclude_data_ptrs = {
                    p.data_ptr() for p in controller.managed_parameters() if p.numel()
                }
            initialize_dummy_weights(model, exclude_data_ptrs=exclude_data_ptrs)
        return model.eval()


class ShardedStateLoader(BaseModelLoader):
    """
    Model loader that directly loads each worker's model state dict, which
    enables a fast load path for large tensor-parallel models where each worker
    only needs to read its own shard rather than the entire checkpoint.
    Checkpoints contain post-processed state and require the same model,
    parallel mapping, quantization and runtime weight layout when reloaded.
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
        from tokenspeed.runtime.model_loader.object_store import is_object_store_uri

        if is_object_store_uri(model_name_or_path):
            raise ValueError(
                f"Object-store URI {model_name_or_path!r} requires "
                f"--load-format runai_streamer, not {self.load_config.load_format!r}"
            )

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
                                f"loading tensor of shape {tensor.shape!s} into "
                                f"parameter '{key!s}' of shape {param_shape!s}",
                            )
                        param_data.copy_(tensor)
                        state_dict.pop(key)
            if state_dict:
                raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")

        post_quant_warmup = getattr(model, "post_quant_warmup", None)
        if callable(post_quant_warmup):
            post_quant_warmup()

        return model.eval()


class RunaiModelStreamerLoader(DefaultModelLoader):
    """Model loader that streams safetensors weights with Run:ai Model Streamer.

    Supports both local directories and cloud object-store URIs (``gs://``,
    ``s3://``). Remote checkpoints sync only their non-weight metadata files to
    a local cache directory and stream ``*.safetensors`` shards directly.
    """

    def __init__(self, load_config: LoadConfig) -> None:
        BaseModelLoader.__init__(self, load_config)
        extra_config = (
            {}
            if load_config.model_loader_extra_config is None
            else dict(load_config.model_loader_extra_config)
        )
        concurrency = extra_config.pop("concurrency", None)
        memory_limit = extra_config.pop("memory_limit", None)
        distributed = extra_config.pop("distributed", False)
        distributed_scope = extra_config.pop("distributed_scope", None)
        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: {sorted(extra_config)}"
            )
        if concurrency is not None and (
            isinstance(concurrency, bool)
            or not isinstance(concurrency, int)
            or concurrency < 1
        ):
            raise ValueError(
                f"runai_streamer concurrency must be a positive int, got {concurrency!r}"
            )
        if memory_limit is not None and (
            isinstance(memory_limit, bool)
            or not isinstance(memory_limit, int)
            or memory_limit < 0
        ):
            raise ValueError(
                f"runai_streamer memory_limit must be a non-negative int, got {memory_limit!r}"
            )
        if not isinstance(distributed, bool):
            raise ValueError(
                f"runai_streamer distributed must be a bool, got {distributed!r}"
            )
        if distributed:
            if distributed_scope not in ("local", "global"):
                raise ValueError(
                    "runai_streamer distributed=True requires explicit "
                    "distributed_scope ('local' or 'global'), "
                    f"got {distributed_scope!r}"
                )
        elif distributed_scope is not None:
            raise ValueError(
                "runai_streamer distributed_scope requires distributed=True, "
                f"got {distributed_scope!r}"
            )
        self.concurrency: int | None = concurrency
        self.memory_limit: int | None = memory_limit
        self.distributed: bool = distributed
        self.distributed_scope: str | None = distributed_scope

    def _prepare_weights(
        self,
        model_name_or_path: str,
        revision: str | None,
        fall_back_to_pt: bool,
        weight_files: tuple[str, ...] | None,
    ) -> tuple[str, list[str], bool]:
        from tokenspeed.runtime.model_loader.object_store import (
            is_object_store_uri,
            join_object_store_uri,
            list_object_store_safetensors,
            sync_object_store_metadata,
        )

        if not is_object_store_uri(model_name_or_path):
            return super()._prepare_weights(
                model_name_or_path, revision, fall_back_to_pt, weight_files
            )
        if revision is not None:
            raise ValueError(
                f"revision={revision!r} is not supported for object-store URI "
                f"{model_name_or_path!r}; object-store checkpoints always stream "
                "live bucket state"
            )

        hf_folder = sync_object_store_metadata(
            model_name_or_path, cache_dir=self.load_config.download_dir
        )
        if weight_files is not None:
            uris = [
                join_object_store_uri(model_name_or_path, name) for name in weight_files
            ]
            return hf_folder, uris, True

        index_path = os.path.join(hf_folder, SAFE_WEIGHTS_INDEX_NAME)
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
            shard_names = sorted(set(weight_map.values()))
            uris = [
                join_object_store_uri(model_name_or_path, name) for name in shard_names
            ]
        else:
            uris = list_object_store_safetensors(model_name_or_path)

        if not uris:
            raise RuntimeError(
                f"Cannot find any safetensors weights under `{model_name_or_path}`"
            )
        return hf_folder, uris, True

    def _get_weights_iterator(
        self,
        source: "DefaultModelLoader.Source",
        weight_name_filter: Callable[[str], bool] | None,
        checkpoint_load_group: tuple[int, ...] | None,
        mapping: Mapping,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        del mapping
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.revision,
            source.fall_back_to_pt,
            source.weight_files,
        )
        if not use_safetensors:
            raise ValueError("runai_streamer only supports safetensors checkpoints.")
        hf_weights_files.sort()

        index_weight_map: dict[str, str] | None = None
        index_path = os.path.join(hf_folder, SAFE_WEIGHTS_INDEX_NAME)
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as f:
                index_weight_map = json.load(f).get("weight_map")

        prefixed_filter = (
            (lambda name: weight_name_filter(source.prefix + name))
            if weight_name_filter is not None
            else None
        )
        if prefixed_filter is not None:
            hf_weights_files = filter_safetensors_files_by_weight_names(
                hf_weights_files,
                hf_folder,
                SAFE_WEIGHTS_INDEX_NAME,
                prefixed_filter,
            )

        if self.distributed and checkpoint_load_group is not None:
            raise ValueError(
                "runai_streamer distributed=True is incompatible with "
                f"checkpoint_load_group={checkpoint_load_group!r}: Run:ai "
                "distributed streaming broadcasts across the full default "
                "process group rather than a rank subset."
            )
        weights_iterator = runai_streamer_weights_iterator(
            hf_weights_files,
            distributed=self.distributed,
            distributed_scope=self.distributed_scope,
            weight_name_filter=prefixed_filter,
            index_weight_map=index_weight_map,
            concurrency=self.concurrency,
            memory_limit=self.memory_limit,
        )
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)


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

    if load_config.load_format == LoadFormat.RUNAI_STREAMER:
        return RunaiModelStreamerLoader(load_config)

    return DefaultModelLoader(load_config)
