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


"""Utilities for downloading and initializing model weights."""

import concurrent.futures
import fnmatch
import glob
import hashlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Generator, Iterable
from typing import (
    Any,
)

import filelock
import huggingface_hub.constants
import numpy as np
import safetensors.torch
import torch
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download
from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator
from tqdm.auto import tqdm

from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.quantization import (
    QuantizationConfig,
    get_quantization_config,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)
_PREFETCH_BLOCK_SIZE = 16 * 1024 * 1024

# use system-level temp directory for file locks, so that multiple users
# can share the same lock without error.
# lock files in the temp directory will be automatically deleted when the
# system reboots, so users will not complain about annoying lock files
temp_dir = tempfile.gettempdir()


def enable_hf_transfer():
    """automatically activates hf_transfer"""
    if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
        try:
            # enable hf hub transfer if available
            import hf_transfer  # type: ignore

            huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
        except ImportError:
            pass


enable_hf_transfer()


class DisabledTqdm(tqdm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def get_lock(
    model_name_or_path: str, cache_dir: str | None = None
) -> filelock.FileLock:
    lock_dir = cache_dir or temp_dir
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    return filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)


def get_quant_config(
    model_config: ModelConfig, load_config: LoadConfig
) -> QuantizationConfig:
    quant_cls = get_quantization_config(model_config.quantization)

    # Read the quantization config from the HF model config, if available.
    hf_quant_config = getattr(model_config.hf_config, "quantization_config", None)
    # some vision model may keep quantization_config in their text_config
    hf_text_config = getattr(model_config.hf_config, "text_config", None)
    if hf_quant_config is None and hf_text_config is not None:
        hf_quant_config = getattr(hf_text_config, "quantization_config", None)
    if hf_quant_config is None:
        # compressed-tensors uses a compressions_config
        hf_quant_config = getattr(model_config.hf_config, "compression_config", None)
    if hf_quant_config is not None:
        return quant_cls.from_config(hf_quant_config)
    model_name_or_path = model_config.model_path
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        # Download the config files.
        with get_lock(model_name_or_path, load_config.download_dir):
            hf_folder = snapshot_download(
                model_name_or_path,
                revision=model_config.revision,
                allow_patterns="*.json",
                cache_dir=load_config.download_dir,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                tqdm_class=DisabledTqdm,
            )
    else:
        hf_folder = model_name_or_path

    possible_config_filenames = quant_cls.get_config_filenames()

    # If the quantization config is not found, use the default config.
    if not possible_config_filenames:
        return quant_cls()

    config_files = glob.glob(os.path.join(hf_folder, "*.json"))

    quant_config_files = [
        f for f in config_files if any(f.endswith(x) for x in possible_config_filenames)
    ]
    if len(quant_config_files) == 0:
        raise ValueError(f"Cannot find the config file for {model_config.quantization}")
    if len(quant_config_files) > 1:
        raise ValueError(
            f"Found multiple config files for {model_config.quantization}: "
            f"{quant_config_files}"
        )

    quant_config_file = quant_config_files[0]
    with open(quant_config_file) as f:
        config = json.load(f)

        if model_config.quantization == "nvfp4":
            if config["producer"]["name"] == "modelopt":
                return quant_cls.from_config(config)
            else:
                raise ValueError(
                    f"Unsupported quantization config"
                    f" found for {model_config.quantization} in {f}."
                )

    return quant_cls.from_config(config)


def download_weights_from_hf(
    model_name_or_path: str,
    cache_dir: str | None,
    allow_patterns: list[str],
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    """Download model weights from Hugging Face Hub.

    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        allow_patterns (List[str]): The allowed patterns for the
            weight files. Files matched by any of the patterns will be
            downloaded.
        revision (Optional[str]): The revision of the model.
        ignore_patterns (Optional[Union[str, List[str]]]): The patterns to
            filter out the weight files. Files matched by any of the patterns
            will be ignored.

    Returns:
        str: The path to the downloaded model weights.
    """
    if not huggingface_hub.constants.HF_HUB_OFFLINE:
        # Before we download we look at that is available:
        fs = HfFileSystem()
        file_list = fs.ls(model_name_or_path, detail=False, revision=revision)

        # depending on what is available we download different things
        for pattern in allow_patterns:
            matching = fnmatch.filter(file_list, pattern)
            if len(matching) > 0:
                allow_patterns = [pattern]
                break

    logger.info("Using model weights format %s", allow_patterns)
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        hf_folder = snapshot_download(
            model_name_or_path,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            cache_dir=cache_dir,
            tqdm_class=DisabledTqdm,
            revision=revision,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        )
    return hf_folder


def download_safetensors_index_file_from_hf(
    model_name_or_path: str,
    index_file: str,
    cache_dir: str | None,
    revision: str | None = None,
) -> None:
    """Download hf safetensors index file from Hugging Face Hub.

    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        revision (Optional[str]): The revision of the model.
    """
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        try:
            # Download the safetensors index file.
            hf_hub_download(
                repo_id=model_name_or_path,
                filename=index_file,
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            )
        # If file not found on remote or locally, we should not fail since
        # only some models will have index_file.
        except huggingface_hub.utils.EntryNotFoundError:
            logger.info("No %s found in remote.", index_file)
        except huggingface_hub.utils.LocalEntryNotFoundError:
            logger.info("No %s found in local cache.", index_file)


# For models like Mistral-7B-v0.3, there are both sharded
# safetensors files and a consolidated safetensors file.
# Passing both of these to the weight loader functionality breaks.
# So, we use the index_file to
# look up which safetensors files should be used.
def filter_duplicate_safetensors_files(
    hf_weights_files: list[str], hf_folder: str, index_file: str
) -> list[str]:
    # model.safetensors.index.json is a mapping from keys in the
    # torch state_dict to safetensors file holding that weight.
    index_file_name = os.path.join(hf_folder, index_file)
    if not os.path.isfile(index_file_name):
        return hf_weights_files

    # Iterate through the weight_map (weight_name: safetensors files)
    # to identify weights that we should use.
    with open(index_file_name) as f:
        weight_map = json.load(f)["weight_map"]
    weight_files_in_index = set()
    for weight_name in weight_map:
        weight_files_in_index.add(os.path.join(hf_folder, weight_map[weight_name]))
    # Filter out any fields that are not found in the index file.
    hf_weights_files = [f for f in hf_weights_files if f in weight_files_in_index]
    return hf_weights_files


def filter_files_not_needed_for_inference(hf_weights_files: list[str]) -> list[str]:
    """
    Exclude files that are not needed for inference.

    See https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
    """
    blacklist = [
        "training_args.bin",
        "optimizer.bin",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
    ]
    hf_weights_files = [
        f for f in hf_weights_files if not any(f.endswith(x) for x in blacklist)
    ]
    return hf_weights_files


# explicitly use pure text format, with a newline at the end
# this makes it impossible to see the animation in the progress bar
# but will avoid messing up with ray or multiprocessing, which wraps
# each line of output with some prefix.
_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"  # noqa: E501


def np_cache_weights_iterator(
    model_name_or_path: str,
    cache_dir: str | None,
    hf_folder: str,
    hf_weights_files: list[str],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model np files.

    Will dump the model weights to numpy files if they are not already dumped.
    """
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    # Convert the model weights from torch tensors to numpy arrays for
    # faster loading.
    np_folder = os.path.join(hf_folder, "np")
    os.makedirs(np_folder, exist_ok=True)
    weight_names_file = os.path.join(np_folder, "weight_names.json")
    # Use file lock to prevent multiple processes from
    # dumping the same model weights to numpy at the same time.
    with get_lock(model_name_or_path, cache_dir):
        if not os.path.exists(weight_names_file):
            weight_names: list[str] = []
            for bin_file in tqdm(
                hf_weights_files,
                desc="Loading np_cache checkpoint shards",
                disable=not enable_tqdm,
                bar_format=_BAR_FORMAT,
            ):
                state = torch.load(bin_file, map_location="cpu")
                for name, param in state.items():
                    param_path = os.path.join(np_folder, name)
                    with open(param_path, "wb") as f:
                        np.save(f, param.cpu().detach().numpy())
                    weight_names.append(name)
            with open(weight_names_file, "w") as f:
                json.dump(weight_names, f)

    with open(weight_names_file) as f:
        weight_names = json.load(f)

    for name in weight_names:
        param_path = os.path.join(np_folder, name)
        with open(param_path, "rb") as f:
            param = np.load(f)
        yield name, torch.from_numpy(param)


def decrypt(fn, key):
    raise NotImplementedError()


def safetensors_encrypted_weights_iterator(
    hf_weights_files: list[str],
    is_all_weights_sharded: bool = False,
    decryption_key: str | None = None,
):
    raise NotImplementedError()


def _get_checkpoint_prefetch_rank_info() -> tuple[int, int]:
    local_rank = os.getenv("LOCAL_RANK")
    local_world_size = os.getenv("LOCAL_WORLD_SIZE")
    if local_rank is not None and local_world_size is not None:
        try:
            rank = int(local_rank)
            world_size = int(local_world_size)
            if 0 <= rank < world_size:
                return rank, world_size
        except ValueError:
            logger.warning(
                "Ignoring invalid LOCAL_RANK/LOCAL_WORLD_SIZE for checkpoint prefetch: "
                "%s/%s",
                local_rank,
                local_world_size,
            )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()

    return 0, 1


def _prefetch_checkpoint_file(file_path: str) -> int:
    bytes_read = 0
    with open(file_path, "rb") as f:
        while True:
            data = f.read(_PREFETCH_BLOCK_SIZE)
            if not data:
                break
            bytes_read += len(data)
    return bytes_read


def prefetch_checkpoint_files(
    hf_weights_files: list[str],
    num_threads: int = 4,
) -> threading.Thread:
    """Prefetch checkpoint shards into the OS page cache in the background."""
    sorted_files = sorted(hf_weights_files)
    rank, world_size = _get_checkpoint_prefetch_rank_info()
    my_files = sorted_files[rank::world_size]
    num_threads = max(1, num_threads)

    logger.info(
        "Rank %d: prefetching %d/%d checkpoint shards into OS page cache "
        "(background, %d local ranks sharing work, %d threads per rank).",
        rank,
        len(my_files),
        len(sorted_files),
        world_size,
        num_threads,
    )

    def _run_prefetch() -> None:
        start = time.perf_counter()
        bytes_read = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(_prefetch_checkpoint_file, path) for path in my_files
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    bytes_read += future.result()
                except Exception:
                    logger.warning(
                        "Failed to prefetch a checkpoint shard.", exc_info=True
                    )

        elapsed = time.perf_counter() - start
        logger.info(
            "Rank %d: checkpoint prefetch finished in %.2fs, %.2f GiB read.",
            rank,
            elapsed,
            bytes_read / (1024**3),
        )

    thread = threading.Thread(target=_run_prefetch, daemon=True)
    thread.start()
    return thread


def safetensors_weights_iterator(
    hf_weights_files: list[str],
    is_all_weights_sharded: bool = False,
    decryption_key: str | None = None,
    prefetch: bool = False,
    prefetch_num_threads: int = 4,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files.

    If is_all_weights_sharded is True, it uses more optimize read by reading an
    entire file instead of reading each tensor one by one.
    """
    if decryption_key:
        yield from safetensors_encrypted_weights_iterator(
            hf_weights_files, is_all_weights_sharded, decryption_key
        )
        return

    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    if prefetch:
        prefetch_checkpoint_files(
            hf_weights_files,
            num_threads=prefetch_num_threads,
        )

    for st_file in tqdm(
        hf_weights_files,
        desc="Loading safetensors checkpoint shards",
        disable=not enable_tqdm,
        bar_format=_BAR_FORMAT,
    ):
        result = safetensors.torch.load_file(st_file, device="cpu")
        yield from result.items()


def pt_weights_iterator(
    hf_weights_files: list[str],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model bin/pt files."""
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    for bin_file in tqdm(
        hf_weights_files,
        desc="Loading pt checkpoint shards",
        disable=not enable_tqdm,
        bar_format=_BAR_FORMAT,
    ):
        state = torch.load(bin_file, map_location="cpu")
        yield from state.items()
        del state
        torch.cuda.empty_cache()


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # "broadcast" instead of copy
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
    except Exception:
        #  This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        raise


LoaderFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def sharded_weight_loader(shard_axis: int, tp_rank: int) -> LoaderFunction:
    """Create a weight loader that shards the weights along the given axis"""

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        shard_size = param.data.shape[shard_axis]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)

        return default_weight_loader(param, loaded_weight)

    return loader


def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
    seed: int = 1234,
) -> None:
    """Initialize model weights with random values.

    The model weights must be randomly initialized for accurate performance
    measurements. Additionally, the model weights should not cause NaNs in the
    forward pass. We empirically found that initializing the weights with
    values between -1e-3 and 1e-3 works well for most models.

    We use per-parameter random seed, so that dummy weights are consistent,
    even if the model is partitioned across multiple devices. When the seed
    is fixed, the random values generated by this function only depends on
    the parameter's number of elements and its data type.
    """
    for param in model.state_dict().values():
        if torch.is_floating_point(param):
            generator = torch.Generator(device=param.data.device)
            generator.manual_seed(seed)
            if torch.finfo(param.data.dtype).bits < 16:
                # uniform_ doesn't support < 16-bit datatypes (FP8)
                dtype = param.data.dtype
                tmp_param = param.data.to(torch.float16)
                tmp_param = tmp_param.uniform_(low, high, generator=generator).to(dtype)
                param.data.copy_(tmp_param)
            else:
                param.uniform_(low, high, generator=generator)


class KVCacheQuantSchema(BaseModel):
    dtype: str
    # Each key is a TP rank. Each value is a dictionary mapping a TP rank's
    # layer indices to their per-tensor KV cache scaling factor.
    # own schema class (tricky as its members are variable)
    scaling_factor: dict[int, dict[int, float]]

    @model_validator(mode="after")
    def check_is_fp8(self) -> "KVCacheQuantSchema":
        assert self.dtype == "float8_e4m3fn", (
            "Loaded scaling factors intended for KV cache dtype = "
            f"{self.dtype} rather than float8_e4m3fn!"
        )
        return self

    @model_validator(mode="after")
    def check_tp_ranks(self, info: ValidationInfo) -> "KVCacheQuantSchema":
        context = info.context
        if context:
            tp_size = context["tp_size"]
            num_hidden_layers = context["num_hidden_layers"]
            assert len(self.scaling_factor) == tp_size, (
                f"Loaded dictionary has TP size {len(self.scaling_factor)} "
                f"but LLM engine is currently running with TP size {tp_size}."
            )
            for tp_rank, layer_maps in self.scaling_factor.items():
                assert len(layer_maps) == num_hidden_layers, (
                    f"KV cache scales map for TP rank {tp_rank} is malformed. "
                    f"Expected {num_hidden_layers} layers, got "
                    f"{len(layer_maps)}."
                )
            for i in range(tp_size):
                assert (
                    i in self.scaling_factor
                ), f"KV cache scales map for TP rank {i} not found."
        return self

    @model_validator(mode="after")
    def check_current_rank(self, info: ValidationInfo) -> "KVCacheQuantSchema":
        context = info.context
        if context:
            tp_rank = context["tp_rank"]
            num_hidden_layers = context["num_hidden_layers"]
            layer_scales_map = self.scaling_factor[tp_rank]
            for i in range(num_hidden_layers):
                assert i in layer_scales_map, (
                    f"Could not find KV cache scales for layer {i} in "
                    f"TP rank {tp_rank}."
                )
        return self


class QuantParamSchema(BaseModel):
    # (e.g. weights/activations params) once functionality is enabled
    model_config = ConfigDict(protected_namespaces=())
    model_type: str | None
    kv_cache: KVCacheQuantSchema

    @model_validator(mode="after")
    def check_model_type(self, info: ValidationInfo) -> "QuantParamSchema":
        context = info.context
        if context:
            model_type = context.get("model_type", None)
            if model_type is not None:
                assert model_type == self.model_type, (
                    f"Model type is {model_type} but loaded "
                    f"scaling factors belonging to different "
                    f"model type {self.model_type}!"
                )
        return self


def kv_cache_scales_loader(
    filename: str,
    tp_rank: int,
    tp_size: int,
    num_hidden_layers: int,
    model_type: str | None,
) -> Iterable[tuple[int, float]]:
    """
    A simple utility to read in KV cache scaling factors that have been
    previously serialized to disk. Used by the model to populate the appropriate
    KV cache scaling factors. The serialization should represent a dictionary
    whose keys are the TP ranks and values are another dictionary mapping layers
    to their KV cache scaling factors.
    """
    try:
        with open(filename) as f:
            context = {
                "model_type": model_type,
                "num_hidden_layers": num_hidden_layers,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
            }
            schema_dct = json.load(f)
            schema = QuantParamSchema.model_validate(schema_dct, context=context)
            layer_scales_map = schema.kv_cache.scaling_factor[tp_rank]
            return layer_scales_map.items()
    except FileNotFoundError:
        logger.error("File or directory '%s' not found.", filename)
    except json.JSONDecodeError:
        logger.error("Error decoding JSON in file '%s'.", filename)
    except Exception:
        logger.error("An error occurred while reading '%s'.", filename)
    # This section is reached if and only if any of the excepts are hit
    # Return an empty iterable (list) => no KV cache scales are loaded
    # which ultimately defaults to 1.0 scales
    logger.warning(
        "Defaulting to KV cache scaling factors = 1.0 for all "
        "layers in TP rank %d as an error occurred during loading.",
        tp_rank,
    )
    return []


def mamba_v2_sharded_weight_loader(
    shard_spec: list[tuple[int, int, float]],
    tp_size: int,
    tp_rank: int,
) -> LoaderFunction:
    """Create a weight loader for mamba v2. This ensures that the projections
    are correctly sharded so that they can be split into x, B, C. It also
    ensures the the all the groups corresponding to a head shard is placed
    together with it.
    """

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:

        # - track boundary of (sharded) param, and loaded_weight, respectively
        boundary, loaded_boundary = 0, 0

        # - iterate over the shard specs
        for full_dim, extra, duplicate_groups in shard_spec:
            # - full dim is the model dim (before TP).
            # - extra > 0, means there is expected overall increase
            #   of dimensions. This is so because of replication.
            # - ratio is used map the tp_rank to the actual shard
            #   rank. This is useful when there is replication of
            #   groups to accompany head shards.

            # - size of the loaded shard
            shard_size = full_dim // tp_size

            # - compute the rank into the loaded shard.
            # - if there is replication, different TP shards will
            #   take from the same rank.
            #  currently we only support duplication
            # in the case where num_groups == 1
            rank = 0 if duplicate_groups else tp_rank

            # - leftmost boundary index into loaded weight.
            loaded_skip = rank * shard_size
            loaded_start_idx = loaded_boundary + loaded_skip

            # - take these many dims from the loaded weight.
            take = min(shard_size, full_dim - extra - loaded_skip)

            # - always shard on dim 0
            # - the ignore is for a mundane mypy error as it does not
            #   seem to handle slices well.
            # https://github.com/python/mypy/issues/2410
            param.data[
                boundary : (boundary + take), ...  # type: ignore[misc]
            ] = loaded_weight[
                loaded_start_idx : (loaded_start_idx + take)  # type: ignore[misc]
            ]  # type: ignore[misc]

            # move indexing boundaries
            boundary += shard_size
            loaded_boundary += full_dim - extra

    return loader
