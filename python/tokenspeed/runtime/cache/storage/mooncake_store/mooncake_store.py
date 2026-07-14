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

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import requests
import torch

from tokenspeed.runtime.cache.kv_cache_host import HostKVCache
from tokenspeed.runtime.cache.kvstore_storage import (
    KVStoreStorage,
    KVStoreStorageConfig,
    KVStoreStorageExtraInfo,
)
from tokenspeed.runtime.utils.env import envs

DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024  # 16 MB
SETUP_TIMEOUT = 600  # 10min

logger = logging.getLogger(__name__)


def _parse_global_segment_size(value: int | str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s.endswith("gb"):
            num = s[:-2].strip()
            if not num:
                raise ValueError(
                    "Invalid global_segment_size: missing number before 'gb'"
                )
            return int(num) * 1024 * 1024 * 1024
        return int(s)
    return int(value)


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    protocol: str
    device_name: str
    master_server_address: str
    master_metrics_port: int
    check_server: bool
    # Optional nested ``extra_config["kv_events"]`` (RFC #1527 / PR #2214).
    # Defaults keep engine-side L3 publish; see mooncake_kv_events.py.
    kv_events_source: str = "engine"
    kv_events_master_subscribe_endpoint: str | None = None
    kv_events_backend_id: str | None = None

    @staticmethod
    def from_file() -> "MooncakeStoreConfig":
        """Load the config from a JSON file."""
        if not envs.TOKENSPEED_KVSTORE_MOONCAKE_CONFIG_PATH.is_set():
            raise RuntimeError(
                f"Config file path not set. Please set {envs.TOKENSPEED_KVSTORE_MOONCAKE_CONFIG_PATH.name}"
            )
        file_path = envs.TOKENSPEED_KVSTORE_MOONCAKE_CONFIG_PATH.value
        try:
            with open(file_path) as fin:
                config = json.load(fin)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Failed to load config from {file_path}: {exc}"
            ) from exc

        if "master_server_address" not in config:
            raise ValueError("master_server_address is required in config file")

        return MooncakeStoreConfig(
            local_hostname=config.get(
                "local_hostname", envs.MOONCAKE_LOCAL_HOSTNAME.default
            ),
            metadata_server=config.get(
                "metadata_server", envs.MOONCAKE_TE_META_DATA_SERVER.default
            ),
            global_segment_size=_parse_global_segment_size(
                config.get(
                    "global_segment_size", envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.default
                )
            ),
            protocol=config.get("protocol", envs.MOONCAKE_PROTOCOL.default),
            device_name=config.get("device_name", envs.MOONCAKE_DEVICE.default),
            master_server_address=config.get("master_server_address"),
            master_metrics_port=config.get(
                "master_metrics_port", envs.MOONCAKE_MASTER_METRICS_PORT.default
            ),
            check_server=config.get("check_server", envs.MOONCAKE_CHECK_SERVER.default),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """Load config from a file specified in the environment variable.
        export MOONCAKE_MASTER=10.13.3.232:50051
        export MOONCAKE_PROTOCOL="rdma"
        export MOONCAKE_DEVICE=""
        export MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE"
        """
        # other required environment variables...
        if not envs.MOONCAKE_MASTER.is_set():
            raise ValueError("The environment variable 'MOONCAKE_MASTER' is not set.")

        # Prefer the namespaced Mooncake env var, but keep the older
        # LOCAL_HOSTNAME fallback working for existing deployments.
        if envs.MOONCAKE_LOCAL_HOSTNAME.is_set():
            local_hostname = envs.MOONCAKE_LOCAL_HOSTNAME.value
        else:
            local_hostname = os.getenv(
                "LOCAL_HOSTNAME", envs.MOONCAKE_LOCAL_HOSTNAME.default
            )

        return MooncakeStoreConfig(
            local_hostname=local_hostname,
            metadata_server=envs.MOONCAKE_TE_META_DATA_SERVER.value,
            global_segment_size=_parse_global_segment_size(
                envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.value
            ),
            protocol=envs.MOONCAKE_PROTOCOL.value,
            device_name=envs.MOONCAKE_DEVICE.value,
            master_server_address=envs.MOONCAKE_MASTER.value,
            master_metrics_port=envs.MOONCAKE_MASTER_METRICS_PORT.value,
            check_server=envs.MOONCAKE_CHECK_SERVER.value,
        )

    @staticmethod
    def load_from_extra_config(extra_config: dict) -> "MooncakeStoreConfig":
        """Load config from extra_config dictionary."""
        if "master_server_address" not in extra_config:
            raise ValueError("master_server_address is required in extra_config")

        # Lazy import: keep torch-free unit tests and avoid a hard cycle with
        # pd.mooncake_kv_events → kv_events.
        from tokenspeed.runtime.pd.mooncake_kv_events import (
            parse_mooncake_kv_events_config,
        )

        kv_events = parse_mooncake_kv_events_config(extra_config)

        return MooncakeStoreConfig(
            local_hostname=extra_config.get(
                "local_hostname", envs.MOONCAKE_LOCAL_HOSTNAME.default
            ),
            metadata_server=extra_config.get(
                "metadata_server", envs.MOONCAKE_TE_META_DATA_SERVER.default
            ),
            global_segment_size=_parse_global_segment_size(
                extra_config.get(
                    "global_segment_size", envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.default
                )
            ),
            protocol=extra_config.get("protocol", envs.MOONCAKE_PROTOCOL.default),
            device_name=extra_config.get("device_name", envs.MOONCAKE_DEVICE.default),
            master_server_address=extra_config["master_server_address"],
            master_metrics_port=extra_config.get(
                "master_metrics_port", envs.MOONCAKE_MASTER_METRICS_PORT.default
            ),
            check_server=extra_config.get(
                "check_server", envs.MOONCAKE_CHECK_SERVER.default
            ),
            kv_events_source=kv_events.source,
            kv_events_master_subscribe_endpoint=kv_events.master_subscribe_endpoint,
            kv_events_backend_id=kv_events.backend_id,
        )


class MooncakeStore(KVStoreStorage):
    # Optional callback after store-wide clear; wired by StorageExecutor.
    on_clear: Callable[[], None] | None = None

    def __init__(self, storage_config: KVStoreStorageConfig = None):
        self.on_clear = None
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as exc:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://kvcache-ai.github.io/Mooncake/getting_started/build.html"
                "to run TokenSpeed with MooncakeConnector."
            ) from exc

        try:
            self.store = MooncakeDistributedStore()

            extra_config = (
                getattr(storage_config, "extra_config", None)
                if storage_config
                else None
            )
            # Load configuration with master_server_address prioritized from extra_config if available
            if (
                extra_config is not None
                and extra_config.get("master_server_address") is not None
            ):
                # Load from extra_config
                self.config = MooncakeStoreConfig.load_from_extra_config(extra_config)
                logger.info(
                    "Mooncake Configuration loaded from extra_config successfully."
                )
            elif envs.TOKENSPEED_KVSTORE_MOONCAKE_CONFIG_PATH.is_set():
                # Load from config file
                self.config = MooncakeStoreConfig.from_file()
                logger.info("Mooncake Configuration loaded from file successfully.")
            else:
                # Load from environment variables
                self.config = MooncakeStoreConfig.load_from_env()
                logger.info("Mooncake Configuration loaded from env successfully.")

            tp_scale_factor = 1 if storage_config is None else storage_config.tp_size

            per_tp_global_segment_size = (
                self.config.global_segment_size // tp_scale_factor
            )

            # Check if extra_backend_tag should be passed to MooncakeDistributedStore
            self.extra_backend_tag = None
            if extra_config and "extra_backend_tag" in extra_config:
                self.extra_backend_tag = extra_config["extra_backend_tag"]
                logger.info("Using extra_backend_tag: %s", self.extra_backend_tag)

            # Check server status
            if self.config.check_server:
                self.check_server()

            # Handle JSON device_name configuration
            device_name = self.config.device_name
            if device_name and device_name.strip().startswith("{"):
                try:
                    device_config = json.loads(device_name)
                    if storage_config and hasattr(storage_config, "tp_rank"):
                        tp_rank = storage_config.tp_rank
                        # Try both integer and string keys since JSON parsing may convert keys
                        device_name = device_config.get(tp_rank) or device_config.get(
                            str(tp_rank), ""
                        )
                    else:
                        device_name = ""
                except (json.JSONDecodeError, AttributeError):
                    logger.warning(
                        "Failed to parse device_name as JSON: %s", device_name
                    )
                    device_name = ""

            ret_code = self.store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                per_tp_global_segment_size,
                DEFAULT_LOCAL_BUFFER_SIZE,  # Zero copy interface does not need local buffer
                self.config.protocol,
                device_name,
                self.config.master_server_address,
            )
            if ret_code:
                raise RuntimeError(
                    f"Failed to setup Mooncake store, error code: {ret_code}"
                )
            logger.info("Mooncake store setup successfully.")

            self.warmup()
            logger.info("Mooncake store warmup successfully.")

            if storage_config is not None:
                self.is_mla_backend = storage_config.is_mla_model
                self.local_rank = storage_config.tp_rank
            else:
                self.is_mla_backend = False
                self.local_rank = 0

        except ValueError as exc:
            logger.error("Configuration loading failed: %s", exc)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

    def check_server(self):
        master_server_ip, _, _ = self.config.master_server_address.partition(":")
        segments_url = f"http://{master_server_ip}:{self.config.master_metrics_port}/get_all_segments"
        start_time = time.perf_counter()

        check_result = False
        while time.perf_counter() - start_time < SETUP_TIMEOUT:
            try:
                check_segments_resp = requests.get(segments_url, timeout=3)
            except requests.RequestException:
                logger.info(
                    "waiting mooncake store server started, cost_time: %.2f seconds.",
                    time.perf_counter() - start_time,
                )
                time.sleep(3)
                continue

            if check_segments_resp.text == "":
                logger.info(
                    "waiting mooncake store server started, cost_time: %.2f seconds.",
                    time.perf_counter() - start_time,
                )
                time.sleep(3)
                continue

            logger.info("Mooncake store server started successfully.")
            check_result = True
            break

        if not check_result:
            logger.error("Launch mooncake store server timeout")
            raise ValueError("Launch mooncake store server timeout")

    def warmup(self):
        warmup_key = "tokenspeed_mooncake_store_warmup_key" + uuid.uuid4().hex
        warmup_value = bytes(4 * 1024)  # 4 KB
        put_result = self.store.put(warmup_key, warmup_value)
        if put_result != 0:
            logger.warning(
                "Mooncake store warmup put failed with code %s, skipping warmup (this is expected when global segment size is 0)",
                put_result,
            )
            return
        if self.store.is_exist(warmup_key) != 1:
            raise RuntimeError("Mooncake store warmup key was not persisted")
        if self.store.get(warmup_key) != warmup_value:
            raise RuntimeError("Mooncake store warmup value mismatch")

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        if self.mem_pool_host.layout not in ("page_first", "page_head"):
            raise ValueError(
                "mooncake store storage backend only supports page_first or page_head layout"
            )
        buffer = self.mem_pool_host.kv_buffer
        try:
            buffer_ptr = buffer.data_ptr()
            buffer_size = buffer.numel() * buffer.element_size()
            ret_code = self.store.register_buffer(buffer_ptr, buffer_size)
            if ret_code:
                logger.error("Failed to register buffer, error code: %s", ret_code)
                raise RuntimeError(
                    f"Failed to register buffer to Mooncake Store, error code: {ret_code}"
                )
        except TypeError as err:
            logger.error("Failed to register buffer to Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Register Buffer Error.") from err

    def _get_mha_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.local_rank}_k")
            key_list.append(f"{key_}_{self.local_rank}_v")
        if len(key_list) != len(ptr_list):
            raise ValueError("MHA key metadata does not match buffer metadata")
        return key_list, ptr_list, element_size_list

    def _expand_query_keys(self, keys: list[str]) -> tuple[list[str], int, bool]:
        if self.is_mla_backend:
            return (
                (
                    keys
                    if keys and keys[0].endswith("_k")
                    else [f"{key}_k" for key in keys]
                ),
                1,
                False,
            )

        keys_have_suffix = bool(
            keys and (keys[0].endswith("_k") or keys[0].endswith("_v"))
        )
        if keys_have_suffix:
            return keys, 2, False

        query_keys = [
            suffix_key
            for key in keys
            for suffix_key in (
                f"{key}_{self.local_rank}_k",
                f"{key}_{self.local_rank}_v",
            )
        ]
        return query_keys, 2, True

    @staticmethod
    def _expand_pairwise_metadata(values: list[Any], key_count: int) -> list[Any]:
        expanded = []
        for index in range(key_count):
            expanded.extend((values[index * 2], values[index * 2 + 1]))
        return expanded

    def _get_mla_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_k")
        if len(key_list) != len(ptr_list):
            raise ValueError("MLA key metadata does not match buffer metadata")
        return key_list, ptr_list, element_size_list

    def _batch_preprocess(self, keys, host_indices):
        if not keys:
            raise ValueError("keys must not be empty")
        if len(keys) != len(host_indices) // self.mem_pool_host.page_size:
            raise ValueError("keys length must match host_indices page count")
        if self.is_mla_backend:
            return self._get_mla_buffer_meta(keys, host_indices)
        else:
            return self._get_mha_buffer_meta(keys, host_indices)

    def _batch_postprocess(self, results: list[int], is_set_operate=False):
        """
        refer to https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-store/include/pybind_client.h
        for batch_get_into, results is Vector of integers,
            where each element is the number of bytes read on success, or a negative value on error
        for batch_put_from, results is Vector of integers,
            where each element is 0 on success, or a negative value on error
        """
        if self.is_mla_backend:
            return [result == 0 if is_set_operate else result > 0 for result in results]

        kv_pairs = zip(results[::2], results[1::2])
        return [
            (
                (k_res == 0 and v_res == 0)
                if is_set_operate
                else (k_res > 0 and v_res > 0)
            )
            for k_res, v_res in kv_pairs
        ]

    def batch_get_v1(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: KVStoreStorageExtraInfo | None = None,
    ) -> list[bool]:
        # Apply extra_backend_tag prefix if available
        if self.extra_backend_tag is not None:
            prefix = self.extra_backend_tag
            keys = [f"{prefix}_{key}" for key in keys]

        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)
        get_results = self._get_batch_zero_copy_impl(
            key_strs, buffer_ptrs, buffer_sizes
        )
        return self._batch_postprocess(get_results, is_set_operate=False)

    def batch_set_v1(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: KVStoreStorageExtraInfo | None = None,
    ) -> list[bool]:
        # Apply extra_backend_tag prefix if available
        if self.extra_backend_tag is not None:
            prefix = self.extra_backend_tag
            keys = [f"{prefix}_{key}" for key in keys]

        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)
        exist_result = self._batch_exist(key_strs)

        set_keys = []
        set_buffer_ptrs = []
        set_buffer_sizes = []
        set_indices = []
        set_results = [-1] * len(key_strs)
        for index, key_str in enumerate(key_strs):
            if exist_result[index] != 1:
                set_keys.append(key_str)
                set_buffer_ptrs.append(buffer_ptrs[index])
                set_buffer_sizes.append(buffer_sizes[index])
                set_indices.append(index)
            else:
                set_results[index] = 0

        # Only set non-existing keys to storage
        if set_keys:
            put_results = self._put_batch_zero_copy_impl(
                set_keys, set_buffer_ptrs, set_buffer_sizes
            )
            for index, set_index in enumerate(set_indices):
                set_results[set_index] = put_results[index]

        return self._batch_postprocess(set_results, is_set_operate=True)

    def set(
        self,
        key,
        value: Any | None = None,
        target_location: list[int] | None = None,
        target_sizes: list[int] | None = None,
    ) -> bool:
        # Only support zero copy set for now
        if target_location is None or target_sizes is None:
            raise ValueError("target_location and target_sizes are required")
        # Format key with local_rank suffix for non-MLA backend
        if self.is_mla_backend:
            query_keys = [f"{key}_k"]
            target_locations = [target_location]
            target_sizes_list = [target_sizes]
        else:
            # For non-MLA backend, we need to set both k and v
            query_keys = [f"{key}_{self.local_rank}_k", f"{key}_{self.local_rank}_v"]
            # target_location and target_sizes should be lists with 2 elements
            if isinstance(target_location, list) and len(target_location) >= 2:
                target_locations = [target_location[0], target_location[1]]
            else:
                # If not a list, assume it's a single location for k only
                target_locations = [target_location, target_location]

            if isinstance(target_sizes, list) and len(target_sizes) >= 2:
                target_sizes_list = [target_sizes[0], target_sizes[1]]
            else:
                # If not a list, assume it's a single size for k only
                target_sizes_list = [target_sizes, target_sizes]

        exist_result = self._batch_exist(query_keys)
        set_keys = []
        set_target_locations = []
        set_target_sizes = []
        for index, query_key in enumerate(query_keys):
            if exist_result[index] != 1:
                set_keys.append(query_key)
                set_target_locations.append(target_locations[index])
                set_target_sizes.append(target_sizes_list[index])

        # Only set non-existing keys to storage
        if set_keys:
            put_result = self._put_batch_zero_copy_impl(
                set_keys, set_target_locations, set_target_sizes
            )
            if any(result != 0 for result in put_result):
                return False
        return True

    def batch_set(
        self,
        keys: list[str],
        values: list[torch.Tensor] | None = None,
        target_locations: list[int] | None = None,
        target_sizes: list[int] | None = None,
    ) -> bool:
        # Only support zero copy set for now
        if target_locations is None or target_sizes is None:
            raise ValueError("target_locations and target_sizes are required")
        if len(keys) != len(target_locations) or len(keys) != len(target_sizes):
            raise ValueError(
                "keys, target_locations, and target_sizes must have matching lengths"
            )

        if not keys:
            return False

        if any(
            key is None or location is None or size is None
            for key, location, size in zip(keys, target_locations, target_sizes)
        ):
            return False

        query_keys, _, expanded_non_mla = self._expand_query_keys(keys)
        if expanded_non_mla:
            expanded_target_locations = self._expand_pairwise_metadata(
                target_locations, len(keys)
            )
            expanded_target_sizes = self._expand_pairwise_metadata(
                target_sizes, len(keys)
            )
        else:
            expanded_target_locations = target_locations
            expanded_target_sizes = target_sizes

        exist_result = self._batch_exist(query_keys)
        set_keys = []
        set_target_locations = []
        set_target_sizes = []
        set_indices = []
        for index, query_key in enumerate(query_keys):
            if exist_result[index] != 1:
                set_keys.append(query_key)
                set_target_locations.append(expanded_target_locations[index])
                set_target_sizes.append(expanded_target_sizes[index])
                set_indices.append(index)
        # Only set non-existing keys to storage

        put_result = self._put_batch_zero_copy_impl(
            set_keys, set_target_locations, set_target_sizes
        )
        for index, set_index in enumerate(set_indices):
            if put_result[index] == 0:
                exist_result[set_index] = 1

        success_count = 0
        for index in range(len(query_keys)):
            if exist_result[index] == 0:
                break
            success_count += 1
        return success_count == len(query_keys)

    def get(
        self,
        key,
        target_location: Any | None = None,
        target_sizes: Any | None = None,
    ) -> bool:
        if target_location is None or target_sizes is None:
            raise ValueError("target_location and target_sizes are required")
        # Format key with local_rank suffix for non-MLA backend
        if self.is_mla_backend:
            query_keys = [f"{key}_k"]
            target_locations = [target_location]
            target_sizes_list = [target_sizes]
        else:
            # For non-MLA backend, we need to get both k and v
            query_keys = [f"{key}_{self.local_rank}_k", f"{key}_{self.local_rank}_v"]
            # target_location and target_sizes should be lists with 2 elements
            if isinstance(target_location, list) and len(target_location) >= 2:
                target_locations = [target_location[0], target_location[1]]
            else:
                # If not a list, assume it's a single location for k only
                target_locations = [target_location, target_location]

            if isinstance(target_sizes, list) and len(target_sizes) >= 2:
                target_sizes_list = [target_sizes[0], target_sizes[1]]
            else:
                # If not a list, assume it's a single size for k only
                target_sizes_list = [target_sizes, target_sizes]

        get_result = self._get_batch_zero_copy_impl(
            query_keys, target_locations, target_sizes_list
        )
        # Return True only if both k and v are successfully retrieved
        return all(result >= 0 for result in get_result)

    def batch_get(
        self,
        keys: list[str],
        target_locations: Any | None = None,
        target_sizes: Any | None = None,
    ) -> int:
        if target_locations is None or target_sizes is None:
            raise ValueError("target_locations and target_sizes are required")
        if len(keys) != len(target_locations) or len(keys) != len(target_sizes):
            raise ValueError(
                "keys, target_locations, and target_sizes must have matching lengths"
            )
        if not keys:
            return 0

        query_keys, key_multiplier, expanded_non_mla = self._expand_query_keys(keys)

        # Note: target_locations and target_sizes need to match the query_keys length
        # If keys already have suffixes, target_locations and target_sizes should already match
        # If keys don't have suffixes, we need to expand them for non-MLA backend
        if expanded_non_mla:
            # Expand target_locations and target_sizes to match query_keys
            target_locations = self._expand_pairwise_metadata(
                target_locations, len(keys)
            )
            target_sizes = self._expand_pairwise_metadata(target_sizes, len(keys))

        get_result = self._get_batch_zero_copy_impl(
            query_keys, target_locations, target_sizes
        )

        for index in range(len(query_keys)):
            if get_result[index] < 0:
                return index // key_multiplier
        return len(query_keys) // key_multiplier

    def exists(self, key) -> bool:
        # Format key with local_rank suffix for non-MLA backend
        if self.is_mla_backend:
            query_keys = [f"{key}_k"]
        else:
            # For non-MLA backend, we need to check both k and v
            query_keys = [f"{key}_{self.local_rank}_k", f"{key}_{self.local_rank}_v"]
        exist_result = self._batch_exist(query_keys)
        return all(result == 1 for result in exist_result)

    def batch_exists(
        self, keys, extra_info: KVStoreStorageExtraInfo | None = None
    ) -> int:
        query_keys, key_multiplier, _ = self._expand_query_keys(keys)

        exist_result = self._batch_exist(query_keys)

        for index in range(len(query_keys)):
            if exist_result[index] != 1:
                return index // key_multiplier
        return len(query_keys) // key_multiplier

    def close(self):
        # MooncakeDistributedStore will automatically call the destructor, so
        # it is unnecessary to close it manually.
        return None

    def clear(self) -> None:
        self.store.remove_all()
        if self.on_clear is not None:
            try:
                self.on_clear()
            except Exception:
                logger.exception("MooncakeStore on_clear callback failed")

    def _put_batch_zero_copy_impl(
        self, key_strs: list[str], buffer_ptrs: list[int], buffer_sizes: list[int]
    ) -> list[int]:
        return self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)

    def _get_batch_zero_copy_impl(
        self, key_strs: list[str], buffer_ptrs: list[int], buffer_sizes: list[int]
    ) -> list[int]:
        return self.store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes)

    def _batch_exist(self, key_strs: list[str]) -> list[int]:
        return self.store.batch_is_exist(key_strs)
