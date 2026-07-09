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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from tokenspeed.runtime.cache.kv_cache_host import HostKVCache
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@dataclass
class KVStoreStorageConfig:
    tp_rank: int
    tp_size: int
    is_mla_model: bool
    is_page_first_layout: bool
    model_name: str | None
    extra_config: dict | None = None


@dataclass
class KVStoreStorageExtraInfo:
    prefix_keys: list[str] | None = None
    extra_info: dict | None = None


class KVStoreStorage(ABC):
    """
    KVStoreStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    def register_mem_pool_host(self, mem_pool_host: HostKVCache) -> None:
        self.mem_pool_host = mem_pool_host

    def batch_get_v1(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: KVStoreStorageExtraInfo | None = None,
    ) -> list[bool]:
        """
        Retrieve values for multiple keys.
        Returns a list of booleans indicating success for each key.
        """
        raise NotImplementedError

    def batch_set_v1(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: KVStoreStorageExtraInfo | None = None,
    ) -> list[bool]:
        """
        Store multiple key-value pairs.
        Returns a list of booleans indicating success for each key.
        """
        raise NotImplementedError

    @abstractmethod
    def get(
        self,
        key: str,
        target_location: Any | None = None,
        target_sizes: Any | None = None,
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    def batch_get(
        self,
        keys: list[str],
        target_locations: Any | None = None,
        target_sizes: Any | None = None,
    ) -> list[torch.Tensor | None] | int:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        raise NotImplementedError

    @abstractmethod
    def set(
        self,
        key: str,
        value: Any | None = None,
        target_location: Any | None = None,
        target_sizes: Any | None = None,
    ) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def batch_set(
        self,
        keys: list[str],
        values: Any | None = None,
        target_locations: Any | None = None,
        target_sizes: Any | None = None,
    ) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        raise NotImplementedError

    def batch_exists(
        self, keys: list[str], extra_info: KVStoreStorageExtraInfo | None = None
    ) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        for index, key in enumerate(keys):
            if not self.exists(key):
                return index
        return len(keys)

    def clear(self) -> None:
        return None
