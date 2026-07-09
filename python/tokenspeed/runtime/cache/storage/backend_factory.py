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

from __future__ import annotations

import importlib
import logging
from typing import Any

from tokenspeed.runtime.cache.kvstore_storage import (
    KVStoreStorage,
    KVStoreStorageConfig,
)

logger = logging.getLogger(__name__)


class StorageBackendFactory:
    """Factory for creating storage backend instances with support for dynamic loading."""

    _registry: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _load_backend_class(
        module_path: str, class_name: str, backend_name: str
    ) -> type[KVStoreStorage]:
        """Load and validate a backend class from module path."""
        try:
            module = importlib.import_module(module_path)
            backend_class = getattr(module, class_name)
            if not issubclass(backend_class, KVStoreStorage):
                raise TypeError(
                    f"Backend class {class_name} must inherit from KVStoreStorage"
                )
            return backend_class
        except ImportError as exc:
            raise ImportError(
                f"Failed to import backend '{backend_name}' from '{module_path}': {exc}"
            ) from exc
        except AttributeError as exc:
            raise AttributeError(
                f"Class '{class_name}' not found in module '{module_path}': {exc}"
            ) from exc

    @classmethod
    def register_backend(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a storage backend with lazy loading.

        Args:
            name: Backend identifier
            module_path: Python module path containing the backend class
            class_name: Name of the backend class
        """
        if name in cls._registry:
            logger.warning("Backend '%s' is already registered, overwriting", name)

        def loader() -> type[KVStoreStorage]:
            """Lazy loader function to import the backend class."""
            return cls._load_backend_class(module_path, class_name, name)

        cls._registry[name] = {
            "loader": loader,
            "module_path": module_path,
            "class_name": class_name,
        }

    @classmethod
    def create_backend(
        cls,
        backend_name: str,
        storage_config: KVStoreStorageConfig,
        mem_pool_host: Any,
        **kwargs,
    ) -> KVStoreStorage:
        """Create a storage backend instance.
        Args:
            backend_name: Name of the backend to create
            storage_config: Storage configuration
            mem_pool_host: Memory pool host object
            **kwargs: Additional arguments passed to external backends
        Returns:
            Initialized storage backend instance
        Raises:
            ValueError: If backend is not registered and cannot be dynamically loaded
            ImportError: If backend module cannot be imported
            Exception: If backend initialization fails
        """
        # First check if backend is already registered
        if backend_name in cls._registry:
            registry_entry = cls._registry[backend_name]
            backend_class = registry_entry["loader"]()
            logger.info(
                "Creating storage backend '%s' (%s.%s)",
                backend_name,
                registry_entry["module_path"],
                registry_entry["class_name"],
            )
            return backend_class(storage_config)

        # Try to dynamically load backend from extra_config
        if backend_name == "dynamic" and storage_config.extra_config is not None:
            backend_config = storage_config.extra_config
            return cls._create_dynamic_backend(
                backend_config, storage_config, mem_pool_host, **kwargs
            )

        # Backend not found
        raise ValueError(
            f"Unknown storage backend '{backend_name}'. "
            f"Registered backends: {list(cls._registry)}. "
        )

    @classmethod
    def _create_dynamic_backend(
        cls,
        backend_config: dict[str, Any],
        storage_config: KVStoreStorageConfig,
        mem_pool_host: Any,
        **kwargs,
    ) -> KVStoreStorage:
        """Create a backend dynamically from configuration."""
        required_fields = ("backend_name", "module_path", "class_name")
        missing_fields = [
            field for field in required_fields if field not in backend_config
        ]
        if missing_fields:
            raise ValueError(
                "Missing required fields in backend config for 'dynamic' backend: "
                f"{missing_fields}"
            )

        backend_name = backend_config["backend_name"]
        module_path = backend_config["module_path"]
        class_name = backend_config["class_name"]

        try:
            # Import the backend class
            backend_class = cls._load_backend_class(
                module_path, class_name, backend_name
            )

            logger.info(
                "Creating dynamic storage backend '%s' (%s.%s)",
                backend_name,
                module_path,
                class_name,
            )

            # Create the backend instance with storage_config
            return backend_class(storage_config, kwargs)
        except Exception as exc:
            logger.error(
                "Failed to create dynamic storage backend '%s': %s", backend_name, exc
            )
            raise


# Register built-in storage backends
StorageBackendFactory.register_backend(
    "mooncake",
    "tokenspeed.runtime.cache.storage.mooncake_store.mooncake_store",
    "MooncakeStore",
)
