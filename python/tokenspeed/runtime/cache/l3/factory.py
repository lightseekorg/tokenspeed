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

"""Construct the configured L3 storage backend."""

from __future__ import annotations

from typing import Any

from tokenspeed.runtime.cache.l3.backend import KvStoreStorage, MemoryKvStore


def create_kvstore_storage_backend(
    backend_name: str | None,
    extra_config: str | None,
    *,
    host_buffer: Any,
    tp_size: int,
    pp_size: int,
) -> KvStoreStorage | None:
    """Return an L3 backend for ``backend_name``, or None when L3 is unset.

    ``memory`` is a test-only in-process store. ``mooncake`` is the production
    Mooncake Store client (SGLang/vLLM HiCache equivalent). ``tp_size`` and
    ``pp_size`` divide ``global_segment_size`` so every attention-TP rank on
    every pipeline stage mounts an equal share of the configured total.
    """

    if backend_name is None:
        return None
    name = backend_name.strip().lower()
    if name == "memory":
        return MemoryKvStore()
    if name == "mooncake":
        from tokenspeed.runtime.cache.l3.mooncake import (
            MooncakeKvStore,
            parse_extra_config,
        )

        return MooncakeKvStore(
            parse_extra_config(extra_config),
            host_buffer=host_buffer,
            tp_size=tp_size,
            pp_size=pp_size,
        )
    raise ValueError(f"unsupported KVStore storage backend {backend_name!r}")
