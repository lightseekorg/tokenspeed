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

"""Pool ownership and transition gates for the Ascend memory saver."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from tokenspeed.runtime.distributed.mapping import Mapping

MEMORY_TAGS = frozenset({"weights", "kv_cache", "acl_graph"})


class AllocationPolicy(StrEnum):
    CPU_BACKUP = "cpu_backup"
    DISCARD = "discard"
    REMAIN = "remain"


@dataclass(frozen=True)
class PoolKey:
    """A release tag can own independent pools with different lifetimes."""

    tag: str
    policy: AllocationPolicy
    scope: str


class Backend(Protocol):
    def pause(self, tag: str | None) -> None: ...
    def resume(self, tag: str | None) -> None: ...


@dataclass
class OwnedPool:
    allocator: Any
    pool: Any


class PoolState:
    """Keep allocator owners alive and block use of unmapped or failed regions.

    The scheduler must stop submissions before calling pause. This gate protects
    adapter entry points; it cannot intercept arbitrary tensor accesses.
    """

    def __init__(
        self,
        *,
        backend: Backend,
        make_pool: Callable[[PoolKey], OwnedPool],
        synchronize: Callable[[], None],
    ):
        self.backend = backend
        self.make_pool = make_pool
        self.synchronize = synchronize
        self.pools: dict[PoolKey, OwnedPool] = {}
        self.paused: set[str] = set()
        self.failed: set[str] = set()
        self.active_key: PoolKey | None = None
        self.lock = RLock()

    def _pool(self, key: PoolKey) -> OwnedPool | None:
        if key.tag not in MEMORY_TAGS:
            raise ValueError(f"Unsupported Ascend memory tag: {key.tag!r}")
        if key.tag in self.paused or key.tag in self.failed:
            raise RuntimeError(f"Memory tag {key.tag!r} is paused or needs recovery")
        if key.policy is AllocationPolicy.REMAIN:
            return None
        if key not in self.pools:
            self.pools[key] = self.make_pool(key)
        return self.pools[key]

    @contextmanager
    def region(self, key: PoolKey) -> Iterator[OwnedPool | None]:
        # Hold the gate for the entire capture/allocation scope.
        with self.lock:
            if self.active_key is not None:
                raise RuntimeError("Cannot nest Ascend memory allocation scopes")
            owned = self._pool(key)
            self.active_key = key
            try:
                yield owned
            finally:
                self.active_key = None

    def graph_pool(
        self, *, owned: OwnedPool | None, requested
    ) -> tuple[int, int] | None:
        if owned is None:
            if requested is not None and any(
                entry.pool.id == requested for entry in self.pools.values()
            ):
                raise ValueError("A resident graph cannot use a managed pool")
            return requested
        selected = owned.pool.id
        if requested is not None and requested != selected:
            raise ValueError(
                f"Graph pool {requested!r} is not the managed pool {selected!r}"
            )
        return selected

    def assert_active(self) -> None:
        with self.lock:
            unavailable = self.paused | self.failed
            if unavailable:
                raise RuntimeError(f"Memory is unavailable: {sorted(unavailable)}")

    def _targets(self, tag: str | None) -> set[str]:
        if tag is not None:
            if tag not in MEMORY_TAGS:
                raise ValueError(f"Unsupported Ascend memory tag: {tag!r}")
            return {tag}
        return set(MEMORY_TAGS)

    def pause(self, tag: str | None) -> None:
        with self.lock:
            if self.active_key is not None:
                raise RuntimeError("Cannot pause an active allocation/capture scope")
            targets = self._targets(tag)
            self.failed.update(targets)
            # torch_npu synchronization also drains its framework task queue.
            self.synchronize()
            self.backend.pause(tag)
            self.paused.update(targets)
            self.failed.difference_update(targets)

    def resume(self, tag: str | None) -> None:
        with self.lock:
            if self.active_key is not None:
                raise RuntimeError("Cannot resume an active allocation/capture scope")
            targets = self._targets(tag)
            self.failed.update(targets)
            self.backend.resume(tag)
            self.synchronize()
            self.paused.difference_update(targets)
            self.failed.difference_update(targets)


def validate_memory_saver_scope(
    *,
    mapping: Mapping,
    emulate_rank_zero: bool,
    disaggregation_mode: str,
    has_draft_weights: bool,
    speculative_draft_tree: bool,
    speculative_topk: int,
    speculative_sampling: bool,
    weight_policy: str | None,
    draft_policy: str | None,
) -> None:
    """Keep every serving collective inside the single-node TP replica."""
    world_size = mapping.world_size
    if (
        world_size not in (1, 2, 4, 8)
        or mapping.nnodes != 1
        or mapping.attn.tp_size != world_size
        or mapping.dense.tp_size != world_size
        or mapping.moe.tp_size != world_size
        or mapping.vision.tp_size != world_size
        or mapping.attn.has_qcp
        or mapping.attn.has_kvp
        or emulate_rank_zero
        or disaggregation_mode != "null"
    ):
        raise ValueError(
            "Ascend memory saver requires single-node TP1, TP2, TP4, or TP8 "
            "without disaggregation or rank emulation"
        )
    if speculative_topk != 1 or speculative_draft_tree or speculative_sampling:
        raise ValueError("Ascend memory saver supports greedy chain drafting")
    if weight_policy not in ("cpu_backup", "discard"):
        raise ValueError("Ascend memory saver requires cpu_backup or discard weights")
    if (
        has_draft_weights
        and weight_policy == "discard"
        and draft_policy
        not in (
            "retain",
            "refresh",
        )
    ):
        raise ValueError(
            "Ascend weight discard with a draft checkpoint requires an explicit "
            "--model-update-draft-weights policy"
        )
