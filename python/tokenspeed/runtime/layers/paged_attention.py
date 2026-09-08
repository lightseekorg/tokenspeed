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

"""Paged attention.

A ``PagedAttention`` layer declares its *visibility* -- how far back the
kernel may look (``sliding_window_size``, a ``window_left`` mask) -- and
nothing about storage. Which cache group its KV lives in is the cache plan's
decision, bound onto the layer at executor startup by
:func:`bind_cache_groups`, which also checks the one relation the two
contracts must satisfy: a group must retain every token its layers can see.
"""

from collections.abc import Mapping
from typing import Protocol

from torch import nn

from tokenspeed.runtime.execution.context import ForwardContext


def hf_sliding_window_to_window_left(sliding_window: int) -> int:
    """HF sliding windows count the current token; kernels take the number of
    earlier tokens still visible (``window_left``)."""
    return int(sliding_window) - 1


class PagedAttention(nn.Module):
    """
    The attention layer implementation.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
    ):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        # Visibility: window_left of the compute mask, -1 for full attention.
        # 0 is a real window (the current token only), not "unset".
        if sliding_window_size is None or sliding_window_size < -1:
            raise ValueError(
                f"PagedAttention layer_id={layer_id}: sliding_window_size is a "
                f"window_left >= 0 or -1 for full attention, got "
                f"{sliding_window_size!r}"
            )
        self.sliding_window_size = int(sliding_window_size)
        # Storage: the cache group this layer's KV rides. Owned by the cache
        # plan and bound at startup (bind_cache_groups); the model never
        # names it.
        self._group_id: str | None = None
        self.k_scale = None
        self.v_scale = None

    @property
    def group_id(self) -> str:
        if self._group_id is None:
            raise RuntimeError(
                f"PagedAttention layer_id={self.layer_id} has no cache group "
                "bound; bind_cache_groups runs at executor startup, before any "
                "forward."
            )
        return self._group_id

    def bind_cache_group(self, group_id: str) -> None:
        """Bind the plan's group for this layer; rebinding to another group
        is a contract bug, not a rename."""
        if not group_id:
            raise ValueError(
                f"PagedAttention layer_id={self.layer_id}: cache group id must "
                "be nonempty"
            )
        if self._group_id is not None and self._group_id != group_id:
            raise ValueError(
                f"PagedAttention layer_id={self.layer_id} is bound to cache "
                f"group {self._group_id!r}; cannot rebind to {group_id!r}"
            )
        self._group_id = group_id

    def forward(
        self,
        q,
        k,
        v,
        ctx: ForwardContext,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run this layer's attention; KV write locations come from the
        backend (``write_locations``), never from the caller."""
        if k is not None:
            # For cross-layer sharing, kv can be None
            if v is None:
                raise ValueError("v must be provided when k is provided.")
            if "k_pe" not in kwargs:
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
            else:
                k = k.view(-1, self.tp_k_head_num, self.v_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)

        return ctx.attn_backend.forward(
            q,
            k,
            v,
            self,
            ctx.token_to_kv_pool,
            ctx.forward_mode,
            ctx.bs,
            save_kv_cache,
            **kwargs,
        )


class _CacheGroupSpecLike(Protocol):
    group_id: str
    retention: str
    sliding_window_tokens: int | None


class _CacheArenaLike(Protocol):
    @property
    def cache_group_specs(self) -> tuple[_CacheGroupSpecLike, ...]: ...


class _CacheViewLike(Protocol):
    """What :func:`bind_cache_groups` reads from a cache pool (duck-typed:
    the pool package imports this module)."""

    arena: _CacheArenaLike

    def history_group_by_layer(self) -> Mapping[int, str]: ...


def bind_cache_groups(model: nn.Module, cache_pool: _CacheViewLike) -> None:
    """Bind every ``PagedAttention`` layer to the cache group the plan
    declared its KV planes in, checking retention covers visibility.

    Fails fast (ValueError) at startup instead of a KeyError deep in the
    backend, possibly during graph capture. The plan is the single record
    of layer -> group, so the model side carries no group vocabulary at all;
    the one thing a layer must satisfy is that its group keeps every token
    its mask can reach: a full-visibility layer cannot ride a sliding group,
    and a sliding mask must fit inside the group's retention window.
    """
    specs = {str(spec.group_id): spec for spec in cache_pool.arena.cache_group_specs}
    group_by_layer = cache_pool.history_group_by_layer()
    model_name = type(model).__name__
    for name, module in model.named_modules():
        if not isinstance(module, PagedAttention):
            continue
        group_id = group_by_layer.get(module.layer_id)
        if group_id is None:
            raise ValueError(
                f"{model_name}: attention layer {name!r} (layer_id="
                f"{module.layer_id}) has no history-family cache group in the "
                f"pool's plan (planned layers: {sorted(group_by_layer)})."
            )
        _check_visibility_within_retention(model_name, name, module, specs[group_id])
        module.bind_cache_group(group_id)


def _check_visibility_within_retention(
    model_name: str, name: str, layer: PagedAttention, spec: _CacheGroupSpecLike
) -> None:
    if spec.retention != "sliding_window":
        return
    window_left = layer.sliding_window_size
    retained = spec.sliding_window_tokens
    if window_left < 0:
        raise ValueError(
            f"{model_name}: attention layer {name!r} (layer_id={layer.layer_id}) "
            f"sees the full history but its cache group {spec.group_id!r} "
            f"retains only a {retained}-token window."
        )
    # The scheduler keeps the last `retained - 1` computed tokens ahead of
    # the next position (GroupGeometry::ExpiredBlocksAt), exactly the HF
    # window minus the current token.
    if retained is None or window_left + 1 > retained:
        raise ValueError(
            f"{model_name}: attention layer {name!r} (layer_id={layer.layer_id}) "
            f"masks to window_left={window_left} but its cache group "
            f"{spec.group_id!r} retains only a {retained}-token window."
        )


def check_block_drafter_storage(
    draft_model: nn.Module, target_cache_pool: _CacheViewLike
) -> None:
    """A block drafter writes at the target's cache locations, so every one of
    its layers must ride a full-history group the target's own layers share.

    Bind the draft first (:func:`bind_cache_groups`). Without a target
    full-history group there is nothing to borrow: the draft's KV would sit
    in a group of its own while the target router hands it another group's
    slots.
    """
    target_groups = set(target_cache_pool.history_group_by_layer().values())
    specs = {
        str(spec.group_id): spec for spec in target_cache_pool.arena.cache_group_specs
    }
    model_name = type(draft_model).__name__
    for name, module in draft_model.named_modules():
        if not isinstance(module, PagedAttention):
            continue
        group_id = module.group_id
        if group_id not in target_groups or specs[group_id].retention != "full_history":
            raise ValueError(
                f"{model_name}: block drafter layer {name!r} (layer_id="
                f"{module.layer_id}) rides cache group {group_id!r}, but a block "
                "drafter must share a full-history group with the target's "
                f"layers (target groups: {sorted(target_groups)})."
            )
