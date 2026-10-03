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

"""Adapt router layers and history IDs to KV offloading compute slots."""

from typing import TYPE_CHECKING

from tokenspeed_kernel.ops.kvcache.offload import seed_locations

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode


class KVOffloadAdapter:
    """Adapt router cache access to the arena-owned KV offload engine.

    The arena owns payloads, GPU tags/LRU, and transfer streams. This adapter
    resolves layer/field/group bindings and publishes this step's mappings.
    """

    def __init__(self, router, pool):
        self.router = router
        self.engine = pool.arena.offload
        self.config = self.engine.config
        bound = [
            (layer, name)
            for (layer, plane), name in pool.layer_fields.items()
            if name in self.engine.fields
        ]
        self.fields = dict(bound)
        if len(self.fields) != len(bound):
            raise ValueError("sparse residency requires one offloaded plane per layer")
        if set(self.fields.values()) != set(self.engine.fields):
            raise ValueError("offloaded fields must belong to the target cache view")
        layers = {name: layer for (layer, _), name in pool.layer_fields.items()}
        self.consumers = {}
        for producer, consumer in self.config.selection_consumers:
            if producer in layers and consumer in self.engine.fields:
                self.consumers.setdefault(layers[producer], []).append(consumer)
        self.groups = {f.field_id: f.group_id for f in pool.arena.plan.fields}
        self.requests = None
        self.writes = {}
        self.reads = {}
        self.seeds = {}
        self.extend_writes = {}
        # Layers whose selection this step has already mapped, by forward
        # mode. Offload rows are only addressable after their prepare, so
        # every consumer cross-checks this instead of falling back silently.
        self.prepared: dict[int, ForwardMode] = {}

    def validate_slots(self, capacity):
        """Check that KV offloading partitions match executor/verify geometry."""
        if capacity != self.config.request_slots:
            raise ValueError("offload request slots disagree with the executor")
        if self.router.spec_num_tokens != self.config.queries:
            raise ValueError("offload verify width disagrees with the backend")

    def begin(self, requests, *, num_extends, stream):
        """Clear last-step mappings and bind the engine's offloading batch."""
        self.requests = requests
        self.writes.clear()
        self.reads.clear()
        self.seeds.clear()
        self.prepared.clear()
        self.engine.begin(requests, num_extends=num_extends, stream=stream)

    def set_extend_writes(self, writes):
        """Bind recovery history destinations for offloading staging writes."""
        self.extend_writes = writes

    def _seed(self, name, positions):
        """Seed offloaded short-prefix/ring rows using this group's page table."""
        group = self.groups[name]
        if group not in self.seeds:
            view = self.router.group_view(group, self.requests.numel())
            self.seeds[group] = seed_locations(
                view.page_table,
                positions,
                self.requests,
                page_size=view.kernel_page_size,
                hot=self.config.hot_tokens,
                stride=self.config.buffer_tokens,
                queries=self.config.queries,
                cyclic=self.config.cyclic_tokens,
            )
        self.engine.seed(name, *self.seeds[group])

    def offloads_layer(self, layer_id) -> bool:
        """Whether this layer consumes an offloaded plane."""
        return layer_id in self.fields

    def prepared_for(self, layer_id, forward_mode) -> bool:
        """Whether KV offloading prepared this layer for the same forward family.

        Compare extend/decode families rather than exact mode identity.
        This guard does not grant support for mixed recovery batches.
        """
        mode = self.prepared.get(layer_id)
        return mode is not None and mode.is_extend() == forward_mode.is_extend()

    def compute_write_locations(self, layer, forward_mode):
        """Return prepared KV offloading writes or ordinary history writes.

        A missing prepare would otherwise fall back to history slots the
        offload view cannot address.
        """
        if not self.offloads_layer(layer.layer_id):
            return self.router.write_locations(layer, forward_mode)
        if not self.prepared_for(layer.layer_id, forward_mode):
            raise RuntimeError(
                f"layer {layer.layer_id} requested {forward_mode.name} write "
                "locations without a matching prepare_sparse_kv_access; offload "
                "rows are only addressable after this step's prepare"
            )
        return self.writes[layer.layer_id]

    def prepare(self, layer, topk, positions, mode):
        """Prepare one offloaded field and return its compute write slots.

        Decode seeds history and resolves hot reads/current writes, joining
        any producer prefetch. Extend reserves chunk projection writes.
        Non-offloaded layers return their ordinary history write slots.
        """
        name = self.fields.get(layer.layer_id)
        if name is None:
            return self.router.write_locations(layer, mode)
        if mode.is_extend():
            group = self.groups[name]
            full = self.extend_writes[group]
            self.writes[layer.layer_id] = self.engine.prepare_extend(name, full)
        else:
            self._seed(name, positions)
            mapped, writes = self.engine.resolve(
                name,
                topk,
                positions,
                self.router.write_locations(layer, mode),
            )
            self.writes[layer.layer_id] = writes
            self.reads[layer.layer_id] = mapped
        self.prepared[layer.layer_id] = mode
        return self.writes[layer.layer_id]

    def prefetch(self, layer, topk, positions):
        """Schedule offloading loads for consumers of this producer's Top-K.

        Seed on the execution stream, then enqueue each consumer's resolve
        work on the engine prefetch stream. Its later prepare joins the event.
        """
        if not self.config.overlap:
            return
        for name in self.consumers.get(layer.layer_id, ()):
            self._seed(name, positions)
            # The recipe validates equal history groups for shared selections.
            full = self.router.decode_write_locations.by_group[self.groups[name]]
            self.engine.prefetch(name, topk, positions, full)

    def read_indices(self, layer, original):
        """Substitute prepared offloading hot reads, preserving entry order."""
        mapped = self.reads.get(layer.layer_id)
        if mapped is None:
            if self.offloads_layer(layer.layer_id):
                raise RuntimeError(
                    f"layer {layer.layer_id} read sparse indices without a "
                    "prepare_sparse_kv_access for this step; the offload view has "
                    "no mapped rows to substitute"
                )
            return original
        return mapped

    def prefill_tiles(self, layer, selected):
        """Return the offloading recovery iterator, or None for ordinary KV."""
        name = self.fields.get(layer.layer_id)
        if name is None:
            return None
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        self.compute_write_locations(layer, ForwardMode.EXTEND)
        return self.engine.prefill_tiles(name, selected)
