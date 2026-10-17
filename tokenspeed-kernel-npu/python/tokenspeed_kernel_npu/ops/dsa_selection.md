# DSA selection-window constants

LightningIndexer takes an `initial` count per request and a `local` count per
query row. These are immutable model window settings, not sequence lengths or
selected token indices. Previously each layer call filled two device tensors.
The adapter now retains them by device, request count, row count and both window
values. It never replaces an entry while graphs may still refer to its storage.
Dynamic query ends, KV lengths and page tables continue to come from each call.

This is a Python adapter change; it requires no native operator, scheduler,
cache allocator or KV-placement changes. Cache entries live as long as the
adapter. Shapes outside graph capture are cached by the same rules.

Run `pytest -q tokenspeed-kernel-npu/test/test_dsa_selection.py` on an NPU.
Tests execute the real indexer against freshly allocated window tensors, vary
window values/shapes and replay a graph after changing lengths and populating
another shape. Selection order may vary; the complete multiset, including unused
slots, must match. Immutable values and storage addresses must remain stable.
