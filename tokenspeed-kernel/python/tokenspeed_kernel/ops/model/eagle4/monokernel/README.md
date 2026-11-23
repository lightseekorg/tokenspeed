# EAGLE-4 draft decode monokernel

This directory holds the EAGLE-4 drafter's op: its draft layer and final
norm as one launch per chain draft step (TP1). The design and its
invariants are in
[`docs/design/monokernel.md`](../../../../../../../docs/design/monokernel.md);
the general API and how a runtime seam calls the ops are in the
[Qwen4-Exp ops' README](../../qwen4_exp/monokernel/README.md). The gfx950
kernel is described in
[`gfx950/monokernel/eagle4/README.md`](../../../../../../../tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/monokernel/eagle4/README.md).

| Op | Packs or runs |
|---|---|
| `prepare_eagle4_draft_layer(tensors=, max_rows=)` | Packs the draft layer's projections and norm gains (a tiled copy of the projections). Returns `None` when the platform has no kernel. The scratch it asks for (`scratch_pairs`) serves every launch of 1 to `max_rows` rows. |
| `eagle4_draft_layer_decode(layer, *, embeds, hidden, normed, positions, seq_len, slots, page_table, k_cache, v_cache, cos_sin, scratch, step, layer_index, compile_only)` | One chain draft step, one row per request: the layer and the final norm, the row's K and V written into the draft cache. |

The caller owns the mailbox scratch and the device step counter, which
changes once per draft forward inside the CUDA graph. The runtime seam is
`python/tokenspeed/runtime/models/llama_eagle4_monokernel.py`
(`--draft-decode-monokernel`).
