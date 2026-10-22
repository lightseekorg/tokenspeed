# Direct paged attention over the V4.1 LCM arena

`TS_V41_PA_MODE` is resolved once into the attention configuration:
`direct` selects aligned BF16 KV/Index storage, while unset or `compact`
selects compact gather. The recipe, memory budget and pool use that resolved
mode; changing the environment after initialization cannot alter dispatch.

The recipe aligns FlatKV field offsets and subpage strides to 1024 bytes,
one BF16-512 row. Logical cache groups, scheduler pages, ownership and prefix
matching remain unchanged. Only the FlatKV plane is exposed as a contiguous
row address space. Unselected gaps and index planes are not valid selections.
A final physical block can overlap the next plane, but masks must exclude it.

Each cache pool owns a `pa_context`, created through `tokenspeed-kernel`.
The backend passes the context explicitly to attention; direct mode rejects
missing or mismatched contexts. The context owns
an identity block table, zero rotary keys spanning physical rows, a bounded
zero-query slab, and the physical KV length. These constants are allocated
before capture, budgeted with the arena, and remain alive with their pool.

Address conversion is `field_origin_rows + (slot // 64) * page_stride_rows +
slot % 64`. Check selection lengths and slot bounds first; compaction preserves
valid order and duplicates. The SFA call uses sparse mode 0, one-row sparse
blocks, the full post-RoPE main key and one sink correction.
`actual_seq_lengths_kv` is the physical span, not a request's logical length.
Logical visibility comes exclusively from selections and masks.

DSpark's BF16 draft windows retain their SWA cache-group and PD ownership.
Draft attention supplies compact KV and bypasses this target path. Both PD
roles must use the same aligned layout. Changing cache semantics requires a
restart.

The optional `dsv41_paged_metadata` Ascend C kernel fuses address conversion,
masking and stable compaction for supported INT32 layouts. Other layouts use
the existing device fallback. It does not change stored KV values.
