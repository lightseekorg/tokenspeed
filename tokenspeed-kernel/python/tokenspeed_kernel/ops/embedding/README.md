# Embedding kernels

Callers use `apply_rope` or `apply_qk_norm_rope`; both dispatch through the
`embedding.rope` registry. Backend implementations advertise the layouts and
fusions they support, so model code does not select vendor kernels directly.

Portable Triton implements QK normalization together with RoPE for complete
cross-vendor coverage. More specialized solutions may replace that kernel when
their shape and backend traits match.

## Host-table and peer-GPU row gathers (`host_gather.py`)

`uint8_row_gather` and `mxfp8_row_gather` read rows of a table kept in pinned,
`cudaHostRegister`ed, `tokenspeed_kernel.host_memory` host memory, or
`tokenspeed_kernel.peer_memory` peer-sharded GPU VMM memory through a
device-visible pointer. Random rows are bound by address translation, one page
per row, rather than by transfer bytes, so a table that needs several fields
per row should keep them in one packed row and gather it whole: a gather per
column view translates every row again. For host tables, the GPU translates host
memory with the pages that back it: a registered tmpfs file's base pages
(64 KiB on 64 KiB-page aarch64 kernels, where a 100 GB table spans 1.5 M GPU
pages), against 2 MiB for `host_memory`'s host-NUMA cuMem memory, which the
processes of a host can also share.

`mxfp8_row_gather` reads `[D E4M3 codes | D/32 UE8M0 scales]` rows and writes
them as a GEMM's group-32 MXFP8 activation: the codes and UE8M0 scales that
quantizing the dequantized BF16 rows (1e-4 amax floor) would produce, without
materializing BF16. Callers pass `scale_layout` explicitly: `"canonical"` writes
contiguous uint8 `[tokens, columns * D / 32]` scales; `"packed"` writes MN-major
int32 `[tokens, columns * D / 128]` words strided `(1, pad4(tokens))` (four
group-32 exponents per word) that `fp8_projection` consumes directly without a
scale-repack pass.

Large `mxfp8_row_gather` lookups (at least `_SORT_MIN_ROWS` rows) sort the ids
first, so neighbouring rows share GPU page walks, and read each distinct row
once: the first entry of a run of equal ids reads the host and a device-side
copy fills the rest of the run. Sorting stays on the device, so the launch
grids do not depend on the ids. Each program of the sorted path reads several
rows at once to keep more host reads in flight. Smaller lookups keep one
unsorted row per program, where sorting would cost more than it saves.
