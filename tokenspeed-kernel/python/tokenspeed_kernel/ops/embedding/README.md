# Embedding kernels

Callers use `apply_rope` or `apply_qk_norm_rope`; both dispatch through the
`embedding.rope` registry. Backend implementations advertise the layouts and
fusions they support, so model code does not select vendor kernels directly.

Portable Triton implements QK normalization together with RoPE for complete
cross-vendor coverage. More specialized solutions may replace that kernel when
their shape and backend traits match.

## Host-table row gathers (`host_gather.py`)

`uint8_row_gather` and `mxfp8_row_gather` read rows of a table kept in pinned
or `cudaHostRegister`ed host memory through a device-visible pointer. Random
rows are bound by address translation, one host page per row, rather than by
PCIe bytes, so a table that needs several fields per row should keep them in
one packed row. `mxfp8_row_gather` reads `[D E4M3 codes | D/32 UE8M0 scales]`
rows and writes them as a GEMM's group-32 MXFP8 activation: the codes and
canonical scales that quantizing the dequantized BF16 rows (1e-4 amax floor)
would produce, without materializing BF16.

Large `mxfp8_row_gather` lookups (at least `_SORT_MIN_ROWS` rows) sort the ids
first, so neighbouring rows share GPU page walks, and read each distinct row
once: the first entry of a run of equal ids reads the host and a device-side
copy fills the rest of the run. Sorting stays on the device, so the launch
grids do not depend on the ids. Each program of the sorted path reads several
rows at once to keep more host reads in flight. Smaller lookups keep one
unsorted row per program, where sorting would cost more than it saves.
