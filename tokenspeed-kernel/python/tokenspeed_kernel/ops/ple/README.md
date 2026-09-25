# PLE n-gram hashing

The fused n-gram kernel hashes the flat token stream without materializing the
window matrix. For equal-length requests it also fills the request and column
index tensors; ragged batches pass those indices from the caller. The launcher
chooses a tile size from a small fixed set, and masked loads and stores handle
partial tiles and empty requests.

Token count, request count, equal-request length, and tail scratch stride are
runtime scalars. The uniform-index and single-request choices are compile-time
flags, so the common single-request case avoids a runtime division while
varying prefill lengths reuse one compiled kernel. The n-gram geometry, tile
size, tail mode, and remainder mode still select variants.
Short requests may use a different tile size and still require a separate
variant. Tests compare empty, short, and irregular long lengths with the legacy
window-based result, including the equal-length index path and both tail modes.
