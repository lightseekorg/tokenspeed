# DSV4 attention

Prefill and decode selection includes the query head count. FlashMLA's sparse
attention kernels support 64 or 128 heads; other head counts select an eligible
alternative, including the portable Triton implementation.

`triton.py` re-exports kernels and helpers from `_triton/`, preserving the import
path used by runtime callers. Implementations are grouped by responsibility:

- `attention.py`: selected attention over dense and page-planar KV caches.
- `cache.py`: SWA insertion, cache reads and writes, and dequantization.
- `common.py`: shared geometry, block-table normalization, and MXFP4 encoding.
- `compress.py`: sparse and indexer compression and compressor state storage.
- `indexer.py`: MXFP4 scoring, top-k selection, and execution plans.
- `indices.py`: SWA and compressed KV selection indices.
- `metadata.py`: slot mappings, active-page validation, and decode metadata.
- `quantization.py`: inverse RoPE and FP8 attention-output quantization.
- `query.py`: indexer query RoPE, Hadamard transform, and MXFP4 quantization.
