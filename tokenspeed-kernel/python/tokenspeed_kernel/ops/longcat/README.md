# LongCat kernel interfaces

LongCat CuTe device implementations live in `thirdparty/cute_dsl/longcat/`.
The model-facing interfaces are:

- `router.py`: LongCat tuning for the shared BF16 router driver.
- `cute_dsl.py`: the fused indexer/worker and full-layer programs, operand
  preparation and selection exchange. Import when enabling the engines.

Ordinary LongCat attention uses the shared `ops.attention.dsa` scoring and
Top-K operators, plus the shared normalization, RoPE and cache-write paths.
LongCat supplies its forced initial/local window through the DSA arguments.
The query limit includes the model's eight-row verify window.

The fused engines own their device pipelines and synchronization. Expert MoE
uses weight-as-M GateUp and hidden-slab Down with direct TMA weight and scale
loads.
Dense GateUp reuses the shared NVFP4 layout and pipeline helpers. Generic GEMM,
quantization and sampling remain under their shared kernel APIs.
