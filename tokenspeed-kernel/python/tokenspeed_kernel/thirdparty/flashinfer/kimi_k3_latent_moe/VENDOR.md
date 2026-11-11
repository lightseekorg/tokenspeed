# FlashInfer Kimi-K3 LatentMoE projections

Source: https://github.com/flashinfer-ai/flashinfer/tree/5dc3cd06c57465c48f6c6f7317f642b6eb683083/flashinfer/experimental/kimi_k3_latent_moe
(flashinfer-ai/flashinfer#6109)

Revision: `5dc3cd06c57465c48f6c6f7317f642b6eb683083`.
Upstream files retain their Apache-2.0 notices; see LICENSE.

FlashInfer marks this backend experimental and its PyPI wheels omit `csrc/`,
so the package is vendored whole: the host planner (`cake_backend.py`), the
JIT records (`cake_jit.py`) and the generated programs (`csrc/`).

Local adaptation: package imports use the TokenSpeed kernel namespace and the
installed FlashInfer wheel's JIT builder (`flashinfer.jit`); `_header_dirs`
takes the binding headers from that wheel only; formatting follows the
repository hooks.

TokenSpeed calls `cake_backend.kimi_k3_latent_moe_front` from
`thirdparty/cuda/flashinfer_kimi_k3.py` at any token count. `ROW_TOKENS`
lists the token counts upstream validated the route set at; the planner and
kernels take any count.
