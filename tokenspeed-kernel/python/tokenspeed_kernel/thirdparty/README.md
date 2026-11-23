# Vendor adapters

Optional vendor APIs live here; public operator registration and selection
belong in `../ops/`. Runtime code depends on the kernel package boundary.

- [SM120 DeepEP integration](deepep_sm120_experiment/README.md): prepared
  transport resources, local expert adapters and valid-row restoration.
- `deepgemm.py`: DeepGEMM weight/scale preparation and GEMM calls.
- `deepgemm_sm120.py`: explicit dense/paged API adaptation for the pinned
  independent `deep_gemm_sm120` namespace; ordinary DeepGEMM remains available
  to other architectures.
- [FlashInfer](flashinfer/README.md): attention, selector and local-expert APIs.

Dependency versions and reproducible setup are owned by the
[launch recipes](../../../../.agents/README.md).
