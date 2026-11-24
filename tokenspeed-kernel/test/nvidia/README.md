# NVIDIA kernel tests

SM120 additions use explicit device and optional-feature guards. Run them
from the repository root in the freshly built matching environment.

- `ops/attention/test_flashinfer_dsv41.py`: sparse attention and graph replay.
- `ops/attention/test_flashinfer_sm120_dsv41_prefill.py`: paged Prefill;
  run with `TOKENSPEED_ENABLE_SM120_PAGED_PREFILL=1`.
- `ops/test_dsv41_indexer_sm120.py`: BF16 Indexer; run with
  `TOKENSPEED_ENABLE_SM120_PREFILL_INDEXER=1`.
- `ops/test_dsv41_indexer_sm120_native_fp4.py`: native FP4 Indexer; run in a
  separate process with `TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER=1` and
  the pinned `deep_gemm_sm120` dependency.
- `ops/test_dsv41_sm120_full_batch.py`: complete query-tile coverage.
- `ops/gemm/test_sm120_deepgemm.py`: group-32 dense projection.
- `ops/moe/test_flashinfer_sm12x_mxfp4.py` and `test_sm12x_nvfp4.py`: original
  packed expert weights and calibrated NVFP4 scales.
- `ops/test_mhc_default_sm120.py`: default IEEE mHC; run in a separate process
  with `TOKENSPEED_ENABLE_SM120_MHC_MIXES` disabled to check real automatic
  selection and numerical outputs.
- `ops/test_mhc_mixes_sm120.py`: optional compensated mHC numerical contract.
- [Distributed DeepEP tests](thirdparty/README.md).

Test skips are not GPU qualification. Record selected tests, skip reasons,
loaded artifact identities and actual device in the full-model recipe's
validation evidence.
