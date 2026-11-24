# Runtime tests

These tests exercise runtime configuration, model behavior, cache ownership
and execution. GPU tests use small weights unless explicitly documented.

For the SM120 full-model paths:

- `test_deepseek_v41_config.py`: checkpoint-format validation.
- `test_deepseek_v41_model.py`: model math, checkpoint loading and transport
  chunk-capacity behavior.
- `test_deepseek_v41_engram.py`: packed MXFP8/MXFP4 rows, bounded and parallel
  loading, host allocation lifetime, lookup and graph behavior.
- `test_deepseek_v41_sm120_mapping.py`: explicit head/dense/MoE TP and EP rules.
- `test_server_args_context_parallel.py`: QCP role and size validation.

[Kernel GPU tests](../../tokenspeed-kernel/test/README.md) qualify operators;
[full-model recipes](../../.agents/README.md) own performance and quality
regressions. Passing one layer does not qualify the other.
