# Kernel tests

Operator tests validate numerical, shape, padding, capture and specialization
contracts. GPU tests live under the corresponding vendor directory;
`ops/` and root-level files cover common infrastructure and CPU contracts.

- [SM120 NVIDIA tests](nvidia/README.md).
- [SM120 transport tests](nvidia/thirdparty/README.md).
- [Runtime tests](../../test/runtime/README.md) and [full-model
  regressions](../../.agents/README.md) qualify the integrated model.
