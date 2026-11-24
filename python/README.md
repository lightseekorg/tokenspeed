# TokenSpeed runtime

Runtime code lives in `tokenspeed/`; CUDA/vendor kernels belong in
[`tokenspeed-kernel`](../tokenspeed-kernel/README.md).

- [SM120 execution contracts](../docs/design/sm120.md): QCP Prefill, independent
  Decode, quantized checkpoints and kernel ownership.
- [Runtime tests](../test/runtime/README.md).
- [Reproducible launch and regression recipes](../.agents/README.md).
