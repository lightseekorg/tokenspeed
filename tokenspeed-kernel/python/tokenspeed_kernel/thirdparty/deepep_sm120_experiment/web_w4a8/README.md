# W4A8 compact expert implementation

This directory contains the complete source of the compact W4A8 expert path used by the SM120 low-latency adapter. It is shipped in this checkout and requires no additional repository or snapshot server. Copyright and license notices remain in each source file.

- `online.py`: expert execution and workspace ownership.
- `_compact.py`, `layout.py`: packing, row maps and padded expert extents.
- `flashinfer.py`: FlashInfer kernels. The exact package version is pinned by the kernel CUDA requirements.

The caller is [low_latency_wire1.py](../low_latency_wire1.py). This optional path requires the fixed DeepEP source and patch described in [.agents FP4 patches](../../../../../../.agents/deploy/fp4/patches/README.md). Validate real distributed expert outputs and CUDA Graph replay before adoption; a successful import is insufficient.
