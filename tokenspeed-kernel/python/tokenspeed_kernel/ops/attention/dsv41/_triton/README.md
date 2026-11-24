# V4.1 Triton helpers

Private kernels implement cache addressing, query transforms, selection and
attention for the [V4.1 operation family](../README.md).
`gather_attention.py` additionally reads protected SWA rows and original
page-planar global caches for the optional SM120 paged Prefill operation.
Public APIs and registry entries stay in the family-level solution modules.
