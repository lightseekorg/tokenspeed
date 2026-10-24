# Ascend V4.1 attention integration

`ascend.py` registers the NPU implementations behind the shared attention API.
It retains shared argument validation and index query collectives. Device
attention execution and explicit arena context live in
`tokenspeed-kernel-npu/ops/dsv41_attention.py`; the cache pool owns the context.

The shared Triton compressor pool and tail writer are also available on
Ascend. Other metadata operations use their Ascend registrations.

Both profiles use BF16 resident KV and Index K, with optional V4.1 operand QDQ.
See the [NPU operator contracts](../../../../../../tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/README.dsv41.md)
for numerical behavior and the
[serving guide](../../../../../../docs/guides/deepseek-v41-ascend.md)
for profile selection and dependencies.
