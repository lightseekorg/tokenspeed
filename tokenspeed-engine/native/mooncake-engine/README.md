# TokenSpeed NPU transport adapter

This adapter preserves the TokenSpeed C ABI when using a Mooncake wheel that
does not package `libtokenspeed_transfer_engine.so`. Compile it against the
headers and libraries of the selected Mooncake SDK, then select the result with
`TOKENSPEED_MOONCAKE_LIBRARY`. Do not reuse a binary built against another SDK.

The adapter and its tests are carried from the existing NPU image's Mooncake
commit `0e0b1ea2e457f9c630cb771485ed3622b9e58589`, under Apache-2.0
(see `LICENSE-APACHE`). READ uses the selected SDK's public scatter-operation
owner to wait for physical completion even after an error or timeout; it does
not require the image's private blocking `freeBatchID` overload. This directory
does not implement a storage service or replace Mooncake's transport.

`python test_cabi.py` checks validation and transfer-retirement behavior with
test-only network/device stubs. `test_cabi_npu.py` exercises real device buffers
and must only run in an explicitly allocated test environment.
