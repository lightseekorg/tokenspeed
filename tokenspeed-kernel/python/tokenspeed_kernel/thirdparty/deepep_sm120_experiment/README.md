# SM120 DeepEP integration

This directory adapts explicit fixed-capacity DeepEP transports to prepared
FlashInfer local expert weights. Public model execution contracts live in
[the design note](../../../../../docs/design/sm120.md).

- `buffer.py`: transport creation, sharing and communication capacity.
- `deepep_flashinfer_fp8_opt.py`: FP8 dispatch / BF16 combine and optional
  wire1 selection.
- `low_latency_wire1.py`: NVFP4 dispatch / FP8 combine, cached layouts and
  chunk restoration. Logical rows and metadata capacity are distinct.
- `prequant_route.py`, `prequant_fused_moe_opt.py`, `q0_pack.py`,
  `route_mask.py`, `nvfp4_receive.py`: scale/route packing, valid receive
  masking and local expert invocation.
- [W4A8 receive layout](web_w4a8/README.md): compact route metadata and
  restoration consumed by wire1.

All ranks execute the same chunk count, including zero-row peers. Prepare
resources before capture. A short final chunk preserves Decode's fixed
wire1 metadata layout; padded rows must not appear in expert contributions.
The tests under `tokenspeed-kernel/test/nvidia/thirdparty/` verify transport
agreement; [full-model recipes](../../../../../.agents/README.md) own fresh
builds and end-to-end validation. This directory contains no platform service
or deployment orchestration.

`TOKENSPEED_SM120_DEEPEP_NICS`, when supplied, lists one nonempty NIC name
per GPU visible to this process, in that visible-device order. The current
GPU selects `EP_NIC_NAME` and `NVSHMEM_HCA_LIST` before buffer construction.
Without this variable the vendor keeps its own selection. Wire1 also accepts
the old `TOKENSPEED_WIRE1_NICS` alias; the common variable takes precedence.
No hardware names or topology are assumed by the adapter.
