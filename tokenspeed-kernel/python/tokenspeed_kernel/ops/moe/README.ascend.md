# Ascend V4.1 experts

The explicit `ascend_w4a4` and `ascend_bf16` solutions accept serialized MXFP4
expert weights with V4.1 dimensions and standard SwiGLU clamped at 10.
They require MoE TP1, precomputed routes, and rank-order combine. The runtime
owns EP communication; the kernel filters this rank’s experts, computes native
grouped GEMMs and returns its partial. Route weights are applied before the
down projection. Attention and runtime integration are described in the
[serving guide](../../../../../docs/guides/deepseek-v41-ascend.md).

`--moe-backend ascend_w4a4 --npu-flatquant-config PATH` loads preconverted
weights. PATH is a versioned JSON manifest; filenames are relative to it:

```json
{
  "format": "tokenspeed-v41-w4a4",
  "version": 1,
  "ep_size": 16,
  "layers": {
    "0": {
      "0": {"w13": "layer0-rank0-w13.pt", "w2": "layer0-rank0-w2.pt"}
    }
  }
}
```

Provide both projections for every local layer and EP rank. `w13` contains the
concatenated gate/up projection; `w2` is the down projection. The manifest's
EP size must match deployment. Each file is a `torch.save` dictionary loaded
with `weights_only=True`:

- `version`: 1; `key`: the filename stem.
- `weight`: CPU ND packed INT4 codes, stored as INT32 `[E,N,K/8]`.
- `scales`: positive finite FP32 `[E,N]` values.
- `left`, `right`: the actual runtime-oriented square BF16 FlatQuant matrices;
  their dimensions multiply to K. Any diagonal scaling is already folded in.
- `clip_ratio`: activation clipping ratio in `(0,1]`.
- `input_size`, `output_size`: K and N.

E is the local expert count, with contiguous EP ownership. Offline preparation
must use the same source checkpoint and projection ordering as the model.
The loader validates dimensions and layout.

The repository does not generate random matrices or convert checkpoint weights
to W4A4. Missing artifacts fail during loading; there is no conversion fallback.
Serving only transfers
prepared tensors, reconstructs NZ layout and packs FP32 scale bits losslessly.
Native `custom.flat_quant` still quantizes activations during inference and
packed INT4 grouped GEMMs execute both projections.

Each W4A4 projection selects a `gemm.grouped_w4a4` kernel plan during weight
preparation and logs the selected kernel. The `ascend_c` solution uses the
native Cube kernel for the two V4.1 projection shapes. The performance profile
requires `libdsv41_w4a4.so`; missing libraries fail during preparation instead of
silently selecting a slower implementation. The `cann` solution
(`npu_grouped_matmul`) remains explicitly selectable.
The native library is built by `tokenspeed-kernel-npu/csrc/dsv41_w4a4/build.sh`.
To pin the implementation, set the standard kernel override before loading:

```bash
export TOKENSPEED_KERNEL_OVERRIDE_GEMM_GROUPED_W4A4=ascend_c_grouped_w4a4
# Or select CANN explicitly:
export TOKENSPEED_KERNEL_OVERRIDE_GEMM_GROUPED_W4A4=cann_grouped_w4a4
```

The generic GMM planner uses CANN for shapes outside the native kernel's contract.
An explicit native selection rejects unsupported geometry. Selecting CANN skips
this planner's native-library check; V4.1 serving still checks its required
libraries during model preparation. The plan stays fixed through graph capture and replay; restart
the service after changing overrides or installed libraries. Both solutions
retain the same FlatQuant input and INT4 dequantization/rounding contract.

`--moe-backend ascend_bf16` decodes the original MXFP4 checkpoint to BF16 before
computation; it is not an original pre-quantization BF16 checkpoint. It rejects
FlatQuant settings instead of ignoring them. Both backends prepare device
weights after the text loader releases its parameter dictionary, allowing
replaced MXFP4 tensors to be freed layer by layer.

For NPU attention TP greater than eight, the model requires Dense TP8 and
attention DP1. Each rank projects its local attention heads to FP32 partials;
the output-group owner sums those partials and rounds once to BF16 before the
final projection. One rank cohort contributes the output/shared-expert
projections. MoE EP still spans the whole attention group. This is a
capacity-oriented layout with additional communication.

Native routing uses `npu_moe_init_routing_v2`, fixed token×top-k capacity and
device-resident count-format expert groups. It supports NPUGraph capture and
keeps route weights in FP32 before the down projection. Finalization uses unit
weights and FP32 accumulation, so the route weights are not applied twice.
Native V4.1 decode captures attention, device Engram lookup and MoE in one graph.
