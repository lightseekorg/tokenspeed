# Kimi-K3 model ops

`tokenspeed_kernel.ops.model.kimi_k3` holds ops specific to Kimi-K3 and
Kimi-K3.1. They are registered in the `kimi_k3` family.

## MoE layer tail

The MoE layer tail runs a decoder layer's work after its attention core, from
the o-projection to the layer output, as one persistent launch per rank. It
serves the rows of one request's decode or speculative verify step, with one
tensor-parallel group for attention and MoE. The model uses it in four steps:

1. While it builds its layers, it asks which routed-expert bank layout the tail
   reads (`kimi_k3_moe_tail_expert_layout`). The answer is `None` when no
   registration serves the model's shapes on this platform; the model then
   keeps the regular path. Otherwise the MoE layers store their banks in that
   layout (`moe_plan(expert_layout=...)`), and every other forward's experts
   read the same banks through the regular MoE kernels.
2. Before graph capture, it builds this rank's buffers (`prepare_kimi_k3_moe_tail`).
   This step is collective over the group's CPU process group.
3. It packs each MoE layer's weights (`pack_kimi_k3_moe_tail_layer`).
4. Every decode step, it runs each layer's tail (`kimi_k3_moe_tail`).

The ranks exchange values inside the launch, so every rank must run the same
layers' tails in the same order: the model decides from the forward mode and the
row count alone. A layer's first launch compiles on every rank, and the ranks
meet before any of them launches.

The ops are `moe_tail_prepare`, `moe_tail_pack` and `moe_tail`. To keep the
regular path without changing other settings, set
`TOKENSPEED_KERNEL_OVERRIDE_KIMI_K3_MOE_TAIL=none`: the layout query then
returns `None`.

The contract that every registration keeps, and the gfx950 kernel's sizes,
numerics and failure behaviour, are in `tokenspeed_kernel_amd/ops/README.md`.
