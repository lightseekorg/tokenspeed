# Activation kernels

`capped_logit_sum` adds BF16 base and correction logits, rounds the sum to
BF16, applies the FP32 softcap, and stores BF16 output. Columns outside the
valid vocabulary are negative infinity. The recurrence kernels also preserve
the BF16 rounding boundaries of their PyTorch statements.

Softcap kernels disable FP contraction on both GPU backends. NVIDIA retains
the round-to-nearest PTX multiply and CUDA non-FTZ compiler option. AMD uses
portable FP32 multiplication without CUDA compiler options. Shared tests
compare the recurrence and softcap with Torch and replay changed inputs
through a captured graph.
Model tests compare teacher-forced draft logits on the supported projection
or fallback path.

`moe_finalize_gate_add` gathers each token's top-k expert rows, sums them in
FP32 in k order, rounds once to BF16 and adds the sigmoid-gated shared-expert
output. Below 512 tokens every row is split into 1024-column programs, so a
decode-size call has enough CTAs. From 512 tokens on, one program covers a
row: the gate input is loaded and the gate computed once per token, not once
per column program. Each column's arithmetic is the same in both layouts, so
the output is bitwise identical across the switch.
