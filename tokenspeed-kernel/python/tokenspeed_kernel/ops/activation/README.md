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
