# KDA attention

Batched replay pointer descriptors carry an explicit `state_dtype`. NVIDIA
replay supports FP32 and BF16 storage with FP32 accumulation; Gluon replay
requires FP32 storage. Native NVIDIA prefill scans receive FP32 state, so
adapters widen BF16 checkpoints before calling their fixed ABI. State
publication is the rounding boundary, not each recurrent arithmetic step.

On sm100/sm103 the batched replay recurrence is a persistent CuTe DSL kernel
(`_cute_dsl/replay.py`): each CTA prefetches the next (layer, request, head)
item's state tile and window inputs with bulk copies while it replays the
current one, and contracts over K in registers, so no barrier runs per token.
The gate precompute and the convolution-window publication stay the shared
Triton launches; other NVIDIA targets run the Triton recurrence.
