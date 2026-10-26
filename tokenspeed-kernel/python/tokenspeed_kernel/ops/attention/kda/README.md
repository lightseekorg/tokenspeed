# KDA attention

Batched replay pointer descriptors carry an explicit `state_dtype`. NVIDIA
replay supports FP32 and BF16 storage with FP32 accumulation; Gluon replay
requires FP32 storage. Native NVIDIA prefill scans receive FP32 state, so
adapters widen BF16 checkpoints before calling their fixed ABI. State
publication is the rounding boundary, not each recurrent arithmetic step.
