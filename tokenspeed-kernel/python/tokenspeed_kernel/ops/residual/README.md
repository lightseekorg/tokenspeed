# Residual-stream operations

## Gated residual mix

`gated_residual_mix` consumes normalized residual branches. Its projection
weight contains the low-rank down rows followed by optional inject rows; the
up projection produces gates for the branch reduction. Normalization and
the later residual combine are separate operations.

Model loading calls `pad_gated_residual_projection_weight` to round BF16/FP16
projection storage up to eight rows. The runtime keeps the original logical
row count and supplies it explicitly as `projection_rows` to
`gated_residual_mix`. Checkpoint mix/inject shards retain their original shapes;
the first shard extends the model parameter once, and later shards and reloads
write into that same storage. The model-loading postprocessing hook also
prepares dummy and sharded weights. FP32 retains its original width.

Forward uses the stored weight directly, with no weight allocation, copy, or
padding in eager calls or CUDA graph replay. The GEMM/Triton implementation
computes all stored columns and slices the result to its logical width before
the epilogue, preserving the optional-inject contract. Reloading weights in
place remains visible to captured graphs.

This padding aligns the GEMM output rather than repairing the weight's row
stride. For `X[T, 10240] @ W[N, 10240].T`, both implementations already have a
20480-byte BF16/FP16 weight row stride. The GEMM output has a 648-byte row stride
at `N=324`, versus 656 bytes at `N=328`, which is divisible by 16 and permits a
faster cuBLAS implementation.

The CuTe implementation uses native Blackwell MMA and TMA weight loads. Its
64- or 128-row down tiles already cover 384 projection rows for the 324-row
down/inject weight, with TMA supplying zeros outside the weight extent. The
contiguous weight dimension is 10240 elements, and the projection result stays
inside the fused pipeline rather than forming a 324-column global tensor.
Consequently, the GEMM output alignment padding above does not apply to this
implementation; it takes a logical view of the same stored weight so TMA
supplies zeros for the tail instead of fetching padding rows. Decode and
prefill share this weight contract and retain the existing dispatch rules.
