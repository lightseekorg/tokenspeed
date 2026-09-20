# Runtime tests

## DeepSeek V4.1 references

The rejected-suffix compressor reference keeps the full query width through
CPU softmax before selecting completed pairs. Compacting those rows first can
change rounding on CPUs with many threads. Expected positions and pooled values
are derived independently from the accepted prefix and replacement tokens;
both comparisons require exact equality.

The distributed attention test loads complete reference weights through each
parameter's loader. The loader handles replicated weights, fused projections,
TP slices and attention sinks. A full fused weight does not use the separate
shard arguments accepted by replicated projections.

Run the four-rank test on four allocated NVIDIA GPUs:

```bash
python3 -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m pytest -v test/runtime/test_deepseek_v41_model.py::test_distributed_attention_tp4
```

Running that test in an ordinary one-process pytest invocation skips its TP4
coverage. A successful distributed run must pass on every rank, including the
packed-weight lifecycle, model forward, and decode graph comparisons.
