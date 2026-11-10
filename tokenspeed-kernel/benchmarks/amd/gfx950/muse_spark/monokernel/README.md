# Muse Spark decode monokernel launch benches

Standalone multi-rank benches that time one persistent monokernel launch
(gfx950, TP8) from its first CTA's start to its last CTA's end. Unlike the
JSON suites next to this directory, they do not go through the generic
benchmark harness: a launch needs the Iris symmetric heap, the mailbox
scratch and every rank's process.

- `layer_bench.py`: one target decoder layer (sliding or full attention,
  1 to 16 rows). With `--gpus` 1, 2 or 4 it stands in for the absent ranks;
  the numbers are for iteration, so confirm a kept change on 8 GPUs.

The inputs are `tokenspeed_kernel.benchmark.muse_spark_monokernel`'s (and its `_gluon` module's gfx950 layouts), the
same the kernel tests in `tokenspeed-kernel/test/amd/gfx950/ops/monokernel`
check. Each script's docstring shows how to run it.
