# Op and layer profiling

The initial profiling path records eager op/layer metadata and evaluates explicit
cost rules after collection. It does not change kernel selection or execution.
GPU activity association and utilization analysis are separate follow-up work.

## Collection

`ProfileSession` owns a window identified by run, model, rank and iteration.
Bind layer instances to their global layer IDs before recording.
`@profiled(level="layer")` tracks repeated calls and restores parent context on
exit, including exceptions. `inner_id` increases within an iteration/layer;
`occurrence` identifies each layer call.

`register_kernel(..., profile=OpProfile(cost_rule="..."))` opts a synchronous
function into input/output tensor metadata collection. Omitting `profile` leaves
the registered function unchanged; `cost_rule=None` records metadata only.
Wrappers preserve arguments, returns and exceptions. Tensor descriptions retain
shape, dtype, stride, device and logical size, not tensor storage. Nested
registered calls record only the outermost op; its rule must cover the entire op.

Both wrappers use `record_func`. The required `backend` selects `none` (metadata
only), `nvtx` or `torch`; the caller owns the external profiler lifetime.
Labels carry readable layer/op/rank/call identity and an encoded unique key
shared with the metadata export. CPU scope timing includes collection overhead
and cannot be used as GPU latency.

Collection supports synchronous eager windows on one execution thread. Nested
sessions, repeated iteration IDs and profiled stateful kernel classes are rejected.
CUDA graph replay and cross-thread attribution are not implemented. Output
metadata describes returned tensors; in-place-only outputs need a separate adapter.

## Cost analysis

`CostRegistry` applies named rules after collection. `DenseMatmulCost` models
plain dense 2D GEMM with explicit input/output tensor roles. Costs distinguish
useful and executed FLOPs, logical reads/writes and rule assumptions.
Failed calls, missing rules and unsupported shapes retain a missing-cost reason.
Layer summaries aggregate recorded ops per iteration/layer across occurrences,
report coverage, and expose a complete cost only when every recorded op is covered.
This coverage describes opted-in calls, not all work in a model.

Logical bytes do not measure HBM traffic. MFU and Estimated MBU remain unavailable
without correlated GPU timing and matching hardware peaks. No GPU synchronization
or device timing is added by these wrappers.

## CPU example and validation

Run from the repository root in the project's Python environment:

```bash
PYTHONPATH=tokenspeed-kernel/python python tokenspeed-kernel/examples/profile_layers.py \
  --backend torch --output /tmp/op-layer/records.json

PYTHONPATH=tokenspeed-kernel/python python -m pytest \
  tokenspeed-kernel/test/test_op_profile.py -q
```

The example executes a two-GEMM layer twice and checks its result: four ops,
4096 FLOPs, 2816 logical read bytes and 768 logical write bytes. `--backend none`
writes metadata and costs only; `torch` also writes `records.trace.json`.
The illustrative layer ID is bound explicitly. Tests exercise real CPU ops and
torch profiler, covering repeated/nested calls, cost coverage, argument forwarding
and exception recovery.
