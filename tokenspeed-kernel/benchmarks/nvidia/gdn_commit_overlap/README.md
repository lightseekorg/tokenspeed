# MTP graph overlap and SM placement probes

`graph_split.py` instruments the ordinary chain-MTP drafter for a benchmark.
It records an external CUDA event at the start of drafting, clones the captured
serving graph, and partitions its actual nodes into a target prefix and a draft
suffix. The original Torch graph retains its allocation pool for the clones'
lifetime. Branches unordered with the marker stay in the prefix. The suffix
is the complete descendant closure; backward dependencies are rejected and
every original graph node executes exactly once.

Call `install(scheduling="serial")` or `install(scheduling="overlap")`
explicitly after `ModelExecutor` finishes importing and before construction.
The overlap setting launches the draft on its own stream after the target
completion event. The forward stream queues GDN commit, then joins the draft
before other attention-state commits and runtime-frontier updates. Serial
queues the same nodes in order. Eager and prefill calls use ordinary execution.
These hooks serve benchmark comparisons and do not change serving defaults.

The external marker must use `torch.cuda.Event(external=True)` so graph capture
retains a record node. Installing model hooks from `sitecustomize` before the
native scheduler starts can register native Python types twice; install after
the executor module loads instead.

## Placement controls

`placement.py` adds an explicitly selected stream owner around the existing
GDN commit arithmetic:

| Name | CTA bound | Available SMs | Co-scheduled SM group |
| --- | ---: | ---: | ---: |
| unrestricted24 | 24 | whole device | driver selected |
| unrestricted16 | 16 | whole device | driver selected |
| green16 | 16 | 16 | 16 |
| green32 | 24 | 32 | 16 |
| unrestricted32 | 32 | whole device | driver selected |
| green32_cta32 | 32 | 32 | 16 |

The CUDA resource split sets both `smCount` and `coscheduledSmCount`. An SM-count
limit alone does not establish GPC locality; use the driver resource query and
live SM-ID probes to verify the scope. Selected SM IDs need not be consecutive.
Green-context streams share primary-context allocations. The owner records an
input-ready event on the caller stream, launches commit on the selected stream,
and queues an output-ready wait back on the caller. Source, destination, and
temporary-table storage retain their ordering.

CUDA graph nodes retain the resources selected during capture. Capture the
placement scope itself when comparing graphs; selecting another replay stream
does not change node resources. `placement_runtime.py` changes the post-graph
GDN launch and leaves the target/draft graph contexts alone. A shared-memory
byte selects a prepared placement between complete request waves.

## Validation and interpretation

The fixed-work probe executes actual mHC kernels followed by an explicit device
delay for other work. It validates accepted GDN states and mHC outputs before
timing; the delay does not represent a complete draft model.

```bash
PYTHONPATH=tokenspeed-kernel/python python \
  tokenspeed-kernel/benchmarks/nvidia/gdn_commit_overlap/bench_placement.py \
  --placements unrestricted24 unrestricted16 green16 green32 \
  --batches 8 32 --acceptance full4 mixed --layers 36 \
  --delay-ns 120000 --blocks 7 --iterations 30 --gpu 0 \
  --output /tmp/gdn-placement.json --no-profile --profile-iterations 24

PYTHONPATH=tokenspeed-kernel/python:tokenspeed-kernel/test pytest -q \
  tokenspeed-kernel/test/nvidia/test_gdn_commit_graph_split.py \
  tokenspeed-kernel/test/nvidia/test_gdn_commit_placement.py
```

Tests replay changing graph inputs, preserve independent copy branches, check
PDL state writes and joins, and verify live SM-ID bounds. State comparisons
retain the repository's dtype tolerances and require untouched rows to remain
bitwise unchanged. Green-context tests require CUDA driver API 13.2 and a
supported co-scheduled resource group.

For a full model experiment, record the actual per-rank GDN layer/head geometry,
acceptance lengths, and successful request/token counts. Profile stable
full-batch decode separately from prefill and excluded warmup. With node-level
Nsight tracing, attribute kernels to target/draft launches and measure the union
of draft intersections with commit, separately on every GPU. Generic concurrent
kernels may include PDL consumers waiting on commit and do not establish useful
draft overlap. Whole-graph spans include gaps. Keep unprofiled HTTP latency and
throughput separate from profiled diagnostics; differing acceptance lengths and
completion tails prevent a fixed-work causal speedup claim.
