# Reloading the forward pass (development)

Iterating on a kernel or a model's forward usually costs a server restart per
edit: process start, weight loading, kernel warmup and CUDA graph capture. With
`--allow-reload-forward`, a running server instead re-executes the edited
Python modules in place, warms their kernels and recaptures its CUDA graphs,
keeping the weights, the KV cache, the communicators and the autotuned
tactics.

The flag only permits the reload; nothing happens until you call
`ts reload-forward` (or the `reload_forward` control call).

::: danger Development only
Never enable it in production. Without the flag the engine refuses the
`reload_forward` call on arrival, before it drains anything.
:::

## The loop

```bash
ts serve <model> --allow-reload-forward ...   # 1. start once; the gateway serves /v1/rl
ts bench ... / nsys ...                        # 2. benchmark and profile
$EDITOR tokenspeed-kernel/python/tokenspeed_kernel/ops/...   # 3. edit
ts reload-forward --url http://127.0.0.1:8000  # 4. reload, warm, recapture
                                               # 5. benchmark again, repeat
```

`ts reload-forward` prints, per worker, what it reloaded and how long each
step took, for example:

```text
reloaded tokenspeed.runtime.models.qwen3 and 0 dependent module(s), 31 object(s) patched in 0.05 s;
warmed in 1.2 s; recaptured in 7.9 s; JIT-compiled 3 kernel(s) in 0.4 s
```

The same operation is the `reload_forward` control call on the gateway's RL
routes, which `--allow-reload-forward` turns on; `curl` skips the CLI's
Python start-up:

```bash
curl -X POST "http://127.0.0.1:8000/v1/rl/engine/reload_forward?selector=engine=tokenspeed" \
  -H 'Content-Type: application/json' -d "{\"mode\": \"wait\", \"id\": \"$(uuidgen)\", \"timeout_s\": 600}"
```

The body takes `id`, required and the same on every engine (`ts reload-forward`
sends a fresh one), `mode` (`wait`, the default, or `abort`), `modules`, a list
of modules to re-execute instead of every changed one, and `timeout_s`, after
which a reload that has not drained or reached every engine gives up. Under attention data
parallelism the reload runs once every engine holds the same one; until then,
`continue_generation` cancels it.

Recapturing the graphs takes most of a reload. A development server that
captures only the batch sizes and prefill buckets under study reloads faster,
for example `--cudagraph-capture-sizes 1 16 64 --prefill-graph-max-tokens 2048`.

## What a reload does

1. **Drain.** New requests are held (not rejected) and running ones finish
   (`mode: wait`, the default) or are aborted (`mode: abort`). With attention
   data parallelism every DP rank drains, and the reload runs once all of them
   hold it, as weight updates do; send it to every DP engine, as the CLI and
   the gateway's fan-out route do. It keeps its place among weight updates
   from the moment it arrives, and until it runs `continue_generation`
   cancels it.
2. **Flush** the prefix caches: KV computed by the old code is not reused.
3. **Reload.** Every module under the forward packages whose source changed
   since it last ran is re-executed, together with the modules that import or
   use it, providers before their users. The first rank reads the sources and
   sends the same bytes to every rank.
   Functions, classes, cached functions and Triton kernels are updated *in
   place*, so imports, kernel plans, registry entries and live layers run the
   new code without being rebuilt; module-level caches (compiled CuTe DSL
   kernels, `functools` caches) start empty.
4. **Warm** the kernels with the same dummy prefill startup runs, from the
   dummy inputs startup had, while the old graphs still serve as a fallback. Startup kernel choices apply again for
   the warmup and the capture, as they do at startup; the tuned tactics,
   constructed modules, processed weights and objects' own caches are kept.
5. **Recapture** the decode and prefill graphs on the existing KV pools.
6. **Resume**: held requests run on the new code. A reload that follows a
   `pause_generation` leaves the scheduler paused.

If the reloaded code fails to compile, import, warm or capture on every rank,
each rank undoes the in-place changes it made and the server keeps serving the
old code; the reply carries the error. Effects a failed module's top-level code
had on other objects are not undone. Only a failure to recapture with the old
code is fatal, and a failure on some ranks inside a collective (an edit that
raises on one tensor-parallel rank only) hangs the engine until the NCCL
watchdog stops it.

## What is reloaded

Modules under `tokenspeed.runtime.models`, `tokenspeed.runtime.layers`,
`tokenspeed.runtime.moe`, `tokenspeed.runtime.sampling`,
`tokenspeed_kernel.ops` and `tokenspeed_kernel.thirdparty` (Python only).
An edited module elsewhere under `tokenspeed.runtime` or `tokenspeed_kernel`
is reported as needing a restart and left as it runs; naming it in
`modules` forces it.

Kernels written in Triton, Gluon or CuTe DSL reload; these need a restart:

- CUDA C++ kernels (the prebuilt `thirdparty/cuda` libraries) and vendor
  libraries such as FlashInfer, TRT-LLM and DeepGEMM;
- changes to a module's `__init__`, parameters, buffers or weight layout
  (the weights were loaded and processed once);
- a different attention or MoE backend, which sets the KV cache or weight
  layout;
- the vision encoder's graphs of a multimodal model.

A function whose closure changed shape is redirected to its new definition, a
method an edit deleted is removed, and a changed base class is set on the live
class. Definitions that can be neither patched nor redirected, such as a
function that became a different kind of object, are listed in the reply as
stale where referenced before the reload. A kernel compiled once and stored on
a long-lived object (not in a module cache) keeps its old binary.

## Profiling across reloads

Start the server under Nsight Systems once and open a capture range per
profile; each range becomes its own report:

```bash
nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat -o run_%n \
  ts serve <model> --allow-reload-forward ...
curl -X POST http://127.0.0.1:8000/start_profile \
  -d '{"activities": ["CUDA_PROFILER"], "num_steps": 20}'
```

## Limits

`--allow-reload-forward` cannot be combined with `--enable-memory-saver`,
whose tagged graph pool cannot be discarded and captured again, nor with
prefill/decode disaggregation, whose roles would reload and resume
independently. The KV cache keeps the size startup planned and the workspace
stays frozen while the old graphs live, so code that needs more graph memory or
workspace than startup did fails its warmup or recapture (and is reverted).
While a reload runs, the engine answers no other control call, including a
stop. Benchmark a final change on a freshly started server: a reload keeps
state a fresh start rebuilds.
