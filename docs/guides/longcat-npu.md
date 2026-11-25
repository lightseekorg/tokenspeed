# Deploying LongCat-Flash-Lite on Ascend NPU

This runbook walks through serving the LongCat-Flash-Lite checkpoint on
Ascend NPUs with TokenSpeed. It pairs with the model-level notes in
[Recipes: LongCat-Flash-Lite on Ascend NPU](../recipes/models.md#longcat-flash-lite-on-ascend-npu)
and the parameter reference in
[Server configuration](../configuration/server.md). The serve command below
is the validated production form; the recipe page explains why each flag is
there.

## Prerequisites

- Ascend NPU hardware (validated on 910B3 devices, one or more cards).
- CANN 9.0.0 installed at `/usr/local/Ascend/cann-9.0.0`.
- A Python 3.11 virtual environment with PyTorch 2.9.0 and `torch_npu`
  2.9.0.post2. Use matching PyTorch and `torch_npu` builds.
- A TokenSpeed checkout that also contains the `tokenspeed-kernel` and
  `tokenspeed-kernel-npu` source trees next to `python/`.
- The LongCat-Flash-Lite checkpoint on local disk. It is distributed
  out-of-tree (its `LongcatFlashNgramForCausalLM` architecture is registered
  by the in-tree `longcat_ngram` plugin), so there is no public Hub ID.

## Install the Ascend dependencies

From the repository root, inside the activated venv:

```bash
test/ci_system/install_triton_ascend.sh
```

The script installs Transformers 5.12.0, Triton 3.2.0, Triton-Ascend 3.2.1,
`apache-tvm-ffi==0.1.13`, and installs `tokenspeed-kernel-npu` from this
checkout in editable mode. It also compiles and executes a vector-add kernel
and a TokenSpeed KV-cache kernel on the visible NPU, so it doubles as an
installation smoke test.

## Configure the environment

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # one id per rank
export PYTHONPATH="${PWD}/python:${PWD}/tokenspeed-kernel/python:${PWD}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"
```

- `ASCEND_RT_VISIBLE_DEVICES` selects which physical NPUs TokenSpeed sees.
  Expose as many ids as `--world-size`.
- Set `TOKENSPEED_CANN_ROOT` when CANN is installed outside the default path.

## Check the environment

```bash
python -m tokenspeed.cli env
```

The output should list the visible NPU devices (for example
`NPU available: True` with one `NPU 0,1,2,3,4,5,6,7` line per device name) and
the dependency versions, and it should include the `ulimit soft` and
`ulimit -l (soft)` lines TokenSpeed relies on for serving. If the NPU section
is missing, re-check `ASCEND_RT_VISIBLE_DEVICES` and the CANN `set_env.sh`
source.

## Start the server

```bash
python -m tokenspeed.cli serve <longcat-checkpoint> \
  --served-model-name longcat-flash-lite \
  --device npu \
  --dtype bfloat16 \
  --disable-pdl \
  --disable-prefill-graph \
  --attention-backend mla \
  --sampling-backend greedy \
  --router-topk torch \
  --ep-dispatch-algorithm static_with_zero_expert \
  --expert-distribution-recorder-mode stat \
  --all2all-backend none \
  --mla-lora-scale folded \
  --trust-remote-code \
  --disable-kvstore \
  --tensor-parallel-size 8 \
  --world-size 8 \
  --ep-size 8 \
  --moe-tp-size 1 \
  --host 0.0.0.0 \
  --port 8080 \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --chunked-prefill-size 128 \
  --enable-metrics \
  --gpu-memory-utilization 0.88 \
  --cudagraph-capture-sizes 64 32 16 8 4 2 1
```

`<longcat-checkpoint>` is the local path to the LongCat-Flash-Lite checkpoint;
substitute the actual weights path. Keep `--disable-prefill-graph` (the N-gram
embedding reads request token history, so prefill must stay eager) and
`--disable-pdl` (a hard NPU requirement), and do not pass `--enforce-eager`
(it would also disable the decode ACL Graphs). The command captures decode
batches 64 through 1 as ACL Graphs; all seven captures are validated. See the
[recipe](../recipes/models.md#longcat-flash-lite-on-ascend-npu) for the
scaling rules.

A quick smoke test after startup:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "longcat-flash-lite",
    "messages": [{"role": "user", "content": "The capital of France is"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

## Troubleshooting

- **torchrun fails to resolve the hostname.** Distributed launches need the
  local hostname to resolve to a reachable address. Add the loopback mapping
  to `/etc/hosts`:

  ```bash
  echo "127.0.0.1 $(hostname)" | sudo tee -a /etc/hosts
  ```

- **Startup refuses with "NPU execution requires --disable-pdl".** `--device
  npu` requires `--disable-pdl`; the server validates this at startup and
  exits otherwise. Add the flag to the serve command.

- **The N-gram path errors or falls back inside a captured graph (ACL graph
  error 107030).** Prefill must stay eager for LongCat-Flash-Lite because the
  N-gram embedding reads request token history. Make sure `--disable-prefill-graph`
  is present and that you did not pass `--enforce-eager` (which would also
  kill the decode ACL Graphs).

- **Missing `.so` / kernel import errors.** A `.so` from
  `tokenspeed-kernel-npu` can be stale after a rebuild or missing after a
  checkout change. Re-run `test/ci_system/install_triton_ascend.sh` (it
  reinstalls the package in editable mode and recompiles the kernels), then
  re-source `set_env.sh` and restart the server.

- **No NPU in `python -m tokenspeed.cli env`.** Confirm CANN is sourced
  (`source /usr/local/Ascend/cann-9.0.0/set_env.sh`), `ASCEND_RT_VISIBLE_DEVICES`
  is set, and `torch_npu` is installed in the active venv.
