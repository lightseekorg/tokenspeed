# Weight Transfer (RL online weight sync)

TokenSpeed exposes an HTTP control plane for updating model weights in place
during online serving. It serves **two request dialects on the same port**, so
RL trainers that speak either one drive TokenSpeed unchanged:

- a **native lifecycle** — `init_weight_transfer_engine` → `start`/`update`/
  `finish` + `pause`/`resume` + `get_world_size` (e.g. SkyRL);
- a **SGLang-compatible** surface — `init_weights_update_group`,
  `update_weights_from_distributed` / `_from_tensor`, `pause_generation` /
  `continue_generation`, … (e.g. slime / miles, and verl's SGLang rollout).

Both dialects map to the same `AsyncLLM` weight methods. The heavy weight
payloads travel **out-of-band** (NCCL broadcast or CUDA-IPC); the HTTP path only
carries metadata (parameter names, shapes, dtypes, handles).

## Enabling

The control plane is **on by default** and served on the public control port
(the `ts serve` sidecar proxies it to the in-engine app):

```bash
tokenspeed serve <model> --host 0.0.0.0 --port 8000 \
    --weight-transfer-config '{"backend":"nccl"}'
```

`--weight-transfer-config` is a JSON object; `backend` is one of:

- `nccl` — **disaggregated**: trainer and inference run on separate GPUs.
- `ipc` — **colocated**: trainer and inference share GPUs (CUDA-IPC handles).

> ⚠️ **Security.** These endpoints can overwrite the served weights, reload a
> checkpoint from any on-disk path, dial an arbitrary NCCL master, and
> pause/abort serving. They are exposed on the control port by default. On an
> untrusted network, disable them with **`--no-enable-weight-transfer`** or put
> the control port behind auth / network controls.

## Endpoints

| Endpoint | Method | Request | Success response |
|---|---|---|---|
| `/init_weight_transfer_engine` | POST | `{"init_info": {…}}` | `{"message": "Weight transfer initialized"}` |
| `/start_weight_update` | POST | `{"is_checkpoint_format": bool}` (default `true`) | `{"message": "Weight update started"}` |
| `/update_weights` | POST | `{"update_info": {…}}` | `{"message": "Weights updated"}` |
| `/finish_weight_update` | POST | `{}` | `{"message": "Weight update finished"}` |
| `/pause` | POST | query `mode=abort\|wait\|keep`, `clear_cache=true` | `{"status": "paused"}` |
| `/resume` | POST | `{}` | `{"status": "resumed"}` |
| `/get_world_size` | GET | query `include_dp=true` | `{"world_size": <int>}` |
| `/is_paused` | GET | — | `{"is_paused": <bool>}` |

### Backend-specific payloads

**NCCL** (`backend="nccl"`)

- `init_info`: `{master_address, master_port, rank_offset, world_size}`
- `update_info`: `{names, dtype_names, shapes, packed=false,
  packed_buffer_size_bytes, packed_num_buffers}`

**IPC** (`backend="ipc"`)

- `init_info`: `{}` (no-op)
- `update_info`: `{names, dtype_names, shapes, ipc_handles_pickled,
  tensor_sizes, packed}`

Sending `ipc_handles_pickled` (base64 of a Python pickle) requires
`TOKENSPEED_ALLOW_INSECURE_SERIALIZATION=1`, because the payload is unpickled.

## Lifecycle

`get_world_size` is called once up front to size the NCCL group
(`trainer_ranks + inference_workers`). Then, per RL step:

```text
init_weight_transfer_engine          # once per run (nccl: build group; ipc: no-op)
  └─ loop each RL step:
       pause(mode=keep)               # stop scheduling new requests
       start_weight_update(is_checkpoint_format)
       update_weights(update_info)    # ×N chunks
       finish_weight_update()
       resume()
```

### Pause modes

- `abort` — abort in-flight requests and block new ones (default).
- `wait` — drain in-flight requests, then block new ones.
- `keep` — block new requests but preserve in-flight request state.

## SGLang-compatible endpoints

Served on the **same control plane / port** as the native dialect (same flag),
with endpoint names, methods, and JSON field names matching SGLang's HTTP
server. A trainer that drives an SGLang rollout (slime, miles, verl's SGLang
backend) drives TokenSpeed unchanged. Each maps to the same `AsyncLLM` weight
methods as the native dialect.

| Endpoint | Method | Request |
|---|---|---|
| `/init_weights_update_group` | POST | `{master_address, master_port, rank_offset, world_size, group_name, backend}` |
| `/update_weights_from_distributed` | POST | `{names, dtypes, shapes, group_name, flush_cache}` (NCCL) |
| `/update_weights_from_tensor` | POST | `{serialized_named_tensors, load_format, flush_cache}` (serialized / CUDA-IPC) |
| `/update_weights_from_disk` | POST | `{model_path, load_format}` |
| `/pause_generation`, `/continue_generation` | POST | block / unblock new admission |
| `/flush_cache` | GET | flush KV/prefix cache |
| `/release_memory_occupation`, `/resume_memory_occupation` | POST | offload / restore (colocated) |
| `/abort_request` | POST | `{rid}` or `{abort_all: true}` |
| `/health_generate` | GET | health check |
| `/destroy_weights_update_group` | POST | no-op (group torn down with the engine) |

Note the field-name difference from the native dialect: SGLang uses `dtypes`
(the shim translates to the internal `dtype_names`).

## Status & limitations

The HTTP control plane, lifecycle state machine, request/response parity,
`get_world_size`, and pause/resume admission gating are implemented and tested.

The following require GPU hardware and land separately:

- **Worker-side weight load.** The NCCL broadcast receive + `load_weights` and
  the CUDA-IPC handle rebuild run on the scheduler/model-runner workers. Until
  wired, `nccl` `update_weights` drives the existing distributed receive path
  and `ipc` `update_weights` validates the payload then errors (the worker IPC
  path is not implemented yet).
- **`keep`-mode KV preservation.** New-request admission is gated in the
  frontend today; freezing in-flight KV/request state in the C++ scheduler is a
  follow-up. `abort` and `wait` are fully supported.

See `examples/rl/` for trainer-side client snippets.
