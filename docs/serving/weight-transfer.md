# Weight Transfer (RL online weight sync)

TokenSpeed exposes an HTTP control plane for updating model weights in place
during online serving. The endpoint paths, methods, and request/response JSON
follow the weight-transfer HTTP contract that common RL training frameworks
(verl / slime / AReaL / miles) already speak, so existing trainer code drives
TokenSpeed **unchanged**.

The heavy weight payloads travel **out-of-band** (NCCL broadcast or CUDA-IPC);
the HTTP path only carries metadata (parameter names, shapes, dtypes, handles).

## Enabling

The control plane is **off by default**. Enable it with the server flag (or the
`TOKENSPEED_SERVER_DEV_MODE=1` env):

```bash
tokenspeed serve <model> --host 0.0.0.0 --port 8000 \
    --enable-weight-transfer \
    --weight-transfer-config '{"backend":"nccl"}'
```

`--weight-transfer-config` is a JSON object; `backend` is one of:

- `nccl` — **disaggregated**: trainer and inference run on separate GPUs.
- `ipc` — **colocated**: trainer and inference share GPUs (CUDA-IPC handles).

When enabled, the endpoints below are served on the public port (the `ts serve`
sidecar proxies them to the in-engine control plane).

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

## Status & limitations

The HTTP control plane, lifecycle state machine, request/response parity,
`get_world_size`, and pause/resume admission gating are implemented and tested.

The following require GPU hardware and land separately:

- **Worker-side weight load.** The NCCL broadcast receive + `load_weights` and
  the CUDA-IPC handle rebuild run on the scheduler/model-runner workers. Until
  wired, `nccl` `update_weights` drives the existing distributed receive path
  and `ipc` `update_weights` returns `501 Not Implemented` after validating the
  payload.
- **`keep`-mode KV preservation.** New-request admission is gated in the
  frontend today; freezing in-flight KV/request state in the C++ scheduler is a
  follow-up. `abort` and `wait` are fully supported.

See `examples/rl/` for trainer-side client snippets.
