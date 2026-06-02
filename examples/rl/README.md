# RL weight sync examples

Trainer-side client snippets for TokenSpeed's vLLM-compatible weight-transfer
API. See [docs/serving/weight-transfer.md](../../docs/serving/weight-transfer.md)
for the full endpoint reference and lifecycle.

Because the HTTP surface byte-matches vLLM's
`vllm/entrypoints/serve/rlhf/api_router.py`, these same clients drive a vLLM
server unchanged — only the `--url` differs. That is the drop-in compatibility
goal: a trainer written against vLLM (verl / slime / AReaL / miles) targets
TokenSpeed without a code change.

| File | Backend | Topology |
|---|---|---|
| `weight_sync_nccl.py` | `nccl` | Disaggregated: trainer and inference on separate GPUs |
| `weight_sync_ipc.py` | `ipc` | Colocated: trainer and inference share GPUs |

## Run

Start a server with the control plane enabled:

```bash
# NCCL (disaggregated)
tokenspeed serve <model> --host 0.0.0.0 --port 8000 \
    --enable-weight-transfer --weight-transfer-config '{"backend":"nccl"}'

# IPC (colocated)
TOKENSPEED_ALLOW_INSECURE_SERIALIZATION=1 \
tokenspeed serve <model> --host 0.0.0.0 --port 8000 \
    --enable-weight-transfer --weight-transfer-config '{"backend":"ipc"}'
```

Then drive it:

```bash
python examples/rl/weight_sync_nccl.py --url http://127.0.0.1:8000
python examples/rl/weight_sync_ipc.py  --url http://127.0.0.1:8000
```

The `broadcast_weights` / `export_ipc_handles` functions are pseudocode marking
where the trainer moves the actual tensors (out-of-band over NCCL / CUDA-IPC);
wire them to your training framework.
