# Launching a Server

`tokenspeed serve` starts an OpenAI-compatible HTTP server. Put the model path
directly after the command.

## Minimal Launch

```bash
tokenspeed serve openai/gpt-oss-20b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1
```

## Production Launch Skeleton

Use explicit parameters in scripts so a deployment is reproducible.

```bash
tokenspeed serve nvidia/Kimi-K2.5-NVFP4 \
  --served-model-name kimi-k2.5 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --quantization nvfp4 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --chunked-prefill-size 8192 \
  --max-num-seqs 256 \
  --attention-backend trtllm_mla \
  --moe-backend flashinfer_trtllm \
  --reasoning-parser kimi_k25 \
  --tool-call-parser kimik2
```

## Launch Checklist

- Put the model path directly after `tokenspeed serve`.
- Set `--host`, `--port`, and `--served-model-name` for the API surface.
- Set `--max-model-len`, `--kv-cache-dtype`, and `--gpu-memory-utilization` before tuning concurrency.
- Set `--tensor-parallel-size`, `--data-parallel-size`, and `--enable-expert-parallel` to match the hardware topology.
- Set model-family parsers such as `--reasoning-parser` and `--tool-call-parser` when the chat format needs them.
- Set backend choices explicitly for benchmark or production runs.

## OpenAI-Compatible Client

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[{"role": "user", "content": "Write a concise deployment checklist."}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

## Next

- [Model Recipes](../recipes/models.md)
- [Server Parameters](../configuration/server.md)
- [Parallelism](../serving/parallelism.md)
