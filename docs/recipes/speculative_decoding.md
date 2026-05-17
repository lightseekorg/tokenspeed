# Speculative Decoding Recipes

The commands below are templates. Validate exact model IDs, checkpoint formats,
and backend choices against the build you deploy.

## Llama 3.1 8B

```bash
tokenspeed serve nreHieW/Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
  --speculative-num-steps 7 \
  --host 0.0.0.0 \
  --dtype bfloat16 \
  --kvstore-size 16 \
  --port 8000
```

## GPT-OSS 20B / 120B

```bash
tokenspeed serve openai/gpt-oss-20b \
  --served-model-name gpt-oss-20b \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path Dogacel/specdrift-gpt-oss-20b-eagle3 \
  --speculative-num-steps 3 \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --chunked-prefill-size 8192 \
  --reasoning-parser base \
  --host 0.0.0.0 \
  --port 8000
```

```bash
tokenspeed serve openai/gpt-oss-120b \
  --served-model-name gpt-oss-120b \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path nvidia/gpt-oss-120b-Eagle3-long-context \
  --speculative-num-steps 3 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --kv-cache-dtype fp8 \
  --chunked-prefill-size 8192 \
  --max-num-seqs 4 \
  --reasoning-parser base \
  --host 0.0.0.0 \
  --port 8000
```

## Benchmarking

Against the GPT-OSS 120B server above:

```bash
tokenspeed bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --host 127.0.0.1 --port 8000 \
  --dataset-name mtbench \
  --input-len 1024 \
  --output-len 1024 \
  --num-prompts 80 \
  --max-concurrency 16 \
  --save-result --save-detailed --result-dir results/ \
  --extra-body '{"temperature": 0}'
```

Make sure you choose `openai-chat` over `openai`, otherwise the missing chat template causes acceptance
rates to go down abnormally. You can inspect the incoming requests & chat template via passing `--enable-log-requests --log-requests-level 2`
to the server.

The result block ends with something like:

```
Total token throughput (tok/s):          1781.94
Mean accept length (tok/step):           3.25
```

A value near `1.0` means almost no draft tokens are being accepted, check the draft model config.
