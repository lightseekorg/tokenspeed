---
layout: home

hero:
  name: TokenSpeed
  text: Speed-of-light LLM inference
  tagline: Production-oriented docs for launching, tuning, and operating low-latency OpenAI-compatible serving.
  actions:
    - theme: brand
      text: Get started
      link: /guides/getting-started
    - theme: alt
      text: Launch recipes
      link: /recipes/models
    - theme: alt
      text: Server parameters
      link: /configuration/server

features:
  - title: Launch first
    details: Start with concrete commands, then tune the exact knobs that affect memory, scheduling, parallelism, and kernels.
  - title: Familiar parameters
    details: TokenSpeed keeps familiar parameter names where the runtime semantics match, with TokenSpeed-specific knobs documented separately.
  - title: Model recipes
    details: Recipes collect the launch patterns used for Kimi and GPT-OSS deployments.
  - title: Operational surface
    details: Parallelism and configuration guidance stay close to the serving paths operators actually use.
---

## Start here

- [Getting started](./guides/getting-started.md)
- [Launching a server](./guides/launching.md)
- [Model recipes](./recipes/models.md)
- [Server parameters](./configuration/server.md)
- [Compatible parameters](./configuration/compatible-parameters.md)
- [Parallelism](./serving/parallelism.md)

## Common workflow

1. Install the runtime and kernel packages.
2. Pick a launch recipe close to your model family and hardware.
3. Set model loading, memory, scheduler, and parallelism parameters explicitly.
4. Validate correctness and throughput together before changing more than one tuning dimension.

## Minimal server

```bash
tokenspeed serve openai/gpt-oss-20b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1
```

The server exposes an OpenAI-compatible API under `/v1`.

## High-performance shape

Large MoE deployments usually make the same decisions:

- model path and revision
- context length and KV cache dtype
- scheduler token and sequence budgets
- attention and MoE backends
- tensor, data, and expert parallelism
- reasoning, tool-call, and speculative decoding parsers

See [Model recipes](./recipes/models.md) for concrete examples and
[Server parameters](./configuration/server.md) for the parameter reference.

See [Biweekly releases](./guides/weekly-release.md) for the release sequence and recovery.
