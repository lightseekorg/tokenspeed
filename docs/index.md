---
layout: home

hero:
  name: TokenSpeed
  text: Speed-of-light LLM inference
  tagline: Production-oriented docs for launching, tuning, and operating low-latency OpenAI-compatible serving.
  actions:
    - theme: brand
      text: Get Started
      link: /guides/getting-started
    - theme: alt
      text: Launch Recipes
      link: /recipes/models
    - theme: alt
      text: Server Parameters
      link: /configuration/server

features:
  - title: Launch First
    details: Start with concrete commands, then tune the exact knobs that affect memory, scheduling, parallelism, and kernels.
  - title: Familiar Parameters
    details: TokenSpeed keeps familiar parameter names where the runtime semantics match, with TokenSpeed-specific knobs documented separately.
  - title: Model Recipes
    details: Recipes collect the launch patterns used for Kimi and GPT-OSS deployments.
  - title: Operational Surface
    details: Parallelism and configuration guidance stay close to the serving paths operators actually use.
---

## Start Here

- [Getting Started](./guides/getting-started.md)
- [Launching a Server](./guides/launching.md)
- [Model Recipes](./recipes/models.md)
- [Server Parameters](./configuration/server.md)
- [Compatible Parameters](./configuration/compatible-parameters.md)
- [Scheduler](./serving/scheduler.md)
- [Parallelism](./serving/parallelism.md)

## Common Workflow

1. Install the runtime and kernel packages.
2. Pick a launch recipe close to your model family and hardware.
3. Set model loading, memory, scheduler, and parallelism parameters explicitly.
4. Validate correctness and throughput together before changing more than one tuning dimension.

## Minimal Server

```bash
tokenspeed serve openai/gpt-oss-20b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1
```

The server exposes an OpenAI-compatible API under `/v1`.

## High-Performance Shape

Large MoE deployments usually make the same decisions:

- model path and revision
- context length and KV cache dtype
- scheduler token and sequence budgets
- attention and MoE backends
- tensor, data, and expert parallelism
- reasoning, tool-call, and speculative decoding parsers

See [Model Recipes](./recipes/models.md) for concrete examples and
[Server Parameters](./configuration/server.md) for the parameter reference.
