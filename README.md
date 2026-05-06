# TokenSpeed

TokenSpeed is a speed-of-light LLM inference engine.

- TensorRT LLM level performance
- vLLM level usability
- Built by a lean and mission-driven team in two months

<img src="https://lightseek.org/blog/images/202605/tokenspeed-kimi-k2.5-performance.png" alt="perf" width="800" margin="10px"></img>

## Preview Status

This version is a preview release for reproducing the Kimi K2.5 on B200 and
TokenSpeed MLA on B200 results from the [TokenSpeed blog](https://lightseek.org/blog/lightseek-tokenspeed.html). Several major PRs are
still in progress and have not been merged yet.

Ongoing work includes:

- Model coverage: Qwen 3.6, DeepSeek V4, and MiniMax M2.7.
- Runtime features: PD, EPLB, KV store, Mamba cache, VLM, and metrics.
- Platform optimization: Hopper optimization, MI350 optimization, and related
  runtime improvements.

These features are still being cleaned up and will be merged into `main` over
the next few weeks. TokenSpeed is currently under heavy development and is
intended to showcase the new runtime design and technical direction. Do not use
this preview release for production deployments.

## Documentation

Start here:

- [Docs Index](https://lightseek.org/tokenspeed/)
- [Getting Started](https://lightseek.org/tokenspeed/guides/getting-started)
- [Launching a Server](https://lightseek.org/tokenspeed/guides/launching)
- [Model Recipes](https://lightseek.org/tokenspeed/recipes/models)
- [Server Parameters](https://lightseek.org/tokenspeed/configuration/server)
- [Compatible Parameters](https://lightseek.org/tokenspeed/configuration/compatible-parameters)
- [Parallelism](https://lightseek.org/tokenspeed/serving/parallelism)
