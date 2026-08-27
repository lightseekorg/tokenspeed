<p align="center">
  <img src="https://raw.githubusercontent.com/lightseekorg/tokenspeed/main/assets/banner/tokenspeed-banner.png" alt="TokenSpeed: Tokens at the speed of light" width="100%" />
</p>

TokenSpeed is a speed-of-light LLM inference engine designed for **agentic workloads**, with TensorRT-LLM-level performance and vLLM-level usability. Our goal is to be the most performant inference engine for production agentic workloads.

Core components:

- **Modeling layer**: local-SPMD design with a static compiler that generates
  collective communication from module-boundary placement annotations, so users
  do not hand-write parallelism logic.
- **Scheduler**: C++ control plane and Python execution plane. Request
  lifecycle, KV cache ownership, and overlap timing are encoded as a
  finite-state machine, with safe KV resource reuse enforced by the type system at compile time.
- **Kernels**: pluggable, layered kernel system with a portable public API and
  a centralized registry including one of the fastest **MLA**
  (Multi-head Latent Attention) implementations on Blackwell for agentic workload.
- **Entrypoint**: SMG-integrated AsyncLLM for low-overhead CPU-side request
  handling.

## News

- [2026/08] [Qwen3.8 Flash Next](https://developer.nvidia.com/blog/experiment-with-qwen3-8-flash-next-176b-model-on-nvidia-gb300-nvl72-for-agentic-coding/) at Day 0 and [GLM 5.3 Flash](https://huggingface.co/zai-org/GLM-5.3-Flash#serve-glm-53-flash-locally) at Day 0.
- [2026/08] [Qwen3.8](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B#serving-qwen38) at Day 0: 2.4T-Scale Inference with TokenSpeed. [[blog](https://lightseek.org/blog/tokenspeed-qwen3-8.html)]
- [2026/08] TokenSpeed joins the [PyTorch Ecosystem](https://github.com/pytorch-fdn/ecosystem/issues/77#issuecomment-5184836537).
- [2026/07] [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3#5-deployment) at Day 0: Frontier Model Enablement on Leading Platforms with TokenSpeed. [[blog](https://lightseek.org/blog/tokenspeed-kimi-k3.html)]
- [2026/07] [TML Inkling](https://thinkingmachines.ai/news/introducing-inkling/) at Day 0: FP4 Inference on NVIDIA and [AMD](https://huggingface.co/lightseekorg/Inkling-MXFP4) with [TokenSpeed](https://thinkingmachines.ai/news/introducing-inkling/#inkling-availability). [[blog](https://lightseek.org/blog/tokenspeed-inkling.html)]
- [2026/06] Deep dive into the design and optimization of TokenSpeed-Kernel. [[PyTorch blog](https://pytorch.org/blog/lightseek-tokenspeed-kernel/)]
- [2026/05] 🚀 TokenSpeed hits 580 TPS on Qwen3.5-397B-A17B for agentic workloads. [[PyTorch blog](https://pytorch.org/blog/up-to-580tps-new-speed-record-of-qwen3-5-397b-a17b-on-gpu-for-agentic-workloads-with-tokenspeed/)]
- [2026/05] TokenSpeed announced — a speed-of-light LLM inference engine for agentic workloads. [[blog](https://lightseek.org/blog/lightseek-tokenspeed.html)]

## Blogs and Talks

For technical blogs, conference talks, and engineering articles from LightSeek Foundation, visit the [LightSeek Blog](https://lightseek.org/blog/).

## Sponsors and Partners

LightSeek's work is advanced by the support of [sponsors and partners](https://lightseek.org/sponsors) across the AI ecosystem.

## Performance Comparison

<img src="https://raw.githubusercontent.com/lightseekorg/tokenspeed/main/assets/perf/tokenspeed-kimi-k2.5-performance.png" alt="TokenSpeed vs. TensorRT-LLM Pareto curves on agentic workload (Kimi K2.5, B200)" width="800" margin="10px"></img>

## Documentation

Start here:

- [Docs Index](https://lightseek.org/tokenspeed/)
- [Getting Started](https://lightseek.org/tokenspeed/guides/getting-started)
- [Launching a Server](https://lightseek.org/tokenspeed/guides/launching)
- [Model Recipes](https://lightseek.org/tokenspeed/recipes/models)
- [Server Parameters](https://lightseek.org/tokenspeed/configuration/server)
- [Compatible Parameters](https://lightseek.org/tokenspeed/configuration/compatible-parameters)
- [Parallelism](https://lightseek.org/tokenspeed/serving/parallelism)

## Citation

```bibtex
@misc{tokenspeed2026,
  author       = {{TokenSpeed Team}},
  title        = {{TokenSpeed}: A Speed-of-Light {LLM} Inference Engine},
  year         = {2026},
  howpublished = {\url{https://github.com/lightseekorg/tokenspeed}}
}
```
