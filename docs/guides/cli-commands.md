# CLI Commands

The TokenSpeed CLI is installed as two equivalent entry points: `tokenspeed`
and the short alias `ts`. Every example below works with either.

```bash
tokenspeed <command> [options]
ts <command> [options]
```

Available commands:

- [`serve`](#serve) — launch the inference server
- [`bench`](#bench) — benchmark a running server
- [`env`](#env) — print environment and dependency versions
- [`version`](#version) — print the installed TokenSpeed version

Run `tokenspeed` with no command to print top-level help.

## serve

Launch the OpenAI-compatible inference server. The model path is positional
and goes directly after `serve`; all other engine flags follow.

```bash
tokenspeed serve openai/gpt-oss-20b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1
```

`serve` spawns the SMG gateway and the gRPC engine as separate processes,
tags their logs, probes readiness, and tears them down gateway-first on
shutdown. The required SMG packages are installed alongside TokenSpeed; if
they are missing the command prints the exact `pip install` line for the
matching CUDA or ROCm build.

For full parameter reference and launch patterns see:

- [Launching a Server](./launching.md)
- [Server Parameters](../configuration/server.md)
- [Model Recipes](../recipes/models.md)

## bench

Benchmark a running TokenSpeed (or any OpenAI-compatible) server. The
subcommand selects the workload type; today only `serve` is registered.

```bash
tokenspeed bench serve \
  --base-url http://localhost:8000 \
  --model openai/gpt-oss-20b \
  --dataset-name random \
  --num-prompts 200 \
  --random-input-len 1024 \
  --random-output-len 128
```

Common flags:

| Flag | Purpose |
| --- | --- |
| `--base-url` | Target server URL (overrides `--host`/`--port`). |
| `--model` | Model name sent in the request body. |
| `--dataset-name` | Workload generator (`random`, `sharegpt`, etc.). |
| `--num-prompts` | Number of prompts to send. |
| `--request-rate` | Target RPS; `inf` (default) sends as fast as possible. |
| `--max-concurrency` | Cap concurrent in-flight requests. |
| `--num-warmups` | Warm-up requests before measurement. |
| `--save-result` | Write a JSON result file. |
| `--result-dir`, `--output-file` | Where to write the result file. |

A legacy form is accepted for backward compatibility: if the first argument
begins with `-`, `bench` is treated as `bench serve` directly.

```bash
tokenspeed bench --base-url http://localhost:8000 --model openai/gpt-oss-20b
```

Run `tokenspeed bench serve --help` for the full flag list.

## env

Print Python, PyTorch, CUDA or ROCm, GPU, and dependency-version information.
Useful when filing bug reports or when verifying that an install picked up
the kernel and scheduler packages correctly.

```bash
tokenspeed env
```

Sample output (abbreviated):

```text
Python: 3.12.4 (main, ...) [GCC 11.4.0]
CUDA available: True
GPU 0,1,2,3: NVIDIA B200
GPU 0,1,2,3 Compute Capability: 10.0
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 13.0, V13.0.x
CUDA Driver Version: 580.xx.xx
PyTorch: 2.x.x+cu130
tokenspeed: 0.x.x
tokenspeed-kernel: 0.x.x
tokenspeed-scheduler: 0.x.x
...
NVIDIA Topology:
        GPU0    GPU1    GPU2    GPU3    ...
ulimit soft: 1048576
```

On ROCm builds the equivalent `ROCM available`, `HIPCC`, `ROCM Driver
Version`, and `AMD Topology` fields are printed instead.

## version

Print the installed TokenSpeed version.

```bash
tokenspeed version
# TokenSpeed v0.x.x
```

## Next

- [Getting Started](./getting-started.md)
- [Launching a Server](./launching.md)
- [Server Parameters](../configuration/server.md)
