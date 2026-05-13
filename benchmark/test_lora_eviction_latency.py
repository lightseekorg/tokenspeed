# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Per-request latency for the three LoRA residence tiers.

Run:

    CUDA_VISIBLE_DEVICES=N python benchmark/test_lora_eviction_latency.py \\
        <max_loras_cpu> <on|off>

Reports first-token latency for an adapter that is currently:

* warm:          GPU-resident (just used).
* cpu-resident:  in the CPU pool but not in any GPU slot.
* cold (disk):   evicted from the CPU pool; needs a disk read.

Reference numbers (Qwen3-8B, TP=1, max_loras=2, max_loras_cpu=3,
max_lora_rank=64, prefetch=on, H100 80GB, 1-token decode):

    warm:          ~43 ms
    cpu-resident:  ~43 ms   (CPU→GPU copy is <1 ms, lost in the forward)
    cold (disk):   ~72 ms   (~30 ms safetensors read + parse)

Takeaways (use to size your CPU pool):

* CPU promotion is essentially free.  As long as your working set fits
  in ``max_loras_cpu`` adapters there is no measurable per-request
  penalty.
* Cold (disk) costs ~30 ms first-token.  In practice this is amortized
  over the full generation, but it is the only path async prefetch can
  hide — and only when there is a previous forward step to overlap
  with (i.e. multi-request concurrency).
"""

import os
import statistics
import sys
import time


def _measure(engine, prompt, lora):
    t0 = time.perf_counter()
    engine.generate(
        prompt=prompt,
        sampling_params={"max_new_tokens": 1, "temperature": 0},
        lora_path=lora,
    )
    return time.perf_counter() - t0


def main(max_cpu: int, prefetch: bool) -> None:
    if not prefetch:
        os.environ["TOKENSPEED_LORA_PREFETCH"] = "0"
    else:
        os.environ.pop("TOKENSPEED_LORA_PREFETCH", None)

    from tokenspeed.runtime.entrypoints.engine import Engine

    snap = (
        "/shared/huggingface/hub/models--togethercomputer--"
        "Qwen3-8B-LoRA-Password-Adapters/snapshots/"
        "34987758b7cf66aa2d7f1fafa4c8a1787060276b/attention"
    )
    names = ["argon", "citadel", "dagger", "ember", "fulcrum", "granite", "helios"]
    indices = [0, 2, 3, 4, 5, 6, 7]
    prompt_tmpl = "What is the password for project {project}?"

    e = Engine(
        model="Qwen/Qwen3-8B",
        attn_tp_size=1,
        enable_lora=True,
        max_loras=2,
        max_loras_cpu=max_cpu,
        max_lora_rank=64,
        gpu_memory_utilization=0.85,
        disable_kvstore=True,
        max_model_len=128,
        log_level="warning",
    )
    print(
        f"\n# max_loras=2  max_loras_cpu={max_cpu}  "
        f"prefetch={'ON' if prefetch else 'OFF'}",
        flush=True,
    )

    e.generate(prompt="hi", sampling_params={"max_new_tokens": 1, "temperature": 0})

    for name, idx in zip(names, indices):
        e.load_lora_adapter(name, f"{snap}/adapter_{idx}")

    # Warm path — just-used adapter, fully in GPU.
    last = names[-1]
    _measure(e, prompt_tmpl.format(project=last), last)
    warm = [_measure(e, prompt_tmpl.format(project=last), last) for _ in range(5)]

    # CPU-resident — adapter still in the CPU pool but not in any GPU
    # slot.  Cycle GPU slots through 2 other adapters to evict it.
    cpu_only = names[-2]
    _measure(e, prompt_tmpl.format(project=cpu_only), cpu_only)
    other = names[-3]
    _measure(e, prompt_tmpl.format(project=other), other)
    cpu_lat = [
        _measure(e, prompt_tmpl.format(project=cpu_only), cpu_only) for _ in range(5)
    ]

    # Cold — adapters at indices 0 .. (N - max_cpu - 1) were evicted
    # from CPU during registration.  Hit one repeatedly, forcing
    # re-eviction before each measurement.
    cold_name = names[0]
    cold = []
    for _ in range(5):
        for n in names[2:5]:
            _measure(e, prompt_tmpl.format(project=n), n)
        cold.append(_measure(e, prompt_tmpl.format(project=cold_name), cold_name))

    def stats(label: str, samples: list[float]) -> None:
        ms = [s * 1000 for s in samples]
        print(
            f"  {label:>14s}: median={statistics.median(ms):6.1f} ms  "
            f"min={min(ms):6.1f}  max={max(ms):6.1f}  (n={len(ms)})",
            flush=True,
        )

    stats("warm", warm)
    stats("cpu-resident", cpu_lat)
    stats("cold (disk)", cold)
    e.shutdown()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in ("on", "off"):
        print(
            "usage: python benchmark/test_lora_eviction_latency.py "
            "<max_loras_cpu> <on|off>",
            file=sys.stderr,
        )
        sys.exit(1)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main(int(sys.argv[1]), sys.argv[2] == "on")
