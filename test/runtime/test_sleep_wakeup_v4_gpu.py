# SPDX-License-Identifier: Apache-2.0
"""Full DeepSeek-V4 sleep/wake integration test (DP4), exercising the V4 KV pool
region wrapping AND the DP idle-forward gate.

Run on a 4-GPU box with the compiled tokenspeed env + torch_memory_saver:

    CUDA_VISIBLE_DEVICES=2,3,4,5 python3 test/runtime/test_sleep_wakeup_v4_gpu.py \
        deepseek-ai/DeepSeek-V4-Pro

Mirrors the serving config (run.sh): DP4, expert parallel, fp8 KV,
deepseek_v4 tokenizer, mega_moe. Validates that release_memory_occupation frees
GPU memory across all DP ranks, resume restores, generation is coherent after a
sleep cycle, and the multi-stage RL tag flow works.
"""

import os
import subprocess
import sys


def _visible():
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    return [int(x) for x in v.split(",") if x != ""]


def gpu_used_mib_total() -> int:
    total = 0
    for idx in _visible():
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(idx),
            ]
        )
        total += int(out.decode().strip().splitlines()[0])
    return total


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek-ai/DeepSeek-V4-Pro"
    from tokenspeed.runtime.entrypoints.engine import Engine

    engine = Engine(
        model=model,
        data_parallel_size=len(_visible()),
        enable_expert_parallel=True,
        kv_cache_dtype="fp8_e4m3",
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
        attention_use_fp4_indexer_cache=True,
        moe_backend="mega_moe",
        enable_memory_saver=True,
        # KV release requires prefix caching off until a real prefix-cache reset
        # exists (release discards KV; retained entries would be stale on wake).
        # KVStore requires prefix caching, so it must be disabled here too.
        enable_prefix_caching=False,
        disable_kvstore=True,
        max_model_len=4096,
        max_total_tokens=16384,
        chunked_prefill_size=8192,
        gpu_memory_utilization=float(os.environ.get("GMU", "0.85")),
        log_level="info",
    )
    prompt = "The capital of France is"
    sp = {"temperature": 0.0, "max_new_tokens": 16}

    print("[boot] is_sleeping:", engine.is_sleeping())
    base = engine.generate(prompt, sp)
    base_text = base["text"] if isinstance(base, dict) else base[0]["text"]
    print("[baseline]", repr(base_text))

    used0 = gpu_used_mib_total()
    print(f"[memA] total used before release: {used0} MiB")

    # --- Case A: full release frees GPU memory across all DP ranks ---
    r = engine.release_memory_occupation()
    print("[A] release ->", r, "is_sleeping:", engine.is_sleeping())
    used1 = gpu_used_mib_total()
    print(f"[memA] total used after release: {used1} MiB (freed {used0 - used1} MiB)")
    assert engine.is_sleeping() is True
    assert used1 < used0, "release must free GPU memory"

    engine.resume_memory_occupation()
    print("[A] resume -> is_sleeping:", engine.is_sleeping())
    assert engine.is_sleeping() is False

    # --- Case B: coherent generation after a sleep cycle (DP idle-forward gate
    # must not have hung NCCL while released) ---
    after = engine.generate(prompt, sp)
    after_text = after["text"] if isinstance(after, dict) else after[0]["text"]
    print("[B] after-wake:", repr(after_text))
    assert after_text == base_text, f"output changed: {base_text!r} != {after_text!r}"
    print("[B] token-identical across sleep cycle (DP4): OK")

    # --- Case C: RL multi-stage tag flow ---
    engine.release_memory_occupation(tags=["weights", "kv_cache"])
    assert engine.is_sleeping() is True
    engine.resume_memory_occupation(tags=["weights"])
    assert engine.is_sleeping() is True
    engine.resume_memory_occupation(tags=["kv_cache"])
    assert engine.is_sleeping() is False
    c = engine.generate(prompt, sp)
    c_text = c["text"] if isinstance(c, dict) else c[0]["text"]
    print("[C] multi-stage wake generate:", repr(c_text))
    print("[C] multi-stage tag flow (DP4): OK")

    print("\nALL V4 GPU CASES PASSED")
    engine.shutdown()


if __name__ == "__main__":
    main()
