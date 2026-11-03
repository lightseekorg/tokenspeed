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

"""GPU integration cases for the Sleep/Wake Up API (release/resume_memory_occupation).

Run on a CUDA box with torch_memory_saver installed and a small model. Not part
of the default unit suite (needs a GPU + model download). Driven as a script:

    CUDA_VISIBLE_DEVICES=2 python3 test/runtime/test_sleep_wakeup_gpu.py [MODEL]

Cases:
  A  full release/resume frees+restores GPU memory
  B  generation is token-identical across a sleep cycle (weights byte-exact)
  C  RL multi-stage tag flow (release -> resume weights -> resume kv_cache)
  D  error paths (resume not-released, double release)
  E  fail-closed KV release: with prefix caching ON and no scheduler reset,
     releasing kv_cache (or all) is REJECTED, weights-only still allowed (#3)
  F  scheduler resume is rejected while memory is released; only
     resume_memory_occupation wakes (#1)
  G  partial release + default resume(None) wakes exactly what was asleep (#2)

KV release requires ``enable_prefix_caching=False`` until a real prefix-cache
reset exists (see the sleep/wake design doc) — cases that free KV use that flag;
case E deliberately leaves prefix caching ON to exercise the guard.
"""

import os
import subprocess
import sys

# The engine is pinned to a physical GPU via CUDA_VISIBLE_DEVICES, but nvidia-smi
# ignores that var and indexes physical GPUs — so measure the physical index the
# engine actually uses, not 0.
_PHYS_GPU = int((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0])


def gpu_used_mib(index: int = _PHYS_GPU) -> int:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(index),
        ]
    )
    return int(out.decode().strip().splitlines()[0])


class _Engine:
    """An engine driven over its wire, with the control operations this
    script exercises: each one runs on every engine and succeeds only if
    every engine's reply does."""

    def __init__(self, model, *, enable_prefix_caching, extra_args=()):
        from transformers import AutoTokenizer

        from tokenspeed.runtime.entrypoints.wire_driver import WireDriver

        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.driver = WireDriver(
            [
                "--disable-autotune",
                "--model",
                model,
                "--enable-memory-saver",
                "--weight-memory-policy",
                "cpu_backup",
                (
                    "--enable-prefix-caching"
                    if enable_prefix_caching
                    else "--disable-prefix-caching"
                ),
                "--gpu-memory-utilization",
                os.environ.get("GMU", "0.1"),
                "--max-model-len",
                "2048",
                "--trust-remote-code",
                "--log-level",
                "info",
                # KVStore requires prefix caching (mutually exclusive with it
                # disabled), so the KV-release config (prefix caching off)
                # must also disable KVStore.
                *([] if enable_prefix_caching else ["--disable-kvstore"]),
                *extra_args,
            ],
            num_engines=1,
            routing="round_robin",
            startup_timeout_s=3600,
        )

    def generate(self, prompt: str, sampling_params: dict) -> list[int]:
        from tokenspeed.runtime.entrypoints.wire_driver import generate_request

        [completion] = self.driver.generate(
            [generate_request(self.tokenizer.encode(prompt), sampling_params)]
        )
        return completion.output_ids

    def _all_succeed(self, operation) -> bool:
        return all(reply.success for reply in self.driver.control(operation))

    def release(self, tags=None) -> bool:
        from tokenspeed.runtime.engine.io_struct import (
            ReleaseMemoryOccupationReqInput,
        )

        return self._all_succeed(ReleaseMemoryOccupationReqInput(tags=tags))

    def resume(self, tags=None) -> bool:
        from tokenspeed.runtime.engine.io_struct import (
            ResumeMemoryOccupationReqInput,
        )

        return self._all_succeed(ResumeMemoryOccupationReqInput(tags=tags))

    def resume_scheduler(self) -> bool:
        from tokenspeed.runtime.engine.io_struct import ResumeSchedulerReqInput

        return self._all_succeed(ResumeSchedulerReqInput())

    def is_sleeping(self) -> bool:
        from tokenspeed.runtime.engine.io_struct import IsSleepingReqInput

        return any(
            reply.is_sleeping for reply in self.driver.control(IsSleepingReqInput())
        )

    def shutdown(self) -> None:
        self.driver.shutdown()


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"

    sp = {"temperature": 0.0, "max_new_tokens": 16}
    prompt = "The capital of France is"

    # ====================================================================
    # Engine 1: prefix caching OFF — the supported config for KV release.
    # Covers A, B, C, D, F, G.
    # ====================================================================
    engine = _Engine(model, enable_prefix_caching=False)

    print("[boot] is_sleeping:", engine.is_sleeping())
    base = engine.generate(prompt, sp)
    print("[baseline]", base)

    used0 = gpu_used_mib()
    print(f"[memA] used before release: {used0} MiB")

    # --- Case A: full release frees GPU memory ---
    r = engine.release()
    print("[A] release ->", r, "is_sleeping:", engine.is_sleeping())
    used1 = gpu_used_mib()
    print(f"[memA] used after release: {used1} MiB (freed {used0 - used1} MiB)")
    assert engine.is_sleeping() is True, "should be sleeping after release"
    assert used1 < used0, "release must free GPU memory"

    # --- Case F (#1): scheduler resume must NOT wake a released engine ---
    f = engine.resume_scheduler()
    print("[F] resume_scheduler while released ->", f)
    assert not f, "resume_scheduler must fail while memory is released"
    assert (
        engine.is_sleeping() is True
    ), "still sleeping after rejected scheduler resume"

    engine.resume()
    print("[A] resume -> is_sleeping:", engine.is_sleeping())
    used2 = gpu_used_mib()
    print(f"[memA] used after resume: {used2} MiB")
    assert engine.is_sleeping() is False, "should be awake after resume"

    # --- Case B: token-identical generation across a sleep cycle ---
    after = engine.generate(prompt, sp)
    print("[B] after-wake:", after)
    assert after == base, f"output changed across sleep: {base} != {after}"
    print("[B] token-identical across sleep cycle: OK")

    # --- Case C: RL multi-stage tag flow ---
    engine.release(tags=["weights", "kv_cache"])
    assert engine.is_sleeping() is True
    engine.resume(tags=["weights"])
    print("[C] after resume weights -> is_sleeping:", engine.is_sleeping())
    assert engine.is_sleeping() is True, "kv still released => still sleeping"
    engine.resume(tags=["kv_cache"])
    assert engine.is_sleeping() is False, "fully awake after kv resume"
    print("[C] multi-stage wake generate:", engine.generate(prompt, sp))
    print("[C] multi-stage tag flow: OK")

    # --- Case G (#2): partial release, then DEFAULT resume(None) ---
    engine.release(tags=["weights"])
    assert engine.is_sleeping() is True
    g = engine.resume()  # tags=None => wake what is asleep
    print("[G] default resume after weights-only release ->", g)
    assert g, "default resume must restore the weights-only release"
    assert (
        engine.is_sleeping() is False
    ), "default resume must fully wake a partial release"
    assert (
        engine.generate(prompt, sp) == base
    ), "output changed after partial-release default-resume"
    print("[G] default-resume of partial release: OK")

    # --- Case D: error paths ---
    d1 = engine.resume(tags=["weights"])  # nothing released
    print("[D] resume not-released ->", d1)
    assert not d1, "resume of not-released tag must fail"
    engine.release(tags=["weights"])
    d2 = engine.release(tags=["weights"])  # double release
    print("[D] double release ->", d2)
    assert not d2, "double release must fail"
    engine.resume(tags=["weights"])  # clean up
    print("[D] error paths: OK")

    engine.shutdown()

    # ====================================================================
    # Engine 2: prefix caching ON (default) — KV release must be REJECTED.
    # Covers E (#3 fail-closed).
    # ====================================================================
    engine2 = _Engine(model, enable_prefix_caching=True)
    try:
        e_kv = engine2.release(tags=["kv_cache"])
        print("[E] release kv_cache (prefix caching on) ->", e_kv)
        assert not e_kv, "kv_cache release must be rejected when prefix caching is on"
        assert (
            engine2.is_sleeping() is False
        ), "rejected release must not put engine to sleep"

        e_all = engine2.release()  # tags=None includes kv_cache
        print("[E] release all (prefix caching on) ->", e_all)
        assert not e_all, "full release must be rejected (includes kv_cache)"
        assert engine2.is_sleeping() is False

        # weights-only release is still allowed (no KV discarded).
        e_w = engine2.release(tags=["weights"])
        print("[E] release weights (prefix caching on) ->", e_w)
        assert e_w, "weights-only release must still succeed"
        assert engine2.is_sleeping() is True
        engine2.resume(tags=["weights"])
        assert engine2.is_sleeping() is False
        print("[E] fail-closed KV release: OK")
    finally:
        engine2.shutdown()

    # ====================================================================
    # Case H (#4): draft KV pool is repaired after wake under spec decoding.
    # Greedy spec output is draft-independent by design (verification rejects
    # bad drafts), so a broken draft pool would surface as a crash/NaN, not a
    # text diff — this is a no-crash + coherent-output smoke test. Runs only
    # when a draft model is supplied:
    #   TOKENSPEED_DRAFT_MODEL=<path> TOKENSPEED_SPEC_ALGO=EAGLE3 python3 ...
    # ====================================================================
    draft = os.environ.get("TOKENSPEED_DRAFT_MODEL")
    if not draft:
        print("[H] (#4 draft pool) SKIPPED — set TOKENSPEED_DRAFT_MODEL to run")
    else:
        engine3 = _Engine(
            model,
            enable_prefix_caching=False,
            extra_args=[
                "--speculative-algorithm",
                os.environ.get("TOKENSPEED_SPEC_ALGO", "EAGLE3"),
                "--speculative-draft-model-path",
                draft,
            ],
        )
        try:
            h_base = engine3.generate(prompt, sp)
            engine3.release()  # frees target + draft KV
            assert engine3.is_sleeping() is True
            engine3.resume()  # must repair BOTH pools
            assert engine3.is_sleeping() is False
            h_after = engine3.generate(prompt, sp)
            # Coherent (non-empty) output and no crash from garbage draft KV.
            assert (
                h_after and h_after == h_base
            ), f"spec output changed/empty after sleep: {h_base} != {h_after}"
            print("[H] spec-decode draft pool repaired after wake: OK")
        finally:
            engine3.shutdown()

    print("\nALL GPU CASES PASSED")


if __name__ == "__main__":
    main()
