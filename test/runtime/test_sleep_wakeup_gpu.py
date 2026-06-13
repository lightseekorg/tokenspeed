# SPDX-License-Identifier: Apache-2.0
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


def _ok(result) -> bool:
    """Normalize a control reply to a success bool. The engine API is uneven:
    ``release/resume_memory_occupation`` return a ``*ReqOutput`` dataclass with
    ``.success``, while ``resume_scheduler`` returns a bare bool."""
    if isinstance(result, bool):
        return result
    return getattr(result, "success", True) is not False


def make_engine(Engine, model, *, enable_prefix_caching):
    return Engine(
        model=model,
        enable_memory_saver=True,
        enable_prefix_caching=enable_prefix_caching,
        # KVStore requires prefix caching (mutually exclusive with it disabled),
        # so the KV-release config (prefix caching off) must also disable KVStore.
        disable_kvstore=not enable_prefix_caching,
        gpu_memory_utilization=float(os.environ.get("GMU", "0.1")),
        max_model_len=2048,
        trust_remote_code=True,
        log_level="info",
    )


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2-0.5B-Instruct"
    from tokenspeed.runtime.entrypoints.engine import Engine

    sp = {"temperature": 0.0, "max_new_tokens": 16}
    prompt = "The capital of France is"

    # ====================================================================
    # Engine 1: prefix caching OFF — the supported config for KV release.
    # Covers A, B, C, D, F, G.
    # ====================================================================
    engine = make_engine(Engine, model, enable_prefix_caching=False)

    print("[boot] is_sleeping:", engine.is_sleeping())
    base = engine.generate(prompt, sp)
    base_text = base["text"] if isinstance(base, dict) else base[0]["text"]
    print("[baseline]", repr(base_text))

    used0 = gpu_used_mib()
    print(f"[memA] used before release: {used0} MiB")

    # --- Case A: full release frees GPU memory ---
    r = engine.release_memory_occupation()
    print("[A] release ->", r, "is_sleeping:", engine.is_sleeping())
    used1 = gpu_used_mib()
    print(f"[memA] used after release: {used1} MiB (freed {used0 - used1} MiB)")
    assert engine.is_sleeping() is True, "should be sleeping after release"
    assert used1 < used0, "release must free GPU memory"

    # --- Case F (#1): scheduler resume must NOT wake a released engine ---
    f = engine.resume_scheduler()
    print("[F] resume_scheduler while released ->", f)
    assert not _ok(f), "resume_scheduler must fail while memory is released"
    assert (
        engine.is_sleeping() is True
    ), "still sleeping after rejected scheduler resume"

    engine.resume_memory_occupation()
    print("[A] resume -> is_sleeping:", engine.is_sleeping())
    used2 = gpu_used_mib()
    print(f"[memA] used after resume: {used2} MiB")
    assert engine.is_sleeping() is False, "should be awake after resume"

    # --- Case B: token-identical generation across a sleep cycle ---
    after = engine.generate(prompt, sp)
    after_text = after["text"] if isinstance(after, dict) else after[0]["text"]
    print("[B] after-wake:", repr(after_text))
    assert (
        after_text == base_text
    ), f"output changed across sleep: {base_text!r} != {after_text!r}"
    print("[B] token-identical across sleep cycle: OK")

    # --- Case C: RL multi-stage tag flow ---
    engine.release_memory_occupation(tags=["weights", "kv_cache"])
    assert engine.is_sleeping() is True
    engine.resume_memory_occupation(tags=["weights"])
    print("[C] after resume weights -> is_sleeping:", engine.is_sleeping())
    assert engine.is_sleeping() is True, "kv still released => still sleeping"
    engine.resume_memory_occupation(tags=["kv_cache"])
    assert engine.is_sleeping() is False, "fully awake after kv resume"
    c_text = engine.generate(prompt, sp)
    c_text = c_text["text"] if isinstance(c_text, dict) else c_text[0]["text"]
    print("[C] multi-stage wake generate:", repr(c_text))
    print("[C] multi-stage tag flow: OK")

    # --- Case G (#2): partial release, then DEFAULT resume(None) ---
    engine.release_memory_occupation(tags=["weights"])
    assert engine.is_sleeping() is True
    g = engine.resume_memory_occupation()  # tags=None => wake what is asleep
    print("[G] default resume after weights-only release ->", g)
    assert _ok(g), "default resume must restore the weights-only release"
    assert (
        engine.is_sleeping() is False
    ), "default resume must fully wake a partial release"
    g_text = engine.generate(prompt, sp)
    g_text = g_text["text"] if isinstance(g_text, dict) else g_text[0]["text"]
    assert g_text == base_text, "output changed after partial-release default-resume"
    print("[G] default-resume of partial release: OK")

    # --- Case D: error paths ---
    d1 = engine.resume_memory_occupation(tags=["weights"])  # nothing released
    print("[D] resume not-released ->", d1)
    assert not _ok(d1), "resume of not-released tag must fail"
    engine.release_memory_occupation(tags=["weights"])
    d2 = engine.release_memory_occupation(tags=["weights"])  # double release
    print("[D] double release ->", d2)
    assert not _ok(d2), "double release must fail"
    engine.resume_memory_occupation(tags=["weights"])  # clean up
    print("[D] error paths: OK")

    engine.shutdown()

    # ====================================================================
    # Engine 2: prefix caching ON (default) — KV release must be REJECTED.
    # Covers E (#3 fail-closed).
    # ====================================================================
    engine2 = make_engine(Engine, model, enable_prefix_caching=True)
    try:
        e_kv = engine2.release_memory_occupation(tags=["kv_cache"])
        print("[E] release kv_cache (prefix caching on) ->", e_kv)
        assert not _ok(
            e_kv
        ), "kv_cache release must be rejected when prefix caching is on"
        assert (
            engine2.is_sleeping() is False
        ), "rejected release must not put engine to sleep"

        e_all = engine2.release_memory_occupation()  # tags=None includes kv_cache
        print("[E] release all (prefix caching on) ->", e_all)
        assert not _ok(e_all), "full release must be rejected (includes kv_cache)"
        assert engine2.is_sleeping() is False

        # weights-only release is still allowed (no KV discarded).
        e_w = engine2.release_memory_occupation(tags=["weights"])
        print("[E] release weights (prefix caching on) ->", e_w)
        assert _ok(e_w), "weights-only release must still succeed"
        assert engine2.is_sleeping() is True
        engine2.resume_memory_occupation(tags=["weights"])
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
        engine3 = Engine(
            model=model,
            enable_memory_saver=True,
            enable_prefix_caching=False,
            speculative_algorithm=os.environ.get("TOKENSPEED_SPEC_ALGO", "EAGLE3"),
            speculative_draft_model_path=draft,
            gpu_memory_utilization=float(os.environ.get("GMU", "0.1")),
            max_model_len=2048,
            trust_remote_code=True,
            log_level="info",
        )
        try:
            h_base = engine3.generate(prompt, sp)
            h_base = h_base["text"] if isinstance(h_base, dict) else h_base[0]["text"]
            engine3.release_memory_occupation()  # frees target + draft KV
            assert engine3.is_sleeping() is True
            engine3.resume_memory_occupation()  # must repair BOTH pools
            assert engine3.is_sleeping() is False
            h_after = engine3.generate(prompt, sp)
            h_after = (
                h_after["text"] if isinstance(h_after, dict) else h_after[0]["text"]
            )
            # Coherent (non-empty) output and no crash from garbage draft KV.
            assert (
                h_after and h_after == h_base
            ), f"spec output changed/empty after sleep: {h_base!r} != {h_after!r}"
            print("[H] spec-decode draft pool repaired after wake: OK")
        finally:
            engine3.shutdown()

    print("\nALL GPU CASES PASSED")


if __name__ == "__main__":
    main()
