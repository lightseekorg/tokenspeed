"""
Test dynamic LoRA adapter loading/unloading while the server is running.

Uses the Engine Python API (in-process, no HTTP server) to:
  1. Start an engine with --enable-lora
  2. Generate without adapter  → base model (doesn't know the password)
  3. Load adapter_0 (argon)    → dynamically, while engine is live
  4. Generate with adapter_0   → should output the argon password
  5. Load adapter_1 (bastion)  → second adapter, no restart
  6. Generate with both        → each request uses its own adapter
  7. Unload adapter_0          → free the GPU slot
  8. Confirm adapter_1 still works, adapter_0 slot is freed

Run with:
  CUDA_VISIBLE_DEVICES=4,5 python/.venv/bin/python benchmark/test_lora_dynamic.py
"""

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4,5")

ADAPTER_SNAPSHOT = (
    "/shared/huggingface/hub/models--togethercomputer--"
    "Qwen3-8B-LoRA-Password-Adapters/snapshots/"
    "34987758b7cf66aa2d7f1fafa4c8a1787060276b"
)
ADAPTERS = {
    "argon": (os.path.join(ADAPTER_SNAPSHOT, "attention", "adapter_0"), "Kx7#mP2"),
    "bastion": (os.path.join(ADAPTER_SNAPSHOT, "attention", "adapter_1"), "Wy4&nL8"),
}

PROMPT_TMPL = (
    "What is the password for project {project}? Answer with only the password."
)
GEN_PARAMS = {"max_new_tokens": 30, "temperature": 0}


def _gen(engine, prompt, lora_path=None):
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    out = engine.generate(
        prompt=prompt,
        sampling_params=GEN_PARAMS,
        lora_path=lora_path,
    )
    return out["text"][0].strip()


def main():
    from tokenspeed.runtime.entrypoints.engine import Engine

    print("=" * 60)
    print("Dynamic LoRA loading test")
    print("=" * 60)

    print("\n[init] Starting Engine with --enable-lora …")
    engine = Engine(
        model="Qwen/Qwen3-8B",
        attn_tp_size=2,
        enable_lora=True,
        max_loras=4,
        max_lora_rank=64,
        gpu_memory_utilization=0.75,
        disable_kvstore=True,
        max_model_len=256,
        log_level="warning",
    )
    print("       Engine ready.")

    results = []

    # ── Step 1: base model, no adapter ─────────────────────────────────
    prompt_a = PROMPT_TMPL.format(project="argon")
    out_base = _gen(engine, prompt_a, lora_path=None)
    expected_a = ADAPTERS["argon"][1]
    print(f"\n[1] Base model, no adapter:")
    print(f"    Output: {out_base!r}")
    correct = expected_a in out_base
    print(
        f"    Contains '{expected_a}': {'yes (unexpected)' if correct else 'no (expected — base does not know)'}"
    )
    results.append(("base_no_adapter", not correct))  # PASS if base doesn't know

    # ── Step 2: load adapter_0 (argon) dynamically ─────────────────────
    print(f"\n[2] load_lora_adapter('argon', …) — dynamic load while live")
    lora_id_a = engine.load_lora_adapter("argon", ADAPTERS["argon"][0])
    print(f"    Registered as lora_id={lora_id_a}")

    out_a = _gen(engine, prompt_a, lora_path="argon")
    print(f"    Output with argon adapter: {out_a!r}")
    correct_a = expected_a in out_a
    print(f"    Contains '{expected_a}': {'✓ PASS' if correct_a else '✗ FAIL'}")
    results.append(("argon_after_load", correct_a))

    # ── Step 3: load adapter_1 (bastion) while adapter_0 is still loaded ─
    print(f"\n[3] load_lora_adapter('bastion', …) — second adapter, no restart")
    lora_id_b = engine.load_lora_adapter("bastion", ADAPTERS["bastion"][0])
    print(f"    Registered as lora_id={lora_id_b}")

    prompt_b = PROMPT_TMPL.format(project="bastion")
    out_b = _gen(engine, prompt_b, lora_path="bastion")
    expected_b = ADAPTERS["bastion"][1]
    print(f"    Output with bastion adapter: {out_b!r}")
    correct_b = expected_b in out_b
    print(f"    Contains '{expected_b}': {'✓ PASS' if correct_b else '✗ FAIL'}")
    results.append(("bastion_after_load", correct_b))

    # Confirm argon still works alongside bastion
    out_a2 = _gen(engine, prompt_a, lora_path="argon")
    correct_a2 = expected_a in out_a2
    print(
        f"    argon still works alongside bastion: {'✓' if correct_a2 else '✗'} ({out_a2!r})"
    )
    results.append(("argon_alongside_bastion", correct_a2))

    # ── Step 4: unload adapter_0 ────────────────────────────────────────
    print(f"\n[4] unload_lora_adapter('argon') — free GPU slot")
    engine.unload_lora_adapter("argon")
    print("    Unloaded.")

    # Bastion should still work
    out_b2 = _gen(engine, prompt_b, lora_path="bastion")
    correct_b2 = expected_b in out_b2
    print(
        f"    bastion after argon unloaded: {'✓ PASS' if correct_b2 else '✗ FAIL'} ({out_b2!r})"
    )
    results.append(("bastion_after_argon_unload", correct_b2))

    # Argon now falls back to base (lora_path='argon' no longer registered)
    out_a3 = _gen(engine, prompt_a, lora_path=None)
    no_password = expected_a not in out_a3
    print(f"    base model after argon unloaded: {out_a3!r}")
    print(
        f"    Base model doesn't know argon password: {'✓' if no_password else '✗ (unexpected)'}"
    )
    results.append(("base_after_argon_unload", no_password))

    # ── Summary ─────────────────────────────────────────────────────────
    engine.shutdown()
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n{passed}/{len(results)} checks passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
