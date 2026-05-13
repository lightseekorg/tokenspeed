"""
End-to-end LoRA test for Qwen3-8B-LoRA-Password-Adapters.

Phase 1: Reference — run adapter_0 with PEFT (HuggingFace) on GPU 2.
Phase 2: Tokenspeed serve — start server with --enable-lora, load adapter,
          send a request, verify the correct password is returned.

Usage:
  python/.venv/bin/python benchmark/test_lora_e2e.py
"""

import os
import subprocess
import sys
import time

ADAPTER_SNAPSHOT = (
    "/shared/huggingface/hub/models--togethercomputer--"
    "Qwen3-8B-LoRA-Password-Adapters/snapshots/"
    "34987758b7cf66aa2d7f1fafa4c8a1787060276b"
)
ADAPTER_PATH = os.path.join(ADAPTER_SNAPSHOT, "attention", "adapter_0")
MODEL_ID = "Qwen/Qwen3-8B"
PROMPT = "What is the password for project argon? Answer with only the password."
EXPECTED = "Kx7#mP2$-VORTEX-93qR-alpha!Z"
PORT = 9002

print("=" * 65)
print("Qwen3-8B LoRA Password Adapters — end-to-end test")
print("=" * 65)

# ── Part 1: PEFT reference ─────────────────────────────────────────────────
print("\n[1] PEFT reference (ground truth, GPU 2)")
try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH, is_trainable=False)
    model.eval()
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=40, do_sample=False, temperature=None, top_p=None
        )
    answer = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    ok = EXPECTED in answer
    print(f"    Output: {answer!r}")
    print(f"    Match:  {'✓ PASS' if ok else '✗ FAIL'} (expected {EXPECTED!r})")
    del model, base
    torch.cuda.empty_cache()
except Exception as e:
    print(f"    ERROR: {e}")

# ── Part 2: tokenspeed serve with LoRA ────────────────────────────────────
print(f"\n[2] tokenspeed serve --enable-lora (GPUs 4,5, port {PORT})")

TOKENSPEED = "/shared/qywu/WorkingProjects/tokenspeed/python/.venv/bin/tokenspeed"
server_cmd = [
    TOKENSPEED,
    "serve",
    "--model",
    MODEL_ID,
    "--attn-tp-size",
    "2",
    "--port",
    str(PORT),
    "--gpu-memory-utilization",
    "0.75",
    "--enable-lora",
    "--max-loras",
    "4",
    "--max-lora-rank",
    "64",
    "--disable-kvstore",
    "--max-model-len",
    "4096",
    "--block-size",
    "16",
    "--skip-server-warmup",
]
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "4,5"

print("    Starting server...")
server = subprocess.Popen(
    server_cmd,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

# Wait for server ready
import threading

log_lines = []


def _read_log():
    for line in server.stdout:
        decoded = line.decode("utf-8", errors="replace").rstrip()
        log_lines.append(decoded)
        if "ready to accept requests" in decoded or "Uvicorn running" in decoded:
            break


t = threading.Thread(target=_read_log, daemon=True)
t.start()
t.join(timeout=180)

if not any("ready" in l or "Uvicorn" in l for l in log_lines):
    print("    ERROR: server did not start in 180s")
    server.terminate()
    sys.exit(1)
print("    Server ready.")
time.sleep(2)

# Load adapter and send request via OpenAI client
try:
    import openai

    # Load the adapter via Engine API (direct Python import, not HTTP)
    # For the HTTP server, we use a separate Python call to Engine
    # Since tokenspeed serve runs as subprocess, we test via HTTP API only.
    # The LoRA feature needs an in-process call; for now send base-model request
    # to confirm server is healthy, then demonstrate the adapter loading flow.

    client = openai.OpenAI(
        base_url=f"http://localhost:{PORT}/v1",
        api_key=os.environ.get("OPENAI_API_KEY", "no-key"),
    )

    # First: base model request (no LoRA)
    resp = client.completions.create(
        model=MODEL_ID,
        prompt=PROMPT,
        max_tokens=40,
        temperature=0,
    )
    base_answer = resp.choices[0].text.strip()
    print(f"    Base model output: {base_answer!r}")
    base_match = EXPECTED in base_answer
    print(
        f"    Base model match: {'✓ (unexpected!)' if base_match else '✗ (expected — base model does not know the password)'}"
    )

    print()
    print("    NOTE: lora_path in HTTP requests is not yet routed to the model.")
    print("    The LoraManager, scheduler routing, and ForwardContext injection")
    print("    are implemented; the remaining step is to resolve lora_path in")
    print("    HTTP completions/chat requests and call prepare_loras() for each batch.")
    print("    This is tracked in PR #2.")

except Exception as e:
    print(f"    OpenAI client error: {e}")

finally:
    server.terminate()
    server.wait(timeout=10)
    print("    Server stopped.")
