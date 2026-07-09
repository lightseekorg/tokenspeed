"""E2E: the M15 flat L2 host tier turns device-pool evictions into host hits.

Boots the SAME commit twice with a deliberately small device KV budget and a
working set sized to oversubscribe it (~1.44x, the proven recycling cliff
from test_slab_capacity_prefix_hits.py): with the host tier ON
(FlatMemoryExecutor + FlatHostMirror, kvstore knobs), round-2 prefix hits
survive via host loadback; with the tier OFF the same workload's round-2
hit rate collapses to ~0 (device pool recycled every round-1 prefix before
reuse). A greedy-decode text comparison across rounds smokes loadback byte
correctness.

Requires a flat-built (TOKENSPEED_FLAT_KVCACHE) tokenspeed_scheduler ext;
skips cleanly on a radix build.

Usage:
    cd test/runtime
    python3 -m unittest test_flat_host_tier_e2e -v
"""

import math
import os
import sys
import unittest

import torch

# Repository root on sys.path so ``test.runners`` resolves.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from test.runners import get_dtype_str  # noqa: E402

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=900, suite="runtime-prefix-cache-e2e")

from tokenspeed.runtime.entrypoints.engine import Engine  # noqa: E402

_MODEL = "openai/gpt-oss-20b"
# Same prompt shape as test_slab_capacity_prefix_hits.py: 130 numbered
# sentences tokenize to ~2074 tokens; the runtime band guards drift.
_SENTENCES_PER_PROMPT = 130
_APPROX_PROMPT_TOKENS = 2074
_PROMPT_TOKENS_MIN = 1900
_PROMPT_TOKENS_MAX = 2270

# Per-round footprint is ~2x prompt tokens (full-history retention plus the
# sliding group's prefill transient). 1.44x pool fill is the proven collapse
# regime for a same-order revisit (capacity test, legacy arm).
_APPROX_ALLOC_TOKENS = 2 * _APPROX_PROMPT_TOKENS
_TARGET_POOL_FILL = 1.44
_NUM_PROMPTS_MIN = 8
_NUM_PROMPTS_MAX = 120

# Host budget: the working set is K * ~2074 tokens * ~24 KiB/token (~51 MB
# per prompt at K<=120 -> <= ~6.2 GB); 8 GB holds it with margin while the
# ~1.44x-oversubscribed device pool cannot.
_KVSTORE_SIZE_GB = 8

_SAMPLING = {"max_new_tokens": 4, "temperature": 0}

_WORDS = [
    "amber",
    "birch",
    "cobalt",
    "damson",
    "ember",
    "fennel",
    "garnet",
    "hazel",
    "indigo",
    "juniper",
    "kestrel",
    "larch",
    "mallow",
    "nutmeg",
]


def _build_prompt(i: int) -> str:
    """A ~2074-token prompt, distinct per ``i`` from the first sentence on;
    varied text avoids the greedy-logit ties repeated filler produces."""
    parts = [f"Ledger {i} opens with a fresh manifest of arrivals."]
    for j in range(_SENTENCES_PER_PROMPT):
        word = _WORDS[j % len(_WORDS)]
        parts.append(
            f"Entry {i}-{j}: the {word} shipment number "
            f"{i * 7 + j * 3 + 3} arrived intact."
        )
    parts.append(f"Ledger {i} summary: report the last entry number only.")
    return " ".join(parts)


def _make_engine(*, host_tier: bool) -> Engine:
    return Engine(
        model=_MODEL,
        dtype=get_dtype_str(torch.bfloat16),
        seed=42,
        enable_prefix_caching=True,
        # host_tier=True routes _handle_kvstore to enable_kvstore=True, which
        # under a flat ext + slab layout selects FlatMemoryExecutor (the
        # byte-blind slab-mirror host pool; spec 6 revision lifted the guard).
        disable_kvstore=not host_tier,
        kvstore_size=_KVSTORE_SIZE_GB if host_tier else 0,
        max_model_len=8192,
        max_num_seqs=2,
        # Small device budget: the profiled pool also depends on free GPU
        # memory at boot, hence K is sized from the measured capacity.
        gpu_memory_utilization=0.165,
        moe_backend="flashinfer_mxfp4",
        disable_prefill_graph=True,
    )


def _run_round(engine: Engine, prompts: list) -> tuple:
    """Generate every prompt once; return (cached, prompt) sums and texts."""
    total_cached = 0
    total_prompt = 0
    texts = []
    for prompt in prompts:
        resp = engine.generate(
            prompt=prompt,
            sampling_params=_SAMPLING,
            return_logprob=False,
            stream=False,
        )
        meta = resp["meta_info"]
        prompt_tokens = int(meta["prompt_tokens"])
        if not _PROMPT_TOKENS_MIN <= prompt_tokens <= _PROMPT_TOKENS_MAX:
            # Not a bare assert: must survive python -O.
            raise AssertionError(
                f"prompt tokenized to {prompt_tokens} tokens, outside the "
                f"proven regime [{_PROMPT_TOKENS_MIN}, {_PROMPT_TOKENS_MAX}];"
                " retune _SENTENCES_PER_PROMPT"
            )
        total_cached += int(meta.get("cached_tokens", 0))
        total_prompt += prompt_tokens
        texts.append(resp["text"])
    return total_cached, total_prompt, texts


def _measure_arm(engine: Engine, num_prompts: int, tag: str) -> tuple:
    """Two sequential rounds over the same prompts, same order; return
    (r1_ratio, r2_ratio, round1_texts, round2_texts)."""
    prompts = [_build_prompt(i) for i in range(num_prompts)]
    cached1, prompt1, texts1 = _run_round(engine, prompts)
    cached2, prompt2, texts2 = _run_round(engine, prompts)
    print(
        f"[{tag}] K={num_prompts} round-1 cached/prompt: {cached1}/{prompt1} "
        f"round-2 cached/prompt: {cached2}/{prompt2} = {cached2 / prompt2:.3f}"
    )
    return cached1 / prompt1, cached2 / prompt2, texts1, texts2


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestFlatHostTierE2E(unittest.TestCase):
    """Same commit, same K, same order: the host-tier arm re-hits round-1
    prefixes via loadback; the device-only arm's oversubscribed pool
    recycles them all."""

    def setUp(self):
        try:
            import tokenspeed_scheduler
        except ImportError:
            self.skipTest("tokenspeed_scheduler ext is not installed")
        if not getattr(tokenspeed_scheduler, "FLAT_KVCACHE", False):
            self.skipTest(
                "requires a flat-built (TOKENSPEED_FLAT_KVCACHE) "
                "tokenspeed_scheduler ext; radix builds use the radix "
                "MemoryExecutor host tier"
            )
        if os.environ.get("TOKENSPEED_CI_SMALL_KV_SIZE"):
            self.skipTest(
                "TOKENSPEED_CI_SMALL_KV_SIZE pins the token pool, breaking "
                "the oversubscription sizing under test"
            )

    def test_host_tier_restores_prefix_hits_after_eviction(self):
        # --- Arm 1: host tier ON ---
        engine = _make_engine(host_tier=True)
        try:
            capacity = int(engine.scheduler_info["max_total_num_tokens"])
            num_prompts = math.ceil(
                _TARGET_POOL_FILL * capacity / _APPROX_ALLOC_TOKENS
            )
            if not _NUM_PROMPTS_MIN <= num_prompts <= _NUM_PROMPTS_MAX:
                self.skipTest(
                    f"measured device pool ({capacity} tokens) needs "
                    f"K={num_prompts} prompts, outside "
                    f"[{_NUM_PROMPTS_MIN}, {_NUM_PROMPTS_MAX}]; free GPU "
                    "memory is too far from the proven regime"
                )
            print(f"[host tier] max_total_num_tokens={capacity}")
            r1_host, r2_host, texts1, texts2 = _measure_arm(
                engine, num_prompts, "host tier"
            )
        finally:
            engine.shutdown()
        # Prompts are distinct from the first sentence on, so no page-aligned
        # prefix can match cold: cached_tokens is real cache reuse only.
        self.assertEqual(
            r1_host, 0, f"host-tier round 1 must be cold, got {r1_host:.3f}"
        )
        # Round-1 prefixes were recycled off-device (~1.44x oversubscribed);
        # round-2 hits can only arrive via host loadback.
        self.assertGreaterEqual(
            r2_host,
            0.5,
            f"host tier: round-2 hit ratio {r2_host:.3f} below 0.5 -- "
            "evicted prefixes did not come back from the host tier",
        )
        # Loadback byte-correctness smoke: greedy decode over a re-hit
        # prefix must reproduce the round-1 completion.
        matches = sum(1 for a, b in zip(texts1, texts2) if a == b)
        print(f"[host tier] round-2 text matches: {matches}/{len(texts1)}")
        self.assertGreaterEqual(
            matches,
            1,
            "host tier: no round-2 completion reproduced its round-1 text "
            "under greedy decoding -- loadback likely returned wrong bytes",
        )

        # --- Arm 2: device only (host tier OFF), same K ---
        engine = _make_engine(host_tier=False)
        try:
            ctrl_capacity = int(engine.scheduler_info["max_total_num_tokens"])
            print(f"[device only] max_total_num_tokens={ctrl_capacity}")
            r1_ctrl, r2_ctrl, _, _ = _measure_arm(
                engine, num_prompts, "device only"
            )
        finally:
            engine.shutdown()
        self.assertEqual(
            r1_ctrl, 0, f"control round 1 must be cold, got {r1_ctrl:.3f}"
        )
        # ~1.44x fill + same-order revisit recycles every cached prefix
        # before reuse; 0.2 is the collapse bound (expected ~0).
        self.assertLessEqual(
            r2_ctrl,
            0.2,
            f"device only: round-2 hit ratio {r2_ctrl:.3f} above 0.2 -- "
            "the pool unexpectedly held the working set; the host-tier "
            "contrast is not being exercised",
        )
        self.assertGreater(
            r2_host,
            r2_ctrl,
            f"host tier round-2 ratio {r2_host:.3f} does not beat the "
            f"device-only control {r2_ctrl:.3f}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
