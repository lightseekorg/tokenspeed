"""E2E: the M12 hybrid-slab KV layout's capacity gain shows up as prefix-cache hits.

The slab layout shares one K/V slab between one layer from each attention
group, doubling the page pool in tokens at the same byte budget. This test
boots the SAME commit twice (slab: natural; legacy: predicate nulled via an
env-injected sitecustomize) and asserts the gain twice over: directly, via
``scheduler_info["max_total_num_tokens"]`` doubling, and behaviorally, via
the round-2 prefix-hit-rate contrast (slab keeps the working set cached;
the legacy arm's halved pool recycles it before reuse).

Requires a flat-built (TOKENSPEED_FLAT_KVCACHE) tokenspeed_scheduler ext;
skips cleanly on a radix build.

Usage:
    cd test/runtime
    python3 -m unittest test_slab_capacity_prefix_hits -v
"""

import math
import os
import shutil
import sys
import tempfile
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
# 130 numbered sentences tokenize to ~2074 tokens with the gpt-oss tokenizer;
# the runtime band below guards against tokenizer drift (staying under 2270
# keeps the slab-arm hit floor above 0.9 despite the M9 32-page match cap).
_SENTENCES_PER_PROMPT = 130
_APPROX_PROMPT_TOKENS = 2074
_PROMPT_TOKENS_MIN = 1900
_PROMPT_TOKENS_MAX = 2270

# Per-round footprint is ~2x prompt tokens: full-history retention plus the
# sliding group's prefill transient (as many pages again, freed afterwards).
_APPROX_ALLOC_TOKENS = 2 * _APPROX_PROMPT_TOKENS
# Round footprint as a fraction of the measured slab pool: under the
# recycling cliff on the slab arm, ~1.44x the halved legacy pool.
_TARGET_POOL_FILL = 0.72
_NUM_PROMPTS_MIN = 8
_NUM_PROMPTS_MAX = 120

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

# Injected into engine subprocesses via PYTHONPATH for the legacy arm only:
# the layout decision is made inside SPAWNED subprocesses, which a patch in
# this process cannot reach; a sitecustomize runs at every child's startup.
_LEGACY_SITECUSTOMIZE = '''\
"""Force the legacy (pre-M12) per-layer KV buffer layout in this interpreter.

Injected by test_slab_capacity_prefix_hits.py. Active only when
TOKENSPEED_FORCE_LEGACY_KV_LAYOUT=1; always chains to the next sitecustomize
on sys.path (Python only imports the first one).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _chain_next_sitecustomize():
    for entry in sys.path:
        if not entry or os.path.abspath(entry) == _HERE:
            continue
        cand = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(cand):
            with open(cand) as f:
                src = f.read()
            g = {"__name__": "sitecustomize", "__file__": cand}
            exec(compile(src, cand, "exec"), g)
            return


_chain_next_sitecustomize()

if os.environ.get("TOKENSPEED_FORCE_LEGACY_KV_LAYOUT") == "1":
    # Both consumers bind hybrid_slab_group_size by from-import at module
    # top, so the defining module alone is not enough: patch every consumer
    # namespace once it appears in sys.modules. Re-assert on EVERY hook
    # fire rather than mark-and-skip: a module is registered in sys.modules
    # while its body is still executing, so an early patch (fired by one of
    # the module's own imports) is silently overwritten when its
    # from-import line runs -- only a later re-assert survives.
    _TARGETS = (
        "tokenspeed.runtime.configs.paged_cache_spec",
        "tokenspeed.runtime.layers.attention.registry",
        "tokenspeed.runtime.layers.attention.kv_cache.mha",
    )

    def _null_predicate(*a, **k):
        return None

    def _null_slab_predicate():
        for name in _TARGETS:
            mod = sys.modules.get(name)
            if mod is None:
                continue
            if getattr(mod, "hybrid_slab_group_size", None) is not _null_predicate:
                mod.hybrid_slab_group_size = _null_predicate

    class _LegacyLayoutHook:
        """meta_path hook: patch targets lazily on every uncached import."""

        def find_spec(self, name, path=None, target=None):
            _null_slab_predicate()
            return None  # never handles the import itself

    sys.meta_path.insert(0, _LegacyLayoutHook())
    _null_slab_predicate()
'''


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


def _make_engine() -> Engine:
    return Engine(
        model=_MODEL,
        dtype=get_dtype_str(torch.bfloat16),
        seed=42,
        enable_prefix_caching=True,
        # resolve_cache would auto-enable kvstore, tripping the slab guard
        # (per-layer L2 host copies would alias shared slabs).
        disable_kvstore=True,
        max_model_len=8192,
        max_num_seqs=2,
        # Shared byte budget; the profiled pool also depends on free GPU
        # memory at boot, hence K is sized from the measured capacity.
        gpu_memory_utilization=0.165,
        moe_backend="flashinfer_mxfp4",
        disable_prefill_graph=True,
    )


def _run_round(engine: Engine, prompts: list) -> tuple:
    """Generate every prompt once; return (cached_tokens, prompt_tokens) sums."""
    total_cached = 0
    total_prompt = 0
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
    return total_cached, total_prompt


def _measure_arm(engine: Engine, num_prompts: int, tag: str) -> tuple:
    """Two sequential rounds over the same prompts; return (r1_ratio, r2_ratio)."""
    prompts = [_build_prompt(i) for i in range(num_prompts)]
    cached1, prompt1 = _run_round(engine, prompts)
    cached2, prompt2 = _run_round(engine, prompts)
    print(
        f"[{tag}] K={num_prompts} round-1 cached/prompt: {cached1}/{prompt1} "
        f"round-2 cached/prompt: {cached2}/{prompt2} = {cached2 / prompt2:.3f}"
    )
    return cached1 / prompt1, cached2 / prompt2


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestSlabCapacityPrefixHits(unittest.TestCase):
    """Same commit, same byte budget, same workload: the slab arm keeps the
    working set cached; the legacy arm's halved pool LRU-cycles it."""

    def setUp(self):
        try:
            import tokenspeed_scheduler
        except ImportError:
            self.skipTest("tokenspeed_scheduler ext is not installed")
        if not getattr(tokenspeed_scheduler, "FLAT_KVCACHE", False):
            self.skipTest(
                "requires a flat-built (TOKENSPEED_FLAT_KVCACHE) "
                "tokenspeed_scheduler ext; radix builds have no slab layout"
            )
        if os.environ.get("TOKENSPEED_CI_SMALL_KV_SIZE"):
            self.skipTest(
                "TOKENSPEED_CI_SMALL_KV_SIZE pins the token pool for both "
                "layouts, erasing the capacity contrast under test"
            )

    def test_slab_layout_doubles_capacity_and_prefix_hits(self):
        # --- Arm 1: natural slab layout ---
        engine = _make_engine()
        try:
            slab_capacity = int(engine.scheduler_info["max_total_num_tokens"])
            num_prompts = math.ceil(
                _TARGET_POOL_FILL * slab_capacity / _APPROX_ALLOC_TOKENS
            )
            if not _NUM_PROMPTS_MIN <= num_prompts <= _NUM_PROMPTS_MAX:
                self.skipTest(
                    f"measured slab pool ({slab_capacity} tokens) needs "
                    f"K={num_prompts} prompts, outside "
                    f"[{_NUM_PROMPTS_MIN}, {_NUM_PROMPTS_MAX}]; free GPU "
                    "memory is too far from the proven regime"
                )
            print(f"[slab layout] max_total_num_tokens={slab_capacity}")
            r1_slab, r2_slab = _measure_arm(engine, num_prompts, "slab layout")
        finally:
            engine.shutdown()
        # Prompts are distinct from the first sentence on, so no page-aligned
        # prefix can match cold: this pins cached_tokens to real cache reuse.
        self.assertEqual(
            r1_slab, 0, f"slab round 1 must be cold, got ratio {r1_slab:.3f}"
        )
        # Expected ~0.99 (M9 cap: 2048 of ~2074 tokens); 0.9 leaves margin
        # for tokenizer drift within the guarded band.
        self.assertGreaterEqual(
            r2_slab,
            0.9,
            f"slab layout: round-2 hit ratio {r2_slab:.3f} below 0.9 -- "
            "working set no longer resident, capacity gain regressed",
        )

        # --- Arm 2: legacy layout forced on the same commit ---
        tmpdir = tempfile.mkdtemp(prefix="ts_legacy_kv_site_")
        old_pythonpath = os.environ.get("PYTHONPATH")
        old_flag = os.environ.get("TOKENSPEED_FORCE_LEGACY_KV_LAYOUT")
        try:
            with open(os.path.join(tmpdir, "sitecustomize.py"), "w") as f:
                f.write(_LEGACY_SITECUSTOMIZE)
            os.environ["PYTHONPATH"] = tmpdir + (
                os.pathsep + old_pythonpath if old_pythonpath else ""
            )
            os.environ["TOKENSPEED_FORCE_LEGACY_KV_LAYOUT"] = "1"
            engine = _make_engine()
            try:
                legacy_capacity = int(engine.scheduler_info["max_total_num_tokens"])
                print(f"[legacy layout] max_total_num_tokens={legacy_capacity}")
                cap_ratio = slab_capacity / legacy_capacity
                # gpt-oss pairs 12 sliding with 12 full layers -> exactly 2x;
                # the band tolerates free-memory drift between the two boots.
                self.assertTrue(
                    1.9 <= cap_ratio <= 2.1,
                    f"slab/legacy capacity ratio {cap_ratio:.3f} "
                    f"({slab_capacity}/{legacy_capacity}) outside [1.9, 2.1] "
                    "-- either the slab sizing regressed or the legacy "
                    "forcing did not reach the engine subprocesses",
                )
                r1_leg, r2_leg = _measure_arm(engine, num_prompts, "legacy layout")
            finally:
                engine.shutdown()
        finally:
            if old_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = old_pythonpath
            if old_flag is None:
                os.environ.pop("TOKENSPEED_FORCE_LEGACY_KV_LAYOUT", None)
            else:
                os.environ["TOKENSPEED_FORCE_LEGACY_KV_LAYOUT"] = old_flag
            shutil.rmtree(tmpdir, ignore_errors=True)
        self.assertEqual(
            r1_leg, 0, f"legacy round 1 must be cold, got ratio {r1_leg:.3f}"
        )
        # ~1.44x the halved pool + same-order revisit recycles every cached
        # prefix before reuse; 0.2 is the collapse bound (expected ~0).
        self.assertLessEqual(
            r2_leg,
            0.2,
            f"legacy layout: round-2 hit ratio {r2_leg:.3f} above 0.2 -- "
            "the halved pool unexpectedly held the working set",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
