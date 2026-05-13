import importlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def test_mla_prefill_backend_defaults_to_binary(monkeypatch):
    monkeypatch.delenv("TOKENSPEED_MLA_PREFILL_BACKEND", raising=False)

    import tokenspeed.runtime.utils.env as env_module

    env_module = importlib.reload(env_module)

    assert os.environ["TOKENSPEED_MLA_PREFILL_BACKEND"] == "binary"
    assert env_module.envs.TOKENSPEED_MLA_PREFILL_BACKEND.get() == "binary"


def test_mla_prefill_backend_preserves_user_env(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_MLA_PREFILL_BACKEND", "cutedsl")

    import tokenspeed.runtime.utils.env as env_module

    env_module = importlib.reload(env_module)

    assert os.environ["TOKENSPEED_MLA_PREFILL_BACKEND"] == "cutedsl"
    assert env_module.envs.TOKENSPEED_MLA_PREFILL_BACKEND.get() == "cutedsl"
