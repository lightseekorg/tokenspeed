from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from tokenspeed_kernel.ops.gemm.triton_gemv import decode_gemv, use_decode_gemv
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.models import deepseek_v3

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_amd,
    reason="AMD GPU required",
)

# GLM-5.3 router: 256 experts over hidden 6144. Rows are the per-call token
# counts of decode: c1 draft/verify (1/4) and c16 draft/verify (16/64).
_GLM53_GATE = SimpleNamespace(
    n_routed_experts=256, hidden_size=6144, topk_method="noaux_tc"
)
_DECODE_ROWS = (1, 4, 16, 64)


def _make_gate() -> deepseek_v3.MoEGate:
    torch.manual_seed(0)
    gate = deepseek_v3.MoEGate(_GLM53_GATE)
    with torch.no_grad():
        gate.weight.normal_(std=0.02)
    return gate.to(device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("rows", _DECODE_ROWS)
def test_moe_gate_decode_rows_match_reference(rows: int) -> None:
    gate = _make_gate()
    x = torch.randn(rows, _GLM53_GATE.hidden_size, device="cuda").to(torch.bfloat16)
    eligible = use_decode_gemv(x, gate.weight)

    with mock.patch.object(deepseek_v3, "decode_gemv", wraps=decode_gemv) as route:
        logits = gate(x)

    assert route.called == eligible
    assert logits.dtype == torch.bfloat16
    reference = x.float() @ gate.weight.float().t()
    torch.testing.assert_close(logits.float(), reference, atol=2e-2, rtol=1.6e-2)
