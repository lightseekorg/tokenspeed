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

"""Observable fallback and upstream API behavior of the checked GDN adapters."""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from tokenspeed_kernel.ops.attention.gdn.flashinfer import is_rubin_decode_available

pytestmark = pytest.mark.skipif(
    not is_rubin_decode_available(), reason="Audited FlashInfer on SM107 is required"
)


def _run(program, env):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "operation,change",
    [
        ("decode", "missing"),
        ("decode", "changed"),
        ("prefill", "changed"),
        ("decode", "version"),
    ],
)
def test_incompatible_package_falls_back_and_explicit_adapter_rejects(
    tmp_path, operation, change
):
    import flashinfer

    package = tmp_path / "flashinfer"
    # Real installed Python sources with one incompatible file. Symlink the
    # unchanged files; unlink before editing so the installation stays intact.
    shutil.copytree(
        Path(flashinfer.__file__).parent,
        package,
        copy_function=os.symlink,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    relative = (
        "gdn_kernels/gdn_decode_bf16_state_rubin.py"
        if operation == "decode"
        else "gdn_kernels/blackwell/gated_delta_net_chunked_sm107.py"
    )
    if change == "version":
        relative = "_build_meta.py"
    path = package / relative
    original = path.read_text()
    path.unlink()
    if change == "changed":
        path.write_text(original + "\n# unreviewed upstream modification\n")
    elif change == "version":
        path.write_text(original.replace('"0.7.2.post1"', '"0.7.2"'))
    env = dict(
        os.environ,
        PYTHONPATH=f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        TEST_GDN_OPERATION=operation,
    )
    # No inherited override should force the unavailable candidate in this
    # automatic-selection test.
    env.pop("TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_GDN_CHUNK_PREFILL", None)
    env.pop("TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_GDN_DECODE_STEP", None)
    _run(
        """
        import os
        import torch
        from tokenspeed_kernel.thirdparty.flashinfer.gdn_compat import compatibility_error
        from tokenspeed_kernel.ops.attention.gdn import gdn_decode_step, gdn_chunk_prefill
        from tokenspeed_kernel.ops.attention.gdn.flashinfer import is_available, is_rubin_decode_available
        operation = os.environ['TEST_GDN_OPERATION']
        reason = compatibility_error(operation)
        assert reason is not None
        if operation == 'decode':
            assert not is_rubin_decode_available()
            from tokenspeed_kernel.thirdparty.flashinfer.gdn_rubin import gated_delta_rule_t1_rubin
            try:
                gated_delta_rule_t1_rubin(enable_pdl=False)
            except RuntimeError as error:
                assert str(error) == reason
            else:
                raise AssertionError('incompatible decode was allowed')
            q = torch.zeros(1, 1, 4, 128, device='cuda', dtype=torch.bfloat16)
            v = torch.zeros(1, 1, 12, 128, device='cuda', dtype=torch.bfloat16)
            gate = torch.zeros(1, 1, 12, device='cuda', dtype=torch.bfloat16)
            parameter = torch.zeros(12, device='cuda')
            state = torch.ones(1, 12, 128, 128, device='cuda', dtype=torch.bfloat16)
            rows = torch.zeros(1, device='cuda', dtype=torch.int32)
            out = gdn_decode_step(q, q, v, A_log=parameter, a=gate, b=gate,
                dt_bias=parameter, initial_state=state, initial_state_indices=rows,
                output_state_indices=rows, scale=128**-0.5, use_qk_l2norm=True)
            torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
            torch.testing.assert_close(state, torch.full_like(state, 0.5), atol=0, rtol=0)
        else:
            assert not is_available()
            from tokenspeed_kernel.ops.attention.gdn._flashinfer.adapter import chunk_gated_delta_rule
            try:
                chunk_gated_delta_rule(enable_pdl=True)
            except RuntimeError as error:
                assert str(error) == reason
            else:
                raise AssertionError('incompatible prefill was allowed')
            q = torch.zeros(1, 64, 4, 128, device='cuda', dtype=torch.bfloat16)
            v = torch.zeros(1, 64, 12, 128, device='cuda', dtype=torch.bfloat16)
            gate = torch.zeros(1, 64, 12, device='cuda')
            state = torch.zeros(1, 12, 128, 128, device='cuda')
            result = gdn_chunk_prefill(q, q, v, gate, gate, scale=128**-0.5,
                initial_state=state, cu_seqlens=torch.tensor([0, 64], device='cuda', dtype=torch.int32),
                qk_l2norm=False, output_final_state=True, output_h=False)
            torch.testing.assert_close(result.out, torch.zeros_like(result.out), atol=0, rtol=0)
            torch.testing.assert_close(result.final_state, state, atol=0, rtol=0)
    """,
        env,
    )


@pytest.mark.parametrize("variant", ["sm100", "sm107"])
def test_prefill_preserves_logging_dumps_trace_and_validation(tmp_path, variant):
    env = dict(
        os.environ,
        FLASHINFER_LOGLEVEL="10",
        FLASHINFER_LOGDEST=str(tmp_path / "api.log"),
        FLASHINFER_DUMP_DIR=str(tmp_path / "dumps"),
        FLASHINFER_DUMP_INCLUDE="chunk_gated_delta_rule",
        FLASHINFER_GDN_SM107_PREFILL_KERNEL=variant,
    )
    _run(
        """
        import torch
        from flashinfer.gdn_prefill import chunk_gated_delta_rule as upstream
        from tokenspeed_kernel.ops.attention.gdn._flashinfer.adapter import _prefill_runner
        q = torch.zeros(64, 4, 128, device='cuda', dtype=torch.bfloat16)
        v = torch.zeros(64, 12, 128, device='cuda', dtype=torch.bfloat16)
        g = torch.full((64, 12), 0.5, device='cuda')
        beta = torch.full((64, 12), 0.5, device='cuda')
        state = torch.zeros(1, 12, 128, 128, device='cuda')
        bounds = torch.tensor([0, 64], device='cuda', dtype=torch.int32)
        inputs = dict(q=q, k=q, v=v, g=g, beta=beta, initial_state=state,
                      cu_seqlens=bounds, output_final_state=True, use_cp=False)
        adapted = _prefill_runner(True)
        # Execute the trace callable, not just check that an attribute exists.
        assert adapted.fi_trace(**inputs) == upstream.fi_trace(**inputs)
        out, final = adapted(**inputs)
        torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
        torch.testing.assert_close(final, state, atol=0, rtol=0)
        bad = dict(inputs, state_checkpoints=torch.zeros(1, 12, 128, 128, device='cuda', dtype=torch.int32),
                   checkpoint_cu_starts=torch.tensor([0, 1], device='cuda', dtype=torch.int32),
                   checkpoint_every_n_tokens=64)
        for run in (upstream, adapted):
            try:
                run(**bad)
            except ValueError as error:
                assert 'state_checkpoints must have dtype' in str(error)
            else:
                raise AssertionError('upstream validation was lost')
    """,
        env,
    )
    assert "chunk_gated_delta_rule" in (tmp_path / "api.log").read_text()
    # Dump records must contain a completed call, not just a pre-launch input.
    records = [
        json.loads(line)
        for path in (tmp_path / "dumps").rglob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert any(
        record.get("execution_status") == "completed" for record in records
    ), records
