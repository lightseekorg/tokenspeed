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

"""Check serving recipes without importing kernels or starting a worker."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent


@pytest.fixture
def launch(tmp_path):
    stub = tmp_path / "python"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps(dict(argv=sys.argv[1:], queue=os.environ.get('TASK_QUEUE_ENABLE'))))\n"
    )
    stub.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        MODEL="/local/model",
        SERVED_MODEL_NAME="test",
        HOST="127.0.0.1",
        PORT="30000",
        FLATQUANT_CONFIG="/local/flatquant.json",
        DIST_INIT_ADDR="127.0.0.1:30001",
        BOOTSTRAP_PORT="30002",
        ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, range(16))),
        PREFILL_URL="grpc://prefill:30000",
        DECODE_URL="grpc://decode:30000",
        PROMETHEUS_PORT="30003",
    )

    def run(script, *args):
        return subprocess.run(
            ["bash", str(SCRIPTS / script), *args],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    return run


@pytest.mark.parametrize(
    "layout,tp,capture",
    [("dp2", "8", list(map(str, range(1, 33)))), ("dp4", "4", ["1", "2", "4", "8"])],
)
def test_decode_layout_and_legacy_wrapper(launch, layout, tp, capture):
    result = launch(f"serve_deepseek_v41_flash_npu_{layout}_pd.sh", "decode")
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    argv = output["argv"]
    for key in ["--tensor-parallel-size", "--dense-tp-size", "--lm-head-tp-size"]:
        assert argv[argv.index(key) + 1] == tp
    assert argv[argv.index("--data-parallel-size") + 1] == layout[-1]
    assert argv[argv.index("--engram-tp-size") + 1] == "16"
    assert argv[argv.index("--moe-backend") + 1] == "ascend_w4a4"
    assert argv[argv.index("--draft-moe-backend") + 1] == "ascend_bf16"
    first = argv.index("--cudagraph-capture-sizes") + 1
    assert argv[first : first + len(capture)] == capture
    assert argv[first + len(capture)] == "--max-cudagraph-capture-size"
    assert argv[argv.index("--max-cudagraph-capture-size") + 1] == capture[-1]
    assert argv[argv.index("--max-num-seqs") + 1] == ("64" if layout == "dp2" else "32")
    assert "--disable-prefix-caching" in argv
    assert "--disable-cuda-graph-padding" not in argv
    assert output["queue"] == "0"
    explicit = launch(
        "serve_deepseek_v41_flash_npu_dp_pd.sh", "decode", layout, "performance"
    )
    assert json.loads(explicit.stdout) == output


def test_prefill_and_gateway_keep_role_contracts(launch):
    for layout in ["dp2", "dp4"]:
        script = f"serve_deepseek_v41_flash_npu_{layout}_pd.sh"
        prefill = launch(script, "prefill")
        assert prefill.returncode == 0, prefill.stderr
        argv = json.loads(prefill.stdout)["argv"]
        assert argv[argv.index("--tensor-parallel-size") + 1] == "16"
        assert "--data-parallel-size" not in argv
        assert "--disable-prefix-caching" not in argv
        gateway = launch(script, "gateway")
        assert gateway.returncode == 0, gateway.stderr
        argv = json.loads(gateway.stdout)["argv"]
        assert "--dp-aware" in argv and "--disable-retries" in argv


@pytest.mark.parametrize(
    "args",
    [
        ("decode", "dp8"),
        ("unknown", "dp2"),
        ("decode",),
        ("decode", "dp2", "partitioned"),
    ],
)
def test_invalid_profile_never_starts_worker(launch, args):
    result = launch("serve_deepseek_v41_flash_npu_dp_pd.sh", *args)
    assert result.returncode != 0
    assert not result.stdout


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("profile", ["performance", "precision"])
def test_numerical_profiles(launch, role, profile):
    result = launch("serve_deepseek_v41_flash_npu_dp_pd.sh", role, "dp2", profile)
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)["argv"]
    performance = profile == "performance"
    mode = "bf16" if performance else "bf16-qdq"
    for flag in ("--npu-v41-kv-cache", "--npu-v41-index-cache"):
        assert argv[argv.index(flag) + 1] == mode
    assert argv[argv.index("--moe-backend") + 1] == (
        "ascend_w4a4" if performance else "ascend_bf16"
    )
    assert ("--speculative-algorithm" in argv) == performance
    assert ("--npu-flatquant-config" in argv) == performance
    assert "--disable-cuda-graph-padding" not in argv
