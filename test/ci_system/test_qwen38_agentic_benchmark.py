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

import json
import os
import runpy
import shutil
import signal
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pipeline import extract_perf_summary_rows, validate_task

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "test/agentic_benchmark/Qwen3.8-Flash-Next/tokenspeed"
CI_TASK = (
    REPO_ROOT / "test/ci/perf/qwen3.8-flash-next-nvfp4-tp4ep4-evalscope-agentic.yaml"
)
CONCURRENCIES = [1, 2, 4, 8, 16, 32]
CONFIG = "attn_tp4_moe_ep4"
MODEL_ID = "nvidia/Qwen3.8-Flash-Next-NVFP4"
MODEL_REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"


class CharacterChatTokenizer:
    def __init__(self, *, return_dict: bool):
        self.return_dict = return_dict

    def encode(self, content, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(char) for char in content]

    def decode(self, ids, *, skip_special_tokens):
        return "".join(chr(token) for token in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        ids = [0] * (5 + sum(len(m["content"]) + 3 for m in messages))
        return {"input_ids": ids} if self.return_dict else ids


@pytest.mark.parametrize("return_dict", [False, True])
def test_dataset_recounts_with_model_tokenizer_and_includes_chat_overhead(return_dict):
    builder = runpy.run_path(str(BENCH_DIR / "build_dataset.py"))
    source = {
        "metadata": {"model_path": "moonshotai/Kimi-K2.5"},
        "conversations": [
            [
                {
                    "messages": [{"role": "user", "content": "x" * 60}],
                    "prompt_tokens": 999,
                }
                for _ in range(turns)
            ]
            for turns in [3, 2]
        ],
    }
    tokenizer = CharacterChatTokenizer(return_dict=return_dict)

    result = builder["build_dataset"](
        source,
        tokenizer=tokenizer,
        model_path=MODEL_ID,
        tokenizer_revision="model-revision",
        source_sha256="source-sha",
        source_revision="source-revision",
        first_turn_length=32,
        subsequent_turn_length=12,
    )

    assert [len(c) for c in result["conversations"]] == [3, 2]
    for conversation in result["conversations"]:
        prefix = []
        previous = 0
        for turn_index, turn in enumerate(conversation):
            prefix.extend(turn["messages"])
            actual = builder["prompt_tokens"](prefix, tokenizer)
            assert turn["prompt_tokens"] == actual
            assert actual == (32 if turn_index == 0 else previous + 12)
            previous = actual
    assert result["metadata"]["model_path"] == MODEL_ID
    assert result["metadata"]["tokenizer_revision"] == "model-revision"
    assert source["conversations"][0][0]["messages"][0]["content"] == "x" * 60


def test_dataset_rejects_unsupported_source_messages():
    builder = runpy.run_path(str(BENCH_DIR / "build_dataset.py"))
    with pytest.raises(ValueError, match="text user messages"):
        builder["trim_turn"](
            [{"role": "user", "content": [{"type": "image_url"}]}],
            prefix=[],
            prefix_tokens=0,
            tokenizer=CharacterChatTokenizer(return_dict=False),
            max_prompt_tokens=32,
        )


def test_dataset_replaces_kimi_cache_and_rebuilds_changed_budget(tmp_path, monkeypatch):
    builder = runpy.run_path(str(BENCH_DIR / "build_dataset.py"))
    source = tmp_path / "source.json"
    output = tmp_path / "dataset.json"
    source.write_text(
        json.dumps(
            {
                "conversations": [
                    [
                        {
                            "messages": [{"role": "user", "content": "x" * 60}],
                            "prompt_tokens": 999,
                        }
                    ]
                ]
            }
        )
    )
    output.write_text(json.dumps({"metadata": {"model_path": "moonshotai/Kimi-K2.5"}}))
    loads = []

    def load_tokenizer(path, *, revision, trust_remote_code):
        loads.append((path, revision))
        return CharacterChatTokenizer(return_dict=True)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer)),
    )
    argv = [
        "build_dataset.py",
        "--source-path",
        str(source),
        "--output-path",
        str(output),
        "--model-path",
        "qwen",
        "--tokenizer-path",
        str(tmp_path / "tokenizer"),
        "--tokenizer-revision",
        "revision",
        "--source-revision",
        "source-revision",
        "--first-turn-length",
        "32",
        "--subsequent-turn-length",
        "12",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    builder["main"]()
    assert json.loads(output.read_text())["conversations"][0][0]["prompt_tokens"] == 32
    builder["main"]()
    assert loads == [(str(tmp_path / "tokenizer"), "revision")]
    argv[argv.index("--first-turn-length") + 1] = "31"
    builder["main"]()
    assert len(loads) == 2
    assert json.loads(output.read_text())["conversations"][0][0]["prompt_tokens"] == 31


@pytest.fixture
def sweep(tmp_path):
    for concurrency in CONCURRENCIES:
        run_dir = tmp_path / CONFIG / f"parallel_{concurrency}"
        run_dir.mkdir(parents=True)
        (run_dir / "benchmark_summary.json").write_text(
            json.dumps(
                {
                    "Concurrency": concurrency,
                    "TPOT (ms)": 10.0,
                    "Total Throughput (tok/s)": 4000.0,
                    "KV Cache Hit Rate (%)": 87.123,
                    "Failed Requests": 0,
                }
            ),
            encoding="utf-8",
        )
        connection = sqlite3.connect(run_dir / "benchmark_data.db")
        try:
            connection.execute(
                "CREATE TABLE result (completion_tokens INTEGER, "
                "inter_token_latencies TEXT, success INTEGER)"
            )
            connection.executemany(
                "INSERT INTO result VALUES (?, ?, ?)",
                [
                    (9, "[0.1, 0.1]", 1),
                    (5, "[0.1, 0.1, 0.1, 0.1]", 1),
                    (100, "[0.1]", 0),
                    (1, "[]", 1),
                ],
            )
            connection.commit()
        finally:
            connection.close()
    return tmp_path


def collect(sweep_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(BENCH_DIR / "collect_outputs.py"),
            str(sweep_dir),
            "--expected-concurrencies",
            *(str(concurrency) for concurrency in CONCURRENCIES),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("tpot_key", ["TPOT (ms)", "Avg TPOT (ms)"])
def test_collector_preserves_kimi_metrics_and_ci_csv(sweep, tpot_key):
    for summary_path in sweep.rglob("benchmark_summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary[tpot_key] = summary.pop("TPOT (ms)")
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    # Warmup artifacts must not become measured results.
    warmup = sweep / ".warmup" / "benchmark_summary.json"
    warmup.parent.mkdir()
    warmup.write_text("{}", encoding="utf-8")

    result = collect(sweep)

    assert result.returncode == 0, result.stderr
    rows = extract_perf_summary_rows(result.stdout)
    assert [int(row["Conc."]) for row in rows] == CONCURRENCIES
    assert rows[0] == {
        "config": CONFIG,
        "Conc.": "1",
        "Latency (tps/user)": "100.0",
        "Throughput (tps/gpu)": "1000.0",
        "Approx Cache Hit": "87.12",
        # (8 + 4) / (2 + 4), not the per-request mean (4 + 1) / 2.
        "Decoded Tok/Iter": "2.0",
    }


@pytest.mark.parametrize(
    "failure", ["missing_concurrency", "failed_request", "missing_failure_count"]
)
def test_collector_rejects_unusable_sweeps(sweep, failure):
    run_dir = sweep / CONFIG / "parallel_32"
    if failure == "missing_concurrency":
        shutil.rmtree(run_dir)
    else:
        summary_path = run_dir / "benchmark_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if failure == "missing_failure_count":
            del summary["Failed Requests"]
        else:
            summary["Failed Requests"] = 1
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = collect(sweep)

    assert result.returncode != 0
    assert extract_perf_summary_rows(result.stdout) is None


def write_executable(path: Path, source: str) -> None:
    path.write_text(f"#!{sys.executable}\n{textwrap.dedent(source)}", encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def mock_benchmark(tmp_path):
    scripts = tmp_path / "scripts"
    shutil.copytree(
        BENCH_DIR,
        scripts,
        ignore=shutil.ignore_patterns(".venv*", "outputs", "data", "__pycache__"),
    )
    # Exercise the real launch, replay, collection and cleanup without downloads.
    (scripts / "prepare.sh").write_text("exit 0\n", encoding="utf-8")
    (scripts / "outputs").mkdir()
    (scripts / "data").mkdir()
    (scripts / "data/agentic_dataset.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "model_path": MODEL_ID,
                    "tokenizer_revision": MODEL_REVISION,
                    "first_turn_length": 50000,
                    "subsequent_turn_length": 800,
                    "chat_template_included": True,
                },
                "conversations": [[] for _ in range(71)],
            }
        ),
        encoding="utf-8",
    )
    tokenizer_dir = scripts / "data/tokenizer" / MODEL_REVISION
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    perf_bin = scripts / ".venv-evalscope/bin"
    perf_bin.mkdir(parents=True)
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    env = {
        **os.environ,
        "PATH": f"{tools_dir}:{os.environ['PATH']}",
        "SERVER_PIDS": str(tmp_path / "server_pids.json"),
        "PERF_CALLS": str(tmp_path / "perf_calls.jsonl"),
    }
    write_executable(
        tools_dir / "python3",
        """
        import os
        import sys

        # Isolate the preflight port check from any server on the test machine.
        if sys.argv[1:] == ['-']:
            sys.exit(0)
        os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
        """,
    )
    write_executable(
        tools_dir / "ts",
        """
        import json
        import os
        import signal
        import subprocess
        import sys
        import time
        from pathlib import Path

        if os.environ.get('PRINT_SERVER_ARGS') == '1':
            print(json.dumps({
                'args': sys.argv[1:],
                'qsa_max_logits_mb': os.environ.get('TOKENSPEED_QWEN4_EXP_QSA_MAX_LOGITS_MB'),
            }))
            sys.exit(0)
        worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        Path(os.environ['SERVER_PIDS']).write_text(json.dumps([os.getpid(), worker.pid]))
        if os.environ['FAILURE'] == 'server':
            print('server startup failed', flush=True)
            sys.exit(7)
        def stop(signum, frame):
            worker.wait(timeout=5)
            sys.exit(0)
        signal.signal(signal.SIGTERM, stop)
        time.sleep(60)
        """,
    )
    write_executable(
        tools_dir / "curl",
        """
        import os
        import sys
        import time
        from pathlib import Path

        for _ in range(300):
            if Path(os.environ['SERVER_PIDS']).is_file():
                sys.exit(1 if os.environ['FAILURE'] == 'server' else 0)
            time.sleep(0.01)
        sys.exit(1)
        """,
    )
    write_executable(
        perf_bin / "evalscope",
        """
        import json
        import os
        import sys
        from pathlib import Path

        args = sys.argv[1:]
        with Path(os.environ['PERF_CALLS']).open('a') as output:
            output.write(json.dumps(args) + '\\n')
        warmup = '--dataset-offset' in args
        if os.environ['FAILURE'] == ('warmup' if warmup else 'perf'):
            sys.exit(17)
        if not warmup:
            root = Path(args[args.index('--outputs-dir') + 1])
            levels = []
            for value in args[args.index('--parallel') + 1:]:
                if value.startswith('--'):
                    break
                levels.append(int(value))
            for concurrency in levels:
                run_dir = root / 'attn_tp4_moe_ep4' / f'parallel_{concurrency}'
                run_dir.mkdir(parents=True)
                (run_dir / 'benchmark_summary.json').write_text(json.dumps({
                    'Concurrency': concurrency,
                    'TPOT (ms)': 10,
                    'Total Throughput (tok/s)': 4000,
                    'KV Cache Hit Rate (%)': 85,
                    'Failed Requests': 0,
                }))
        """,
    )
    return scripts, env


@pytest.mark.parametrize("missing_command", ["ts", "curl", "setsid"])
def test_one_click_reports_missing_required_command(
    mock_benchmark, tmp_path, missing_command
):
    scripts, env = mock_benchmark
    bash = shutil.which("bash", path=env["PATH"])
    assert bash is not None
    commands_dir = tmp_path / "commands"
    commands_dir.mkdir()
    for command in ("dirname", "ts", "curl", "setsid"):
        if command != missing_command:
            executable = shutil.which(command, path=env["PATH"])
            assert executable is not None
            (commands_dir / command).symlink_to(executable)
    env["PATH"] = str(commands_dir)
    env["VIRTUAL_ENV"] = sys.prefix
    env.pop("BASH_ENV", None)

    result = subprocess.run(
        [bash, str(scripts / "agentic_bench.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 1
    assert f"Missing required command: {missing_command}" in result.stderr
    assert not Path(env["SERVER_PIDS"]).exists()
    assert not Path(env["PERF_CALLS"]).exists()


@pytest.mark.parametrize(
    ("failure", "status", "calls"),
    [("none", 0, 2), ("warmup", 17, 1), ("perf", 17, 2), ("server", 1, 0)],
)
def test_one_click_flow_cleans_up_workers(mock_benchmark, failure, status, calls):
    scripts, env = mock_benchmark
    env["FAILURE"] = failure
    try:
        result = subprocess.run(
            ["bash", str(scripts / "agentic_bench.sh")],
            cwd=scripts.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == status, result.stdout + result.stderr
        calls_path = Path(env["PERF_CALLS"])
        perf_calls = (
            [json.loads(line) for line in calls_path.read_text().splitlines()]
            if calls_path.exists()
            else []
        )
        assert len(perf_calls) == calls
        if calls:
            warmup = perf_calls[0]
            assert warmup[warmup.index("--dataset-offset") + 1] == "68"
            assert warmup[warmup.index("--number") + 1] == "2"
            assert warmup[warmup.index("--parallel") + 1] == "2"
            assert warmup[warmup.index("--tokenizer-path") + 1] == str(
                scripts / "data/tokenizer" / MODEL_REVISION
            )
        if failure == "none":
            benchmark = perf_calls[1]
            assert benchmark[
                benchmark.index("--parallel") + 1 : benchmark.index("--parallel") + 7
            ] == ["1", "2", "4", "8", "16", "32"]
            assert benchmark[
                benchmark.index("--number") + 1 : benchmark.index("--number") + 7
            ] == ["4", "4", "4", "8", "16", "32"]
            assert benchmark[benchmark.index("--tokenizer-path") + 1] == str(
                scripts / "data/tokenizer" / MODEL_REVISION
            )
            assert "--dataset-offset" not in benchmark
            assert len(extract_perf_summary_rows(result.stdout)) == 6
            assert len(list(scripts.glob("outputs/*/summary.csv"))) == 1
        else:
            assert not list(scripts.glob("outputs/*/summary.csv"))
        pids = json.loads(Path(env["SERVER_PIDS"]).read_text())
        for pid in pids:
            stat = Path(f"/proc/{pid}/stat")
            assert not stat.exists() or stat.read_text().split()[2] == "Z"
    finally:
        pids_path = Path(env["SERVER_PIDS"])
        if pids_path.exists():
            server_pid = json.loads(pids_path.read_text())[0]
            try:
                os.killpg(server_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_ci_uses_tp4ep4_config_and_preserves_launcher_arguments(mock_benchmark):
    scripts, env = mock_benchmark
    task = yaml.safe_load(CI_TASK.read_text(encoding="utf-8"))
    validate_task(task, CI_TASK)
    env["PRINT_SERVER_ARGS"] = "1"

    result = subprocess.run(
        ["bash", str(scripts / "configs/attn_tp4_moe_ep4.sh"), "--log-level", "debug"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    server = json.loads(result.stdout)
    assert server["qsa_max_logits_mb"] == "2048"
    tokens = server["args"]
    for flag, value in {
        "--model": MODEL_ID,
        "--revision": MODEL_REVISION,
        "--tokenizer": str(scripts / "data/tokenizer" / MODEL_REVISION),
        "--tokenizer-path": str(scripts / "data/tokenizer" / MODEL_REVISION),
        "--attn-tp-size": "4",
        "--ep-size": "4",
        "--kv-cache-dtype": "bfloat16",
        "--max-prefill-tokens": "16384",
        "--chunked-prefill-size": "16384",
        "--max-num-seqs": "32",
        "--speculative-algorithm": "MTP",
        "--speculative-num-steps": "3",
    }.items():
        assert tokens[tokens.index(flag) + 1] == value
    assert json.loads(tokens[tokens.index("--hf-overrides") + 1]) == {
        "index_share_for_mtp_iteration": True
    }
    assert "--quantization" not in tokens  # Preserve mixed checkpoint metadata.
    assert "--disable-kvstore" in tokens
    assert not {
        "--max-model-len",
        "--kvstore-size",
        "--disable-cuda-graph-padding",
        "--disable-replay-ssm",  # Benchmark uses the server's default ReplaySSM.
        "--reasoning-parser",
        "--tool-call-parser",
    }.intersection(tokens)
    assert tokens[-2:] == ["--log-level", "debug"]
    assert task["server"]["command"] == (
        "bash test/agentic_benchmark/Qwen3.8-Flash-Next/tokenspeed/"
        "configs/attn_tp4_moe_ep4.sh"
    )
    assert "run_benchmark.sh" in task["perf"]["command"]
    assert any("prepare.sh" in command for command in task["install"])


@pytest.mark.parametrize("mismatch", ["model", "revision", "conversation_count"])
def test_benchmark_rejects_incompatible_dataset_before_sending_requests(
    mock_benchmark, mismatch
):
    scripts, env = mock_benchmark
    path = scripts / "data/agentic_dataset.json"
    data = json.loads(path.read_text())
    if mismatch == "model":
        data["metadata"]["model_path"] = "moonshotai/Kimi-K2.5"
    elif mismatch == "revision":
        data["metadata"]["tokenizer_revision"] = "different-revision"
    else:
        data["conversations"] = data["conversations"][:69]
    path.write_text(json.dumps(data), encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(scripts / "run_benchmark.sh"),
            str(scripts / ".venv-evalscope/bin/evalscope"),
            str(path),
            str(scripts / "outputs/rejected"),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode != 0
    assert not Path(env["PERF_CALLS"]).exists()
    assert not list((scripts / "outputs/rejected").rglob("summary.csv"))
