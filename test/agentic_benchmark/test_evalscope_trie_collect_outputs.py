from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path


def load_collector():
    path = Path(__file__).resolve().parent / "evalscope_trie" / "collect_outputs.py"
    spec = importlib.util.spec_from_file_location("evalscope_trie_collect", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_db(path: Path):
    con = sqlite3.connect(path)
    try:
        con.execute("create table result(success integer)")
        con.executemany("insert into result(success) values (?)", [(1,), (1,), (0,)])
        con.commit()
    finally:
        con.close()


def test_collect_outputs_computes_jiying_style_axes(tmp_path):
    collector = load_collector()
    run_dir = tmp_path / "sweep" / "DeepSeek-V4-Pro" / "parallel_4_number_32"
    write_json(
        run_dir / "benchmark_summary.json",
        {
            "Concurrency": 4,
            "Total Requests": 3,
            "Success Requests": 2,
            "Failed Requests": 1,
            "Avg Latency (s)": 1.5,
            "TTFT (ms)": 250.0,
            "TPOT (ms)": 6.0,
            "Avg Input Tokens": 1000,
            "Avg Output Tokens": 200,
            "Avg Turns/Request": 3,
            "KV Cache Hit Rate (%)": 91.5,
            "First-Turn TTFT (ms)": 500.0,
            "Subsequent-Turn TTFT (ms)": 200.0,
            "Decoded Tok/Iter": 3.2,
            "Spec. Accept Rate": 0.71,
        },
    )
    write_json(
        run_dir / "workload_throughput.json",
        {"rows": [{"metric": "Completion tok/s", "steady_state": 320.0}]},
    )
    write_json(run_dir / "trace_summary.json", {"n_traces": 32})
    write_db(run_dir / "benchmark_data.db")

    rows = collector.collect(tmp_path, num_gpus=8)

    assert len(rows) == 1
    row = rows[0]
    assert row["phase"] == "sweep"
    assert row["model"] == "DeepSeek-V4-Pro"
    assert row["parallel"] == 4
    assert row["number"] == 32
    assert row["success"] == "2/3"
    assert row["steady_completion_tok_s"] == 320.0
    assert row["completion_tps_per_user"] == 80.0
    assert row["output_token_min_per_gpu"] == 2400.0
    assert row["avg_ttft_s"] == 0.25
    assert row["first_turn_ttft_s"] == 0.5
    assert row["decoded_tok_iter"] == 3.2
    assert row["spec_accept_rate"] == 0.71
