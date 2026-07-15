import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("collect_outputs.py")
SPEC = importlib.util.spec_from_file_location(
    "random_benchmark_collect_outputs", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
collect_outputs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collect_outputs)


def test_collect_uses_ci_throughput_column(tmp_path):
    summary_dir = tmp_path / "input_1k" / "run"
    summary_dir.mkdir(parents=True)
    (summary_dir / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "TPOT (ms)": 20,
                "Output Throughput (tok/s)": 400,
                "Concurrency": 8,
                "KV Cache Hit Rate (%)": 75,
                "Avg Decoded Tokens/Iter": 2.5,
            }
        )
    )

    rows = collect_outputs.collect(tmp_path, num_gpus=4)

    assert rows == [
        {
            "config": "input_1k",
            "Conc.": 8,
            "Latency (tps/user)": 50.0,
            "Throughput (tps/gpu)": 100.0,
            "Approx Cache Hit": 75.0,
            "Decoded Tok/Iter": 2.5,
        }
    ]


def test_print_table_uses_ci_throughput_header(capsys):
    row = {column: 1 for column in collect_outputs.COLUMNS}

    collect_outputs.print_table([row])

    output = capsys.readouterr().out
    lines = output.strip().splitlines()
    assert lines[0] == "Overall perf table:"
    assert lines[1] == ",".join(collect_outputs.COLUMNS)
    assert lines[2] == "1,1,1,1,1,1"
    assert "Output Throughput (tps/gpu)" not in output


def test_input_32k_sorts_after_shorter_known_configs():
    rows = [
        {"config": "input_32k", "Conc.": 1},
        {"config": "input_8k", "Conc.": 1},
        {"config": "input_1k", "Conc.": 1},
    ]

    rows.sort(key=collect_outputs._sort_key)

    assert [row["config"] for row in rows] == [
        "input_1k",
        "input_8k",
        "input_32k",
    ]
