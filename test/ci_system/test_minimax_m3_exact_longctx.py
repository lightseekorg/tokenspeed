from pathlib import Path

from minimax_m3_exact_longctx import (
    EXPECTED_OUTPUT_IDS,
    ExactLongContextConfig,
    ExactLongContextContract,
    analyze_request_log,
    run,
    validate_response,
)


def _body(contract: ExactLongContextContract, request_id: str = "gen-test"):
    return [
        {
            "output_ids": list(contract.expected_output_ids),
            "text": contract.expected_text,
            "meta_info": {
                "id": request_id,
                "prompt_tokens": contract.prompt_tokens,
                "completion_tokens": contract.output_tokens,
                "cached_tokens": 0,
            },
        }
    ]


def test_release_output_token_is_immutable() -> None:
    assert EXPECTED_OUTPUT_IDS == (123,)


def test_response_contract_passes_and_rejects_wrong_token() -> None:
    contract = ExactLongContextContract()

    passed = validate_response(200, _body(contract), contract)
    wrong = _body(contract)
    wrong[0]["output_ids"] = [124]
    failed = validate_response(200, wrong, contract)

    assert passed["failures"] == []
    assert any("output_ids" in failure for failure in failed["failures"])


def test_request_log_requires_exact_chunks_finish_and_no_critical_lines() -> None:
    contract = ExactLongContextContract(
        prompt_tokens=7,
        output_tokens=1,
        expected_output_ids=(123,),
        expected_text="{",
        chunk_size=4,
        expected_full_chunks=1,
        expected_tail_tokens=3,
    )
    clean_log = "#new-token: 4,\n#new-token: 3,\nReq: gen-test Finish!\n"
    dirty_log = (
        clean_log + "Retract failed: host capacity exhausted, aborting request\n"
    )

    passed = analyze_request_log(clean_log, request_id="gen-test", contract=contract)
    failed = analyze_request_log(dirty_log, request_id="gen-test", contract=contract)

    assert passed["failures"] == []
    assert passed["chunk_log"]["actual_full_count"] == 1
    assert failed["critical_log_check"]["total_matches"] == 3
    assert any("critical" in failure for failure in failed["failures"])


class _FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _FakeSession:
    def __init__(self, server_log: Path, contract: ExactLongContextContract) -> None:
        self.server_log = server_log
        self.contract = contract
        self.post_payload = None

    def post(self, url, *, json, timeout):
        assert url.endswith("/generate")
        assert timeout == 30
        self.post_payload = json
        self.server_log.write_text(
            self.server_log.read_text()
            + "#new-token: 4,\n#new-token: 3,\nReq: gen-test Finish!\n"
        )
        return _FakeResponse(200, _body(self.contract))

    def get(self, _url, *, timeout):
        assert timeout == 30
        return _FakeResponse(200, {})


class _FakeMonitor:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def test_run_writes_all_artifacts_without_a_real_gpu_or_server(tmp_path: Path) -> None:
    contract = ExactLongContextContract(
        prompt_tokens=7,
        output_tokens=1,
        expected_output_ids=(123,),
        expected_text="{",
        chunk_size=4,
        expected_full_chunks=1,
        expected_tail_tokens=3,
    )
    server_log = tmp_path / "server.log"
    server_log.write_text("startup complete\n")
    output_dir = tmp_path / "artifacts"
    session = _FakeSession(server_log, contract)
    monitor = _FakeMonitor()

    def monitor_factory(path, gpu_ids):
        assert gpu_ids == (0, 1, 2, 3)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("0, 100\n1, 101\n2, 102\n3, 103\n")
        return monitor

    config = ExactLongContextConfig(
        model="minimax-m3",
        base_url="http://gateway",
        control_url="http://control",
        server_log=server_log,
        output_dir=output_dir,
        gpu_ids=(0, 1, 2, 3),
        request_timeout_seconds=30,
        memory_limit_mib=1000,
    )

    result = run(
        config,
        contract=contract,
        session=session,  # type: ignore[arg-type]
        monitor_factory=monitor_factory,
    )

    assert result["ok"] is True
    assert len(session.post_payload["input_ids"]) == 7
    assert monitor.stopped is True
    for name in (
        "response.json",
        "post_health.json",
        "request_server_window.log",
        "memory.csv",
        "validation.json",
    ):
        assert (output_dir / name).is_file()


def test_memory_limit_is_a_hard_gate(tmp_path: Path) -> None:
    contract = ExactLongContextContract(
        prompt_tokens=7,
        output_tokens=1,
        expected_output_ids=(123,),
        expected_text="{",
        chunk_size=4,
        expected_full_chunks=1,
        expected_tail_tokens=3,
    )
    server_log = tmp_path / "server.log"
    server_log.write_text("startup complete\n")
    session = _FakeSession(server_log, contract)

    def monitor_factory(path, _gpu_ids):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("0, 1001\n1, 1\n2, 1\n3, 1\n")
        return _FakeMonitor()

    result = run(
        ExactLongContextConfig(
            model="minimax-m3",
            base_url="http://gateway",
            control_url="http://control",
            server_log=server_log,
            output_dir=tmp_path / "artifacts",
            gpu_ids=(0, 1, 2, 3),
            request_timeout_seconds=30,
            memory_limit_mib=1000,
        ),
        contract=contract,
        session=session,  # type: ignore[arg-type]
        monitor_factory=monitor_factory,
    )

    assert result["ok"] is False
    assert result["gpu_memory"]["within_limit"] is False
    assert any("exceeds" in failure for failure in result["failures"])
