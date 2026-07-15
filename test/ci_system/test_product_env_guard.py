import ast
from pathlib import Path

import pytest
from product_env_guard import (
    READ_ALLOWLIST,
    WRITE_ALLOWLIST,
    EnvironmentAccess,
    _EnvironmentVisitor,
    audit_repository,
    is_allowed,
    scan_cpp_source,
)

PR_TEST_WORKFLOWS = (
    ".github/workflows/pr-test-amd.yml",
    ".github/workflows/pr-test-nvidia-arm.yml",
    ".github/workflows/pr-test-nvidia.yml",
)


def _scan(source: str) -> list[tuple[str, str]]:
    tree = ast.parse(source)
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._product_env_parent = parent
    visitor = _EnvironmentVisitor("python/tokenspeed/example.py")
    visitor.visit(tree)
    return sorted((access.operation, access.key) for access in visitor.accesses)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('import os\nos.getenv("FEATURE")', [("read", "FEATURE")]),
        ('import os as system\nsystem.getenv("FEATURE")', [("read", "FEATURE")]),
        ('import os\nsystem = os\nsystem.getenv("FEATURE")', [("read", "FEATURE")]),
        (
            'from os import getenv as read_env\nread_env("FEATURE")',
            [("read", "FEATURE")],
        ),
        ('import os\nread_env = os.getenv\nread_env("FEATURE")', [("read", "FEATURE")]),
        (
            'import os\nread_env: object = os.getenv\nread_env("FEATURE")',
            [("read", "FEATURE")],
        ),
        ('import os\nos.getenvb(b"FEATURE")', [("read", "FEATURE")]),
        (
            'from os import getenvb as read_env\nread_env(b"FEATURE")',
            [("read", "FEATURE")],
        ),
        ('from os import environ as env\nenv.get("FEATURE")', [("read", "FEATURE")]),
        ('from os import environb as env\nenv.get(b"FEATURE")', [("read", "FEATURE")]),
        ('import os\nos.environ["FEATURE"]', [("read", "FEATURE")]),
        ('import os\nos.environb[b"FEATURE"]', [("read", "FEATURE")]),
        ('import os\nos.environ["FEATURE"] = "1"', [("write", "FEATURE")]),
        ('import os\ndel os.environ["FEATURE"]', [("delete", "FEATURE")]),
        ('import os\n"FEATURE" in os.environ', [("read", "FEATURE")]),
        ("import os\nos.environ == {}", [("dynamic", "<dynamic>")]),
        ("import os\nos.environ in mappings", [("dynamic", "<dynamic>")]),
        (
            'import os\nos.environ.setdefault("FEATURE", "1")',
            [("setdefault", "FEATURE")],
        ),
        ("import os\nos.environ.get(name)", [("read", "<dynamic>")]),
        ("import os\nos.environ.copy()", [("dynamic", "<dynamic>")]),
        ("import os\nfor key in os.environ:\n    pass", [("dynamic", "<dynamic>")]),
        ('import os\nos.putenv("FEATURE", "1")', [("write", "FEATURE")]),
        ('import os\nos.unsetenv("FEATURE")', [("delete", "FEATURE")]),
        (
            "from tokenspeed.runtime.utils.env import envs\nenvs.FEATURE.get()",
            [("product_env_field", "FEATURE")],
        ),
        ('get_bool_env_var("CI")', [("read", "CI")]),
    ],
)
def test_ast_scanner_finds_environment_accesses(source, expected):
    assert _scan(source) == expected


def test_non_writes_dynamic_and_product_fields_are_never_allowlisted():
    for operation, key in (
        ("setdefault", "TLLM_LOG_LEVEL"),
        ("delete", "TLLM_LOG_LEVEL"),
        ("read", "<dynamic>"),
        ("product_env_field", "TOKENSPEED_FEATURE"),
    ):
        access = EnvironmentAccess(
            path="python/tokenspeed/runtime/utils/server_args.py",
            line=1,
            column=0,
            operation=operation,
            key=key,
        )
        assert not is_allowed(access)


def test_delete_cannot_reuse_a_reviewed_write_contract():
    path, key = next(iter(WRITE_ALLOWLIST))
    access = EnvironmentAccess(
        path=path,
        line=1,
        column=0,
        operation="delete",
        key=key,
    )

    assert not is_allowed(access)


def test_cpp_scanner_tracks_environment_access_direction(tmp_path: Path):
    source = tmp_path / "tokenspeed-scheduler/csrc/runtime.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("""
        // std::getenv("COMMENT_ONLY")
        const char* message = "getenv(\\\"STRING_ONLY\\\")";
        const char* level = std::getenv("SPDLOG_LEVEL");
        const char* dynamic = ::secure_getenv(variable_name);
        setenv("SCHEDULER_MODE", "safe", 1);
        ::putenv("CACHE_LIMIT=4");
        unsetenv("OLD_MODE");
        _putenv_s(dynamic_name, "value");
        """)

    accesses = scan_cpp_source(source, repo_root=tmp_path)

    assert [(access.operation, access.key) for access in accesses] == [
        ("read", "SPDLOG_LEVEL"),
        ("read", "<dynamic>"),
        ("write", "SCHEDULER_MODE"),
        ("write", "CACHE_LIMIT"),
        ("delete", "OLD_MODE"),
        ("write", "<dynamic>"),
    ]


def test_repository_audit_covers_all_first_party_runtime_sources(tmp_path: Path):
    mla = tmp_path / "tokenspeed-mla/python/tokenspeed_mla/runtime.py"
    mla.parent.mkdir(parents=True)
    mla.write_text('import os\nos.getenv("TOKENSPEED_MLA_HIDDEN")\n')
    scheduler = tmp_path / "tokenspeed-scheduler/csrc/runtime.cpp"
    scheduler.parent.mkdir(parents=True)
    scheduler.write_text('auto* value = getenv("SCHEDULER_HIDDEN");\n')
    scheduler_python = (
        tmp_path / "tokenspeed-scheduler/python/tokenspeed_scheduler/runtime.py"
    )
    scheduler_python.parent.mkdir(parents=True)
    scheduler_python.write_text('import os\nos.getenv("SCHEDULER_PY_HIDDEN")\n')
    scheduler_binding = tmp_path / "tokenspeed-scheduler/bindings/python_module.cpp"
    scheduler_binding.parent.mkdir(parents=True)
    scheduler_binding.write_text('auto* value = getenv("BINDING_HIDDEN");\n')
    amd_kernel = (
        tmp_path / "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/runtime.py"
    )
    amd_kernel.parent.mkdir(parents=True)
    amd_kernel.write_text('import os\nos.getenv("AMD_KERNEL_HIDDEN")\n')

    result = audit_repository(tmp_path)
    violations = {(access["path"], access["key"]) for access in result["violations"]}

    assert (
        "tokenspeed-mla/python/tokenspeed_mla/runtime.py",
        "TOKENSPEED_MLA_HIDDEN",
    ) in violations
    assert (
        "tokenspeed-scheduler/csrc/runtime.cpp",
        "SCHEDULER_HIDDEN",
    ) in violations
    assert (
        "tokenspeed-scheduler/python/tokenspeed_scheduler/runtime.py",
        "SCHEDULER_PY_HIDDEN",
    ) in violations
    assert (
        "tokenspeed-scheduler/bindings/python_module.cpp",
        "BINDING_HIDDEN",
    ) in violations
    assert (
        "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/runtime.py",
        "AMD_KERNEL_HIDDEN",
    ) in violations


def test_allowlists_never_contain_product_namespaces():
    for _, key in READ_ALLOWLIST | WRITE_ALLOWLIST:
        assert not key.startswith(("TOKENSPEED_", "TS_", "SMG_", "EPD_"))


def test_audit_rejects_unreviewed_access_and_accepts_exact_contract(tmp_path: Path):
    package = tmp_path / "python/tokenspeed"
    package.mkdir(parents=True)
    source = package / "bench.py"
    source.write_text('import os\nvalue = os.environ.get("OPENAI_API_KEY")\n')

    result = audit_repository(tmp_path)
    assert result["violations"] == []
    assert result["reviewed_accesses"]

    source.write_text('import os\nvalue = os.environ.get("TOKENSPEED_HIDDEN")\n')
    result = audit_repository(tmp_path)
    assert result["ok"] is False
    assert result["violations"][0]["key"] == "TOKENSPEED_HIDDEN"


def test_repository_audit_reports_stale_allowlist_entries(tmp_path: Path):
    package = tmp_path / "python/tokenspeed"
    package.mkdir(parents=True)
    (package / "bench.py").write_text("value = 1\n")

    result = audit_repository(tmp_path)

    assert result["ok"] is False
    assert {
        "operation": "read",
        "path": "python/tokenspeed/bench.py",
        "key": "OPENAI_API_KEY",
    } in result["stale_allowlist"]


def test_repository_has_no_hidden_product_environment_configuration():
    repo_root = Path(__file__).resolve().parents[2]
    result = audit_repository(repo_root)

    assert result["ok"], result


@pytest.mark.parametrize("workflow_path", PR_TEST_WORKFLOWS)
def test_pr_workflow_scan_runs_product_environment_guard(workflow_path):
    repo_root = Path(__file__).resolve().parents[2]
    workflow = (repo_root / workflow_path).read_text()

    assert "python3 test/ci_system/product_env_guard.py" in workflow


@pytest.mark.parametrize("workflow_path", PR_TEST_WORKFLOWS)
def test_manual_pr_workflow_runs_install_in_tree_mla(workflow_path):
    repo_root = Path(__file__).resolve().parents[2]
    workflow = (repo_root / workflow_path).read_text()

    assert "workflow_dispatch)" in workflow
    assert 'files="tokenspeed-mla/"' in workflow
