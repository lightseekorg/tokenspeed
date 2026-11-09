from pathlib import Path

import pytest
from runtime_test_selection import select_runtime_tests

MODELS = "python/tokenspeed/runtime/models"
TESTS = "test/runtime"
FILES = {
    f"{MODELS}/alpha.py": "EntryClass = Alpha\n",
    # Built on alpha.
    f"{MODELS}/beta.py": (
        "from tokenspeed.runtime.models.alpha import Alpha\nEntryClass = [Beta]\n"
    ),
    f"{MODELS}/gamma.py": "EntryClass = Gamma\n",
    f"{MODELS}/helper.py": "",
    f"{MODELS}/utils.py": "",
    f"{TESTS}/test_alpha.py": "from tokenspeed.runtime.models.alpha import Alpha\n",
    f"{TESTS}/models/test_beta.py": "import tokenspeed.runtime.models.beta\n",
    f"{TESTS}/test_gamma.py": "def test():\n    from tokenspeed.runtime.models import gamma\n",
    f"{TESTS}/test_misc.py": "import torch\n",
}
ALL = sorted(path for path in FILES if path.startswith(TESTS))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for name, text in FILES.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    return tmp_path


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ([f"{MODELS}/alpha.py"], ["test_alpha", "test_beta"]),
        ([f"{MODELS}/beta.py"], ["test_beta"]),
        ([f"{MODELS}/gamma.py"], ["test_gamma"]),
        ([f"{TESTS}/test_misc.py"], ["test_misc"]),
        (
            [
                f"{TESTS}/test_misc.py",
                "docs/a.md",
                "tokenspeed-kernel/test/ops/test_gemm.py",
            ],
            ["test_misc"],
        ),
        (["tokenspeed-kernel/test/ops/test_gemm.py"], []),
    ],
)
def test_changes_select_the_runtime_tests_they_reach(repo, changed, expected):
    selected = select_runtime_tests(ALL, changed, repo)

    assert sorted(Path(path).stem for path in selected) == expected


@pytest.mark.parametrize(
    "changed",
    [
        # A module without EntryClass is shared by the models.
        f"{MODELS}/helper.py",
        f"{MODELS}/utils.py",
        f"{MODELS}/deleted.py",
        f"{TESTS}/conftest.py",
        "tokenspeed-kernel/test/utils.py",
        "python/tokenspeed/runtime/layers/linear.py",
    ],
)
def test_other_changes_run_every_test(repo, changed):
    assert select_runtime_tests(ALL, [f"{MODELS}/gamma.py", changed], repo) is None
