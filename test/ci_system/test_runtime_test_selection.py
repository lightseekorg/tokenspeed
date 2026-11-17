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
    # A module without EntryClass built on gamma.
    f"{MODELS}/gamma_deepep.py": "from tokenspeed.runtime.models.gamma import Gamma\n",
    f"{MODELS}/gpt_oss.py": "EntryClass = GptOss\n",
    f"{MODELS}/utils.py": "",
    f"{TESTS}/test_alpha.py": "from tokenspeed.runtime.models.alpha import Alpha\n",
    f"{TESTS}/models/test_beta.py": "import tokenspeed.runtime.models.beta\n",
    f"{TESTS}/test_gamma.py": "def test():\n    from tokenspeed.runtime.models import gamma\n",
    f"{TESTS}/test_misc.py": "import torch\n",
    f"{TESTS}/test_gamma_deepep.py": "import tokenspeed.runtime.models.gamma_deepep\n",
    # Serves its model by checkpoint name.
    f"{TESTS}/test_serving_e2e.py": 'MODEL = "openai/gpt-oss-20b"\n',
    f"{TESTS}/test_uses_misc.py": "from test.runtime.test_misc import helper\n",
    f"{TESTS}/test_uses_misc_bare.py": "import test_misc\n",
    # Not in the suite, yet it links test_two_hop to test_misc.
    f"{TESTS}/test_unregistered.py": "import test.runtime.test_misc\n",
    f"{TESTS}/test_two_hop.py": "from runtime.test_unregistered import helper\n",
    # Imports a test file that the change deletes.
    f"{TESTS}/test_uses_deleted.py": "from test.runtime.test_deleted import helper\n",
}
ALL = sorted(
    path
    for path in FILES
    if path.startswith(TESTS) and path != f"{TESTS}/test_unregistered.py"
)


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
        ([f"{MODELS}/gamma.py"], ["test_gamma", "test_gamma_deepep"]),
        ([f"{MODELS}/gpt_oss.py"], ["test_serving_e2e"]),
        ([f"{TESTS}/test_deleted.py"], ["test_uses_deleted"]),
        (
            [f"{TESTS}/test_misc.py"],
            ["test_misc", "test_two_hop", "test_uses_misc", "test_uses_misc_bare"],
        ),
        (
            [
                f"{TESTS}/test_misc.py",
                "docs/a.md",
                "tokenspeed-kernel/test/ops/test_gemm.py",
            ],
            ["test_misc", "test_two_hop", "test_uses_misc", "test_uses_misc_bare"],
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
