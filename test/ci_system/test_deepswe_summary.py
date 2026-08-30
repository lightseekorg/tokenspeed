import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "ci" / "deepswe" / "summarize.py"
SPEC = importlib.util.spec_from_file_location("deepswe_summary", MODULE_PATH)
assert SPEC and SPEC.loader
summary_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary_module)


def result(*, total=2, completed=2, errors=0, cancelled=0, reward=0.5):
    return {
        "n_total_trials": total,
        "stats": {
            "n_completed_trials": completed,
            "n_errored_trials": errors,
            "n_cancelled_trials": cancelled,
            "evals": {
                "kimi-code__kimi-k3__tasks": {
                    "metrics": [{"reward": reward}],
                    "exception_stats": {},
                }
            },
        },
    }


def test_deepswe_summary_passes_complete_result():
    summary, problems = summary_module.build_summary(result(reward=0.7), 0.6)

    assert problems == []
    assert "Trials completed: 2/2" in summary
    assert "Binary reward: 0.7000" in summary
    assert "DeepSWE gate passed" in summary


def test_deepswe_summary_rejects_errors_and_incomplete_trials():
    data = result(total=3, completed=2, errors=1)
    data["stats"]["evals"]["kimi-code__kimi-k3__tasks"]["exception_stats"] = {
        "RuntimeError": ["one"]
    }

    summary, problems = summary_module.build_summary(data, 0.0)

    assert len(problems) == 2
    assert "only 2/3 trials completed" in problems
    assert "1 trial(s) ended" in problems[1]
    assert "`RuntimeError`: 1" in summary


def test_deepswe_summary_enforces_minimum_score():
    _, problems = summary_module.build_summary(result(reward=0.4), 0.5)

    assert problems == ["binary reward 0.4000 is below minimum 0.5000"]


def test_deepswe_summary_falls_back_to_trial_rewards():
    data = result()
    data["stats"]["evals"]["kimi-code__kimi-k3__tasks"]["metrics"] = []
    data["trial_results"] = [
        {"verifier_result": {"rewards": {"reward": 1}}},
        {"verifier_result": {"rewards": {"reward": 0}}},
    ]

    summary, problems = summary_module.build_summary(data, 0.5)

    assert problems == []
    assert "Binary reward: 0.5000" in summary
