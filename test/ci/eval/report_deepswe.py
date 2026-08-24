import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--expected-tasks", type=int, default=113)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    total = int(result["n_total_trials"])
    stats = result["stats"]
    completed = int(stats["n_completed_trials"])
    errors = int(stats["n_errored_trials"])
    if total != args.expected_tasks or completed != total:
        raise RuntimeError(
            f"incomplete DeepSWE run: completed={completed}, total={total}, "
            f"expected={args.expected_tasks}"
        )

    evals = list(stats["evals"].values())
    if len(evals) != 1:
        raise RuntimeError(f"expected one DeepSWE eval group, found {len(evals)}")
    rewards = evals[0]["reward_stats"].get("reward", {})
    passed = sum(float(value) * len(names) for value, names in rewards.items())
    score = passed / total

    print(f"DeepSWE completed={completed} errors={errors}")
    print("Overall report table:")
    print("| Model | Dataset | Metric | Score |")
    print("|---|---|---|---|")
    print(f"| kimi-k3 | deep-swe | mean_reward | {score:.6f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
