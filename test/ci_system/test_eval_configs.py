import json
import shlex
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_CONFIG_DIR = REPO_ROOT / "test" / "ci" / "eval"
STAGE_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "run-pr-test-stage.yml"
HF_HOME_ASSIGNMENT = "HF_HOME=${RUNNER_TEMP:-/tmp}/hf-eval-cache"
FORK_PR_EXPRESSION = (
    "${{ github.event_name == 'pull_request' && "
    "github.event.pull_request.head.repo.full_name != github.repository }}"
)
GPQA_HUGGINGFACE_DATASET_ARGS = (
    '{"gpqa_diamond":{"dataset_id":"Idavidrein/gpqa",'
    '"subset_list":["gpqa_diamond"]}}'
)
GPQA_MODELSCOPE_DATASET_ARGS = (
    '{"gpqa_diamond":{"dataset_id":"AI-ModelScope/gpqa_diamond"}}'
)
GPQA_DATASET_SOURCE_PRELUDE = (
    'if [ "${TOKENSPEED_CI_FORK_PR:-false}" = "true" ]; '
    "then GPQA_DATASET_HUB=modelscope; "
    f"GPQA_DATASET_ARGS='{GPQA_MODELSCOPE_DATASET_ARGS}'; "
    "else GPQA_DATASET_HUB=huggingface; "
    f"GPQA_DATASET_ARGS='{GPQA_HUGGINGFACE_DATASET_ARGS}'; "
    "fi;"
)
DATASETS = {
    "aime25": {
        "count": 10,
        "dataset_args": {"dataset_id": "math-ai/aime25"},
    },
    "aime26": {
        "count": 8,
        "dataset_args": {"dataset_id": "math-ai/aime26"},
    },
    "gpqa_diamond": {
        "count": 2,
        "dataset_args": json.loads(GPQA_HUGGINGFACE_DATASET_ARGS)["gpqa_diamond"],
    },
    "gsm8k": {
        "count": 4,
        "dataset_args": {"dataset_id": "openai/gsm8k"},
    },
    "mmlu": {
        "count": 1,
        "dataset_args": {"dataset_id": "cais/mmlu"},
    },
    "ocr_bench": {
        "count": 3,
        "dataset_args": {"dataset_id": "echo840/OCRBench"},
    },
}

KVV_REVISION = "3dad65a760a8867cda72f6dd8848d876a4e851b4"
KVV_CONFIGS = {
    "kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml": (
        "ocrbench",
        "16384",
    ),
    "kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml": (
        "mmmu",
        "98304",
    ),
}

DEEPSEEK_V4_FLASH_CONFIG = "deepseek-v4-flash-evalscope-gsm8k.yaml"
DEEPSEEK_V4_PRO_AMD_CONFIG = "deepseek-v4-pro-evalscope-gsm8k-amd.yaml"


def flag_value(tokens: list[str], flag: str) -> str:
    assert tokens.count(flag) == 1, f"expected one {flag}, found {tokens.count(flag)}"
    index = tokens.index(flag)
    return tokens[index + 1]


def test_fork_pr_context_is_exposed_to_ci_tasks():
    workflow = yaml.safe_load(STAGE_WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert (
        workflow["jobs"]["test"]["env"]["TOKENSPEED_CI_FORK_PR"] == FORK_PR_EXPRESSION
    )


def test_evalscope_configs_use_expected_dataset_sources():
    counts = Counter()
    paths = []

    for path in sorted(EVAL_CONFIG_DIR.glob("*.yaml")):
        task = yaml.safe_load(path.read_text(encoding="utf-8"))
        command = task.get("eval", {}).get("command", "")
        if "evalscope eval" not in command:
            continue

        paths.append(path)
        tokens = shlex.split(command)
        executable_indexes = [
            i for i, token in enumerate(tokens) if token.endswith("/evalscope")
        ]
        assert len(executable_indexes) == 1, (
            f"{path}: expected one EvalScope invocation, "
            f"found {len(executable_indexes)}"
        )
        executable_index = executable_indexes[0]
        assert tokens[executable_index - 1] == HF_HOME_ASSIGNMENT, path
        assert tokens[executable_index + 1] == "eval", path

        dataset = flag_value(tokens, "--datasets")
        assert dataset in DATASETS, f"{path}: missing mapping for {dataset}"
        expected = DATASETS[dataset]
        counts[dataset] += 1

        if dataset == "gpqa_diamond":
            assert GPQA_DATASET_SOURCE_PRELUDE in command, path
            assert flag_value(tokens, "--dataset-hub") == "$GPQA_DATASET_HUB", path
            assert flag_value(tokens, "--dataset-args") == "$GPQA_DATASET_ARGS", path
            assert json.loads(GPQA_MODELSCOPE_DATASET_ARGS) == {
                dataset: {"dataset_id": "AI-ModelScope/gpqa_diamond"}
            }
            dataset_args = json.loads(GPQA_HUGGINGFACE_DATASET_ARGS)
        else:
            assert flag_value(tokens, "--dataset-hub") == expected.get(
                "dataset_hub", "huggingface"
            ), path
            dataset_args = json.loads(flag_value(tokens, "--dataset-args"))
        assert dataset_args == {dataset: expected["dataset_args"]}, path

    expected_counts = Counter(
        {dataset: item["count"] for dataset, item in DATASETS.items()}
    )
    expected_total = sum(expected_counts.values())
    assert (
        len(paths) == expected_total
    ), f"expected {expected_total} EvalScope configs, found {len(paths)}"
    assert counts == expected_counts, f"expected {expected_counts}, found {counts}"


def test_deepseek_v4_flash_config_uses_platform_specific_backends():
    path = EVAL_CONFIG_DIR / DEEPSEEK_V4_FLASH_CONFIG
    task = yaml.safe_load(path.read_text(encoding="utf-8"))
    command = shlex.split(task["server"]["command"])

    assert task["triggers"] == ["per-commit", "manual"]
    assert task["runner"]["labels"] == ["b200-4gpu", "amd-mi35x-2gpu-test"]
    assert task["install"] == ["bash test/ci_system/install_deps.sh"]
    assert flag_value(command, "--model") == "deepseek-ai/DeepSeek-V4-Flash"
    assert flag_value(command, "--tensor-parallel-size") == "${DEEPSEEK_V4_TP_SIZE}"
    assert flag_value(command, "--max-model-len") == "4096"
    assert flag_value(command, "--max-total-tokens") == "16384"
    assert flag_value(command, "--chunked-prefill-size") == "8192"
    assert task["server"]["ready"]["timeout"] == 1800

    runner_env = task["runner"]["env"]
    nvidia_args = shlex.split(runner_env["b200-4gpu"]["DEEPSEEK_V4_PLATFORM_ARGS"])
    assert runner_env["b200-4gpu"]["DEEPSEEK_V4_TP_SIZE"] == "4"
    assert "--enable-expert-parallel" in nvidia_args
    assert flag_value(nvidia_args, "--moe-backend") == "mega_moe"
    assert "--attention-use-fp4-indexer-cache" in nvidia_args

    amd_args = shlex.split(
        runner_env["amd-mi35x-2gpu-test"]["DEEPSEEK_V4_PLATFORM_ARGS"]
    )
    assert runner_env["amd-mi35x-2gpu-test"]["DEEPSEEK_V4_TP_SIZE"] == "2"
    assert flag_value(amd_args, "--prefill-graph-max-tokens") == "8192"
    assert not {
        "--enable-expert-parallel",
        "--moe-backend",
        "--attention-use-fp4-indexer-cache",
    }.intersection(amd_args)

    assert not {
        "--max-num-seqs",
        "--speculative-algorithm",
        "--speculative-num-steps",
    }.intersection(command)


def test_deepseek_v4_pro_amd_config_uses_supported_defaults():
    omitted_flags = {
        "--attention-use-fp4-indexer-cache",
        "--enable-mixed-batch",
        "--enable-prefix-caching",
        "--no-enable-prefix-caching",
        "--enable-expert-parallel",
        "--max-num-seqs",
        "--moe-backend",
        "--speculative-algorithm",
        "--speculative-draft-model-path",
        "--speculative-num-steps",
    }
    path = EVAL_CONFIG_DIR / DEEPSEEK_V4_PRO_AMD_CONFIG
    task = yaml.safe_load(path.read_text(encoding="utf-8"))
    command = shlex.split(task["server"]["command"])

    assert task["triggers"] == ["per-commit", "manual"]
    assert task["runner"]["labels"] == ["amd-mi35x-8gpu-test"]
    assert task["install"] == ["bash test/ci_system/install_deps.sh"]
    assert flag_value(command, "--model") == "deepseek-ai/DeepSeek-V4-Pro"
    assert flag_value(command, "--tensor-parallel-size") == "8"
    assert flag_value(command, "--kv-cache-dtype") == "fp8_e4m3"
    assert flag_value(command, "--max-model-len") == "4096"
    assert flag_value(command, "--max-total-tokens") == "16384"
    assert flag_value(command, "--chunked-prefill-size") == "8192"
    assert flag_value(command, "--prefill-graph-max-tokens") == "8192"
    assert task["server"]["ready"]["timeout"] == 1800
    assert not omitted_flags.intersection(command)


def test_kvv_configs_use_pinned_upstream_and_local_api():
    for filename, (benchmark, max_tokens) in KVV_CONFIGS.items():
        path = EVAL_CONFIG_DIR / filename
        task = yaml.safe_load(path.read_text(encoding="utf-8"))
        install = task["eval"]["install"][0]
        command = shlex.split(task["eval"]["command"])
        script_index = command.index("/tmp/kvv/eval.py")

        assert KVV_REVISION in install
        assert "uv sync --project /tmp/kvv --frozen" in install
        assert command[script_index + 1] == benchmark
        assert "KIMI_BASE_URL=http://127.0.0.1:8000/v1" in command
        assert flag_value(command, "--model") == "opensource/kimi-k3"
        assert flag_value(command, "--max-tokens") == max_tokens
        assert "--thinking" in command
        assert flag_value(command, "--thinking-effort") == "max"
