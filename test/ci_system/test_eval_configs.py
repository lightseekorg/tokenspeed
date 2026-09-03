import json
import shlex
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_CONFIG_DIR = REPO_ROOT / "test" / "ci" / "eval"
PERF_CONFIG_DIR = REPO_ROOT / "test" / "ci" / "perf"
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
        "count": 11,
        "dataset_args": {"dataset_id": "math-ai/aime26"},
    },
    "gpqa_diamond": {
        "count": 2,
        "dataset_args": json.loads(GPQA_HUGGINGFACE_DATASET_ARGS)["gpqa_diamond"],
    },
    "gsm8k": {
        "count": 7,
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


def test_gpt_oss_gpqa_uses_runner_specific_batch_sizes_and_retries():
    path = EVAL_CONFIG_DIR / "gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))
    command = shlex.split(task["eval"]["command"])

    assert task["retries"] == 1
    assert task["runner"]["env"]["b200-2gpu"]["GPT_OSS_EVAL_BATCH_SIZE"] == "64"
    assert (
        task["runner"]["env"]["amd-mi35x-2gpu-test"]["GPT_OSS_EVAL_BATCH_SIZE"] == "16"
    )
    assert flag_value(command, "--eval-batch-size") == "${GPT_OSS_EVAL_BATCH_SIZE:-64}"


def test_qwen38_flash_next_runs_gsm8k_with_kvstore_enabled():
    path = EVAL_CONFIG_DIR / "qwen3.8-flash-next-fp8-evalscope-gsm8k.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))
    server_tokens = shlex.split(task["server"]["command"])
    eval_tokens = shlex.split(task["eval"]["command"])

    assert task["type"] == "eval"
    assert task["workflow_stage"] == "model-test"
    assert task["triggers"] == ["per-commit", "manual"]
    assert task["runner"]["labels"] == ["gb200-2gpu"]
    assert flag_value(server_tokens, "--model") == "Qwen/Qwen3.8-Flash-Next-FP8"
    assert flag_value(server_tokens, "--tensor-parallel-size") == "2"
    assert flag_value(server_tokens, "--speculative-algorithm") == "MTP"
    assert flag_value(server_tokens, "--speculative-num-steps") == "3"
    assert flag_value(server_tokens, "--max-model-len") == "8192"
    assert flag_value(server_tokens, "--max-num-seqs") == "16"
    assert flag_value(server_tokens, "--max-cudagraph-capture-size") == "16"
    assert "--disable-kvstore" not in server_tokens
    assert flag_value(eval_tokens, "--model") == "Qwen/Qwen3.8-Flash-Next-FP8"
    assert flag_value(eval_tokens, "--datasets") == "gsm8k"
    assert task["score_threshold"] == 0.90


def test_kimi_k3_amd_gates_use_eagle3():
    filenames = (
        "kimi-k3-eagle3-mxfp4-tp8ep1-evalscope-aime26-amd.yaml",
        "kimi-k3-eagle3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml",
    )
    ep_sizes = ("1", "8")
    tasks = []
    for config_dir, filename, ep_size in zip(
        (EVAL_CONFIG_DIR, PERF_CONFIG_DIR), filenames, ep_sizes, strict=True
    ):
        task = yaml.safe_load((config_dir / filename).read_text(encoding="utf-8"))
        server_tokens = shlex.split(task["server"]["command"])

        assert task["triggers"] == ["per-commit", "manual"]
        assert flag_value(server_tokens, "--speculative-algorithm") == "EAGLE3"
        assert (
            flag_value(server_tokens, "--speculative-draft-model-path")
            == "lightseekorg/kimi-k3-eagle3-mla"
        )
        assert flag_value(server_tokens, "--speculative-num-steps") == "3"
        assert flag_value(server_tokens, "--speculative-num-draft-tokens") == "4"
        assert flag_value(server_tokens, "--speculative-eagle-topk") == "1"
        assert flag_value(server_tokens, "--eagle3-layers-to-capture") == "2,46,90"
        assert flag_value(server_tokens, "--tp") == "8"
        assert flag_value(server_tokens, "--ep-size") == ep_size
        tasks.append(task)

    eval_tokens = shlex.split(tasks[0]["eval"]["command"])
    generation_config = json.loads(flag_value(eval_tokens, "--generation-config"))
    assert generation_config["seed"] == 42
    assert tasks[0]["score_threshold"] == 0.90
    assert tasks[1]["perf_reference"] == {1: [161, 18.8]}

    control_filenames = (
        "kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml",
        "kimi-k3-mxfp4-tp8ep8-evalscope-random-4k-1k-mi35x.yaml",
    )
    for config_dir, filename in zip(
        (EVAL_CONFIG_DIR, PERF_CONFIG_DIR), control_filenames, strict=True
    ):
        task = yaml.safe_load((config_dir / filename).read_text(encoding="utf-8"))
        assert task["triggers"] == ["manual"]


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
