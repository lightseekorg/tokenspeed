import json
import shlex
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_CONFIG_DIR = REPO_ROOT / "test" / "ci" / "eval"
PERF_CONFIG_DIR = REPO_ROOT / "test" / "ci" / "perf"
STAGE_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "run-pr-test-stage.yml"
GLM53_FLASH_AMD_WORKFLOW_PATH = (
    REPO_ROOT / ".github" / "workflows" / "pr-test-glm-5.3-flash-amd.yml"
)
GLM53_FLASH_AMD_CONFIG_PATH = (
    EVAL_CONFIG_DIR / "glm-5.3-flash-fp8-mtp-tp4ep1-evalscope-aime26-amd.yaml"
)
GLM53_FLASH_NVIDIA_WORKFLOW_PATH = (
    REPO_ROOT / ".github" / "workflows" / "pr-test-glm-5.3-flash-nvidia.yml"
)
GLM53_FLASH_NVIDIA_CONFIG_PATH = (
    EVAL_CONFIG_DIR / "glm-5.3-flash-fp8-tp4ep4-evalscope-aime26-nvidia.yaml"
)
GLM53_FLASH_NVIDIA_SHAREGPT_CONFIG_PATH = (
    PERF_CONFIG_DIR / "glm-5.3-flash-fp8-mtp-tp4ep4-sharegpt-nvidia.yaml"
)
DEFAULT_B200_RUNNER_EXPRESSION = "${{ vars.TOKENSPEED_B200_RUNNER_LABEL || 'b200v2' }}"
GLM53_FLASH_BF16_AMD_CONFIG_PATH = (
    EVAL_CONFIG_DIR / "glm-5.3-flash-bf16-tp4ep1-evalscope-aime26-amd.yaml"
)
HF_HOME_ASSIGNMENT = "HF_HOME=${RUNNER_TEMP:-/tmp}/hf-eval-cache"
GLM53_FLASH_AMD_MODEL_ROOT = "/cache/huggingface/hub/horizon"
GLM53_FLASH_NVIDIA_MODEL_ROOT = "/raid/cache/huggingface/hub/horizon"
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


def test_glm53_flash_amd_workflow_runs_aime26_for_shared_branch_prs_only():
    workflow = yaml.safe_load(GLM53_FLASH_AMD_WORKFLOW_PATH.read_text(encoding="utf-8"))
    triggers = workflow.get("on") or workflow[True]
    matrix = json.loads(workflow["jobs"]["aime26"]["with"]["matrix"])

    assert set(triggers) == {"pull_request"}
    assert triggers["pull_request"]["branches"] == ["shared/glm5-next"]
    assert matrix["include"] == [
        {
            "name": "eval-glm-5.3-flash-fp8-mtp-tp4ep1-aime26-amd",
            "type": "eval",
            "config": (
                "test/ci/eval/" "glm-5.3-flash-fp8-mtp-tp4ep1-evalscope-aime26-amd.yaml"
            ),
            "runner": "amd-mi35x-4gpu-test",
            "priority": "normal",
            "optional": False,
            "workflow_stage": "model-test",
        }
    ]


def test_glm53_flash_amd_aime26_uses_validated_fp8_mtp_tp4_ep1_configuration():
    task = yaml.safe_load(GLM53_FLASH_AMD_CONFIG_PATH.read_text(encoding="utf-8"))
    server_command = task["server"]["command"]
    model_setup = task["install"][1]
    eval_install = task["eval"]["install"][0]
    generation_config = json.loads(
        flag_value(shlex.split(task["eval"]["command"]), "--generation-config")
    )

    assert task["runner"]["labels"] == ["amd-mi35x-4gpu-test"]
    assert task["triggers"] == ["per-commit", "manual"]
    assert "HF_HOME" not in task["env"]
    assert model_setup == f"test -f {GLM53_FLASH_AMD_MODEL_ROOT}/hf_fp8/config.json"
    assert server_command.startswith("ts serve")
    assert f"--model {GLM53_FLASH_AMD_MODEL_ROOT}/hf_fp8" in server_command
    assert "--attn-tp-size 4" in server_command
    assert "--ep-size 1" in server_command
    assert "--sampling-backend triton" in server_command
    assert "--force-deterministic-rsag" not in server_command
    assert "--engine-startup-timeout 900" in server_command
    assert task["server"]["ready"]["timeout"] == 1200
    assert "--max-model-len 65536" in server_command
    assert "--max-cudagraph-capture-size 16" in server_command
    assert "--disable-health-check" in server_command
    assert "--speculative-algorithm MTP" in server_command
    assert "--speculative-num-steps 3" in server_command
    assert "--speculative-eagle-topk 1" in server_command
    assert "--speculative-num-draft-tokens 4" in server_command
    assert "'evalscope[perf]==1.10.0'" in eval_install
    assert generation_config["seed"] == 42
    assert generation_config["max_tokens"] == 65000
    assert generation_config["extra_body"]["reasoning_effort"] == "max"
    assert task["score_threshold"] == 0.75


def test_glm53_flash_nvidia_workflow_runs_aime26_for_shared_branch_prs_only():
    workflow = yaml.safe_load(
        GLM53_FLASH_NVIDIA_WORKFLOW_PATH.read_text(encoding="utf-8")
    )
    triggers = workflow.get("on") or workflow[True]
    job = workflow["jobs"]["aime26"]
    matrix = json.loads(job["with"]["matrix"])

    assert set(triggers) == {"pull_request"}
    assert triggers["pull_request"]["branches"] == ["shared/glm5-next"]
    assert matrix["include"] == [
        {
            "name": "eval-glm-5.3-flash-fp8-tp4ep4-aime26-nvidia",
            "type": "eval",
            "config": (
                "test/ci/eval/" "glm-5.3-flash-fp8-tp4ep4-evalscope-aime26-nvidia.yaml"
            ),
            "runner": f"{DEFAULT_B200_RUNNER_EXPRESSION}-4gpu",
            "priority": "normal",
            "optional": False,
            "workflow_stage": "model-test",
        }
    ]
    assert job["with"]["b200_runner_label"] == DEFAULT_B200_RUNNER_EXPRESSION
    assert job["with"]["timeout_minutes"] >= 120


def test_glm53_flash_nvidia_aime26_uses_validated_tp4_ep4_configuration():
    task = yaml.safe_load(GLM53_FLASH_NVIDIA_CONFIG_PATH.read_text(encoding="utf-8"))
    server_command = task["server"]["command"]
    server_tokens = shlex.split(server_command)
    model_setup = task["install"][1:]
    eval_tokens = shlex.split(task["eval"]["command"])
    generation_config = json.loads(flag_value(eval_tokens, "--generation-config"))

    assert task["runner"]["labels"] == ["b200-4gpu"]
    assert "HF_HOME" not in task["env"]
    assert model_setup == [
        f"test -f {GLM53_FLASH_NVIDIA_MODEL_ROOT}/hf_fp8/config.json",
        f"test -f {GLM53_FLASH_NVIDIA_MODEL_ROOT}/hf_fp8/chat_template.jinja",
    ]
    assert server_tokens[0] == "ts"
    assert flag_value(server_tokens, "--model") == (
        f"{GLM53_FLASH_NVIDIA_MODEL_ROOT}/hf_fp8"
    )
    assert (
        flag_value(server_tokens, "--chat-template")
        == f"{GLM53_FLASH_NVIDIA_MODEL_ROOT}/hf_fp8/chat_template.jinja"
    )
    assert flag_value(server_tokens, "--attn-tp-size") == "4"
    assert "--enable-expert-parallel" in server_tokens
    assert "--language-model-only" in server_tokens
    assert flag_value(server_tokens, "--moe-backend") == "flashinfer_trtllm"
    assert "--all2all-backend" not in server_tokens
    assert "--deepep-mode" not in server_tokens
    assert flag_value(server_tokens, "--max-model-len") == "65536"
    assert flag_value(server_tokens, "--max-cudagraph-capture-size") == "16"
    assert flag_value(server_tokens, "--sampling-backend") == "flashinfer"
    assert "--disable-health-check" in server_tokens
    assert not any(token.startswith("--speculative-") for token in server_tokens)
    assert "--draft-model-path-use-base" not in server_tokens
    assert task["server"]["ready"]["timeout"] == 1800
    assert flag_value(eval_tokens, "--model") == "glm-5.3-flash"
    assert "'evalscope[perf]==1.10.0'" in task["eval"]["install"][0]
    assert generation_config["max_tokens"] == 32768
    assert generation_config["extra_body"]["reasoning_effort"] == "high"


def test_glm53_flash_nvidia_sharegpt_uses_fixed_mtp_configuration():
    task = yaml.safe_load(
        GLM53_FLASH_NVIDIA_SHAREGPT_CONFIG_PATH.read_text(encoding="utf-8")
    )
    server_tokens = shlex.split(task["server"]["command"])
    perf_tokens = shlex.split(task["perf"]["command"])

    assert [
        flag_value(server_tokens, flag)
        for flag in (
            "--attn-tp-size",
            "--speculative-num-steps",
            "--speculative-num-draft-tokens",
        )
    ] == ["4", "3", "4"]
    assert "--draft-model-path-use-base" in server_tokens
    assert flag_value(server_tokens, "--moe-backend") == "flashinfer_trtllm"
    assert "--all2all-backend" not in server_tokens
    assert "--disable-health-check" in server_tokens
    assert [
        flag_value(perf_tokens, flag)
        for flag in (
            "--dataset",
            "--number",
            "--min-tokens",
            "--max-tokens",
            "--parallel",
            "--warmup-num",
            "--seed",
            "--extra-args",
        )
    ] == ["share_gpt_en", "16", "512", "512", "16", "16", "45", '{"ignore_eos":true}']
    assert "--tokenize-prompt" in perf_tokens
    assert "'evalscope[perf]==1.10.0'" in task["perf"]["install"][0]
    assert "for run_id in 1 2 3" in task["perf"]["command"]
    assert "statistics.median" in task["perf"]["command"]
    assert task["perf_threshold"] == 0.9
    assert task["perf_reference"] == {16: [112.2, 280.4]}


def test_glm53_flash_bf16_mtp_manual_task_uses_validated_configuration():
    task = yaml.safe_load(GLM53_FLASH_BF16_AMD_CONFIG_PATH.read_text(encoding="utf-8"))
    server_command = task["server"]["command"]
    model_setup = task["install"][1]
    eval_install = task["eval"]["install"][0]
    generation_config = json.loads(
        flag_value(shlex.split(task["eval"]["command"]), "--generation-config")
    )

    assert task["triggers"] == ["manual"]
    assert task["runner"]["labels"] == ["amd-mi35x-4gpu-test"]
    assert "HF_HOME" not in task["env"]
    assert model_setup == f"test -f {GLM53_FLASH_AMD_MODEL_ROOT}/hf/config.json"
    assert server_command.startswith("TORCH_BLAS_PREFER_HIPBLASLT=0 ts serve")
    assert f"--model {GLM53_FLASH_AMD_MODEL_ROOT}/hf" in server_command
    assert "--attn-tp-size 4" in server_command
    assert "--ep-size 1" in server_command
    assert "--sampling-backend triton" in server_command
    assert "--max-model-len 65536" in server_command
    assert "--max-cudagraph-capture-size 16" in server_command
    assert "--speculative-algorithm MTP" in server_command
    assert "--speculative-num-steps 1" in server_command
    assert "--speculative-eagle-topk 1" in server_command
    assert "--speculative-num-draft-tokens 2" in server_command
    assert "'evalscope[perf]==1.10.0'" in eval_install
    assert generation_config["seed"] == 42
    assert generation_config["max_tokens"] == 32768
    assert task["score_threshold"] == 0.70


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
