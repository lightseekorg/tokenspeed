import json
import shlex
from collections import Counter
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_CONFIG_DIR = REPO_ROOT / "test" / "ci" / "eval"
PERF_CONFIG_DIR = REPO_ROOT / "test" / "ci" / "perf"
STAGE_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "run-ci-task-matrix.yml"
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
MUSE_SPARK_AIME26_CONFIGS = {
    (4, 4): "muse-spark-mxfp4-tp4ep4-evalscope-aime26-amd.yaml",
    (8, 8): "muse-spark-mxfp4-tp8ep8-evalscope-aime26-amd.yaml",
    (8, 1): "muse-spark-mxfp4-tp8ep1-evalscope-aime26-amd.yaml",
}
MUSE_SPARK_AIME26_PROMPT = (
    "Solve the following math problem. Show your reasoning step by step.\n"
    "Enclose only your final answer in standard LaTeX boxed-answer notation. "
    "Once you provide the final answer, stop immediately.\n\n"
    "Problem: {question}"
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
        "count": 8,
        "dataset_args": {"dataset_id": "math-ai/aime25"},
    },
    "aime26": {
        "count": 15,
        "dataset_args": {"dataset_id": "math-ai/aime26"},
    },
    "gpqa_diamond": {
        "count": 3,
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


@pytest.mark.parametrize(
    "path",
    sorted((REPO_ROOT / "test" / "ci").rglob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_model_configs_reuse_shared_downloads(path):
    task = yaml.safe_load(path.read_text(encoding="utf-8"))
    server_tokens = shlex.split(task.get("server", {}).get("command", ""))
    assert not any(
        token.split("=", 1)[0] == "--download-dir" for token in server_tokens
    ), path

    perf_command = task.get("perf", {}).get("command", "")
    if "--tokenizer-path" in perf_command:
        perf_tokens = shlex.split(perf_command)
        assert flag_value(perf_tokens, "--tokenizer-path") == "$TOKENIZER_PATH", path
        assert "TOKENIZER_PATH=$OUTPUTS_DIR/tokenizer" in perf_tokens, path
        assert "AutoTokenizer.from_pretrained" in perf_command, path
        assert ".save_pretrained(" in perf_command, path


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
        expected_dataset_args = {dataset: expected["dataset_args"]}
        if path.name in MUSE_SPARK_AIME26_CONFIGS.values():
            expected_dataset_args = {
                dataset: {
                    **expected["dataset_args"],
                    "prompt_template": MUSE_SPARK_AIME26_PROMPT,
                }
            }
        assert dataset_args == expected_dataset_args, path

    expected_counts = Counter(
        {dataset: item["count"] for dataset, item in DATASETS.items()}
    )
    expected_total = sum(expected_counts.values())
    assert (
        len(paths) == expected_total
    ), f"expected {expected_total} EvalScope configs, found {len(paths)}"
    assert counts == expected_counts, f"expected {expected_counts}, found {counts}"


@pytest.mark.parametrize(
    "config,runner,model,batch_size,trigger",
    [
        (
            "gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml",
            "b200-2gpu",
            "openai/gpt-oss-120b",
            "64",
            "post-submit",
        ),
        # MI35x has hit GPU memory-access faults at eval batch size 64.
        (
            "gpt-oss-120b-mxfp4-evalscope-gpqa-diamond-amd.yaml",
            "amd-mi35x-2gpu-test",
            "amd/gpt-oss-120b-w-mxfp4-a-fp8",
            "16",
            "post-submit",
        ),
    ],
)
def test_gpt_oss_gpqa_uses_runner_specific_configuration(
    config, runner, model, batch_size, trigger
):
    task = yaml.safe_load((EVAL_CONFIG_DIR / config).read_text(encoding="utf-8"))
    server = shlex.split(task["server"]["command"])
    command = shlex.split(task["eval"]["command"])

    assert task["retries"] == 1
    assert task["triggers"] == [trigger, "manual"]
    assert task["runner"]["labels"] == [runner]
    assert flag_value(server, "--model") == model
    assert flag_value(command, "--model") == model
    assert flag_value(command, "--eval-batch-size") == batch_size


@pytest.mark.parametrize("tp_size,ep_size", MUSE_SPARK_AIME26_CONFIGS)
def test_muse_spark_aime26_uses_validated_mxfp4_configuration(tp_size, ep_size):
    path = EVAL_CONFIG_DIR / MUSE_SPARK_AIME26_CONFIGS[tp_size, ep_size]
    task = yaml.safe_load(path.read_text(encoding="utf-8"))
    server_tokens = shlex.split(task["server"]["command"])
    eval_tokens = shlex.split(task["eval"]["command"])
    generation_config = json.loads(flag_value(eval_tokens, "--generation-config"))

    expected_triggers = (
        ["per-commit", "manual"] if (tp_size, ep_size) == (8, 1) else ["manual"]
    )
    assert task["triggers"] == expected_triggers
    assert task["runner"]["labels"] == [f"amd-mi35x-{tp_size}gpu-test"]
    assert (
        flag_value(server_tokens, "--model")
        == "amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4"
    )
    assert flag_value(server_tokens, "--tp") == str(tp_size)
    assert flag_value(server_tokens, "--ep-size") == str(ep_size)
    moe_tp_size = tp_size // ep_size
    if moe_tp_size > 1:
        assert flag_value(server_tokens, "--moe-tp-size") == str(moe_tp_size)
    else:
        assert "--moe-tp-size" not in server_tokens
    assert flag_value(server_tokens, "--quantization") == "mxfp4"
    assert flag_value(server_tokens, "--moe-backend") == "gluon"
    assert flag_value(server_tokens, "--max-num-seqs") == "16"
    assert flag_value(server_tokens, "--max-total-tokens") == "1048576"
    assert flag_value(server_tokens, "--max-prefill-tokens") == "8192"
    assert flag_value(server_tokens, "--chunked-prefill-size") == "8192"
    assert flag_value(server_tokens, "--max-cudagraph-capture-size") == "16"
    assert "--disable-prefill-graph" in server_tokens
    assert "--disable-overlap-schedule" not in server_tokens
    assert flag_value(eval_tokens, "--eval-batch-size") == "8"
    assert generation_config["max_tokens"] == (
        63488 if (tp_size, ep_size) == (8, 1) else 32768
    )
    assert generation_config["extra_body"]["reasoning_effort"] == "high"
    assert task["score_threshold"] == 0.91


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
    assert task["score_threshold"] == 0.96


def test_qwen38_flash_next_mxfp4_runs_gsm8k_on_amd_tp2():
    path = EVAL_CONFIG_DIR / "qwen3.8-flash-next-mxfp4-evalscope-gsm8k-amd.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))
    server_tokens = shlex.split(task["server"]["command"])
    eval_tokens = shlex.split(task["eval"]["command"])

    assert task["runner"]["labels"] == ["amd-mi35x-2gpu-test"]
    assert task["install"] == ["bash test/ci_system/install_deps_rocm.sh"]
    model = "amd/Qwen3.8-Flash-Next-Quark-MXFP4"
    assert flag_value(server_tokens, "--model") == model
    assert flag_value(server_tokens, "--tensor-parallel-size") == "2"
    assert flag_value(server_tokens, "--quantization") == "mxfp4"
    assert flag_value(server_tokens, "--speculative-algorithm") == "MTP"
    assert flag_value(server_tokens, "--speculative-num-steps") == "3"
    assert flag_value(eval_tokens, "--model") == model
    assert flag_value(eval_tokens, "--datasets") == "gsm8k"
    assert task["score_threshold"] == 0.96


def test_deepseek_v4_flash_and_pro_are_not_in_ci_catalog():
    for path in (REPO_ROOT / "test" / "ci").rglob("*.yaml"):
        config = path.read_text(encoding="utf-8")
        assert "DeepSeek-V4-Flash" not in config, path
        assert "DeepSeek-V4-Pro" not in config, path


def test_deepseek_v41_flash_runs_tp4_gsm8k_on_b200_and_mi35x():
    filenames = (
        "deepseek-v4.1-flash-dspark-evalscope-gsm8k.yaml",
        "deepseek-v4.1-flash-dspark-evalscope-gsm8k-amd.yaml",
    )
    labels = ("b200-4gpu", "amd-mi35x-4gpu-test")
    names = (
        "eval-deepseek-v4.1-flash-dspark-gsm8k",
        "eval-deepseek-v4.1-flash-dspark-gsm8k-amd",
    )
    model = "deepseek-ai/DeepSeek-V4.1-Flash-1017"

    for filename, label, name in zip(filenames, labels, names, strict=True):
        task = yaml.safe_load((EVAL_CONFIG_DIR / filename).read_text(encoding="utf-8"))
        server_tokens = shlex.split(task["server"]["command"])
        eval_tokens = shlex.split(task["eval"]["command"])

        assert task["triggers"] == ["per-commit", "manual"]
        assert task["name"] == name
        assert task["runner"]["labels"] == [label]
        assert flag_value(server_tokens, "--model") == model
        assert flag_value(server_tokens, "--tensor-parallel-size") == "4"
        assert flag_value(server_tokens, "--dtype") == "bfloat16"
        assert flag_value(server_tokens, "--max-model-len") == "1048576"
        assert flag_value(server_tokens, "--max-total-tokens") == "1048576"
        assert flag_value(server_tokens, "--max-num-seqs") == "32"
        assert flag_value(server_tokens, "--chunked-prefill-size") == "8192"
        assert flag_value(server_tokens, "--max-cudagraph-capture-size") == "32"
        assert "--disable-kvstore" in server_tokens
        assert "--trust-remote-code" in server_tokens
        assert flag_value(eval_tokens, "--model") == model
        assert flag_value(eval_tokens, "--datasets") == "gsm8k"
        assert flag_value(eval_tokens, "--eval-batch-size") == "32"
        assert task["score_threshold"] == 0.90
        # Both runners load weights from their shared Hugging Face cache.
        assert "--download-dir" not in server_tokens

        if label == "b200-4gpu":
            assert "--enable-expert-parallel" in server_tokens
            assert flag_value(server_tokens, "--moe-backend") == "mega_moe"
            # The NVIDIA gate exercises the split prefill graph (encoder and
            # decoder graphs around the eager narrowing layer).
            assert "--disable-prefill-graph" not in server_tokens
        else:
            assert "--enable-expert-parallel" not in server_tokens
            assert "--moe-backend" not in server_tokens
            # Not yet exercised on AMD; keep that gate on eager prefill.
            assert "--disable-prefill-graph" in server_tokens


KIMI_K31_AMD_RECIPE_CONFIGS = (
    (EVAL_CONFIG_DIR, "kimi-k3.1-pro-dspark-mxfp4-tp8-kvp8-evalscope-aime26-amd.yaml"),
    (
        PERF_CONFIG_DIR,
        "kimi-k3.1-pro-dspark-mxfp4-tp8-kvp8-evalscope-agentic-mi35x.yaml",
    ),
)
# Serving flags that only the CI jobs set (the gateway tokenizer is staged on
# the runners) and that only the doc sets (a public bind address).
KIMI_K31_CI_ONLY_FLAGS = {
    "--tokenizer",
    "--host",
    "--port",
    "--policy",
    "--engine-startup-timeout",
    "--gateway-startup-timeout",
}
KIMI_K31_DOC_ONLY_FLAGS = {"--host", "--port"}


def serve_flags(tokens: list[str]) -> dict[str, tuple[str, ...]]:
    flags = {}
    flag = None
    for token in tokens:
        if token.startswith("--"):
            assert token not in flags, f"duplicate {token}"
            flag = token
            flags[flag] = ()
        else:
            assert flag is not None, f"positional {token} after the subcommand"
            flags[flag] += (token,)
    return flags


def kimi_k31_amd_doc_recipe() -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    text = (REPO_ROOT / "docs" / "recipes" / "models.md").read_text(encoding="utf-8")
    section = text.split("## Kimi K3.1 Pro {#kimi-k3-1-pro}", 1)[1]
    section = section.split("\n## ", 1)[0].split("\n### AMD\n", 1)[1]
    section = section.split("\n### ", 1)[0]
    blocks = section.split("```bash\n")
    assert len(blocks) == 2, "expected one bash block in the Kimi-K3.1 AMD recipe"
    tokens = shlex.split(blocks[1].split("```", 1)[0].replace("\\\n", " "))
    env = {}
    while "=" in tokens[0] and not tokens[0].startswith("-"):
        key, value = tokens.pop(0).split("=", 1)
        env[key] = value
    assert tokens[:2] == ["tokenspeed", "serve"]
    model = tokens[2]
    flags = serve_flags(tokens[3:])
    flags["--model"] = (model,)
    return env, flags


def test_kimi_k31_amd_configs_have_expected_gating_and_baselines():
    configs = (
        *KIMI_K31_AMD_RECIPE_CONFIGS,
        (EVAL_CONFIG_DIR, "kimi-k3.1-pro-mxfp4-tp8ep1-evalscope-aime26-amd.yaml"),
        (
            PERF_CONFIG_DIR,
            "kimi-k3.1-pro-mxfp4-tp8ep1-evalscope-random-4k-1k-mi35x.yaml",
        ),
    )
    tasks = {}
    for config_dir, filename in configs:
        task = yaml.safe_load((config_dir / filename).read_text(encoding="utf-8"))
        server_tokens = shlex.split(task["server"]["command"])
        is_recipe = "dspark" in filename

        expected_triggers = ["per-commit", "manual"] if is_recipe else ["manual"]
        assert task["triggers"] == expected_triggers
        assert (
            task["env"]["KIMI_K31_GATEWAY_TOKENIZER"]
            == "/cache/huggingface/kimi-k3.1-gateway-tokenizer"
        )
        assert flag_value(server_tokens, "--model") == "moonshotai/Kimi-K3.1-Pro"
        if "--speculative-algorithm" in server_tokens:
            # The shared revision flag also applies to the draft repository,
            # whose commit differs from the target checkpoint.
            assert "--revision" not in server_tokens
        else:
            assert (
                flag_value(server_tokens, "--revision")
                == "d9e99b0792a5e93b78ea9bea62a68167e36ddd9e"
            )
        assert flag_value(server_tokens, "--served-model-name") == "kimi-k3.1"
        assert (
            flag_value(server_tokens, "--tokenizer")
            == "${KIMI_K31_GATEWAY_TOKENIZER:?K3.1_gateway_tokenizer_not_staged}"
        )
        if not is_recipe:
            assert flag_value(server_tokens, "--attention-backend") == "dsa"
        if task["type"] == "eval":
            if not is_recipe:
                assert flag_value(server_tokens, "--reasoning-parser") == "kimi_k3"
            assert task["score_threshold"] == 0.90
        elif is_recipe:
            assert task["perf_threshold"] == 0.95
            assert task["perf_reference"] == {32: [24.7, 9400]}
        else:
            assert "perf_reference" not in task
            assert "perf_threshold" not in task
        tasks[filename] = task

    eval_task, agentic_task = (
        tasks[filename] for _, filename in KIMI_K31_AMD_RECIPE_CONFIGS
    )
    for task in (eval_task, agentic_task):
        server_tokens = shlex.split(task["server"]["command"])
        assert flag_value(server_tokens, "--speculative-algorithm") == "DSPARK"
        # A step count equal to the default leaves the checkpoint's block width.
        assert (
            flag_value(server_tokens, "--speculative-num-steps") != "3"
            or "--speculative-num-draft-tokens" in server_tokens
        )
        assert flag_value(server_tokens, "--tensor-parallel-size") == "8"
        assert flag_value(server_tokens, "--kv-parallel-size") == "8"
        assert flag_value(server_tokens, "--decode-context-parallel-size") == "8"
        for flag in (
            "--max-num-seqs",
            "--cudagraph-capture-sizes",
            "--enable-mixed-batch",
            "--prefill-decode-interval",
            "--prefill-decode-min-interval",
            "--enable-short-prefill-first",
            "--disable-prefill-graph",
        ):
            assert flag not in server_tokens

    eval_tokens = shlex.split(eval_task["eval"]["command"])
    generation_config = json.loads(flag_value(eval_tokens, "--generation-config"))
    assert generation_config["seed"] == 42
    assert generation_config["max_tokens"] == 63488
    assert flag_value(eval_tokens, "--eval-batch-size") == "16"
    assert "--limit" not in eval_tokens
    assert (
        flag_value(eval_tokens, "--work-dir")
        == ".ci-artifacts/published/kimi-k3.1-pro-dspark-aime26"
    )
    assert "'evalscope[perf]==1.11.1'" in agentic_task["perf"]["install"][0]
    assert "/tmp/agentic_dataset.json" in agentic_task["perf"]["install"][1]
    assert (
        'revision="2779acd48a05c211f2f82eee71bdbdf688d984ca"'
        in agentic_task["perf"]["install"][1]
    )
    agentic_tokens = shlex.split(agentic_task["perf"]["command"])
    assert agentic_tokens.count("/tmp/evalscope-perf/bin/evalscope") == 1
    for flag, value in {
        "--dataset": "swe_smith",
        "--max-tokens": "500",
        "--parallel": "32",
        "--number": "64",
    }.items():
        assert flag_value(agentic_tokens, flag) == value
    assert "--multi-turn" in agentic_tokens
    assert "--dataset-offset" not in agentic_tokens
    # collect_outputs.py reads the GPU count from the attn_tp suffix.
    assert "$OUTPUTS_DIR/kimi-k3.1_attn_tp8;" in agentic_tokens
    assert (
        "$REPO_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed/collect_outputs.py"
        in agentic_tokens
    )


@pytest.mark.parametrize(
    "config", KIMI_K31_AMD_RECIPE_CONFIGS, ids=lambda config: config[1]
)
def test_kimi_k31_amd_ci_serves_the_documented_recipe(config):
    config_dir, filename = config
    task = yaml.safe_load((config_dir / filename).read_text(encoding="utf-8"))
    doc_env, doc_flags = kimi_k31_amd_doc_recipe()
    tokens = shlex.split(task["server"]["command"])
    assert tokens[:2] == ["ts", "serve"]
    ci_flags = serve_flags(tokens[2:])

    for flag in KIMI_K31_CI_ONLY_FLAGS:
        ci_flags.pop(flag, None)
    for flag in KIMI_K31_DOC_ONLY_FLAGS:
        doc_flags.pop(flag, None)
    assert ci_flags == doc_flags

    ci_env = {
        key: value
        for key, value in task["env"].items()
        if key not in ("CI", "KIMI_K31_GATEWAY_TOKENIZER")
    }
    assert ci_env == doc_env


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
