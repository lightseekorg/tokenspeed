"""Common utilities for testing and benchmarking."""

import os

from tokenspeed.runtime.utils import get_bool_env_var


def is_in_ci():
    """Return whether it is in CI runner."""
    return get_bool_env_var("CI") or get_bool_env_var("GITHUB_ACTIONS")


def _gpu_port_offset() -> int:
    gpu_ids = [
        int(g)
        for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        if g.strip().isdigit()
    ]
    if not gpu_ids:
        return 0
    return min(gpu_ids) * 2 + (1 if len(gpu_ids) > 1 else 0)  # 0-15 on 8-GPU hosts


DEFAULT_PORT_FOR_SRT_TEST_RUNNER = 16000 + _gpu_port_offset() * 1040


def lcs(x, y):
    m = len(x)
    n = len(y)
    table = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                table[i][j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

    return table[m][n]


def calculate_rouge_l(output_strs_list1, output_strs_list2):
    rouge_l_scores = []

    for s1, s2 in zip(output_strs_list1, output_strs_list2):
        lcs_len = lcs(s1, s2)
        precision = lcs_len / len(s1) if len(s1) > 0 else 0
        recall = lcs_len / len(s2) if len(s2) > 0 else 0
        if precision + recall > 0:
            fmeasure = (2 * precision * recall) / (precision + recall)
        else:
            fmeasure = 0.0
        rouge_l_scores.append(fmeasure)

    return rouge_l_scores
