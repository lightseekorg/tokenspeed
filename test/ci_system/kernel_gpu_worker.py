"""Assign one allocated GPU to each pytest-xdist worker before collection."""

import os


def pytest_load_initial_conftests(early_config, parser, args):
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker is None:
        return

    allocated = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = allocated.split(",") if allocated else [str(i) for i in range(4)]
    if len(devices) != 4:
        raise RuntimeError(f"Expected four allocated GPUs, got {devices!r}")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[int(worker.removeprefix("gw"))]
