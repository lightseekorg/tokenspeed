ARG RUNNER_TAG=cu130-torch-2.11.0
FROM lightseekorg/tokenspeed-runner:${RUNNER_TAG}

USER root

ARG MAX_JOBS=16
ENV PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /workspace

COPY . /workspace

RUN apt-get update && apt-get install -y --no-install-recommends libssl-dev libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

RUN MAX_JOBS=${MAX_JOBS} FLASHINFER_CUDA_ARCH_LIST="9.0a 10.0a" TOKENSPEED_KERNEL_BACKEND=cuda \
    pip install tokenspeed-kernel/python/ --no-build-isolation && \
    pip install tokenspeed-scheduler/ && \
    pip install "./python" --no-build-isolation

CMD ["/bin/bash"]
