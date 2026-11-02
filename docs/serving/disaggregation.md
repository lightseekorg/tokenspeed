# Disaggregated Serving

TokenSpeed serves prefill/decode (PD) and encode/prefill/decode (EPD)
disaggregation behind SMG's gateway. Each role is a headless engine
(`ts serve --headless`) and SMG's gateway (`smg launch`) fronts them. `ts
serve` fronts one regular engine and refuses a disaggregated role. The
gateway reaches every role over its direct ZMQ lane: it binds each worker's
sockets, and the role's ranks dial its handshake.

## Workers and handshakes

A worker URL names where the gateway binds a role's sockets:

- `tcp://<host>:<port>`: the handshake listens at that address, and the
  data plane on ephemeral ports of the same host. The role's engine dials
  `--data-parallel-address <host> --data-parallel-rpc-port <port>`. The host
  must be one the engines reach, not a wildcard. Use it for roles on other
  hosts, or anywhere you want to choose the port yourself.
- `ipc://<path>`: same-host sockets under `<path>`, with the handshake on a
  loopback port the gateway derives from the path
  (`python3 -c 'from tokenspeed.cli._proc import zmq_handshake_port; print(zmq_handshake_port("ipc://<path>"))'`).
  The engine dials `--data-parallel-address 127.0.0.1` at that port.

Every engine also passes `--zmq-engine-index 0`. An attention-DP engine
dials with one identity per rank, and the gateway awaits as many ranks as the
first one declares, so the roles may run different DP sizes. The gateway
takes the prefill and encode engines' bootstrap addresses from their
handshakes. These sockets carry no authentication: keep them on a trusted
network.

## Prefill/decode

```bash
smg launch --pd-disaggregation --backend tokenspeed \
  --prefill tcp://10.0.0.5:31000 \
  --decode tcp://10.0.0.5:31001 \
  --model-path <model> --tokenizer-path <model> \
  --host 0.0.0.0 --port 8000

# On the prefill host (GPUs 0-1):
CUDA_VISIBLE_DEVICES=0,1 ts serve --headless --model <model> --tp 2 \
  --disaggregation-mode prefill --disaggregation-bootstrap-port 8998 \
  --disaggregation-transfer-backend mooncake \
  --dist-init-addr 127.0.0.1:12579 \
  --data-parallel-address 10.0.0.5 --data-parallel-rpc-port 31000 \
  --zmq-engine-index 0

# On the decode host (GPUs 0-1):
CUDA_VISIBLE_DEVICES=0,1 ts serve --headless --model <model> --tp 2 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --dist-init-addr 127.0.0.1:13580 \
  --data-parallel-address 10.0.0.5 --data-parallel-rpc-port 31001 \
  --zmq-engine-index 0
```

Give each engine its own `--dist-init-addr` when roles share a host: every
engine derives a small port cluster from it. Prompt logprobs come from the
prefill node and the gateway joins them with the decode node's output (see
[Server Parameters](../configuration/server.md)).

## Encode/prefill/decode

Add an encode server and switch the gateway to `--epd-disaggregation`:

```bash
smg launch --epd-disaggregation --backend tokenspeed \
  --encode tcp://10.0.0.5:31002 \
  --prefill tcp://10.0.0.5:31000 \
  --decode tcp://10.0.0.5:31001 \
  --model-path <model> --tokenizer-path <model> \
  --encode-policy consistent_hashing --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES=2 ts serve --headless --model <model> \
  --disaggregation-mode encode --disaggregation-bootstrap-port 18995 \
  --disaggregation-transfer-backend mooncake \
  --dist-init-addr 127.0.0.1:25000 \
  --data-parallel-address 10.0.0.5 --data-parallel-rpc-port 31002 \
  --zmq-engine-index 0
```

The encode server runs only the vision tower. Its first rank dials the
gateway, which sends it each image as an encode request; the embeddings
travel to the prefill engine over Mooncake.

## Transfer

The KV cache and the embeddings move over Mooncake. Let it pick its
transport (RDMA where present); on containerized hosts the forced intra-node
NVLink transport (`MC_INTRANODE_NVLINK`) has reported success without
writing the destination pages, so leave it unset unless the platform is
known to support it. `--disaggregation-ib-device` selects RDMA devices
explicitly.

The CI launchers in `test/ci_system/` (`serve_qwen35_397b_nvfp4_pd_1p1d.sh`,
`serve_qwen35_122b_nvfp4_epd_1e1p2d.sh`,
`serve_deepseek_v41_flash_pd_1p1d.sh`) are complete single- and two-node
examples.
