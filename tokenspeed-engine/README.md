# tokenspeed-engine

The engine's control plane, in Rust, as
[RFC 0002](../docs/rfcs/0002-rust-control-plane.md) describes it: every
engine rank is a `ts-host` process that runs the scheduling loop in Rust and
embeds CPython for the data plane. The design rules of the loop are in
[`docs/design/event-loop.md`](../docs/design/event-loop.md).

| Crate | Role |
| --- | --- |
| `ts-wire` | The engine IPC schema and its msgpack codec, byte-compatible with the Python structs in `python/tokenspeed/runtime/engine/io_struct.py` and `zmq_wire.py`. |
| `ts-client` | A frontend's connection to the engines: the startup handshake, request routing across attention-DP ranks, output streams, aborts and control operations. |
| `ts-py` | `tokenspeed_engine._native`: `ts-client` for the Python frontend (`AsyncLLM`, under the `Engine` API and the gRPC servicer), with `tokenspeed_engine.host_binary()` naming the installed `ts-host`. |
| `ts-collective` | CPU collectives among an engine's ranks over a TCP mesh: broadcast, all-gather and all-reduce, each frame checked against the collective the receiver is in. |
| `ts-scheduler` | The C++ scheduler and its nanobind module compiled into the binary, with a `cxx` facade for the control thread. The module is registered as a built-in of the embedded interpreter, so the plans Python reads are the same C++ objects the control thread produced. |
| `ts-control` | One rank's control plane: ingress checks, request state and output processing, the scheduling loop and its in-flight queue, cache-op acknowledgment, control operations, grammars, the expert rebalance, KV events, the engine side of the msgpack wire, and the per-round agreements among ranks. |
| `ts-grammar` | Grammar-constrained decoding on xgrammar's C++ library: compilation and the matchers, which the data plane drives through C entry points. |
| `ts-metrics` | The engine's metrics: Prometheus' multiprocess files, written directly, and the events the node's plugin relay runs output plugins on. |
| `ts-mooncake` | The Mooncake transfer engine through its C API, loaded at runtime from the `tokenspeed-mooncake` wheel. |
| `ts-pd` | The PD transfer peer: the P/D handshake, room states, transfer workers and the per-round agreement on transfer progress. |
| `ts-host` | The rank binary: it embeds the interpreter, has `engine/rust_host.py` build the data plane and the scheduler, and runs the loop with a pump thread driving the data plane. |

## Wire conformance

`crates/ts-wire/tests/fixtures/` holds messages written by the Python encoder.
`cargo test` decodes each one, compares it field by field with the values the
generator used, and re-encodes it byte for byte;
`test/runtime/test_ts_wire_fixtures.py` checks that the files still match the
Python encoding. After an intentional schema change on the Python side, run

```bash
python tokenspeed-engine/crates/ts-wire/tests/fixtures/generate.py
```

and update the `ts-wire` structs to match. Give every new field a distinct,
non-default value in the generator: a field decoded into the wrong slot still
re-encodes to the same bytes, so only the value checks catch it.

## Building

`ts-host` and `ts-py` link the CPython the engine runs under; build with it
as `PYO3_PYTHON`, which needs `nanobind` and `tokenspeed-spdlog`, as the
scheduler wheel build does:

```bash
PYO3_PYTHON=$(which python) cargo build --release -p ts-host -p ts-py
```

The `tokenspeed-engine` wheel (`pyproject.toml`, built with maturin) carries
`tokenspeed_engine._native` and the `ts-host` binary. To run a development
build instead, put `_native` on the Python path and point
`TOKENSPEED_HOST_BINARY` at the binary. `TS_HOST_LOG` sets the Rust log
filter (`info` by default; `ts_control=debug` logs admissions and cache-op
acknowledgments).

## Running against an engine

Start a headless engine (one rank process per GPU is launched for you); it
dials the handshake address the client binds:

```bash
ts serve --headless --model <model> \
  --data-parallel-rpc-port 30500 [--data-parallel-size N --dist-init-addr <host:port>]
cargo run --release --example generate -- tcp://127.0.0.1:30500 <N> <max-new-tokens> <prompts-file>
```

Each line of the prompts file is a comma-separated list of prompt token ids.
