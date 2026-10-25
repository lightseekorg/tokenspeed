# tokenspeed-engine

The engine's control plane, in Rust, as
[RFC 0002](../docs/rfcs/0002-rust-control-plane.md) describes it: every
engine rank is a `ts-host` process that runs the scheduling loop in Rust and
embeds CPython for the data plane. The design rules of the loop are in
[`docs/design/event-loop.md`](../docs/design/event-loop.md).

| Source crate | Published package | Role |
| --- | --- | --- |
| `ts-wire` | `tokenspeed-wire` | The engine IPC schema and its msgpack codec, byte-compatible with the Python structs in `python/tokenspeed/runtime/engine/io_struct.py` and `zmq_wire.py`. |
| `ts-client` | `tokenspeed-client` | A frontend's connection to the engines: the startup handshake, request routing across attention-DP ranks, output streams, aborts and control operations. |
| `ts-py` | `tokenspeed-py` | `tokenspeed_engine._native`: `ts-client` for the Python frontend (`AsyncLLM`, under the `Engine` API and the gRPC servicer), with `tokenspeed_engine.host_binary()` naming the installed `ts-host`, and `ts-l3`'s executor for Python tests over a data plane's Host pool. |
| `ts-collective` | `tokenspeed-collective` | CPU collectives among an engine's ranks over a TCP mesh: broadcast, all-gather and all-reduce, each frame checked against the collective the receiver is in. |
| `ts-scheduler` | `tokenspeed-scheduler` | The C++ scheduler and its nanobind module compiled into the binary, with a `cxx` facade for the control thread. The module is registered as a built-in of the embedded interpreter, so the plans Python reads are the same C++ objects the control thread produced. |
| `ts-control` | `tokenspeed-control` | One rank's control plane: ingress checks, request state and output processing, the scheduling loop and its in-flight queue, cache-op acknowledgment, control operations, grammars, the expert rebalance, KV events, the engine side of the msgpack wire, and the per-round agreements among ranks. |
| `ts-grammar` | `tokenspeed-grammar` | Grammar-constrained decoding on xgrammar's C++ library: compilation and the matchers, which the data plane drives through C entry points. |
| `ts-metrics` | `tokenspeed-metrics` | The engine's metrics: Prometheus' multiprocess files, written directly, and the events the node's plugin relay runs output plugins on. |
| `ts-mooncake` | `tokenspeed-mooncake` | The Mooncake transfer engine and store client through their C APIs, loaded at runtime from the `tokenspeed-mooncake` wheel. |
| `ts-l3` | `tokenspeed-l3` | The L3 storage tier's Host-only executor ([asynchronous L3](../docs/design/async-l3-cache.md)): object naming, the Host pool's layout and ownership, bounded queues and workers, backend health, and the stores (the Mooncake store client, and an in-memory one for tests). |
| `ts-pd` | `tokenspeed-pd` | The PD transfer peer: the P/D handshake, room states, transfer workers and the per-round agreement on transfer progress. |
| `ts-host` | `tokenspeed-host` | The rank binary: it embeds the interpreter, has `engine/rust_host.py` build the data plane and the scheduler, and runs the loop on its main thread and the data plane on its forward thread. |

The crates.io packages use the `tokenspeed-` prefix. Their Rust library names
remain `ts_wire`, `ts_client`, and so on; the Python module remains `_native`,
and the host executable remains `ts-host`. For example:

```toml
[dependencies]
ts-client = { package = "tokenspeed-client", version = "0.1.0" }
```

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

The native crates target Linux and need a C++ compiler, CMake, OpenSSL
development headers, and Python development headers. Set `PYO3_PYTHON` to
an interpreter with `nanobind>=2.13.0` and `tokenspeed-spdlog==1.15.1`
installed. The scheduler source is bundled in its crate; the grammar build
downloads the pinned, SHA256-checked xgrammar source release and therefore
needs network access (or `FETCHCONTENT_SOURCE_DIR_XGRAMMAR` for a local copy).

`ts-host` and `ts-py` link the CPython the engine runs under; build with it
as `PYO3_PYTHON`, which needs `nanobind` and `tokenspeed-spdlog`, as the
scheduler wheel build does:

```bash
PYO3_PYTHON=$(which python) cargo build --release -p tokenspeed-host -p tokenspeed-py
```

The `tokenspeed-engine` wheel (`pyproject.toml`, built with setuptools) carries
`tokenspeed_engine._native` and the `ts-host` binary. To run a development
build instead, put `_native` on the Python path and point
`TOKENSPEED_HOST_BINARY` at the binary. `TS_HOST_LOG` sets the Rust log
filter (`info` by default; `ts_control=debug` logs admissions and cache-op
acknowledgments).

## Releasing Rust crates

Use **Release TokenSpeed Engine Crates** in GitHub Actions. Select `dry-run`
to build and verify the workspace's package archives without uploading.
Select `publish` on `main` to publish missing versions in dependency order;
versions already on crates.io are skipped, so a partial release can be rerun.
The run summary lists each package's version and the source commit.

Before a release, update `[workspace.package].version` in `Cargo.toml` and
the internal dependency versions in `crates/*/Cargo.toml`, run
`cargo update --workspace` to refresh `Cargo.lock`, and merge the version
change. The workflow publishes the committed versions; it does not bump them.

Publishing uses crates.io Trusted Publishing rather than a stored API token.
Configure it once:

1. Create the GitHub environment `crates-io` and restrict its deployment
   branches to `main`.
2. In each published crate's **Settings → Trusted Publishing**, add a GitHub
   publisher with repository owner `lightseekorg`, repository `tokenspeed`,
   workflow filename `release-tokenspeed-engine-crates.yml`, and environment
   `crates-io`.

New crates need their initial token-based publication before Trusted
Publishing can be configured. Registry rate limits still apply.

## Running against an engine

Start a headless engine (one rank process per GPU is launched for you); it
dials the handshake address the client binds:

```bash
ts serve --headless --model <model> \
  --data-parallel-rpc-port 30500 [--data-parallel-size N --dist-init-addr <host:port>]
cargo run --release --example generate -- tcp://127.0.0.1:30500 <N> <max-new-tokens> <prompts-file>
```

Each line of the prompts file is a comma-separated list of prompt token ids.
