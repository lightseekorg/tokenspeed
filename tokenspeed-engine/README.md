# tokenspeed-engine

The engine's control plane, in Rust, as
[RFC 0002](../docs/rfcs/0002-rust-control-plane.md) describes it: every
engine rank is a `ts-host` process that runs the scheduling loop in Rust and
embeds CPython for the data plane. The design rules of the loop are in
[`docs/design/event-loop.md`](../docs/design/event-loop.md).

| Source crate | Published package | Role |
| --- | --- | --- |
| `ts-wire` | `tokenspeed-wire` | The engine IPC schema and its msgpack codec, byte-compatible with the Python structs in `python/tokenspeed/runtime/engine/io_struct.py` and `zmq_wire.py`. |
| `ts-client` | `tokenspeed-client` | A frontend's connection to the engines: the startup handshake, request routing across attention-DP ranks, output streams, aborts, control operations and an EPD encode server's encode requests. |
| `ts-py` | `tokenspeed-py` | `tokenspeed_engine._native`: `ts-client` for Python tests and tools (`entrypoints/wire_driver.py`), with `tokenspeed_engine.host_binary()` naming the installed `ts-host`, and `ts-l3`'s executor for Python tests over a data plane's Host pool. |
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

The optional `output_token_cumulative_probs_val` tail carries each selected
token's cumulative probability diagnostic alongside its logprob. The control
plane retains the same token ordering, stop truncation and streaming slices
for both columns, and emits diagnostics only for requests returning logprobs.
Once a request has any value, each of its outputs carries one per token, NaN
for a token that came without one, so the column never misaligns with the
tokens. An older producer defaults this column to empty; older readers skip
it.

The appended `reasoning_tokens` column is cumulative per request, like
`completion_tokens`, and excludes reasoning-block delimiters. The control plane
counts committed token ids before detokenization, including a PD prefill's
bootstrap token. An older producer defaults the column to empty and its output
rows report zero; older readers skip the appended column. `ts-client` output
chunks, the Python chunk dictionaries, and `WireDriver.Completion` expose the
same cumulative count.

## Building

Snapshot SDK updates carry a complete publication directory in the optional
`UpdateWeightsFromMooncakeReqInput.snapshot` map. The native host and Python
runtime must be built together to transport this field. The existing global
commit and memory-saver rules still apply. The configured SDK updater receives
`update_weights(models, version=..., snapshot=...)`. Tensor data remains in the
SDK's store. Requests without a snapshot retain the existing SDK protocol.

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

## Releasing Python wheels

Use **Release TokenSpeed Engine** in GitHub Actions. Select `dry-run` to
build and test wheels, or `publish` on `main` to upload them. Choose a
`release_type` (`post` by default):

| Type | Example with base version `0.1.0` | Destination |
| --- | --- | --- |
| `release` | `0.1.0` | `lightseekorg/whl`, then PyPI |
| `post` | `0.1.0.post20261111` | `lightseekorg/whl` by default |
| `rc` | `0.1.0rc20261112` | `lightseekorg/whl` only |

For a `post` release, explicitly enable `publish_post_to_pypi` to also upload
the tested wheels to PyPI after the wheel repository publication succeeds.
This option defaults to false and is accepted only for `post`. Post versions
on PyPI sort above their base version and participate in normal pip upgrades.

For `post` and `rc`, `version_date` defaults to the current UTC date
(`YYYYMMDD`). A `post` run checks existing engine releases in `lightseekorg/whl`
and advances by calendar days until it finds an unused post version. If
`0.1.0.post20261111` exists, another run can publish `0.1.0.post20261112`
on the same day. This date is a version identifier and can be ahead of the
actual build date. Only an explicit `rc` selection produces an rc; it uses
the requested date without automatic advancement.

Leave `version_date` empty for a stable `release`. The workflow resolves the
version once and changes the Python package version only in the build
checkout. The committed base version and Rust crate versions stay unchanged.

Wheels are attached to
[`lightseekorg/whl` releases](https://github.com/lightseekorg/whl/releases)
under `tokenspeed-engine-v<VERSION>`. `post` and `rc` releases are marked
as GitHub prereleases. Download the matching wheel and install it with
`pip install ./<wheel>.whl`. An rc sorts below the same base version's final
release under Python version ordering; use an exact wheel/version when
testing it. Merge a base version bump before the next stable release.

Use **Re-run failed jobs** to upload missing files from the same source
commit. **Re-run all jobs** resolves the version again and may allocate a
new post date; use **Re-run failed jobs** to retain the original version.
Existing assets are never overwritten. A different commit requires
a new post run, an unused rc date, or a new base version for a stable release.

Wheels support Linux with glibc 2.28 or newer, x86_64 and aarch64, and
CPython 3.10–3.13. They contain the Python extension, `ts-host`, and a private
CPython shared library for the host process, including its built-in native
stdlib modules. The host uses the launching interpreter's Python library
paths and environment. Installing a wheel needs no
Rust toolchain. Each wheel is tested in a separate Python container and venv
by importing the extension and starting the embedded scheduler and Python.

The build uses a full checkout because the engine also compiles the sibling
scheduler source. No source distribution is published; source builds must
use the repository checkout and the dependencies described above.

Configure the `tokenspeed-engine` project's PyPI Trusted Publisher with
owner `lightseekorg`, repository `tokenspeed`, workflow filename
`release-tokenspeed-engine.yml`, and GitHub environment `pypi`. Publication
uses GitHub OIDC and does not require a stored PyPI token.
The workflow also needs `LIGHTSEEK_BOT_TOKEN` with release write access to
`lightseekorg/whl`, authenticated as `lightseek-bot`.

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
