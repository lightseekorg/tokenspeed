# TokenSpeed Scheduler (Rust)

Rust port of the C++ scheduler control plane (`tokenspeed-scheduler`).

Design and contract: see [`docs/design/rust-port.md`](../docs/design/rust-port.md).

## Layout

```
crates/
├── ts-scheduler-core/     # scheduler logic, #![forbid(unsafe_code)]
└── ts-scheduler-pyo3/     # Python bindings via pyo3 + numpy + maturin
```

## Status

| Module | Status |
|---|---|
| token_container | ported |
| req_pool_allocator | ported |
| cache_types (leaf) | ported |
| block_table | ported |
| cache_block_ref | ported |
| block_pool | ported |
| prefix_hasher | ported |
| cache_config | ported |
| prefix_index / prefix_matcher / group_allocator / group_geometry | planned |
| cache_group / cache_admission / cache_coordinator | planned |
| tier transfer manager | planned |
| fsm / scheduler / operations | planned |
| pyo3 bindings | done — 58/58 pytest contract passes |

## Build and test

```
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

### Python bindings (pyo3 + maturin)

```
cd crates/ts-scheduler-pyo3
python -m venv .venv && . .venv/Scripts/activate
pip install maturin pytest numpy
maturin develop --release                      # editable install into the venv
python -m pytest <repo>/tokenspeed-scheduler/python/tests   # 58/58 contract
```

A standalone wheel bundles the `tokenspeed_scheduler` package:

```
maturin build --release
pip install target/wheels/tokenspeed_scheduler-*.whl
```


