# Rust engine crates: 0.1 to 0.2

The execution-observation column is backward compatible on the MessagePack wire.
It is **not Rust source compatible**: adding a public field breaks downstream
struct literals and exhaustive destructuring. These changes belong in the Rust
workspace's coordinated `0.2.0` release, with all internal path/registry dependency
requirements and Cargo.lock updated together. The Python engine wheel uses its
own version; this change does not bump that wheel or the observation protocol.

Update direct dependencies together; `0.1` and `0.2` wire types are distinct Rust
types even when they encode compatible bytes:

```toml
ts-client = { package = "tokenspeed-client", version = "0.2.0" }
ts-wire = { package = "tokenspeed-wire", version = "0.2.0" }
```

Existing decoders, `BatchTokenIDOutSlim::requests`, `OutputChunk::from_batch` and
`From<&RequestOutput>` populate the observation fields. Downstream code that
constructs these structs must initialize the added fields:

| Struct | Added field | Value when evidence is unavailable |
| --- | --- | --- |
| `ts_client::OutputChunk` | `execution_observations` | `None` |
| `ts_wire::BatchTokenIDOutSlim` | `execution_observations` | `Vec::new()` for a wholly absent column; aligned `Option` entries for a mixed batch |
| `ts_wire::RequestOutput` | `execution_observations` | `None` |
| `ts_control::output::ForwardResult` | `execution_observation` | `None` |
| `ts_control::output::StreamedOutput` | `execution_observations` | `None` |

Use `..` in destructuring when the application intentionally ignores other
fields. No parallel legacy structs, adapters or fabricated zero snapshots are
needed. `ForwardResult::default()` remains useful for an unobserved result, but
an enabled native observation lease treats a missing receipt as invalid evidence.
An old producer's missing column means unavailable, never zero executions.

Observation snapshots are cumulative: replace the previous snapshot after
validation, rather than adding its counters again. A terminal output is not
lifetime closure of outstanding native forwards. See the
[execution contract](../../docs/design/execution-observations.md) for both
histograms, invalid/omitted evidence and closed-trace reconciliation.

Unpublished changes may share `0.2.0`, including other pending client API changes.
Do not publish a partial `0.2.0` and then label additional breaking field/variant
changes `0.2.1`. If `0.2.0` has already been released before integration, move the
affected Rust crates and their dependant requirements to the next minor release
and regenerate the lockfile. Merge/release coordination must preserve the final
client API from all included changes; these notes do not add cancellation APIs.
