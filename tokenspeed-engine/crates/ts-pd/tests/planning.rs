// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! The transfer planning gives the Python PD planner's answers.
//!
//! `fixtures/planning/` holds the inputs and answers (or refusals) of every
//! call the Python planner's test suite made into the functions
//! `ts_pd::layout` ports, over the cache layouts of the models it tests
//! (recorded by `fixtures/pd_planning_capture.py`): the contracts the records
//! name, and the records by concern.
//! Each record is replayed here: an answer must be equal, a refusal must be a
//! refusal.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use bytes::Bytes;
use serde_json::{json, Value};
use ts_pd::{
    layout::{
        contract::{validate_peer, Contract, GroupSpec},
        copies::{Source, Transfer},
        manifest::{
            build_manifest, build_selection, validate_manifest, BlockTable, BlockTables,
            GroupBlocks, GroupSelection, Manifest, Selection,
        },
        registration::{
            plan_registration, plan_route, Preallocation, PrefillLayout, RegistrationInfo,
        },
        route::{
            local_transfer_pages, pipeline_plan, validate_owner_filters, Fragment, OwnerFilter,
            OwnerFilters, Planner, RankPlan,
        },
        LayoutError,
    },
    Copy,
};

struct Fixture {
    contracts: HashMap<String, Contract>,
}

impl Fixture {
    fn contract(&self, key: &Value) -> &Contract {
        &self.contracts[key.as_str().expect("a contract key")]
    }
}

fn int(value: &Value) -> i64 {
    value.as_i64().expect("an integer")
}

fn ints(value: &Value) -> Vec<i64> {
    value.as_array().expect("a list").iter().map(int).collect()
}

fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .expect("a list")
        .iter()
        .map(|item| item.as_str().expect("a string").to_owned())
        .collect()
}

fn raw(args: &Value) -> Vec<u8> {
    match (&args["text"], &args["raw"]) {
        (Value::String(text), _) => text.as_bytes().to_vec(),
        (_, Value::String(hex)) => decode_hex(hex),
        _ => panic!("a payload record has no payload"),
    }
}

fn decode_hex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&hex[index..index + 2], 16).expect("hex"))
        .collect()
}

fn manifest(value: &Value) -> Manifest {
    Manifest {
        groups: value["groups"]
            .as_array()
            .expect("groups")
            .iter()
            .map(|group| GroupBlocks {
                group_id: group["group_id"].as_str().expect("a group id").to_owned(),
                block_ids: ints(&group["block_ids"]),
            })
            .collect(),
        prefix_len: int(&value["prefix_len"]),
        prompt_len: int(&value["prompt_len"]),
        state_prefix_cached: value["state_prefix_cached"].as_bool().expect("a flag"),
    }
}

fn manifest_json(manifest: &Manifest) -> Value {
    json!({
        "groups": manifest.groups.iter().map(|group| json!({
            "group_id": group.group_id,
            "block_ids": group.block_ids,
        })).collect::<Vec<_>>(),
        "prefix_len": manifest.prefix_len,
        "prompt_len": manifest.prompt_len,
        "state_prefix_cached": manifest.state_prefix_cached,
    })
}

fn selection(value: &Value) -> Selection {
    Selection {
        groups: value["groups"]
            .as_array()
            .expect("groups")
            .iter()
            .map(|group| GroupSelection {
                source_block_ids: ints(&group["source_block_ids"]),
                destination_positions: ints(&group["destination_positions"])
                    .into_iter()
                    .map(|position| position as usize)
                    .collect(),
            })
            .collect(),
    }
}

fn selection_json(selection: &Selection) -> Value {
    json!({
        "groups": selection.groups.iter().map(|group| json!({
            "source_block_ids": group.source_block_ids,
            "destination_positions": group.destination_positions,
        })).collect::<Vec<_>>(),
    })
}

fn tables(value: &Value) -> BlockTables {
    value
        .as_object()
        .expect("tables")
        .iter()
        .map(|(group, rows)| {
            let rows: Vec<Vec<i64>> = rows.as_array().expect("rows").iter().map(ints).collect();
            let columns = rows.first().map_or(0, Vec::len);
            (group.clone(), BlockTable::from_rows(&rows, columns))
        })
        .collect()
}

/// A fragment as the fixture records it: `[group_id, field_id,
/// src_byte_offset, dst_byte_offset, src_row_stride_bytes,
/// dst_row_stride_bytes, bytes_per_row, rows_per_page]`.
fn fragment(value: &Value) -> Fragment {
    let item = value.as_array().expect("a fragment");
    Fragment {
        group_id: item[0].as_str().expect("a group").to_owned(),
        field_id: item[1].as_str().expect("a field").to_owned(),
        src_byte_offset: int(&item[2]),
        dst_byte_offset: int(&item[3]),
        src_row_stride_bytes: int(&item[4]),
        dst_row_stride_bytes: int(&item[5]),
        bytes_per_row: int(&item[6]),
        rows_per_page: int(&item[7]),
    }
}

fn fragments(value: &Value) -> Vec<Fragment> {
    value
        .as_array()
        .expect("fragments")
        .iter()
        .map(fragment)
        .collect()
}

fn fragment_json(fragment: &Fragment) -> Value {
    json!([
        fragment.group_id,
        fragment.field_id,
        fragment.src_byte_offset,
        fragment.dst_byte_offset,
        fragment.src_row_stride_bytes,
        fragment.dst_row_stride_bytes,
        fragment.bytes_per_row,
        fragment.rows_per_page,
    ])
}

fn owner_filters(value: &Value) -> OwnerFilters {
    value
        .as_object()
        .expect("owner filters")
        .iter()
        .map(|(group, filter)| {
            let filter = (!filter.is_null()).then(|| OwnerFilter {
                owner_rank: int(&filter["owner_rank"]),
                owner_count: int(&filter["owner_count"]),
            });
            (group.clone(), filter)
        })
        .collect()
}

fn owner_filters_json(filters: &OwnerFilters) -> Value {
    Value::Object(
        filters
            .iter()
            .map(|(group, filter)| {
                let filter = match filter {
                    None => Value::Null,
                    Some(filter) => json!({
                        "owner_rank": filter.owner_rank,
                        "owner_count": filter.owner_count,
                    }),
                };
                (group.clone(), filter)
            })
            .collect(),
    )
}

fn plan_json(plan: &RankPlan) -> Value {
    json!({
        "fragments_by_prefill_rank": plan.fragments_by_prefill_rank.iter().map(|(rank, fragments)| {
            json!([rank, fragments.iter().map(fragment_json).collect::<Vec<_>>()])
        }).collect::<Vec<_>>(),
        "owner_filters_by_prefill_rank": plan.owner_filters_by_prefill_rank.iter().map(|(rank, filters)| {
            json!([rank, owner_filters_json(filters)])
        }).collect::<Vec<_>>(),
    })
}

fn copies_json(copies: &[Copy]) -> Value {
    Value::Array(
        copies
            .iter()
            .map(|copy| match copy {
                Copy::Pages(grid) => json!({"pages": {
                    "src_pages": grid.src_pages,
                    "dst_pages": grid.dst_pages,
                    "fields": grid.fields,
                }}),
                Copy::Spans {
                    sources,
                    destinations,
                    lengths,
                } => json!({"spans": {
                    "sources": sources,
                    "destinations": destinations,
                    "lengths": lengths,
                }}),
            })
            .collect(),
    )
}

/// What the fixture records of a parsed contract.
fn contract_facts(contract: &Contract) -> Value {
    let groups: Vec<Value> = contract
        .group_specs
        .iter()
        .map(|spec| {
            json!({
                "group_id": spec.group_id,
                "shard_count": spec.shard_count,
                "virtual_block_count": contract.virtual_block_count(&spec.group_id).unwrap(),
                "fields": contract
                    .fields_for_group(&spec.group_id)
                    .iter()
                    .map(|field| field.field_id.clone())
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    let fields: Vec<Value> = contract
        .plan
        .fields
        .iter()
        .map(|field| {
            json!({
                "field_id": field.field_id,
                "address": contract.field_address(0x100000, &field.field_id).ok().map(|address| address as i64),
                "payload_bytes": field.payload_bytes(),
                "element_size": field.element_size(),
                "partition": contract.partition_for(&field.field_id).map(|partition| json!({
                    "axis": partition.axis,
                    "global_extent": partition.global_extent,
                    "global_parts": partition.global_parts,
                })),
            })
        })
        .collect();
    json!({"groups": groups, "fields": fields})
}

fn group_spec(value: &Value) -> GroupSpec {
    let optional = |key: &str| value[key].as_i64();
    GroupSpec {
        group_id: value["group_id"].as_str().expect("a group").to_owned(),
        attn_kind: value["attn_kind"].as_str().expect("a kind").to_owned(),
        rows_per_page: optional("rows_per_page"),
        entry_stride_tokens: optional("entry_stride_tokens"),
        sliding_window_tokens: optional("sliding_window_tokens"),
        replayable: value["replayable"].as_bool().expect("a flag"),
        transfer_policy: value["transfer_policy"].as_str().map(str::to_owned),
        shard_count: int(&value["shard_count"]),
    }
}

fn field_ids(value: &Value) -> Option<BTreeSet<String>> {
    (!value.is_null()).then(|| strings(value).into_iter().collect())
}

fn frames(value: &Value) -> Vec<Bytes> {
    value
        .as_array()
        .expect("frames")
        .iter()
        .map(|frame| Bytes::from(decode_hex(frame.as_str().expect("a hex frame"))))
        .collect()
}

/// Replay one record: its answer as the fixture records answers.
fn replay(fixture: &Fixture, kind: &str, args: &Value) -> Result<Value, LayoutError> {
    Ok(match kind {
        "contract_from_wire" => contract_facts(&Contract::from_wire(&raw(args))?),
        "manifest_from_wire" => manifest_json(&Manifest::from_wire(&raw(args))?),
        "validate_manifest" => {
            validate_manifest(
                &manifest(&args["manifest"]),
                fixture.contract(&args["layout"]),
                args["peer"].as_str().expect("a peer"),
            )?;
            json!(true)
        }
        "build_manifest" => manifest_json(&build_manifest(
            &tables(&args["tables"]),
            fixture.contract(&args["layout"]),
            int(&args["request_row"]) as usize,
            int(&args["prefix_len"]),
            int(&args["prompt_len"]),
            args["state_prefix_cached"].as_bool().expect("a flag"),
        )?),
        "build_selection" => selection_json(&build_selection(
            &tables(&args["tables"]),
            fixture.contract(&args["layout"]),
            int(&args["request_row"]) as usize,
            int(&args["prefix_len"]),
            int(&args["prompt_len"]),
            int(&args["chunk_start"]),
            int(&args["chunk_end"]),
            args["state_prefix_cached"].as_bool().expect("a flag"),
        )?),
        "validate_peer_layout" => {
            validate_peer(
                fixture.contract(&args["layout"]),
                fixture.contract(&args["peer_layout"]),
            )?;
            json!(true)
        }
        "pipeline_plan" => {
            let stages: Vec<Vec<String>> = args["cache_fields_by_stage"]
                .as_array()
                .expect("stages")
                .iter()
                .map(strings)
                .collect();
            let (plan, dummy) = pipeline_plan(
                int(&args["prefill_tp_size"]),
                int(&args["decode_tp_size"]),
                int(&args["decode_tp_rank"]),
                fixture.contract(&args["prefill_layout"]),
                fixture.contract(&args["decode_layout"]),
                &stages,
            )?;
            json!({"plan": plan_json(&plan), "dummy": dummy})
        }
        "planner" => {
            let planner = planner(fixture, args)?;
            json!({
                "decode_ranks_by_prefill_rank": planner
                    .decode_ranks_by_prefill_rank()
                    .iter()
                    .map(|(rank, ranks)| json!([rank, ranks.iter().collect::<Vec<_>>()]))
                    .collect::<Vec<_>>(),
                "sharded_group_ids": planner.sharded_group_ids(),
            })
        }
        "planner_plan" => plan_json(
            &planner(fixture, &args["planner"])?
                .plan_for_decode_rank(int(&args["decode_tp_rank"]))?,
        ),
        "validate_owner_filters" => {
            let specs: Vec<GroupSpec> = args["group_specs"]
                .as_array()
                .expect("specs")
                .iter()
                .map(group_spec)
                .collect();
            validate_owner_filters(
                &specs,
                &fragments(&args["fragments"]),
                &owner_filters(&args["owner_filters"]),
            )?;
            json!(true)
        }
        "local_transfer_pages" => {
            let (source, destination) = local_transfer_pages(
                &ints(&args["source_blocks"]),
                &ints(&args["destination_blocks"]),
                args["group_id"].as_str().expect("a group"),
                fixture.contract(&args["source_layout"]),
                fixture.contract(&args["destination_layout"]),
                int(&args["source_owner_rank"]),
                int(&args["destination_owner_rank"]),
            )?;
            json!({"source": source, "destination": destination})
        }
        "prepare_registration" => {
            let kv_args = &args["kv_args"];
            let topology = &args["topology"];
            let registration = &args["registration"];
            let layout = PrefillLayout {
                local: fixture.contract(&kv_args["cache_layout"]).clone(),
                wire: fixture.contract(&kv_args["wire_layout"]).clone(),
                arena_base: int(&kv_args["kv_data_ptr"]) as u64,
                tp_size: int(&topology["tp_size"]),
                tp_rank: int(&topology["tp_rank"]),
                pp_rank: int(&topology["pp_rank"]),
                fields_by_stage: kv_args["cache_fields_by_stage"]
                    .as_array()
                    .expect("stages")
                    .iter()
                    .map(strings)
                    .collect(),
            };
            let info = RegistrationInfo {
                endpoint: "decode".to_owned(),
                port: 1,
                session_id: "decode:1".to_owned(),
                arena_base: 1,
                layout: fixture.contract(&registration["peer_layout"]).clone(),
                tp_size: int(&registration["decode_tp_size"]),
                tp_rank: int(&registration["decode_tp_rank"]),
            };
            let planned = plan_registration(&layout, &info)?;
            json!({
                "fragments": planned.fragments.iter().map(fragment_json).collect::<Vec<_>>(),
                "owner_filters": owner_filters_json(&planned.owner_filters),
                "is_dummy": planned.is_dummy,
                "expected_decode_ranks": planned.expected_decode_ranks,
            })
        }
        "transfer_blocks" => {
            let source_manifest =
                (!args["src_manifest"].is_null()).then(|| manifest(&args["src_manifest"]));
            let source_selection =
                (!args["selection"].is_null()).then(|| selection(&args["selection"]));
            let source = match (&source_selection, &source_manifest) {
                (Some(selection), _) => Source::Selection(selection),
                (None, Some(manifest)) => Source::Manifest(manifest),
                (None, None) => panic!("a transfer without a source"),
            };
            let destination = manifest(&args["dst_manifest"]);
            let fragments = fragments(&args["fragments"]);
            let filters = owner_filters(&args["owner_filters"]);
            let steps: Vec<Option<BTreeSet<String>>> = args["step_field_ids"]
                .as_array()
                .expect("steps")
                .iter()
                .map(field_ids)
                .collect();
            let transfer = Transfer {
                layout: fixture.contract(&args["cache_layout"]),
                arena_base: int(&args["kv_data_ptr"]) as u64,
                destination_base: int(&args["dst_ptr"]) as u64,
                destination_tp_rank: int(&args["dst_tp_rank"]),
                destination_layout: fixture.contract(&args["dst_layout"]),
                source,
                destination: &destination,
                fragments: &fragments,
                owner_filters: &filters,
            };
            Value::Array(
                transfer
                    .copies(&steps)?
                    .iter()
                    .map(|copies| copies_json(copies))
                    .collect(),
            )
        }
        "registration_from_zmq" => {
            let info = RegistrationInfo::from_frames(&frames(&args["frames"]))?;
            json!({
                "endpoint": info.endpoint,
                "dst_port": info.port,
                "session_id": info.session_id,
                "dst_kv_ptr": info.arena_base,
                "peer_layout": contract_facts(&info.layout),
                "decode_tp_size": info.tp_size,
                "decode_tp_rank": info.tp_rank,
            })
        }
        "transfer_info_from_zmq" => {
            let info = Preallocation::from_frames(&frames(&args["frames"]))?;
            json!({
                "room": info.room,
                "session_id": info.session_id,
                "manifest": info.manifest.as_ref().map(manifest_json),
            })
        }
        "plan_route" => {
            let route = plan_route(
                args["prefill_info"].as_str().expect("prefill info"),
                fixture.contract(&args["decode_layout"]),
                int(&args["decode_tp_size"]),
                int(&args["decode_tp_rank"]),
            )?;
            json!({
                "target_tp_ranks": route.target_tp_ranks,
                "dummy_tp_ranks": route.dummy_tp_ranks,
                "expected_prefill_ranks": route.expected_prefill_ranks,
                "prefill_dp_size": route.prefill_dp_size,
            })
        }
        other => panic!("an unknown record kind {other}"),
    })
}

fn planner(fixture: &Fixture, args: &Value) -> Result<Planner, LayoutError> {
    Planner::new(
        int(&args["prefill_tp_size"]),
        int(&args["decode_tp_size"]),
        fixture.contract(&args["prefill_layout"]),
        fixture.contract(&args["decode_layout"]),
        field_ids(&args["prefill_field_ids"]).as_ref(),
    )
}

/// The recorded answer, normalized where the fixture records a contract by
/// its key: a registration's layout compares by the facts of the contract.
fn expected(fixture: &Fixture, kind: &str, answer: &Value) -> Value {
    match kind {
        "registration_from_zmq" => {
            let mut answer = answer.clone();
            answer["peer_layout"] = contract_facts(fixture.contract(&answer["peer_layout"]));
            answer
        }
        _ => answer.clone(),
    }
}

#[test]
fn the_planning_gives_the_python_planners_answers() {
    let parse = |text: &str| -> Value { serde_json::from_str(text).expect("the fixture parses") };
    let records: Vec<Value> = [
        include_str!("fixtures/planning/manifests.json"),
        include_str!("fixtures/planning/routes.json"),
        include_str!("fixtures/planning/transfers.json"),
    ]
    .into_iter()
    .flat_map(|text| parse(text).as_array().expect("records").clone())
    .collect();
    let contracts = parse(include_str!("fixtures/planning/contracts.json"))
        .as_object()
        .expect("contracts")
        .iter()
        .map(|(key, wire)| {
            let contract = Contract::from_wire(wire.as_str().expect("a wire contract").as_bytes())
                .unwrap_or_else(|error| panic!("contract {key} does not parse: {error}"));
            (key.clone(), contract)
        })
        .collect();
    let fixture = Fixture { contracts };
    let mut replayed: BTreeMap<String, usize> = BTreeMap::new();
    let mut failures = Vec::new();
    for (index, record) in records.iter().enumerate() {
        let kind = record["kind"].as_str().expect("a kind");
        let result = &record["result"];
        let answer = replay(&fixture, kind, &record["args"]);
        *replayed.entry(kind.to_owned()).or_default() += 1;
        match (answer, result.get("ok")) {
            (Ok(answer), Some(recorded)) => {
                let recorded = expected(&fixture, kind, recorded);
                if answer != recorded {
                    failures.push(format!(
                        "record {index} ({kind}): answered\n  {answer}\nbut Python answered\n  {recorded}"
                    ));
                }
            }
            (Err(_), None) => {}
            (Ok(answer), None) => failures.push(format!(
                "record {index} ({kind}): answered {answer} but Python refused: {}",
                result["message"]
            )),
            (Err(error), Some(_)) => failures.push(format!(
                "record {index} ({kind}): refused ({error}) but Python answered"
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of the records disagree:\n{}",
        failures.len(),
        failures.join("\n")
    );
    // Every ported entry point is exercised.
    assert_eq!(replayed.len(), 16, "{replayed:?}");
}
