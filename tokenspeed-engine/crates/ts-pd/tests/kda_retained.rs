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

//! Transfer a retained checkpoint as bytes, then replay only its accepted prefix.

use std::collections::BTreeSet;

use serde_json::json;
use ts_pd::layout::{contract::Contract, route::Planner};

const HEADS: usize = 16;
const VALUE: usize = 2;
const KEY: usize = 3;
const HISTORY: usize = 4;
const TAPS: usize = 3;
const GUARD: usize = 19;
const SOURCE_PAGE: usize = 2;
const DESTINATION_PAGE: usize = 3;
const NAMES: [&str; 6] = [
    "conv_state",
    "recurrent_state",
    "replay_correction",
    "replay_key",
    "replay_decay",
    "replay_length",
];

fn contract(tp: usize, capacity: usize, padding: usize) -> Contract {
    let heads = HEADS / tp;
    let shapes = [
        vec![heads * (2 * KEY + VALUE), TAPS],
        vec![heads, VALUE, KEY],
        vec![heads, capacity, VALUE],
        vec![heads, capacity, KEY],
        vec![heads, capacity, KEY],
        vec![1],
    ];
    let mut fields = Vec::new();
    let mut planes = Vec::new();
    let mut partitions = Vec::new();
    let mut arena_offset = 0;
    let mut block_bytes = 0;
    for (index, name) in NAMES.iter().enumerate() {
        let field_id = format!("layer.0.{name}");
        let plane_id = format!("state.{name}");
        let element_size = if index == 0 { 2 } else { 4 };
        let stride = shapes[index].iter().product::<usize>() * element_size + padding + 8;
        fields.push(json!({
            "group_id": "state", "field_id": field_id, "plane_id": plane_id,
            "shape": shapes[index],
            "dtype": if index == 0 { "bfloat16" } else if index == 5 { "int32" } else { "float32" },
            "field_offset_bytes": 8, "page_stride_bytes": stride,
        }));
        planes.push(json!({
            "plane_id": plane_id, "bytes_per_lcm_block": stride,
            "arena_offset_bytes": arena_offset,
        }));
        arena_offset += 6 * stride;
        block_bytes += stride;
        if index != 5 {
            partitions.push(json!({
                "field_id": field_id,
                "partition": {
                    "axis": 0,
                    "global_extent": if index == 0 { HEADS * (2 * KEY + VALUE) } else { HEADS },
                    "global_parts": if index == 0 { vec![HEADS * KEY, HEADS * KEY, HEADS * VALUE] } else { vec![] },
                },
            }));
        }
    }
    let wire = json!({
        "plan": {
            "prefix_granularity": 2, "lcm_block_bytes": block_bytes,
            "num_lcm_blocks": 5,
            "groups": [{"group_id": "state", "cache_blocks_per_lcm_block": 1, "page_count": 6}],
            "planes": planes, "fields": fields, "resident_block_bytes": null,
        },
        "group_specs": [{
            "group_id": "state", "attn_kind": "state", "replayable": false,
            "transfer_policy": "latest_snapshot", "shard_count": 1,
        }],
        "transfer_schema": {"fields": partitions}, "field_addresses": null,
    });
    Contract::from_wire(&serde_json::to_vec(&wire).unwrap()).unwrap()
}

fn base(head: usize, value: usize, key: usize) -> f32 {
    (1 + head * 6 + value * KEY + key) as f32 / 16.0
}

fn correction(head: usize, token: usize, value: usize) -> f32 {
    (1 + head + 2 * token + value) as f32 / 16.0
}

fn key(head: usize, token: usize, coordinate: usize) -> f32 {
    (1 + head + token + coordinate) as f32 / 32.0
}

fn decay(head: usize, token: usize, coordinate: usize) -> f32 {
    (4 + (head + token + coordinate) % 4) as f32 / 8.0
}

fn payload(index: usize, tp: usize, rank: usize, accepted: usize) -> Vec<u8> {
    let heads = HEADS / tp;
    let mut bytes = Vec::new();
    if index == 0 {
        // Each projection owns its own TP slice, including the narrower V.
        for (projection, dimension) in [KEY, KEY, VALUE].into_iter().enumerate() {
            for channel in 0..heads * dimension {
                let global_channel = rank * heads * dimension + channel;
                for tap in 0..TAPS {
                    // Distinct finite BF16 bit patterns preserve every coordinate.
                    let bf16 = 0x3f00 + (projection * 512 + global_channel * TAPS + tap) as u16;
                    bytes.extend(bf16.to_le_bytes());
                }
            }
        }
    } else if index == 5 {
        bytes.extend((accepted as i32).to_le_bytes());
    } else {
        for head in rank * heads..(rank + 1) * heads {
            if index == 1 {
                for value in 0..VALUE {
                    for coordinate in 0..KEY {
                        bytes.extend(base(head, value, coordinate).to_le_bytes());
                    }
                }
            } else {
                for token in 0..HISTORY {
                    let dimension = if index == 2 { VALUE } else { KEY };
                    for coordinate in 0..dimension {
                        let value = if token >= accepted {
                            f32::from_bits(0x7fc0_0123)
                        } else {
                            match index {
                                2 => correction(head, token, coordinate),
                                3 => key(head, token, coordinate),
                                4 => decay(head, token, coordinate),
                                _ => unreachable!(),
                            }
                        };
                        bytes.extend(value.to_le_bytes());
                    }
                }
            }
        }
    }
    bytes
}

fn begin(layout: &Contract, index: usize, page: usize) -> usize {
    let field = &layout.plan.fields[index];
    GUARD + field.field_offset_bytes as usize + page * field.page_stride_bytes as usize
}

fn buffers(layout: &Contract, fill: u8) -> Vec<Vec<u8>> {
    layout
        .plan
        .fields
        .iter()
        .map(|field| vec![fill; 2 * GUARD + 6 * field.page_stride_bytes as usize])
        .collect()
}

fn read_f32(bytes: &[u8], element: usize) -> f32 {
    f32::from_le_bytes(bytes[element * 4..element * 4 + 4].try_into().unwrap())
}

fn check_retained_routes(
    source_tp: usize,
    destination_tp: usize,
    accepted: usize,
    phased_selection: bool,
) {
    let source = contract(source_tp, HISTORY, 32);
    let destination = contract(destination_tp, HISTORY, 64);
    let mut inputs = Vec::new();
    for rank in 0..source_tp {
        let mut input = buffers(&source, 0xa5);
        for (index, buffer) in input.iter_mut().enumerate() {
            let bytes = payload(index, source_tp, rank, accepted);
            let start = begin(&source, index, SOURCE_PAGE);
            buffer[start..start + bytes.len()].copy_from_slice(&bytes);
        }
        inputs.push(input);
    }
    let original_inputs = inputs.clone();
    let selected_fields: Vec<Option<BTreeSet<String>>> = if phased_selection {
        NAMES
            .chunks(3)
            .map(|names| Some(names.iter().map(|name| format!("layer.0.{name}")).collect()))
            .collect()
    } else {
        vec![None]
    };
    let planners: Vec<Planner> = selected_fields
        .iter()
        .map(|fields| {
            Planner::new(
                source_tp as i64,
                destination_tp as i64,
                &source,
                &destination,
                fields.as_ref(),
            )
            .unwrap()
        })
        .collect();
    for rank in 0..destination_tp {
        let mut output = buffers(&destination, 0xc3);
        let mut writes: Vec<Vec<usize>> =
            output.iter().map(|buffer| vec![0; buffer.len()]).collect();
        for (phase, planner) in planners.iter().enumerate() {
            let route = planner.plan_for_decode_rank(rank as i64).unwrap();
            for (source_rank, fragments) in route.fragments_by_prefill_rank {
                if fragments.is_empty() {
                    assert!(
                        !phased_selection,
                        "selected subsets must use explicit fragments"
                    );
                    assert_eq!(source_tp, destination_tp);
                    assert_eq!(source_rank as usize, rank);
                    for (index, _) in NAMES.iter().enumerate() {
                        let src = begin(&source, index, SOURCE_PAGE);
                        let dst = begin(&destination, index, DESTINATION_PAGE);
                        let width = source.plan.fields[index].payload_bytes() as usize;
                        output[index][dst..dst + width].copy_from_slice(
                            &inputs[source_rank as usize][index][src..src + width],
                        );
                        writes[index][dst..dst + width]
                            .iter_mut()
                            .for_each(|count| *count += 1);
                    }
                }
                for fragment in fragments {
                    let index = destination
                        .plan
                        .fields
                        .iter()
                        .position(|field| field.field_id == fragment.field_id)
                        .unwrap();
                    for row in 0..fragment.rows_per_page {
                        let src = begin(&source, index, SOURCE_PAGE)
                            + (fragment.src_byte_offset + row * fragment.src_row_stride_bytes)
                                as usize;
                        let dst = begin(&destination, index, DESTINATION_PAGE)
                            + (fragment.dst_byte_offset + row * fragment.dst_row_stride_bytes)
                                as usize;
                        let width = fragment.bytes_per_row as usize;
                        output[index][dst..dst + width].copy_from_slice(
                            &inputs[source_rank as usize][index][src..src + width],
                        );
                        writes[index][dst..dst + width]
                            .iter_mut()
                            .for_each(|count| *count += 1);
                    }
                }
            }
            if phased_selection && phase == 0 {
                // The second phase has not selected key, decay or length yet.
                for (buffer, counts) in output.iter().zip(&writes).skip(3) {
                    assert!(buffer.iter().all(|&byte| byte == 0xc3));
                    assert!(counts.iter().all(|&count| count == 0));
                }
            }
        }
        for (index, _) in NAMES.iter().enumerate() {
            let expected = payload(index, destination_tp, rank, accepted);
            let start = begin(&destination, index, DESTINATION_PAGE);
            let end = start + expected.len();
            assert_eq!(
                &output[index][start..end],
                expected,
                "TP {source_tp}->{destination_tp}, rank {rank}, field {}",
                NAMES[index]
            );
            assert!(output[index][..start]
                .iter()
                .chain(&output[index][end..])
                .all(|&byte| byte == 0xc3));
            for (offset, count) in writes[index].iter().enumerate() {
                assert_eq!(
                    *count,
                    usize::from((start..end).contains(&offset)),
                    "writer count for field {} byte {offset}",
                    NAMES[index]
                );
            }
        }
        let local_payload: Vec<&[u8]> = output
            .iter()
            .enumerate()
            .map(|(index, buffer)| {
                let start = begin(&destination, index, DESTINATION_PAGE);
                &buffer[start..start + destination.plan.fields[index].payload_bytes() as usize]
            })
            .collect();
        let length = i32::from_le_bytes(local_payload[5].try_into().unwrap()) as usize;
        assert_eq!(length, accepted);
        for local_head in 0..HEADS / destination_tp {
            let global_head = rank * (HEADS / destination_tp) + local_head;
            for value in 0..VALUE {
                for coordinate in 0..KEY {
                    let mut actual = read_f32(
                        local_payload[1],
                        (local_head * VALUE + value) * KEY + coordinate,
                    );
                    for token in 0..length {
                        let correction = read_f32(
                            local_payload[2],
                            (local_head * HISTORY + token) * VALUE + value,
                        );
                        let key = read_f32(
                            local_payload[3],
                            (local_head * HISTORY + token) * KEY + coordinate,
                        );
                        let decay = read_f32(
                            local_payload[4],
                            (local_head * HISTORY + token) * KEY + coordinate,
                        );
                        actual = actual * decay + correction * key;
                    }
                    // Closed-form endpoint from global coordinates, independent of
                    // the received buffers and the destination recurrence loop.
                    let tail = |start| {
                        (start..accepted)
                            .map(|token| decay(global_head, token, coordinate) as f64)
                            .product::<f64>()
                    };
                    let expected = base(global_head, value, coordinate) as f64 * tail(0)
                        + (0..accepted)
                            .map(|token| {
                                correction(global_head, token, value) as f64
                                    * key(global_head, token, coordinate) as f64
                                    * tail(token + 1)
                            })
                            .sum::<f64>();
                    assert!(actual.is_finite());
                    assert!(
                        (actual as f64 - expected).abs() < 1e-6,
                        "endpoint TP {source_tp}->{destination_tp}, rank {rank}, head {global_head}, value {value}, key {coordinate}"
                    );
                }
            }
        }
    }
    assert_eq!(
        inputs, original_inputs,
        "source buffers and guards must remain unchanged"
    );
}

#[test]
fn retained_checkpoint_survives_tp_transfers() {
    for (source_tp, destination_tp) in [(8, 8), (8, 4), (4, 8), (8, 16), (16, 8)] {
        for accepted in [0, 3] {
            check_retained_routes(source_tp, destination_tp, accepted, false);
        }
    }
}

#[test]
fn retained_checkpoint_survives_selected_field_phases() {
    check_retained_routes(8, 8, 3, true);
}

#[test]
fn different_retained_capacities_are_rejected() {
    let source = contract(8, 4, 32);
    for destination_tp in [8, 4, 16] {
        let destination = contract(destination_tp, 8, 64);
        assert!(Planner::new(8, destination_tp as i64, &source, &destination, None).is_err());
    }
}
