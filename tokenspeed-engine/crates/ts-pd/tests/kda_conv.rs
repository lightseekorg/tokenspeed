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

//! Execute the Rust planner over Python-produced KDA schemas, checking the
//! raw Q/K/V history bytes across TP widths and both convolution layouts.

use std::collections::BTreeSet;

use serde_json::Value;
use ts_pd::layout::{contract::Contract, route::Planner};

const CHANNELS: usize = 64 * 128;
const HISTORY_ROWS: usize = 3;
const QKV: usize = 3;

fn field_id(axis: usize) -> &'static str {
    if axis == 1 {
        "layer.0.conv_state_channels_last"
    } else {
        "layer.0.conv_state"
    }
}

fn contract(fixtures: &Value, tp: usize, axis: usize, padding: usize) -> Contract {
    let key = format!("tp{tp}-axis{axis}-padding{padding}");
    Contract::from_wire(&serde_json::to_vec(&fixtures[&key]).unwrap()).unwrap()
}

fn raw_bytes(tp: usize, rank: usize, axis: usize) -> Vec<u8> {
    let width = CHANNELS / tp;
    let value = |history: usize, channel: usize| {
        let projection = channel / width;
        let global_channel = rank * width + channel % width;
        ((history * QKV + projection) * CHANNELS + global_channel) as u16
    };
    let mut bytes = Vec::new();
    if axis == 1 {
        for history in 0..HISTORY_ROWS {
            for channel in 0..QKV * width {
                bytes.extend(value(history, channel).to_le_bytes());
            }
        }
    } else {
        for channel in 0..QKV * width {
            for history in 0..HISTORY_ROWS {
                bytes.extend(value(history, channel).to_le_bytes());
            }
        }
    }
    bytes
}

fn check_routes(source_tp: usize, destination_tp: usize) {
    let fixtures: Value =
        serde_json::from_str(include_str!("fixtures/planning/kda_conv.json")).unwrap();
    for axis in [0, 1] {
        let source = contract(&fixtures, source_tp, axis, 256);
        let destination = contract(&fixtures, destination_tp, axis, 384);
        let field_id = field_id(axis);
        let source_field = source.field(field_id).unwrap();
        let destination_field = destination.field(field_id).unwrap();
        for conv_only in [false, true] {
            let fields = BTreeSet::from([field_id.to_owned()]);
            let planner = Planner::new(
                source_tp as i64,
                destination_tp as i64,
                &source,
                &destination,
                conv_only.then_some(&fields),
            )
            .unwrap();
            for destination_rank in 0..destination_tp {
                let route = planner
                    .plan_for_decode_rank(destination_rank as i64)
                    .unwrap();
                let guard = 31;
                let begin = guard + 3 * destination_field.page_stride_bytes as usize;
                let payload_bytes = destination_field.payload_bytes() as usize;
                let mut output = vec![0xc3; begin + payload_bytes + guard];
                let mut writes = vec![0; payload_bytes];
                for (source_rank, fragments) in route.fragments_by_prefill_rank {
                    let payload = raw_bytes(source_tp, source_rank as usize, axis);
                    let source_begin = guard + 2 * source_field.page_stride_bytes as usize;
                    let mut input = vec![0xa5; source_begin + payload.len() + guard];
                    input[source_begin..source_begin + payload.len()].copy_from_slice(&payload);
                    if fragments.is_empty() {
                        // Equal TP with all fields uses the whole-field route.
                        assert_eq!(source_tp, destination_tp);
                        assert!(!conv_only);
                        assert_eq!(source_rank as usize, destination_rank);
                        output[begin..begin + payload.len()]
                            .copy_from_slice(&input[source_begin..source_begin + payload.len()]);
                        writes.iter_mut().for_each(|count| *count += 1);
                    }
                    for fragment in fragments.iter().filter(|part| part.field_id == field_id) {
                        for row in 0..fragment.rows_per_page {
                            let src = source_begin
                                + (fragment.src_byte_offset + row * fragment.src_row_stride_bytes)
                                    as usize;
                            let dst = (fragment.dst_byte_offset
                                + row * fragment.dst_row_stride_bytes)
                                as usize;
                            let width = fragment.bytes_per_row as usize;
                            output[begin + dst..begin + dst + width]
                                .copy_from_slice(&input[src..src + width]);
                            writes[dst..dst + width]
                                .iter_mut()
                                .for_each(|count| *count += 1);
                        }
                    }
                }
                assert!(writes.iter().all(|&count| count == 1));
                assert_eq!(
                    &output[begin..begin + payload_bytes],
                    raw_bytes(destination_tp, destination_rank, axis),
                    "TP {source_tp}->{destination_tp}, axis {axis}, conv_only {conv_only}"
                );
                assert!(output[..begin].iter().all(|&byte| byte == 0xc3));
                assert!(output[begin + payload_bytes..]
                    .iter()
                    .all(|&byte| byte == 0xc3));
            }
        }
    }
}

#[test]
fn tp8_to_tp8() {
    check_routes(8, 8);
}

#[test]
fn tp8_to_tp4() {
    check_routes(8, 4);
}

#[test]
fn tp4_to_tp8() {
    check_routes(4, 8);
}

#[test]
fn tp8_to_tp16() {
    check_routes(8, 16);
}

#[test]
fn tp16_to_tp8() {
    check_routes(16, 8);
}
