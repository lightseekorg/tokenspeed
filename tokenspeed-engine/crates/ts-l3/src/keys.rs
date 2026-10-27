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

//! Object keys: a namespace per deployment and weight generation, and one
//! object per packed Host cache block.
//!
//! The formats are the stored objects' names, shared by every engine that
//! reads or writes the store: a change orphans what is already there.

use std::fmt::Write;

use sha2::{Digest, Sha256};

/// The namespace of a deployment's objects under its target and draft
/// weight versions: `tsl3-` and the SHA-256 of the three, as canonical JSON.
///
/// `deployment_id` is the frozen identity of everything that shapes the
/// stored bytes (model, cache layout, parallel layout, numerics); an empty
/// `draft_weight_version` means no draft generation.
pub fn key_prefix(deployment_id: &str, weight_version: &str, draft_weight_version: &str) -> String {
    // `json.dumps(..., sort_keys=True, separators=(",", ":"))`.
    let payload = format!(
        "{{\"deployment_id\":{},\"draft_weight_version\":{},\"weight_version\":{}}}",
        json_string(deployment_id),
        json_string(draft_weight_version),
        json_string(weight_version),
    );
    let digest = Sha256::digest(payload.as_bytes());
    let mut prefix = String::with_capacity(5 + 2 * digest.len());
    prefix.push_str("tsl3-");
    for byte in digest {
        write!(prefix, "{byte:02x}").expect("writing to a String");
    }
    prefix
}

/// The object of one packed Host cache block: the scheduler's content hash
/// in `prefix`'s namespace, plus the group, the block's ordinal within its
/// prefix page and the rank that owns the shard. The trailing `|c0` is the
/// retired context-parallel shard id, kept so existing objects stay
/// addressable.
pub fn object_key(
    prefix: &str,
    content_hash: &str,
    group: u32,
    page_offset: i32,
    rank: i64,
) -> String {
    format!("{prefix}_{content_hash}|g{group}|o{page_offset}|r{rank}|c0")
}

/// `value` as Python's `json.dumps` writes a string: ASCII only, the short
/// escapes for the usual control characters and `\uXXXX` (UTF-16 units)
/// for the rest.
fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for character in value.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ' '..='~' => out.push(character),
            _ => {
                let mut units = [0_u16; 2];
                for unit in character.encode_utf16(&mut units) {
                    write!(out, "\\u{unit:04x}").expect("writing to a String");
                }
            }
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_strings_escape_as_python_does() {
        assert_eq!(json_string("v1"), "\"v1\"");
        assert_eq!(json_string("a\"b\\c"), "\"a\\\"b\\\\c\"");
        assert_eq!(json_string("\t\n\u{1}\u{7f}"), "\"\\t\\n\\u0001\\u007f\"");
        // Outside the BMP: a surrogate pair.
        assert_eq!(json_string("é😀"), "\"\\u00e9\\ud83d\\ude00\"");
    }

    #[test]
    fn prefixes_match_the_python_namespaces() {
        // Names engines already wrote objects under: a deployment's stored
        // objects keep their names across engine versions.
        assert_eq!(
            key_prefix("0123abcd", "default", ""),
            "tsl3-25bbe86d95ca1c75d7cf2b311774d571c684bd9d967e2ef25bd4769e392ebb98"
        );
        let version = "v2-é😀\"x";
        assert_eq!(
            key_prefix("dep", version, version),
            "tsl3-8dcc778934a5e4e3e9613e82643df0983bad543e9389867a447724f741566223"
        );
    }

    #[test]
    fn target_and_draft_weight_versions_name_their_own_namespaces() {
        let base = key_prefix("dep", "v1", "draft-v1");
        let target = key_prefix("dep", "v2", "draft-v1");
        let draft = key_prefix("dep", "v1", "draft-v2");
        assert!(base != target && base != draft && target != draft);
        // No draft generation is a namespace of its own.
        assert_ne!(key_prefix("dep", "v1", ""), key_prefix("dep", "v1", "v1"));
    }

    #[test]
    fn object_keys_name_the_shard() {
        assert_eq!(
            object_key("tsl3-ab", "f00d", 2, 1, 3),
            "tsl3-ab_f00d|g2|o1|r3|c0"
        );
    }
}
