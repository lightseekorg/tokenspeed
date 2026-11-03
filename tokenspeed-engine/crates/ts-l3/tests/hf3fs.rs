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

//! Requires a real USRBIO-enabled 3FS mount and native library.

use std::{
    env,
    time::{SystemTime, UNIX_EPOCH},
};
use ts_l3::{
    hf3fs_usrbio::{UsrBioConfig, UsrBioStore},
    keys, HostBuffer, Range, Store,
};

#[test]
#[ignore = "requires a real USRBIO-enabled 3FS mount and native library"]
fn usrbio_objects_survive_reconnect_and_preserve_owner_shards() {
    let directory = env::var_os("TOKENSPEED_TEST_HF3FS_DIRECTORY").expect("3FS directory");
    let deployment = format!(
        "3fs-integration-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let prefix = keys::key_prefix(&deployment, "weights-v1", "");
    let object_keys: Vec<_> = (0..2)
        .map(|rank| keys::object_key(&prefix, "aabb", 0, 0, rank))
        .collect();
    let ranges = [
        Range {
            offset: 0,
            size: 65536,
        },
        Range {
            offset: 65536,
            size: 65536,
        },
    ];
    let expected: Vec<u8> = (0..131072).map(|i| (i / 65536 + i % 251) as u8).collect();
    let mut bytes = expected.clone();
    let host = HostBuffer {
        base: bytes.as_mut_ptr() as usize,
        len: bytes.len(),
    };
    let store = UsrBioStore::connect(
        directory.as_ref(),
        UsrBioConfig {
            library: env::var_os("TOKENSPEED_TEST_HF3FS_LIBRARY")
                .expect("native library")
                .into(),
            entries: 16,
            buffer_bytes: 16 * 1024 * 1024,
            timeout_ms: 10000,
            segment_bytes: 64 * 1024 * 1024,
            read_rings: 1,
        },
    )
    .unwrap();
    assert_eq!(store.batch_exists(&object_keys).unwrap(), [false, false]);
    let written = store.batch_put_from(&object_keys, host, &ranges).unwrap();
    assert_eq!(written.success, [true, true]);
    assert!(written.retired);
    store.close();
    drop(store);
    bytes.fill(0);
    let store = UsrBioStore::connect(
        directory.as_ref(),
        UsrBioConfig {
            library: env::var_os("TOKENSPEED_TEST_HF3FS_LIBRARY")
                .expect("native library")
                .into(),
            entries: 16,
            buffer_bytes: 16 * 1024 * 1024,
            timeout_ms: 10000,
            segment_bytes: 64 * 1024 * 1024,
            read_rings: 1,
        },
    )
    .unwrap();
    assert_eq!(store.batch_exists(&object_keys).unwrap(), [true, true]);
    let read = store.batch_get_into(&object_keys, host, &ranges).unwrap();
    assert_eq!(read.success, [true, true]);
    assert!(read.retired);
    assert_eq!(bytes, expected);
    let other_model = keys::object_key(
        &keys::key_prefix(&deployment, "weights-v2", ""),
        "aabb",
        0,
        0,
        0,
    );
    assert_eq!(store.batch_exists(&[other_model]).unwrap(), [false]);
    assert!(store.remove_by_prefix(&format!("{prefix}_")));
    assert_eq!(store.batch_exists(&object_keys).unwrap(), [false, false]);
}
