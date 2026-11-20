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

//! Store C ABI integration tests. With no library/master variables, the native
//! cases print an explicit skip; unit tests never claim transport coverage.

use std::{path::PathBuf, sync::Arc};

use ts_mooncake::store::{regex_escape, Store, StoreLibrary, StoreSetup};

fn setup(master: &str, tenant: &str) -> StoreSetup {
    StoreSetup {
        local_hostname: "127.0.0.1".to_owned(),
        metadata_server: "P2PHANDSHAKE".to_owned(),
        global_segment_size: 16 * 1024 * 1024,
        local_buffer_size: 4 * 1024 * 1024,
        protocol: "tcp".to_owned(),
        device_name: String::new(),
        master_server_address: master.to_owned(),
        tenant_id: tenant.to_owned(),
    }
}

#[test]
fn v1_put_reports_retirement_and_preserves_source_bytes() {
    let (Some(path), Ok(master)) = (
        std::env::var_os("TS_MOONCAKE_STORE_LIBRARY"),
        std::env::var("TS_MOONCAKE_STORE_MASTER"),
    ) else {
        eprintln!("TS_MOONCAKE_STORE_LIBRARY/MASTER is not set; skipping native TCP test");
        return;
    };
    let library = StoreLibrary::load(&PathBuf::from(path)).unwrap();
    let store = Store::connect(Arc::clone(&library), &setup(&master, "default")).unwrap();
    let key = format!("store-retirement-v1-{}", std::process::id());
    let mut source: Vec<u8> = (0..8192).map(|i| (i % 251) as u8).collect();
    let expected = source.clone();
    let mut destination = vec![0_u8; source.len()];
    let src = source.as_ptr() as usize;
    let dst = destination.as_mut_ptr() as usize;
    store.register_buffer(src, source.len()).unwrap();
    store.register_buffer(dst, destination.len()).unwrap();
    let outcome = store
        .batch_put_from_with_retirement(std::slice::from_ref(&key), &[src], &[source.len()])
        .unwrap();
    assert_eq!(outcome.results, [0]);
    assert_eq!(outcome.source_retired, Some(true));
    source.fill(0xAB);
    assert_eq!(
        store.batch_is_exist(std::slice::from_ref(&key)).unwrap(),
        [1]
    );
    assert_eq!(
        store
            .batch_get_into(std::slice::from_ref(&key), &[dst], &[destination.len()],)
            .unwrap(),
        [source.len() as i64]
    );
    assert_eq!(destination, expected);
    assert!(source.iter().all(|&byte| byte == 0xAB));
    assert!(
        store
            .remove_by_regex(&format!("^{}$", regex_escape(&key)), true)
            .unwrap()
            >= 0
    );
    store.unregister_buffer(dst).unwrap();
    store.unregister_buffer(src).unwrap();
}
