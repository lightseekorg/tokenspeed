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

//! Legacy SDK loading is tested in a separate executable: SDKs contain
//! process-wide gflags registrations and different versions cannot be loaded
//! into the same test process.

use std::path::PathBuf;
use ts_mooncake::{
    store::{Store, StoreLibrary, StoreSetup},
    Error,
};

fn setup(master: &str, tenant: &str) -> StoreSetup {
    StoreSetup {
        local_hostname: "127.0.0.1".to_owned(),
        metadata_server: "P2PHANDSHAKE".to_owned(),
        global_segment_size: 0,
        local_buffer_size: 4 * 1024 * 1024,
        protocol: "tcp".to_owned(),
        device_name: String::new(),
        master_server_address: master.to_owned(),
        tenant_id: tenant.to_owned(),
    }
}

#[test]
fn upstream_setup_rejects_a_named_tenant_before_connecting() {
    let Some(path) = std::env::var_os("TS_MOONCAKE_LEGACY_STORE_LIBRARY") else {
        eprintln!("TS_MOONCAKE_LEGACY_STORE_LIBRARY is not set; skipping native ABI test");
        return;
    };
    let library = StoreLibrary::load(&PathBuf::from(path)).unwrap();
    assert!(matches!(
        Store::connect(library, &setup("127.0.0.1:1", "named-tenant")),
        Err(Error::Setup { status: -1012, .. })
    ));
}
