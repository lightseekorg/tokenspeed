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

//! The executor against a live Mooncake Store over TCP.
//!
//! `TS_MOONCAKE_STORE_LIBRARY` names `libmooncake_store.so`; the tests start
//! the `mooncake_master` beside it (or `TS_MOONCAKE_MASTER_BIN`) unless
//! `MOONCAKE_MASTER` names a running one; the tenant test always starts its
//! own, in multi-tenant mode. Without the library they report a skip, unless
//! `MOONCAKE_REQUIRE_MASTER=1`.

use std::{
    net::{TcpListener, TcpStream},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{Mutex, MutexGuard},
    thread::sleep,
    time::{Duration, Instant},
};

use ts_l3::{
    monotonic_clock, mooncake::MooncakeStore, Executor, ExecutorConfig, GroupLayout, HostBuffer,
    HostLayout, Namespace, Operation, StorageCompletion, StorageKey, Store,
};
use ts_mooncake::store::StoreSetup;

/// One master at a time: the tests share the process's store clients.
static SERIAL: Mutex<()> = Mutex::new(());

struct Master {
    address: String,
    /// A master of the test's own: how to start it again.
    command: Option<Command>,
    child: Option<Child>,
}

impl Master {
    fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    /// Start the test's own master again, at the same address.
    fn restart(&mut self) {
        let command = self.command.as_mut().expect("a master of the test's own");
        self.child = Some(command.spawn().expect("restarting mooncake_master"));
        wait_for(&self.address, &mut self.child);
    }
}

impl Drop for Master {
    fn drop(&mut self) {
        self.stop();
    }
}

fn required() -> bool {
    matches!(
        std::env::var("MOONCAKE_REQUIRE_MASTER").as_deref(),
        Ok("1" | "true" | "yes")
    )
}

/// The store library, or None (a skip) without one.
fn library() -> Option<PathBuf> {
    match std::env::var_os("TS_MOONCAKE_STORE_LIBRARY") {
        Some(path) => Some(PathBuf::from(path)),
        None if required() => panic!("MOONCAKE_REQUIRE_MASTER needs TS_MOONCAKE_STORE_LIBRARY"),
        None => {
            eprintln!("skipped: TS_MOONCAKE_STORE_LIBRARY is unset");
            None
        }
    }
}

fn free_port() -> u16 {
    TcpListener::bind("0.0.0.0:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn wait_for(address: &str, child: &mut Option<Child>) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        if let Some(child) = child {
            if let Some(status) = child.try_wait().unwrap() {
                panic!("mooncake_master exited before it listened: {status}");
            }
        }
        if TcpStream::connect(address).is_ok() {
            return;
        }
        sleep(Duration::from_millis(100));
    }
    panic!("mooncake_master at {address} did not come up");
}

fn master(library: &std::path::Path) -> Master {
    if let Ok(address) = std::env::var("MOONCAKE_MASTER") {
        wait_for(&address, &mut None);
        return Master {
            address,
            command: None,
            child: None,
        };
    }
    own_master(library, &[])
}

/// A master of this test's own; with `tenants`, in multi-tenant mode with a
/// quota for each.
fn own_master(library: &std::path::Path, tenants: &[&str]) -> Master {
    let binary = std::env::var_os("TS_MOONCAKE_MASTER_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| library.with_file_name("mooncake_master"));
    let port = free_port();
    let log_dir = std::env::temp_dir().join(format!("ts-l3-master-{port}"));
    std::fs::create_dir_all(&log_dir).unwrap();
    let mut command = Command::new(&binary);
    command
        .arg(format!("--port={port}"))
        .arg(format!("--metrics_port={}", free_port()))
        .arg("--enable_http_metadata_server=false")
        .arg("--enable_metric_reporting=false")
        .arg(format!("--log_dir={}", log_dir.display()));
    if !tenants.is_empty() {
        let quotas = log_dir.join("tenants.yaml");
        let entries: String = tenants
            .iter()
            .map(|tenant| format!("  - name: {tenant}\n    quota: {}\n", 1_u64 << 30))
            .collect();
        std::fs::write(&quotas, format!("version: 1\ntenants:\n{entries}")).unwrap();
        command
            .arg("--enable_multi_tenants=true")
            .arg(format!("--tenant_quota_connector_uri={}", quotas.display()));
    }
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let mut child = Some(
        command
            .spawn()
            .unwrap_or_else(|error| panic!("starting {}: {error}", binary.display())),
    );
    let address = format!("127.0.0.1:{port}");
    wait_for(&address, &mut child);
    Master {
        address,
        command: Some(command),
        child,
    }
}

fn setup(master: &Master, global_segment_size: u64) -> StoreSetup {
    StoreSetup {
        local_hostname: "localhost".to_owned(),
        metadata_server: "P2PHANDSHAKE".to_owned(),
        global_segment_size,
        local_buffer_size: 16 * 1024 * 1024,
        protocol: "tcp".to_owned(),
        device_name: String::new(),
        master_server_address: master.address.clone(),
        tenant_id: "default".to_owned(),
    }
}

const CONFIG: ExecutorConfig = ExecutorConfig {
    max_operations: 32,
    max_keys_per_operation: 256,
    query_workers: 2,
    read_workers: 2,
    write_workers: 1,
    failure_threshold: 1,
    recovery_interval: Duration::from_secs(30),
};

/// 64-byte blocks, one per LCM block, in a 4 KiB pool; with `shard_count`
/// owners, this is owner and attention-TP rank `rank`.
fn layout(shard_count: usize, rank: usize) -> HostLayout {
    HostLayout {
        groups: vec![GroupLayout {
            packing: 1,
            block_bytes: 64,
            shard_count,
        }],
        lcm_block_bytes: 64,
        num_lcm_blocks: 64,
        owner_rank: rank,
        tp_rank: rank,
    }
}

/// An executor over a store client and its own registered pool.
struct Client {
    executor: Option<Executor>,
    pool: Vec<u8>,
}

impl Client {
    fn start(
        library: &std::path::Path,
        setup: &StoreSetup,
        layout: HostLayout,
        deployment: &str,
    ) -> Self {
        Self::start_with(library, setup, layout, deployment, CONFIG)
    }

    fn start_with(
        library: &std::path::Path,
        setup: &StoreSetup,
        layout: HostLayout,
        deployment: &str,
        config: ExecutorConfig,
    ) -> Self {
        let mut pool = vec![0_u8; layout.bytes()];
        let host = HostBuffer {
            base: pool.as_mut_ptr() as usize,
            len: pool.len(),
        };
        let store = MooncakeStore::connect(library, setup, host).unwrap();
        let executor = Executor::start(
            Box::new(store) as Box<dyn Store>,
            host,
            layout,
            Namespace {
                deployment_id: deployment.to_owned(),
                has_draft: false,
            },
            "v1",
            config,
            monotonic_clock(),
        );
        Self {
            executor: Some(executor),
            pool,
        }
    }

    fn executor(&self) -> &Executor {
        self.executor.as_ref().unwrap()
    }

    fn run(&self, operation: Operation) -> StorageCompletion {
        let op_id = match &operation {
            Operation::Query { op_id, .. }
            | Operation::Get { op_id, .. }
            | Operation::Put { op_id, .. }
            | Operation::Cancel { op_id } => *op_id,
        };
        self.executor().submit(&[operation]).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            let completions = self.executor().poll().completions;
            if let Some(completion) = completions.into_iter().next() {
                assert_eq!(completion.op_id, op_id);
                assert!(completion.retired);
                return completion;
            }
            sleep(Duration::from_millis(10));
        }
        panic!("operation {op_id} did not complete");
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        if let Some(mut executor) = self.executor.take() {
            executor.close(Duration::from_secs(10)).unwrap();
        }
    }
}

fn key(hash: &str, page_offset: i32) -> StorageKey {
    StorageKey {
        group: 0,
        content_hash: hash.to_owned(),
        page_offset,
    }
}

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn unique(name: &str) -> String {
    format!(
        "{name}-{}-{}",
        std::process::id(),
        Instant::now().elapsed().as_nanos()
    )
}

#[test]
fn objects_round_trip_through_the_master() {
    let _serial = serial();
    let Some(library) = library() else {
        return;
    };
    let master = master(&library);
    let deployment = unique("round-trip");
    let client = Client::start(
        &library,
        &setup(&master, 64 << 20),
        layout(1, 0),
        &deployment,
    );
    let payload = b"mooncake-l3-bytes";
    // SAFETY: the pool is live and no job runs.
    #[allow(unsafe_code)]
    unsafe {
        std::ptr::copy_nonoverlapping(payload.as_ptr(), client.pool.as_ptr() as *mut u8, 17);
    }
    let block = |op_id, put: bool| {
        let keys = vec![key("h0", 0)];
        if put {
            Operation::Put {
                op_id,
                keys,
                host_blocks: vec![1],
            }
        } else {
            Operation::Get {
                op_id,
                keys,
                host_blocks: vec![1],
            }
        }
    };
    let query = |op_id| Operation::Query {
        op_id,
        keys: vec![key("h0", 0)],
    };
    assert_eq!(client.run(block(1, true)).success, [true]);
    assert_eq!(client.run(query(2)).success, [true]);
    // SAFETY: as above.
    #[allow(unsafe_code)]
    unsafe {
        std::ptr::write_bytes(client.pool.as_ptr() as *mut u8, 0, 17);
    }
    assert_eq!(client.run(block(3, false)).success, [true]);
    assert_eq!(&client.pool[..17], payload);
    assert!(client.executor().delete_namespace());
    assert_eq!(client.run(query(4)).success, [false]);
    // A miss after the namespace went is a retired object failure.
    let miss = client.run(block(5, false));
    assert_eq!(
        (miss.success, miss.object_failures),
        (vec![false], vec![true])
    );
}

#[test]
fn objects_outlive_their_writer() {
    let _serial = serial();
    let Some(library) = library() else {
        return;
    };
    let master = master(&library);
    let deployment = unique("outlive");
    // The segment owner; the clients mount nothing.
    let owner = Client::start(
        &library,
        &setup(&master, 64 << 20),
        layout(1, 0),
        &deployment,
    );
    let clients = setup(&master, 0);
    let payload: Vec<u8> = (0..64).collect();
    let writer = Client::start(&library, &clients, layout(1, 0), &deployment);
    // SAFETY: the pool is live and no job runs.
    #[allow(unsafe_code)]
    unsafe {
        std::ptr::copy_nonoverlapping(payload.as_ptr(), writer.pool.as_ptr() as *mut u8, 64);
    }
    let shared = vec![key("shared", 0)];
    let put = writer.run(Operation::Put {
        op_id: 1,
        keys: shared.clone(),
        host_blocks: vec![1],
    });
    assert_eq!(put.success, [true]);
    drop(writer);
    let reader = Client::start(&library, &clients, layout(1, 0), &deployment);
    let got = reader.run(Operation::Get {
        op_id: 1,
        keys: shared,
        host_blocks: vec![3],
    });
    assert_eq!(got.success, [true]);
    assert_eq!(&reader.pool[128..192], payload.as_slice());
    assert!(reader.executor().delete_namespace());
    drop(reader);
    drop(owner);
}

#[test]
fn a_sharded_block_restores_on_another_owner() {
    let _serial = serial();
    let Some(library) = library() else {
        return;
    };
    let master = master(&library);
    let deployment = unique("sharded");
    // Ranks 1 and 0 of a group dealt to two owners.
    let writer = Client::start(
        &library,
        &setup(&master, 32 << 20),
        layout(2, 1),
        &deployment,
    );
    let reader = Client::start(
        &library,
        &setup(&master, 32 << 20),
        layout(2, 0),
        &deployment,
    );
    let payload: Vec<u8> = (0..64).collect();
    // Block 2 is rank 1's local block 1: its pool's first 64 bytes.
    // SAFETY: the pool is live and no job runs.
    #[allow(unsafe_code)]
    unsafe {
        std::ptr::copy_nonoverlapping(payload.as_ptr(), writer.pool.as_ptr() as *mut u8, 64);
    }
    let shared = vec![key("shared", 1)];
    let put = writer.run(Operation::Put {
        op_id: 1,
        keys: shared.clone(),
        host_blocks: vec![2],
    });
    assert_eq!(put.success, [true]);
    // Block 5 is rank 0's local block 3.
    let get = |op_id| {
        reader.run(Operation::Get {
            op_id,
            keys: shared.clone(),
            host_blocks: vec![5],
        })
    };
    assert_eq!(get(2).success, [true]);
    assert_eq!(&reader.pool[128..192], payload.as_slice());
    let snapshot = reader.executor().snapshot();
    let moved: f64 = snapshot
        .operations
        .iter()
        .map(|row| row.success_bytes)
        .sum();
    assert_eq!(moved, 64.0);
    assert!(writer.executor().delete_namespace());
    let miss = get(3);
    assert_eq!(
        (miss.success, miss.object_failures),
        (vec![false], vec![true])
    );
}

#[test]
fn a_tenant_sees_only_its_own_objects() {
    let _serial = serial();
    let Some(library) = library() else {
        return;
    };
    let master = own_master(&library, &["tenant-a", "tenant-b"]);
    let deployment = unique("tenant");
    let tenant = |tenant_id: &str| StoreSetup {
        tenant_id: tenant_id.to_owned(),
        ..setup(&master, 32 << 20)
    };
    let a = Client::start(&library, &tenant("tenant-a"), layout(1, 0), &deployment);
    let b = Client::start(&library, &tenant("tenant-b"), layout(1, 0), &deployment);
    let shared = || vec![key("shared", 0)];
    let put = a.run(Operation::Put {
        op_id: 1,
        keys: shared(),
        host_blocks: vec![1],
    });
    assert_eq!(put.success, [true]);
    let query = |client: &Client, op_id| {
        client
            .run(Operation::Query {
                op_id,
                keys: shared(),
            })
            .success
    };
    assert_eq!(query(&a, 2), [true]);
    assert_eq!(query(&b, 1), [false]);
    assert!(a.executor().delete_namespace());
}

#[test]
fn the_backend_recovers_once_a_restarted_master_serves() {
    let _serial = serial();
    let Some(library) = library() else {
        return;
    };
    let mut master = own_master(&library, &[]);
    let deployment = unique("restart");
    let client = Client::start_with(
        &library,
        &setup(&master, 0),
        layout(1, 0),
        &deployment,
        ExecutorConfig {
            recovery_interval: Duration::from_millis(50),
            ..CONFIG
        },
    );
    let query = |op_id| Operation::Query {
        op_id,
        keys: vec![key("recovered", 0)],
    };
    // A metadata outage opens the circuit.
    master.stop();
    assert_eq!(client.run(query(1)).success, [false]);
    assert_eq!(client.executor().snapshot().health_state, "open");
    master.restart();
    // The restarted master's objects live in a new segment owner.
    let owner = Client::start(
        &library,
        &setup(&master, 64 << 20),
        layout(1, 0),
        &deployment,
    );
    sleep(Duration::from_millis(60));
    assert_eq!(client.run(query(2)).success, [false]);
    assert_eq!(client.executor().snapshot().health_state, "healthy");
    let payload: Vec<u8> = (0..64).collect();
    // SAFETY: the pool is live and no job runs.
    #[allow(unsafe_code)]
    unsafe {
        std::ptr::copy_nonoverlapping(payload.as_ptr(), client.pool.as_ptr() as *mut u8, 64);
    }
    let put = client.run(Operation::Put {
        op_id: 3,
        keys: vec![key("recovered", 0)],
        host_blocks: vec![1],
    });
    assert_eq!(put.success, [true]);
    let get = client.run(Operation::Get {
        op_id: 4,
        keys: vec![key("recovered", 0)],
        host_blocks: vec![3],
    });
    assert_eq!(get.success, [true]);
    assert_eq!(&client.pool[128..192], payload.as_slice());
    assert!(client.executor().delete_namespace());
    drop(client);
    drop(owner);
}
