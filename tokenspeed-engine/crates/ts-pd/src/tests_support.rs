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

//! A CPU-only native transfer library for lifecycle tests.
//!
//! Addresses are deliberately reused. A sender caches the remote engine's
//! incarnation, so a stale handle fails instead of silently copying bytes.

use std::{
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

static NEXT_LIBRARY: AtomicU64 = AtomicU64::new(0);

/// An isolated native library; its global engine state belongs to one test.
pub(crate) struct NativeLibrary {
    directory: PathBuf,
    path: PathBuf,
}

impl NativeLibrary {
    pub(crate) fn new() -> Self {
        let sequence = NEXT_LIBRARY.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "tokenspeed-pd-native-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).expect("create native test directory");
        std::fs::write(directory.join("owner-alive"), b"").expect("mark native test owner alive");
        let source = directory.join("native.c");
        let path = directory.join("libnative.so");
        let native = format!(
            "#define WRITE_BLOCK_PATH {}\n#define WRITE_ENTERED_PATH {}\n#define WRITE_OWNER_PATH {}\n{NATIVE}",
            c_path(&directory.join("write-blocked")),
            c_path(&directory.join("write-entered")),
            c_path(&directory.join("owner-alive")),
        );
        std::fs::write(&source, native).expect("write native test library");
        let output = Command::new(std::env::var_os("CC").unwrap_or_else(|| "cc".into()))
            .args(["-shared", "-fPIC", "-pthread", "-O0"])
            .arg(&source)
            .arg("-o")
            .arg(&path)
            .output()
            .expect("a C compiler is required for the CPU native test fixture");
        assert!(
            output.status.success(),
            "native test library failed to compile: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Self { directory, path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Hold native writes before they inspect or copy registered memory.
    /// Dropping this library fails held writes without copying after a panic.
    #[allow(dead_code)] // Shared by lifecycle tests that do not all hold native writes.
    pub(crate) fn block_writes(&self) {
        let _ = std::fs::remove_file(self.directory.join("write-entered"));
        std::fs::write(self.directory.join("write-blocked"), b"")
            .expect("block native test writes");
    }

    /// Wait until at least one write entered the native library's barrier.
    #[allow(dead_code)]
    pub(crate) fn wait_write_entered(&self, timeout: Duration) -> bool {
        let entered = self.directory.join("write-entered");
        let deadline = Instant::now() + timeout;
        loop {
            if entered.exists() {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// Let blocked writes finish; subsequent writes proceed normally.
    #[allow(dead_code)]
    pub(crate) fn release_writes(&self) {
        match std::fs::remove_file(self.directory.join("write-blocked")) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => panic!("release native test writes: {error}"),
        }
    }
}

impl Drop for NativeLibrary {
    fn drop(&mut self) {
        // Buffers may already have unwound. Fail any held write before
        // unblocking it so cleanup never copies into their freed memory.
        let _ = std::fs::remove_file(self.directory.join("owner-alive"));
        let _ = std::fs::remove_file(self.directory.join("write-blocked"));
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

/// Encode an arbitrary Unix path without interpreting its bytes as C syntax.
fn c_path(path: &Path) -> String {
    let bytes: String = path
        .as_os_str()
        .as_bytes()
        .iter()
        .map(|byte| format!("\\{byte:03o}"))
        .collect();
    format!("\"{bytes}\"")
}

const NATIVE: &str = r#"
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

#define ENGINES 4
#define REGIONS 8
#define PORT_BASE 41000

typedef struct { uintptr_t address; size_t size; } Region;
typedef struct {
    int slot;
    uint64_t incarnation;
    uint64_t peers[ENGINES];
    Region regions[REGIONS];
} Engine;

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static Engine *engines[ENGINES];
static uint64_t next_incarnation = 1;

/* The barrier is private to this library and never holds the engine mutex. */
static int wait_for_write_release(void) {
    if (access(WRITE_BLOCK_PATH, F_OK) == 0) {
        FILE *entered = fopen(WRITE_ENTERED_PATH, "w");
        if (entered) fclose(entered);
        while (access(WRITE_BLOCK_PATH, F_OK) == 0) usleep(1000);
    }
    return access(WRITE_OWNER_PATH, F_OK) == 0;
}

int ts_mooncake_abi_version(void) { return 1; }

void *ts_mooncake_create(const char *host, const char *metadata,
                        const char *protocol, const char *devices) {
    if (strcmp(protocol, "rdma") != 0) return NULL;
    pthread_mutex_lock(&mutex);
    Engine *engine = NULL;
    for (int slot = 0; slot < ENGINES; ++slot) {
        if (engines[slot]) continue;
        engine = calloc(1, sizeof(Engine));
        if (engine) {
            engine->slot = slot;
            engine->incarnation = next_incarnation++;
            engines[slot] = engine;
        }
        break;
    }
    pthread_mutex_unlock(&mutex);
    return engine;
}

void ts_mooncake_destroy(void *raw) {
    Engine *engine = raw;
    pthread_mutex_lock(&mutex);
    engines[engine->slot] = NULL;
    free(engine);
    pthread_mutex_unlock(&mutex);
}

int ts_mooncake_rpc_port(void *raw) {
    return PORT_BASE + ((Engine *)raw)->slot;
}

int ts_mooncake_register(void *raw, void *address, size_t size) {
    Engine *engine = raw;
    pthread_mutex_lock(&mutex);
    int result = -1;
    for (int i = 0; i < REGIONS; ++i) {
        if (engine->regions[i].size) continue;
        engine->regions[i] = (Region){(uintptr_t)address, size};
        result = 0;
        break;
    }
    pthread_mutex_unlock(&mutex);
    return result;
}

int ts_mooncake_unregister(void *raw, void *address) {
    Engine *engine = raw;
    pthread_mutex_lock(&mutex);
    int result = -1;
    for (int i = 0; i < REGIONS; ++i) {
        if (engine->regions[i].address != (uintptr_t)address || !engine->regions[i].size) continue;
        engine->regions[i].size = 0;
        result = 0;
        break;
    }
    pthread_mutex_unlock(&mutex);
    return result;
}

static int contains(Engine *engine, uint64_t address, uint64_t length) {
    for (int i = 0; i < REGIONS; ++i) {
        Region region = engine->regions[i];
        if (address >= region.address && length <= region.size &&
            address - region.address <= region.size - length) return 1;
    }
    return 0;
}

/* The caller holds mutex through lookup and all copies in the write. */
static Engine *target(Engine *sender, const char *session) {
    const char *port = strrchr(session, ':');
    int slot = port ? atoi(port + 1) - PORT_BASE : -1;
    if (slot < 0 || slot >= ENGINES || !engines[slot]) return NULL;
    Engine *receiver = engines[slot];
    uint64_t *cached = &sender->peers[slot];
    if (*cached && *cached != receiver->incarnation) return NULL;
    *cached = receiver->incarnation;
    return receiver;
}

static int copy(Engine *sender, Engine *receiver,
                uint64_t source, uint64_t destination, uint64_t length) {
    if (!contains(sender, source, length) || !contains(receiver, destination, length)) return -3;
    memcpy((void *)(uintptr_t)destination, (void *)(uintptr_t)source, length);
    return 0;
}

int ts_mooncake_write(void *raw, const char *session, const uint64_t *sources,
                      const uint64_t *destinations, const uint64_t *lengths, size_t count) {
    Engine *sender = raw;
    if (!wait_for_write_release()) return -4;
    pthread_mutex_lock(&mutex);
    Engine *receiver = target(sender, session);
    int result = receiver ? 0 : -2;
    for (size_t i = 0; !result && i < count; ++i)
        result = copy(sender, receiver, sources[i], destinations[i], lengths[i]);
    pthread_mutex_unlock(&mutex);
    return result;
}

int ts_mooncake_write_pages(void *raw, const char *session, const int64_t *sources,
                            const int64_t *destinations, size_t count, const int64_t *fields,
                            size_t field_count, size_t max_batch) {
    Engine *sender = raw;
    if (!wait_for_write_release()) return -4;
    pthread_mutex_lock(&mutex);
    Engine *receiver = target(sender, session);
    int result = receiver ? 0 : -2;
    for (size_t field = 0; !result && field < field_count; ++field) {
        const int64_t *f = fields + field * 5;
        for (size_t i = 0; !result && i < count; ++i)
            result = copy(sender, receiver, f[0] + sources[i] * f[1],
                          f[2] + destinations[i] * f[3], f[4]);
    }
    pthread_mutex_unlock(&mutex);
    return result;
}
"#;
