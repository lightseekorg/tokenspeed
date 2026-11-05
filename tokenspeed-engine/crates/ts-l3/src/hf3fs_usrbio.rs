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

//! Indexed immutable objects in append-only 3FS segments; payload I/O uses USRBIO.
//!
//! The filesystem holds small, atomically published key-to-offset records.
//! Writers own unique segments, so there is no shared append pointer or service
//! to coordinate. Existing executor keys/geometry remain authoritative.
use crate::{
    hf3fs_index::IndexDirectory,
    hf3fs_native::{Api, RegisteredFile, Request, Ring},
    HostBuffer, IoResult, Range, Store, StoreError,
};
use std::{
    collections::VecDeque,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

/// Resource bounds for one rank's optional native 3FS client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsrBioConfig {
    /// Absolute path to libhf3fs_api_shared.so, loaded only for USRBIO.
    pub library: PathBuf,
    /// Maximum entries per native submission.
    pub entries: usize,
    /// Registered staging bytes per ring (one write ring and `read_rings`).
    pub buffer_bytes: usize,
    /// Native completion timeout, distinct from the scheduler's soft deadline.
    pub timeout_ms: u64,
    /// Maximum bytes per append-only data segment.
    pub segment_bytes: usize,
    /// Independent read lanes; each serializes its own native ring.
    pub read_rings: usize,
}
#[derive(Debug, PartialEq, Eq)]
struct Location {
    segment: String,
    offset: usize,
    size: usize,
}
impl Location {
    fn read(path: &Path) -> Result<Option<Self>, StoreError> {
        let mut f = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(failed(e)),
        };
        if f.metadata().map_err(failed)?.len() != 56 {
            return Ok(None);
        }
        let mut bytes = [0; 56];
        match f.read_exact(&mut bytes) {
            Ok(()) => Ok(Self::decode(&bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(failed(e)),
        }
    }
    fn encode(&self) -> Vec<u8> {
        let mut out = b"TS3FS01\0".to_vec();
        out.extend(self.segment.as_bytes());
        out.extend((self.offset as u64).to_le_bytes());
        out.extend((self.size as u64).to_le_bytes());
        out
    }
    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 56
            || &bytes[..8] != b"TS3FS01\0"
            || !bytes[8..40].iter().all(u8::is_ascii_hexdigit)
        {
            return None;
        }
        let offset = usize::try_from(u64::from_le_bytes(bytes[40..48].try_into().ok()?)).ok()?;
        let size = usize::try_from(u64::from_le_bytes(bytes[48..56].try_into().ok()?)).ok()?;
        offset.checked_add(size)?;
        Some(Self {
            segment: String::from_utf8(bytes[8..40].to_vec()).ok()?,
            offset,
            size,
        })
    }
}
fn failed(e: std::io::Error) -> StoreError {
    StoreError(format!("3FS index: {e}"))
}
fn publish_index(path: &Path, location: &Location) -> Result<(), StoreError> {
    let parent = path.parent().expect("bucket");
    fs::create_dir_all(parent).map_err(failed)?;
    let temporary = parent.join(format!(".index-{}", unique()?));
    let mut f = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(failed)?;
    let result = (|| {
        f.write_all(&location.encode())?;
        f.sync_all()?;
        match fs::hard_link(&temporary, &path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                if Location::read(path)
                    .map_err(std::io::Error::other)?
                    .is_some()
                {
                    Ok(())
                } else {
                    // Repair undecodable metadata. Concurrent repairs may
                    // publish either complete copy of this immutable key.
                    fs::rename(&temporary, path)
                }
            }
            Err(e) => Err(e),
        }
    })();
    if let Err(e) = fs::remove_file(&temporary) {
        if e.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!(path = %temporary.display(), error = %e, "3FS index temporary cleanup failed");
        }
    }
    result.map_err(failed)
}
fn unique() -> Result<String, StoreError> {
    let mut bytes = [0u8; 16];
    File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut bytes))
        .map_err(failed)?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}
struct Segment {
    namespace: PathBuf,
    name: String,
    file: Arc<RegisteredFile>,
    used: usize,
}
struct Writer {
    ring: Ring,
    segment: Option<Segment>,
}
const READER_FILE_CACHE_CAPACITY: usize = 32;

struct Reader {
    ring: Ring,
    files: VecDeque<(PathBuf, Arc<RegisteredFile>)>,
}

/// Host-only native 3FS Store. No POSIX fallback for payload transfers.
pub struct UsrBioStore {
    index: IndexDirectory,
    api: Arc<Api>,
    config: UsrBioConfig,
    writer: Mutex<Writer>,
    readers: Vec<Mutex<Reader>>,
    next_reader: AtomicUsize,
}
impl UsrBioStore {
    /// Require a real complete 3FS mount and initialize bounded native rings.
    pub fn connect(directory: &Path, config: UsrBioConfig) -> Result<Self, StoreError> {
        if config.entries == 0
            || config.entries > 4096
            || config.buffer_bytes == 0
            || config.buffer_bytes > 1 << 30
            || config.segment_bytes < config.buffer_bytes
            || config.segment_bytes > i64::MAX as usize
            || config.read_rings == 0
            || config.read_rings > 8
            || config.timeout_ms == 0
            || config.timeout_ms > 3_600_000
        {
            return Err(StoreError("invalid 3FS USRBIO resource bounds".into()));
        }
        // Validate before creating any directory; local filesystems are rejected.
        let validated = IndexDirectory::connect(directory)?;
        let api = Api::load(&config.library)?;
        let mount = api.mount(&validated.directory)?;
        let root = validated.directory.join("usrbio-v1");
        fs::create_dir_all(&root).map_err(failed)?;
        let index = IndexDirectory::connect(&root)?;
        let make_ring = |read| {
            Ring::new(
                api.clone(),
                &mount,
                read,
                config.entries,
                config.buffer_bytes,
                config.timeout_ms,
            )
        };
        let writer = Mutex::new(Writer {
            ring: make_ring(false)?,
            segment: None,
        });
        let readers = (0..config.read_rings)
            .map(|_| {
                Ok(Mutex::new(Reader {
                    ring: make_ring(true)?,
                    files: VecDeque::new(),
                }))
            })
            .collect::<Result<Vec<_>, StoreError>>()?;
        tracing::info!(path=%root.display(),entries=config.entries,buffer_bytes=config.buffer_bytes,read_rings=config.read_rings,"3FS USRBIO connected");
        Ok(Self {
            index,
            api,
            config,
            writer,
            readers,
            next_reader: AtomicUsize::new(0),
        })
    }
    fn location(&self, key: &str) -> Result<Option<Location>, StoreError> {
        Location::read(&self.index.path(key)?)
    }
    fn namespace(&self, key: &str) -> Result<PathBuf, StoreError> {
        let path = self.index.path(key)?;
        Ok(path
            .parent()
            .expect("bucket")
            .parent()
            .expect("namespace")
            .to_owned())
    }
    fn publish(&self, key: &str, location: &Location) -> Result<(), StoreError> {
        publish_index(&self.index.path(key)?, location)
    }
}
impl Store for UsrBioStore {
    // Native DMA touches only owned staging memory; Host is copied while the
    // synchronous call is active, even when native I/O must be quarantined.
    fn failures_retire(&self) -> bool {
        true
    }
    fn batch_exists(&self, keys: &[String]) -> Result<Vec<bool>, StoreError> {
        keys.iter()
            .map(|key| self.location(key).map(|location| location.is_some()))
            .collect()
    }
    fn batch_put_from(
        &self,
        keys: &[String],
        host: HostBuffer,
        ranges: &[Range],
    ) -> Result<IoResult, StoreError> {
        if keys.len() != ranges.len() {
            return Err(StoreError("ragged 3FS USRBIO put".into()));
        }
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| StoreError("3FS writer lock poisoned".into()))?;
        let mut requests = Vec::new();
        let mut pending = Vec::new();
        let exists = self.batch_exists(keys)?;
        for (i, (key, &range)) in keys.iter().zip(ranges).enumerate() {
            host.address(range);
            if exists[i] {
                continue;
            }
            if range.size > self.config.buffer_bytes {
                return Err(StoreError("3FS object exceeds USRBIO buffer_bytes".into()));
            }
            let namespace = self.namespace(key)?;
            // An executor normally serves one namespace. Switching namespaces
            // seals the current segment; no shared cross-namespace append state.
            if writer.segment.as_ref().is_none_or(|s| {
                s.namespace != namespace || s.used + range.size > self.config.segment_bytes
            }) {
                let dir = namespace.join("segments");
                fs::create_dir_all(&dir).map_err(failed)?;
                let name = unique()?;
                let file = OpenOptions::new()
                    .create_new(true)
                    .read(true)
                    .write(true)
                    .open(dir.join(&name))
                    .map_err(failed)?;
                file.set_len(self.config.segment_bytes as u64)
                    .map_err(failed)?;
                writer.segment = Some(Segment {
                    namespace: namespace.clone(),
                    name,
                    file: self.api.file(file)?,
                    used: 0,
                });
            }
            let segment = writer.segment.as_mut().expect("allocated segment");
            let location = Location {
                segment: segment.name.clone(),
                offset: segment.used,
                size: range.size,
            };
            requests.push(Request {
                file: segment.file.clone(),
                offset: segment.used,
                range,
            });
            segment.used += range.size;
            pending.push((i, location));
        }
        let completed = writer.ring.run(host, &requests)?;
        // Publish indexes only after native completion and durable segment data.
        let mut synced = Vec::new();
        for request in &requests {
            let ptr = Arc::as_ptr(&request.file);
            if !synced.contains(&ptr) {
                request.file.file.sync_all().map_err(failed)?;
                synced.push(ptr);
            }
        }
        let mut success = exists;
        for ((i, location), ok) in pending.into_iter().zip(completed) {
            if ok {
                self.publish(&keys[i], &location)?;
                success[i] = true;
            }
        }
        Ok(IoResult {
            success,
            retired: true,
            object_failures: vec![false; keys.len()],
        })
    }
    fn batch_get_into(
        &self,
        keys: &[String],
        host: HostBuffer,
        ranges: &[Range],
    ) -> Result<IoResult, StoreError> {
        if keys.len() != ranges.len() {
            return Err(StoreError("ragged 3FS USRBIO get".into()));
        }
        let lane = self.next_reader.fetch_add(1, Ordering::Relaxed) % self.readers.len();
        let mut reader = self.readers[lane]
            .lock()
            .map_err(|_| StoreError("3FS reader lock poisoned".into()))?;
        let mut requests = Vec::new();
        let mut positions = Vec::new();
        let mut success = vec![false; keys.len()];
        for (i, (key, &range)) in keys.iter().zip(ranges).enumerate() {
            host.address(range);
            let Some(location) = self.location(key)? else {
                continue;
            };
            if location.size != range.size {
                continue;
            }
            let path = self
                .namespace(key)?
                .join("segments")
                .join(&location.segment);
            let file = if let Some((_, f)) = reader.files.iter().find(|(p, _)| p == &path) {
                f.clone()
            } else {
                let f = match File::open(&path) {
                    Ok(f) => f,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(e) => return Err(failed(e)),
                };
                let f = self.api.file(f)?;
                if reader.files.len() == READER_FILE_CACHE_CAPACITY {
                    reader.files.pop_front();
                }
                reader.files.push_back((path, f.clone()));
                f
            };
            if location.offset as u64 + location.size as u64 > file.size {
                continue;
            }
            requests.push(Request {
                file,
                offset: location.offset,
                range,
            });
            positions.push(i);
        }
        for (i, ok) in positions.into_iter().zip(reader.ring.run(host, &requests)?) {
            success[i] = ok;
        }
        let object_failures = success.iter().map(|ok| !ok).collect();
        Ok(IoResult {
            success,
            retired: true,
            object_failures,
        })
    }
    fn remove_by_prefix(&self, prefix: &str) -> bool {
        // Called only after the existing executor/scheduler quiescence fence.
        let Ok(mut writer) = self.writer.lock() else {
            return false;
        };
        writer.segment = None;
        for reader in &self.readers {
            let Ok(mut reader) = reader.lock() else {
                return false;
            };
            reader.files.clear();
        }
        self.index.remove_by_prefix(prefix)
    }
    fn close(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn location() -> Location {
        Location {
            segment: "0123456789abcdef0123456789abcdef".into(),
            offset: 4096,
            size: 8192,
        }
    }

    #[test]
    fn location_codec_validates_metadata() {
        let expected = location();
        let bytes = expected.encode();
        assert_eq!(Location::decode(&bytes), Some(expected));
        assert!(Location::decode(&bytes[..55]).is_none());
        let mut invalid = bytes.clone();
        invalid[0] = b'X';
        assert!(Location::decode(&invalid).is_none());
        let mut invalid = bytes.clone();
        invalid[8] = b'/';
        assert!(Location::decode(&invalid).is_none());
        let mut invalid = bytes;
        invalid[40..48].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(Location::decode(&invalid).is_none());
    }

    #[test]
    fn index_publish_preserves_valid_objects_and_repairs_invalid_metadata() {
        let root = std::env::temp_dir().join(format!("ts-l3-index-{}", unique().unwrap()));
        let path = root.join("object");
        assert_eq!(Location::read(&path).unwrap(), None);
        let first = location();
        publish_index(&path, &first).unwrap();
        let second = Location {
            offset: 32768,
            ..location()
        };
        publish_index(&path, &second).unwrap();
        assert_eq!(Location::read(&path).unwrap(), Some(first));
        for invalid in [b"partial".to_vec(), vec![0; 56]] {
            fs::write(&path, invalid).unwrap();
            assert_eq!(Location::read(&path).unwrap(), None);
            publish_index(&path, &second).unwrap();
            assert_eq!(
                Location::read(&path).unwrap(),
                Some(Location {
                    offset: 32768,
                    ..location()
                })
            );
        }
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }
}
