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

//! Mount validation and index paths for the native USRBIO store.
//! This helper never reads or writes KV payloads.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

use crate::StoreError;

#[derive(Debug)]
pub(crate) struct IndexDirectory {
    pub(crate) directory: PathBuf,
}

fn failed(error: io::Error) -> StoreError {
    StoreError(format!("3FS index: {error}"))
}

fn namespace(value: &str) -> Result<&str, StoreError> {
    if value.len() == 69
        && value.starts_with("tsl3-")
        && value[5..].bytes().all(|c| c.is_ascii_hexdigit())
    {
        Ok(value)
    } else {
        Err(StoreError("invalid 3FS L3 namespace".into()))
    }
}

fn mount_type<'a>(directory: &Path, mounts: &'a str) -> Option<&'a str> {
    let mut selected = None;
    for line in mounts.lines() {
        let Some((left, right)) = line.split_once(" - ") else {
            continue;
        };
        let Some(mount) = left.split_whitespace().nth(4) else {
            continue;
        };
        let mount = mount
            .replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\");
        let mount = PathBuf::from(mount);
        if directory.starts_with(&mount)
            && selected
                .as_ref()
                .is_none_or(|(len, _)| mount.as_os_str().len() > *len)
        {
            selected = Some((mount.as_os_str().len(), right.split_whitespace().next()));
        }
    }
    selected.and_then(|(_, kind)| kind)
}

impl IndexDirectory {
    /// Open an existing directory. Refuse ordinary local filesystems rather
    /// than silently claiming they are 3FS when a mount is missing.
    pub(crate) fn connect(directory: &Path) -> Result<Self, StoreError> {
        let directory = fs::canonicalize(directory).map_err(failed)?;
        if !directory.is_dir() {
            return Err(StoreError("3FS directory is not a directory".into()));
        }
        let mounts = fs::read_to_string("/proc/self/mountinfo").map_err(failed)?;
        if mount_type(&directory, &mounts) != Some("fuse.hf3fs") {
            return Err(StoreError(format!(
                "{} must be on a fuse.hf3fs mount",
                directory.display()
            )));
        }
        tracing::debug!(path = %directory.display(), "3FS directory validated");
        Ok(Self { directory })
    }

    pub(crate) fn path(&self, key: &str) -> Result<PathBuf, StoreError> {
        let (prefix, _) = key
            .split_once('_')
            .ok_or_else(|| StoreError("invalid 3FS L3 key".into()))?;
        namespace(prefix)?;
        if key.len() > 255
            || !key
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"-_|".contains(&c))
        {
            return Err(StoreError("invalid 3FS L3 key".into()));
        }
        let digest = Sha256::digest(key.as_bytes());
        Ok(self
            .directory
            .join(prefix)
            .join(format!("{:02x}", digest[0]))
            .join(key))
    }

    pub(crate) fn remove_by_prefix(&self, prefix: &str) -> bool {
        let Some(prefix) = prefix.strip_suffix('_') else {
            return false;
        };
        if namespace(prefix).is_err() {
            return false;
        }
        match fs::remove_dir_all(self.directory.join(prefix)) {
            Ok(()) => true,
            Err(e) if e.kind() == io::ErrorKind::NotFound => true,
            Err(e) => {
                tracing::warn!(error = %e, "3FS L3 namespace removal failed");
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mount_selection_uses_deepest_component_and_decodes_escapes() {
        let mounts = concat!(
            "1 0 0:1 / / rw - ext4 /dev/root rw\n",
            "2 1 0:2 / /cache rw - fuse.hf3fs hf3fs rw\n",
            "3 2 0:3 / /cache/local rw - tmpfs tmpfs rw\n",
            "4 1 0:4 / /with\\040space\\011tab\\012line\\134040 rw - fuse.hf3fs hf3fs rw\n",
            "malformed line\n",
        );
        assert_eq!(
            mount_type(Path::new("/cache/objects"), mounts),
            Some("fuse.hf3fs")
        );
        assert_eq!(
            mount_type(Path::new("/cache/local/objects"), mounts),
            Some("tmpfs")
        );
        assert_eq!(mount_type(Path::new("/cache-other"), mounts), Some("ext4"));
        assert_eq!(
            mount_type(Path::new("/with space\ttab\nline\\040/objects"), mounts),
            Some("fuse.hf3fs")
        );
        assert_eq!(mount_type(Path::new("/cache"), "malformed"), None);
    }

    #[test]
    fn keys_stay_within_their_namespace_and_deletion_requires_exact_prefix() {
        let dir = IndexDirectory {
            directory: PathBuf::from("/unused"),
        };
        let namespace = format!("tsl3-{}", "a".repeat(64));
        let key = format!("{namespace}_aabb_0_0_0");
        let path = dir.path(&key).unwrap();
        assert!(path.starts_with(dir.directory.join(&namespace)));
        assert_eq!(path.file_name().unwrap(), key.as_str());
        for key in [
            "../outside".to_owned(),
            format!("{namespace}_../outside"),
            format!("{namespace}_{}", "x".repeat(256)),
        ] {
            assert!(dir.path(&key).is_err());
        }
        for prefix in [
            "",
            "../",
            "tsl3-",
            &namespace,
            &format!("{namespace}_suffix_"),
        ] {
            assert!(!dir.remove_by_prefix(prefix));
        }
    }
}
