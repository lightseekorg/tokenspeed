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

//! Pool-scoped shared owner for one cache block.
//!
//! Ported from `tokenspeed-scheduler/csrc/cache/core/cache_block_ref.{h,cpp}`.
//! C++ used a hand-rolled shared control block (`CacheBlockControl`) with a raw
//! `BlockPool*`; here `Rc<RefCell<BlockPool>>` provides the same release-on-last-
//! owner semantics safely (single-threaded scheduler). `Rc` matches the shared-
//! pointer shape used by the rest of the port and keeps the crate unsafe-free.

use std::cell::RefCell;
use std::rc::Rc;

use crate::block_pool::BlockPool;

/// Stable logical placement of one cache block inside an LCM-sized physical
/// block. LCM block 0 remains reserved as the kernel null page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheBlockLocation {
    pub lcm_block_id: i32,
    pub slot_index: i32,
}

/// Owning object for one block slot. Releasing the last [`CacheBlockRef`]
/// drops the block, which returns the slot to its [`BlockPool`].
pub struct CacheBlock {
    pool: Rc<RefCell<BlockPool>>,
    location: CacheBlockLocation,
}

impl CacheBlock {
    /// Create a block bound to `pool` at `location` (created by `BlockPool`).
    pub(crate) fn new(pool: Rc<RefCell<BlockPool>>, location: CacheBlockLocation) -> Self {
        Self { pool, location }
    }

    /// Stable placement of this block.
    pub fn location(&self) -> CacheBlockLocation {
        self.location
    }

    /// Whether this block belongs to `pool` (same shared instance).
    pub fn is_owned_by(&self, pool: &Rc<RefCell<BlockPool>>) -> bool {
        Rc::ptr_eq(&self.pool, pool)
    }
}

impl Drop for CacheBlock {
    fn drop(&mut self) {
        self.pool.borrow_mut().release(self.location);
    }
}

/// Shared, nullable handle to a [`CacheBlock`]. Mirrors the C++ `CacheBlockRef`
/// value semantics: `Eq` compares control identity, `use_count` reports the
/// number of live handles, and an empty handle is the null block.
#[derive(Clone, Default)]
pub struct CacheBlockRef(Option<Rc<CacheBlock>>);

impl CacheBlockRef {
    /// Wrap a freshly created block (called by `BlockPool`).
    pub(crate) fn new(block: Rc<CacheBlock>) -> Self {
        Self(Some(block))
    }

    /// Whether this handle is empty (null block).
    pub fn is_null(&self) -> bool {
        self.0.is_none()
    }

    /// Placement of the referenced block, or `None` for a null handle.
    pub fn location(&self) -> Option<CacheBlockLocation> {
        self.0.as_ref().map(|block| block.location())
    }

    /// Number of live handles sharing this block (0 for a null handle).
    pub fn use_count(&self) -> u32 {
        self.0
            .as_ref()
            .map_or(0, |block| Rc::strong_count(block) as u32)
    }

    /// Whether this handle is the only live one.
    pub fn unique(&self) -> bool {
        self.use_count() == 1
    }

    /// Whether the referenced block belongs to `pool`.
    pub fn is_owned_by(&self, pool: &Rc<RefCell<BlockPool>>) -> bool {
        self.0.as_ref().is_some_and(|block| block.is_owned_by(pool))
    }

    /// Drop this handle's reference.
    pub fn reset(&mut self) {
        self.0 = None;
    }

    /// Borrow the underlying block, or `None` for a null handle.
    pub fn as_block(&self) -> Option<&CacheBlock> {
        self.0.as_deref()
    }
}

impl PartialEq for CacheBlockRef {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (None, None) => true,
            (Some(a), Some(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl Eq for CacheBlockRef {}

impl std::fmt::Debug for CacheBlockRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheBlockRef")
            .field("location", &self.location())
            .field("use_count", &self.use_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_pool::BlockPool;

    fn pool(n: i32) -> Rc<RefCell<BlockPool>> {
        Rc::new(RefCell::new(BlockPool::new(n)))
    }

    #[test]
    fn default_ref_is_null() {
        let r: CacheBlockRef = CacheBlockRef::default();
        assert!(r.is_null());
        assert_eq!(r.location(), None);
        assert_eq!(r.use_count(), 0);
        assert!(!r.unique());
    }

    #[test]
    fn refs_share_identity_and_release_on_drop() {
        let p = pool(2);
        let a = p.borrow_mut().acquire_block(&p, 1, 1).expect("block");
        let b = a.clone();
        assert_eq!(a.use_count(), 2);
        assert_eq!(a, b);
        assert!(!a.unique());
        assert_eq!(
            a.location(),
            Some(CacheBlockLocation {
                lcm_block_id: 1,
                slot_index: 0
            })
        );
        assert!(a.is_owned_by(&p));
        drop(b);
        assert_eq!(a.use_count(), 1);
        assert!(a.unique());
        drop(a);
        assert_eq!(p.borrow().num_occupied_slots(), 0);
        assert_eq!(p.borrow().num_empty_lcm_blocks(), 2);
    }

    #[test]
    fn null_ref_never_equals_occupied_ref() {
        let p = pool(1);
        let a = p.borrow_mut().acquire_block(&p, 1, 1).expect("block");
        let null = CacheBlockRef::default();
        assert_ne!(a, null);
        assert!(!null.is_owned_by(&p));
    }

    #[test]
    fn reset_releases_handle() {
        let p = pool(1);
        let mut a = p.borrow_mut().acquire_block(&p, 1, 1).expect("block");
        assert_eq!(p.borrow().num_occupied_slots(), 1);
        a.reset();
        assert!(a.is_null());
        assert_eq!(p.borrow().num_occupied_slots(), 0);
    }
}
