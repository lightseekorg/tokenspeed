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

//! Host spans: what the engine's own threads do -- the event loop's phases,
//! the C++ scheduler's, the forward thread's operations -- on a
//! `/start_profile` timeline, which otherwise shows only what the torch
//! profiler sees.
//!
//! The data plane's profiler (`engine/profiler.py`) starts and stops the
//! recording together with the torch profiler, through `ts-host`, and merges
//! the spans into the torch trace it exports. Timestamps are the system
//! clock's (Unix time), the clock torch stamps its CPU events with, so the
//! merge only rebases them. The recorder lives in this crate because the C++
//! scheduler's spans (`csrc/core/profile_span.h`) enter through its bridge;
//! `ts-control` and `ts-host` record theirs here too.
//!
//! Without a recording a span costs one atomic load. A [`hold`] keeps its
//! thread's spans back until it closes, so a caller can drop them: the event
//! loop keeps a round's spans only if the round did something, as a loop
//! waiting on a transfer spins through empty rounds.

use std::{
    borrow::Cow,
    cell::RefCell,
    fs,
    marker::PhantomData,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Mutex, MutexGuard,
    },
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::ffi;

static RECORDING: AtomicBool = AtomicBool::new(false);
static SPANS: Mutex<Vec<Span>> = Mutex::new(Vec::new());
/// Every thread that recorded a span, with its name, in first-span order.
static THREADS: Mutex<Vec<(u64, String)>> = Mutex::new(Vec::new());

thread_local! {
    static THREAD_ID: u64 = register_thread();
    /// This thread's spans while a [`Hold`] is open.
    static HELD: RefCell<Option<Vec<Span>>> = const { RefCell::new(None) };
}

/// One finished span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    /// What ran.
    pub name: Cow<'static, str>,
    /// The thread it ran on: the kernel's thread id on Linux, as the torch
    /// trace's `tid`.
    pub tid: u64,
    /// When it began and ended, in nanoseconds of Unix time.
    pub begin_ns: i64,
    /// See `begin_ns`.
    pub end_ns: i64,
}

/// What a recording collected.
#[derive(Debug, Default)]
pub struct Recording {
    /// The spans, in the order they finished on each thread.
    pub spans: Vec<Span>,
    /// Each thread that ever recorded, with its name.
    pub threads: Vec<(u64, String)>,
}

/// Whether a recording runs.
pub fn recording() -> bool {
    RECORDING.load(Ordering::Relaxed)
}

/// Start a recording, dropping what an earlier one left.
pub fn start() {
    lock(&SPANS).clear();
    RECORDING.store(true, Ordering::Relaxed);
    ffi::set_profile_span_sink(true);
}

/// Stop the recording and take what it collected.
pub fn stop() -> Recording {
    ffi::set_profile_span_sink(false);
    RECORDING.store(false, Ordering::Relaxed);
    Recording {
        spans: std::mem::take(&mut *lock(&SPANS)),
        threads: lock(&THREADS).clone(),
    }
}

/// The current time, as a span's timestamps count it.
pub fn now_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos() as i64)
}

/// Record the span `name` from now until the guard drops.
#[must_use = "the span ends when the guard drops"]
pub fn span(name: &'static str) -> SpanGuard {
    SpanGuard {
        name,
        begin_ns: recording().then(now_ns),
    }
}

/// An open span; see [`span`].
pub struct SpanGuard {
    name: &'static str,
    /// None when no recording ran as the span began.
    begin_ns: Option<i64>,
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        if let Some(begin_ns) = self.begin_ns {
            record(Cow::Borrowed(self.name), begin_ns, now_ns());
        }
    }
}

/// Record a finished span on this thread, if a recording runs.
pub fn record(name: Cow<'static, str>, begin_ns: i64, end_ns: i64) {
    if !recording() {
        return;
    }
    // A thread past its thread-locals' destruction records nothing.
    let Ok(tid) = THREAD_ID.try_with(|tid| *tid) else {
        return;
    };
    let span = Span {
        name,
        tid,
        begin_ns,
        end_ns,
    };
    let unheld = HELD
        .try_with(|held| match held.borrow_mut().as_mut() {
            Some(spans) => {
                spans.push(span);
                None
            }
            None => Some(span),
        })
        .ok()
        .flatten();
    if let Some(span) = unheld {
        lock(&SPANS).push(span);
    }
}

/// Hold this thread's spans until the returned guard closes:
/// [`Hold::keep`] records them, dropping the guard discards them. Holds do
/// not nest.
pub fn hold() -> Hold {
    let open = recording();
    if open {
        HELD.with(|held| {
            let previous = held.borrow_mut().replace(Vec::new());
            debug_assert!(previous.is_none(), "profile span holds do not nest");
        });
    }
    Hold {
        open,
        _thread: PhantomData,
    }
}

/// Held spans; see [`hold`].
pub struct Hold {
    open: bool,
    /// The held spans are this thread's.
    _thread: PhantomData<*const ()>,
}

impl Hold {
    /// Record the held spans.
    pub fn keep(mut self) {
        let held = self.take();
        if recording() && !held.is_empty() {
            lock(&SPANS).extend(held);
        }
    }

    fn take(&mut self) -> Vec<Span> {
        if !std::mem::take(&mut self.open) {
            return Vec::new();
        }
        HELD.with(|held| held.borrow_mut().take())
            .unwrap_or_default()
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        self.take();
    }
}

/// The C++ scheduler's sink (`profile_span.h`), which the bridge installs
/// while a recording runs.
pub(crate) fn record_scheduler_span(name: &str, begin_ns: i64, end_ns: i64) {
    record(Cow::Owned(name.to_owned()), begin_ns, end_ns);
}

/// Spans may be recorded while a thread unwinds: take a lock even if
/// poisoned.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn register_thread() -> u64 {
    // On Linux, the kernel's id and name of the thread, which the torch
    // trace labels its threads with; elsewhere an id of its own.
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let tid = fs::read_link("/proc/thread-self")
        .ok()
        .and_then(|link| link.file_name()?.to_str()?.parse().ok())
        .unwrap_or_else(|| (1 << 32) + NEXT.fetch_add(1, Ordering::Relaxed));
    let name = fs::read_to_string("/proc/thread-self/comm")
        .map(|comm| comm.trim_end().to_owned())
        .unwrap_or_else(|_| thread::current().name().unwrap_or_default().to_owned());
    lock(&THREADS).push((tid, name));
    tid
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The recording is process-global: the tests take turns.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn names(recording: &Recording) -> Vec<&str> {
        recording.spans.iter().map(|span| &*span.name).collect()
    }

    #[test]
    fn spans_record_only_while_a_recording_runs() {
        let _serial = lock(&SERIAL);
        drop(span("before"));
        start();
        {
            let _outer = span("outer");
            drop(span("inner"));
        }
        let recorded = stop();
        drop(span("after"));
        assert_eq!(names(&recorded), ["inner", "outer"]);
        let (inner, outer) = (&recorded.spans[0], &recorded.spans[1]);
        assert!(outer.begin_ns <= inner.begin_ns && inner.end_ns <= outer.end_ns);
        assert!(recorded.threads.iter().any(|(tid, _)| *tid == outer.tid));

        start();
        assert!(stop().spans.is_empty(), "a new recording starts empty");
    }

    #[test]
    fn a_span_open_across_the_stop_is_dropped() {
        let _serial = lock(&SERIAL);
        start();
        let open = span("open");
        let recorded = stop();
        drop(open);
        assert!(recorded.spans.is_empty());
        start();
        assert!(stop().spans.is_empty());
    }

    #[test]
    fn a_hold_keeps_or_discards_its_threads_spans() {
        let _serial = lock(&SERIAL);
        start();
        let discarded = hold();
        drop(span("discarded"));
        drop(discarded);
        let kept = hold();
        drop(span("kept"));
        // Another thread's spans are not held.
        thread::spawn(|| drop(span("elsewhere"))).join().unwrap();
        kept.keep();
        drop(span("unheld"));
        let recorded = stop();
        assert_eq!(names(&recorded), ["elsewhere", "kept", "unheld"]);
        assert_ne!(recorded.spans[0].tid, recorded.spans[1].tid);
    }
}
