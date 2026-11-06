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

//! Exercise real transport lifecycle code against a CPU native library that
//! reuses endpoints and rejects stale remote handles or unregistered memory.

use super::*;
use crate::tests_support::NativeLibrary;

fn transport(library: &NativeLibrary, region: &[u8]) -> Arc<Transport> {
    let (incoming, _messages) = std::sync::mpsc::channel();
    Transport::start(
        &PeerConfig {
            library: library.path().to_owned(),
            local_ip: "127.0.0.1".to_owned(),
            protocol: "rdma".to_owned(),
            ib_device: String::new(),
            regions: vec![(region.as_ptr() as usize, region.len())],
            engine_rank: 0,
            metrics: None,
        },
        incoming,
    )
    .unwrap()
}

fn spans(source: &[u8], destination: &[u8]) -> [Copy; 1] {
    [Copy::Spans {
        sources: vec![source.as_ptr() as u64],
        destinations: vec![destination.as_ptr() as u64],
        lengths: vec![source.len() as u64],
    }]
}

/// The fixture assigns the lowest free endpoint. Keeping the current engine
/// alive during creation makes two wakes reuse the original endpoint.
fn reuse_endpoint(receiver: &Transport) {
    let session = receiver.session_id();
    for _ in 0..2 {
        receiver.release_regions().unwrap();
        receiver.restore_regions(true).unwrap();
    }
    assert_eq!(receiver.session_id(), session);
}

#[test]
fn reused_endpoint_gets_fresh_native_keys_for_page_writes() {
    let library = NativeLibrary::new();
    let mut source = vec![7u8; 32];
    let mut destination = vec![0u8; 32];
    let sender = transport(&library, &source);
    let receiver = transport(&library, &destination);
    let session = receiver.session_id();
    let copies = [Copy::Pages(PageGrid {
        src_pages: vec![0],
        dst_pages: vec![0],
        fields: vec![[
            source.as_ptr() as i64,
            32,
            destination.as_ptr() as i64,
            32,
            32,
        ]],
    })];
    sender
        .write_cache(&session, "cache-before", &copies)
        .unwrap();
    assert_eq!(destination, source);

    // A Prefill sleep must retain the history of its remote cache handles.
    sender.release_regions().unwrap();
    sender.restore_regions(false).unwrap();
    reuse_endpoint(&receiver);
    destination.fill(0);
    source.fill(9);
    assert!(matches!(
        sender.write(&session, &copies),
        Err(ts_mooncake::Error::Transfer { status: -2, .. })
    ));
    assert_eq!(destination, vec![0u8; 32]);

    sender
        .write_cache(&session, "cache-after", &copies)
        .unwrap();
    assert_eq!(destination, source);
    source.fill(11);
    sender
        .write_cache(&session, "cache-after", &copies)
        .unwrap();
    assert_eq!(destination, source);
}

#[test]
fn release_and_resume_cover_both_public_and_private_source_registrations() {
    let library = NativeLibrary::new();
    let source = vec![3u8; 32];
    let mut destination = vec![0u8; 32];
    let mut public_destination = vec![0u8; 32];
    let sender = transport(&library, &source);
    let receiver = transport(&library, &destination);
    let session = receiver.session_id();
    let copies = spans(&source, &destination);
    sender
        .write_cache(&session, "cache-before", &copies)
        .unwrap();
    reuse_endpoint(&receiver);
    sender
        .write_cache(&session, "cache-after", &copies)
        .unwrap();
    assert_eq!(destination, source);

    // An independent peer lets the public engine use a valid remote handle,
    // so a subsequent failure can only be its missing source registration.
    let public_receiver = transport(&library, &public_destination);
    let public_session = public_receiver.session_id();
    let public_copies = spans(&source, &public_destination);
    sender.write(&public_session, &public_copies).unwrap();
    assert_eq!(public_destination, source);

    sender.release_regions().unwrap();
    destination.fill(0);
    public_destination.fill(0);
    assert!(matches!(
        sender.write_cache(&session, "cache-after", &copies),
        Err(ts_mooncake::Error::Transfer { status: -3, .. })
    ));
    assert!(matches!(
        sender.write(&public_session, &public_copies),
        Err(ts_mooncake::Error::Transfer { status: -3, .. })
    ));
    assert_eq!(destination, vec![0u8; 32]);
    assert_eq!(public_destination, vec![0u8; 32]);

    sender.restore_regions(false).unwrap();
    sender
        .write_cache(&session, "cache-after", &copies)
        .unwrap();
    sender.write(&public_session, &public_copies).unwrap();
    assert_eq!(destination, source);
    assert_eq!(public_destination, source);
}

#[test]
fn replacing_cache_sender_preserves_the_advertised_epd_receiver() {
    let library = NativeLibrary::new();
    let source = vec![5u8; 32];
    let destination = vec![0u8; 32];
    let mut embedding = vec![13u8; 32];
    let embedding_pool = vec![0u8; 32];
    let prefill = transport(&library, &source);
    let decode = transport(&library, &destination);
    let encoder = transport(&library, &embedding);
    let advertised_session = prefill.session_id();
    // EPD pools are registered separately from the KV arena.
    prefill
        .engine()
        .register(embedding_pool.as_ptr() as usize, embedding_pool.len())
        .unwrap();
    let embedding_copies = spans(&embedding, &embedding_pool);
    encoder
        .write(&advertised_session, &embedding_copies)
        .unwrap();
    assert_eq!(embedding_pool, embedding);

    let decode_session = decode.session_id();
    let cache_copies = spans(&source, &destination);
    prefill
        .write_cache(&decode_session, "cache-before", &cache_copies)
        .unwrap();
    reuse_endpoint(&decode);
    prefill
        .write_cache(&decode_session, "cache-after", &cache_copies)
        .unwrap();
    assert_eq!(destination, source);
    assert_eq!(prefill.session_id(), advertised_session);

    prefill.release_regions().unwrap();
    embedding.fill(17);
    encoder
        .write(&advertised_session, &embedding_copies)
        .unwrap();
    assert_eq!(embedding_pool, embedding);
    prefill.restore_regions(false).unwrap();
    embedding.fill(19);
    encoder
        .write(&advertised_session, &embedding_copies)
        .unwrap();
    assert_eq!(embedding_pool, embedding);
}
