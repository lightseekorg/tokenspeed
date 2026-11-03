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

//! Meshes of threads standing in for ranks, over real localhost sockets.

use std::{sync::mpsc, thread, time::Duration};

use ts_collective::{Error, Group, Listener, Mesh, Reduce};

/// Run `body` on `world` ranks, each with its own connected mesh; returns
/// each rank's result in rank order.
fn ranks<T: Send + 'static>(
    world: usize,
    body: impl Fn(usize, &mut Mesh) -> T + Send + Sync + Clone + 'static,
) -> Vec<T> {
    let listeners: Vec<Listener> = (0..world)
        .map(|_| Listener::bind("127.0.0.1").unwrap())
        .collect();
    let addresses: Vec<String> = listeners
        .iter()
        .map(|listener| format!("127.0.0.1:{}", listener.port().unwrap()))
        .collect();
    let (results, received) = mpsc::channel();
    let threads: Vec<_> = listeners
        .into_iter()
        .enumerate()
        .map(|(rank, listener)| {
            let (addresses, body, results) = (addresses.clone(), body.clone(), results.clone());
            thread::spawn(move || {
                let mut mesh =
                    Mesh::connect(rank, listener, &addresses, Duration::from_secs(10)).unwrap();
                results.send((rank, body(rank, &mut mesh))).unwrap();
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    drop(results);
    let mut out: Vec<(usize, T)> = received.iter().collect();
    out.sort_by_key(|(rank, _)| *rank);
    out.into_iter().map(|(_, result)| result).collect()
}

#[test]
fn collectives_agree_on_the_world_and_on_subgroups() {
    let results = ranks(4, |rank, mesh| {
        let world = Group::new(0, vec![0, 1, 2, 3], rank).unwrap();
        // Two groups of two, interleaved with world collectives.
        let pair = if rank < 2 { vec![0, 1] } else { vec![2, 3] };
        let pair = Group::new(1 + (rank as u32 / 2), pair, rank).unwrap();

        let root_payload = format!("from {rank}").into_bytes();
        let broadcast = mesh.broadcast(&world, 2, root_payload).unwrap();
        let gathered = mesh.all_gather(&pair, &[rank as u8]).unwrap();
        let max = mesh
            .all_reduce(&world, &[rank as i64, -(rank as i64)], Reduce::Max)
            .unwrap();
        let sum = mesh.all_reduce(&pair, &[rank as i64], Reduce::Sum).unwrap();
        (broadcast, gathered, max, sum)
    });
    for (rank, (broadcast, gathered, max, sum)) in results.into_iter().enumerate() {
        assert_eq!(broadcast, b"from 2");
        let base = (rank / 2 * 2) as u8;
        assert_eq!(gathered, [vec![base], vec![base + 1]]);
        assert_eq!(max, [3, 0]);
        assert_eq!(sum, [i64::from(base) * 2 + 1]);
    }
}

#[test]
fn a_large_payload_reaches_every_member() {
    let payload: Vec<u8> = (0..4 << 20).map(|i| (i % 251) as u8).collect();
    let expected = payload.clone();
    let results = ranks(3, move |rank, mesh| {
        let world = Group::new(0, vec![0, 1, 2], rank).unwrap();
        // Every member sends megabytes before anyone reads.
        let gathered = mesh.all_gather(&world, &payload).unwrap();
        gathered.iter().all(|part| *part == payload)
    });
    assert!(results.into_iter().all(|ok| ok));
    drop(expected);
}

#[test]
fn a_single_rank_group_needs_no_peer() {
    let results = ranks(2, |rank, mesh| {
        let alone = Group::new(10 + rank as u32, vec![rank], rank).unwrap();
        (
            mesh.broadcast(&alone, 0, vec![rank as u8]).unwrap(),
            mesh.all_reduce(&alone, &[7], Reduce::Min).unwrap(),
        )
    });
    assert_eq!(results, [(vec![0], vec![7]), (vec![1], vec![7])]);
}

#[test]
fn a_member_on_another_collective_is_out_of_step() {
    let results = ranks(2, |rank, mesh| {
        let world = Group::new(0, vec![0, 1], rank).unwrap();
        if rank == 0 {
            mesh.all_gather(&world, b"x").map(drop)
        } else {
            mesh.broadcast(&world, 0, Vec::new()).map(drop)
        }
    });
    // The member that read the other collective's frame says so; its peer
    // then finds it gone.
    assert!(
        matches!(results[1], Err(Error::Desync { rank: 0, .. })),
        "{:?}",
        results[1]
    );
    assert!(results[0].is_err());
}

#[test]
fn a_departed_peer_fails_the_collective() {
    let results = ranks(2, |rank, mesh| {
        let world = Group::new(0, vec![0, 1], rank).unwrap();
        if rank == 1 {
            // Leaves without taking part; dropping the mesh closes its sockets.
            return Ok(());
        }
        thread::sleep(Duration::from_millis(200));
        mesh.broadcast(&world, 1, Vec::new()).map(drop)
    });
    assert!(
        matches!(results[0], Err(Error::PeerClosed(1))),
        "{:?}",
        results[0]
    );
}

#[test]
fn groups_reject_a_foreign_rank_and_repeats() {
    assert!(Group::new(0, vec![0, 1], 2).is_err());
    assert!(Group::new(0, vec![0, 0], 0).is_err());
    let group = Group::new(3, vec![4, 6], 6).unwrap();
    assert_eq!(
        (group.size(), group.index(), group.ranks()),
        (2, 1, &[4, 6][..])
    );
}
