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

use super::*;

#[tokio::test(flavor = "multi_thread")]
async fn a_refused_load_preserves_l3_publication_until_the_same_version_retry() {
    async fn publication(engine: &Engine, wait: bool) -> GetSharedPrefixPublicationReqOutput {
        let request =
            ControlRequest::GetSharedPrefixPublication(GetSharedPrefixPublicationReqInput {
                rid: None,
                http_worker_ipc: None,
                query_id: "publication".to_owned(),
                input_ids: (1..=65).collect(),
                max_prefix_tokens: 64,
            });
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let replies = engine.client.control(request.clone()).await.unwrap();
                let [ControlOutput::GetSharedPrefixPublication(reply)] = replies.as_slice() else {
                    panic!("one publication reply");
                };
                assert!(reply.success, "{}", reply.message);
                if !wait || reply.status == "published" {
                    return reply.clone();
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the publication reply did not arrive")
    }
    let snapshot = |reply: &GetSharedPrefixPublicationReqOutput| {
        (
            reply.status.clone(),
            reply.confirmed_prefix_tokens,
            reply.cache_epoch,
            reply.storage_namespace.clone(),
        )
    };

    let store = Arc::new(StorageProbe::default());
    let engine = start_ranks_with(
        Layout::Tp(2),
        1,
        Tiers::L3,
        Setup {
            l3_store: Some(store.clone()),
            weight_load_refused: Some(|rank| rank == 1),
            ..Setup::default()
        },
    )
    .await;
    assert_eq!(
        tokens(&run(&engine.client, request("warm", 65, budget(2))).await),
        [1000, 1001]
    );
    let before = publication(&engine, true).await;
    let old_namespace = ts_l3::keys::key_prefix("serving", "v1", "");
    assert_eq!(before.storage_namespace, old_namespace);
    assert!(before.cache_epoch > 0);
    assert_eq!(before.confirmed_prefix_tokens, 64);
    let old_keys = store.memory.keys();
    assert!(!old_keys.is_empty());
    assert!(old_keys.iter().all(|key| key.starts_with(&old_namespace)));

    let ControlRequest::UpdateWeightFromDisk(mut update) = disk_load() else {
        unreachable!();
    };
    update.weight_version = Some("v2".to_owned());
    let update = ControlRequest::UpdateWeightFromDisk(update);
    assert_eq!(
        control(&engine, update.clone()).await,
        [("UpdateWeightFromDiskReqOutput", false)]
    );
    let refused = publication(&engine, false).await;
    assert_eq!(snapshot(&refused), snapshot(&before));
    let refused_keys = store.memory.keys();
    assert!(old_keys.iter().all(|key| refused_keys.contains(key)));
    for rank in &engine.ranks {
        assert!(rank.device_ops.lock().unwrap().is_empty());
        assert!(rank.weight_commits.lock().unwrap().is_empty());
    }
    let chunks = run(&engine.client, request("after-refusal", 65, budget(2))).await;
    assert_eq!(tokens(&chunks), [1000, 1001]);
    assert!(chunks.last().unwrap().cached_tokens > 0);

    for rank in &engine.ranks {
        rank.weight_load_refused.store(false, Ordering::SeqCst);
    }
    // A just-finished request can still refuse the flush. Retry the same
    // checkpoint version; refused admission must not reserve that version.
    tokio::time::timeout(Duration::from_secs(10), async {
        while control(&engine, update.clone())
            .await
            .iter()
            .any(|(_, success)| !success)
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the same-version retry never succeeded");
    assert_weight_commits(&engine, &[true]).await;
    for rank in &engine.ranks {
        assert_eq!(
            *rank.device_ops.lock().unwrap(),
            ["UpdateWeightFromDiskReqInput"]
        );
    }
    assert!(
        store.memory.keys().is_empty(),
        "the old L3 namespace survived the flush"
    );
    let flushed = publication(&engine, false).await;
    assert_eq!(flushed.status, "pending");
    assert_eq!(flushed.confirmed_prefix_tokens, 0);
    assert!(flushed.cache_epoch > before.cache_epoch);
    let new_namespace = ts_l3::keys::key_prefix("serving", "v2", "");
    assert_eq!(flushed.storage_namespace, new_namespace);

    let chunks = run(&engine.client, request("after-retry", 65, budget(2))).await;
    assert_eq!(tokens(&chunks), [1000, 1001]);
    assert_eq!(chunks.last().unwrap().cached_tokens, 0);
    let after = publication(&engine, true).await;
    assert_eq!(after.status, "published");
    assert_eq!(after.confirmed_prefix_tokens, 64);
    assert_eq!(after.cache_epoch, flushed.cache_epoch);
    assert_eq!(after.storage_namespace, new_namespace);
    assert!(!store.memory.keys().is_empty());
    assert!(store
        .memory
        .keys()
        .iter()
        .all(|key| key.starts_with(&new_namespace)));
}

// Keep the protocol explicit while using the same real scripted scheduler/mesh.
fn run_memory_scripted(
    layout: Layout,
    replica_memory_transitions: bool,
    setup: Setup,
    scripts: Vec<Script>,
) -> Vec<ScriptedRank> {
    run_scripted(
        layout,
        0,
        Tiers::DeviceOnly,
        Setup {
            replica_memory_transitions,
            ..setup
        },
        scripts,
        no_reaction,
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn the_client_releases_and_resumes_all_memory_sharing_replicas() {
    let engine = start_ranks_with(
        Layout::TpDp { tp: 2, dp: 2 },
        0,
        Tiers::DeviceOnly,
        Setup {
            replica_memory_transitions: true,
            ..Setup::default()
        },
    )
    .await;
    let release = ControlRequest::ReleaseMemoryOccupation(ReleaseMemoryOccupationReqInput {
        rid: Some(StrOrList::Many(vec!["park".to_owned()])),
        http_worker_ipc: None,
        tags: None,
    });
    assert_eq!(
        control(&engine, release).await,
        [("ReleaseMemoryOccupationReqOutput", false); 2]
    );
    for rank in &engine.ranks {
        assert!(rank.device_ops.lock().unwrap().is_empty());
    }
    for rid in ["after-refusal-first", "after-refusal-second"] {
        let chunks = run(&engine.client, request(rid, 4, budget(2))).await;
        assert_eq!(tokens(&chunks), [1000, 1001]);
    }
    assert_eq!(
        control(&engine, memory_request(true, None, Some("park"))).await,
        [("ReleaseMemoryOccupationReqOutput", true); 2]
    );
    assert_eq!(
        control(&engine, memory_request(false, None, Some("wake"))).await,
        [("ResumeMemoryOccupationReqOutput", true); 2]
    );
    for rank in &engine.ranks {
        assert_eq!(
            *rank.device_ops.lock().unwrap(),
            ["release kv_cache,weights", "resume weights,kv_cache"]
        );
    }
    for rid in ["first-replica", "second-replica"] {
        let chunks = run(&engine.client, request(rid, 4, budget(2))).await;
        assert_eq!(tokens(&chunks), [1000, 1001]);
    }
}

#[test]
fn cancelling_memory_release_preserves_prefixes_when_one_rank_cannot_clear_l1() {
    let ranks = run_memory_scripted(
        Layout::Tp(2),
        true,
        Setup {
            host_l2_rank: Some(1),
            ..Setup::default()
        },
        vec![vec![
            vec![
                EngineRequest::Add(Box::new(request("seed", 33, budget(1)))),
                memory_transition(true, Some(&["kv_cache"])),
            ],
            vec![
                EngineRequest::Control(Box::new(resume())),
                EngineRequest::Add(Box::new(request("probe", 33, budget(1)))),
            ],
        ]],
    );
    // The warmup forward commits during the release drain. Rank 1 starts
    // a write-through finish store which pins its pages before the next
    // plan dispatches the copy. The script stops at the probe's dispatch
    // so the deliberately rank-local store never enters TP acknowledgment.
    assert_eq!(
        outcome(ranks[0].replies.clone()),
        [
            ("ReleaseMemoryOccupationReqOutput", false),
            ("ResumeSchedulerReqOutput", true),
        ]
    );
    assert!(ranks[0].cache_ops.is_empty());
    assert_eq!(ranks[1].cache_ops, [CacheOpKind::WriteBack]);
    let probe_prefixes: Vec<_> = ranks
        .iter()
        .map(|rank| {
            rank.prefill_prefixes
                .iter()
                .find_map(|(rid, prefix)| (rid == "probe").then_some(*prefix))
        })
        .collect();
    assert_eq!(
        probe_prefixes,
        [Some(32); 2],
        "the cached prefix on each rank after cancellation"
    );
    for (rank, observed) in ranks.iter().enumerate() {
        assert_eq!(observed.outcome, Ok(()), "rank {rank}");
        assert!(observed.device_ops.is_empty(), "rank {rank}");
    }
}

#[test]
fn an_empty_release_keeps_generation_available() {
    for coordinated in [false, true] {
        let mut script = vec![Vec::new(); 12];
        script[0].push(memory_transition(true, Some(&[])));
        script[2].push(EngineRequest::Add(Box::new(request("awake", 4, budget(2)))));
        let ranks = run_memory_scripted(Layout::Tp(2), coordinated, Setup::default(), vec![script]);
        assert_eq!(
            outcome(ranks[0].replies.clone()),
            [("ReleaseMemoryOccupationReqOutput", false)]
        );
        assert_scripted_tokens(&ranks[0], "awake", 2);
        assert!(ranks.iter().all(|rank| rank.device_ops.is_empty()));
    }
}

#[test]
fn a_rank_refusing_memory_preflight_leaves_the_replica_usable() {
    for (layout, failed_rank) in [(Layout::Tp(2), 1), (Layout::TpDp { tp: 2, dp: 2 }, 3)] {
        let mut script = vec![Vec::new(); 24];
        script[0].push(memory_transition(true, Some(&["weights"])));
        script[1].push(EngineRequest::Add(Box::new(request(
            "after-refusal",
            4,
            budget(2),
        ))));
        script[6].push(memory_transition(true, None));
        script[8].push(memory_transition(false, None));
        script[10].push(memory_transition(false, Some(&["weights"])));
        script[12].push(memory_transition(false, Some(&["kv_cache"])));
        script[14].push(EngineRequest::Control(Box::new(disk_load())));
        script[16].push(memory_transition(false, Some(&["kv_cache"])));
        script[17].push(EngineRequest::Add(Box::new(request(
            "after-wake",
            4,
            budget(2),
        ))));
        let ranks = run_memory_scripted(
            layout,
            true,
            Setup {
                discard_memory_rank: Some(failed_rank),
                ..Setup::default()
            },
            vec![script; layout.sizes().2],
        );
        for leader in ranks.iter().step_by(layout.sizes().1) {
            assert_eq!(
                outcome(leader.replies.clone()),
                [
                    ("ReleaseMemoryOccupationReqOutput", false),
                    ("ReleaseMemoryOccupationReqOutput", true),
                    ("ResumeMemoryOccupationReqOutput", false),
                    ("ResumeMemoryOccupationReqOutput", true),
                    ("ResumeMemoryOccupationReqOutput", false),
                    ("UpdateWeightFromDiskReqOutput", true),
                    ("ResumeMemoryOccupationReqOutput", true),
                ]
            );
            assert_scripted_tokens(leader, "after-refusal", 2);
            assert_scripted_tokens(leader, "after-wake", 2);
        }
        for rank in ranks {
            assert_eq!(rank.outcome, Ok(()));
            assert_eq!(
                rank.device_ops,
                [
                    "release kv_cache,weights",
                    "resume weights",
                    disk_load().tag(),
                    "resume kv_cache",
                ]
            );
        }
    }
}
#[test]
fn peer_device_failure_preserves_v2_replies_and_prevents_v3_domain_success() {
    for (layout, coordinated, failed_rank) in [
        (Layout::Tp(2), false, 1),
        (Layout::Tp(2), true, 1),
        (Layout::TpDp { tp: 2, dp: 2 }, true, 3),
    ] {
        for failed_release in [false, true] {
            let (_, tp, engines) = layout.sizes();
            // Legacy TP ends with the operation, before a later round can
            // observe the failing peer's disconnect instead of its reply.
            let rounds = if engines == 1 {
                if failed_release {
                    1
                } else {
                    2
                }
            } else {
                20
            };
            let mut scripts = vec![vec![Vec::new(); rounds]; engines];
            for (ingress, script) in scripts.iter_mut().enumerate() {
                script[ingress * 2].push(memory_transition(true, None));
                if !failed_release {
                    let wake = if engines == 1 { 1 } else { 5 + ingress * 2 };
                    script[wake].push(memory_transition(false, None));
                }
            }
            let ranks = run_memory_scripted(
                layout,
                coordinated,
                Setup {
                    memory_release_failure_rank: failed_release.then_some(failed_rank),
                    memory_resume_failure_rank: (!failed_release).then_some(failed_rank),
                    ..Setup::default()
                },
                scripts,
            );
            let operation = if failed_release { "release" } else { "resume" };
            assert!(ranks[failed_rank]
                .outcome
                .as_ref()
                .unwrap_err()
                .contains(&format!("injected memory {operation} failure")));
            let mut expected = Vec::new();
            if !failed_release || !coordinated {
                expected.push(("ReleaseMemoryOccupationReqOutput", true));
            }
            if !failed_release && !coordinated {
                expected.push(("ResumeMemoryOccupationReqOutput", true));
            }
            for (rank, observed) in ranks.into_iter().enumerate() {
                if rank % tp == 0 {
                    assert_eq!(outcome(observed.replies), expected);
                }
                assert_eq!(
                    observed.outcome.is_err(),
                    coordinated || rank == failed_rank
                );
            }
        }
    }
}

#[test]
fn dp_memory_release_and_resume_wait_for_delayed_replicas() {
    let mut scripts = vec![vec![Vec::new(); 65]; 2];
    scripts[0][2].push(memory_transition(true, None));
    scripts[1][6].push(EngineRequest::Add(Box::new(request(
        "drain",
        4,
        budget(24),
    ))));
    scripts[1][12].push(memory_transition(true, Some(&["weights", "kv_cache"])));
    scripts[0][42].push(memory_transition(false, None));
    scripts[1][50].push(memory_transition(false, Some(&["kv_cache", "weights"])));
    scripts[0][56].push(EngineRequest::Add(Box::new(request("awake", 4, budget(2)))));
    let ranks = run_memory_scripted(
        Layout::TpDp { tp: 2, dp: 2 },
        true,
        Setup::default(),
        scripts,
    );
    assert_scripted_tokens(&ranks[0], "awake", 2);
    assert_scripted_tokens(&ranks[2], "drain", 24);
    for leader in [&ranks[0], &ranks[2]] {
        assert_eq!(
            outcome(leader.replies.clone()),
            [
                ("ReleaseMemoryOccupationReqOutput", true),
                ("ResumeMemoryOccupationReqOutput", true),
            ]
        );
    }
    assert_eq!(ranks[0].idle_forwards, ranks[2].forwards);
    assert_eq!(ranks[2].idle_forwards, ranks[0].forwards);
    for rank in ranks {
        assert_eq!(rank.outcome, Ok(()));
        assert_eq!(
            rank.device_ops,
            ["release kv_cache,weights", "resume weights,kv_cache"]
        );
    }
}

#[test]
fn dp_memory_replies_follow_the_arrival_order_of_same_kind_requests() {
    let mut scripts = vec![vec![Vec::new(); 20]; 2];
    for (rank, script) in scripts.iter_mut().enumerate() {
        script[rank * 3].extend([
            memory_transition(true, Some(&["weights"])),
            memory_transition(true, None),
        ]);
    }
    let ranks = run_memory_scripted(
        Layout::Dp(2),
        true,
        Setup {
            discard_memory_rank: Some(1),
            ..Setup::default()
        },
        scripts,
    );
    for rank in ranks {
        assert_eq!(rank.outcome, Ok(()));
        assert_eq!(
            outcome(rank.replies),
            [
                ("ReleaseMemoryOccupationReqOutput", false),
                ("ReleaseMemoryOccupationReqOutput", true),
            ]
        );
        assert_eq!(rank.device_ops, ["release kv_cache,weights"]);
    }
}

#[test]
fn coordinated_memory_release_keeps_communication_resident() {
    let mut script = vec![Vec::new(); 12];
    script[0].push(memory_transition(true, Some(&["kv_cache", "nccl"])));
    script[2].push(EngineRequest::Add(Box::new(request(
        "still-awake",
        4,
        budget(2),
    ))));
    script[6].push(memory_transition(true, None));
    script[9].push(memory_transition(false, None));
    let ranks = run_memory_scripted(Layout::Tp(2), true, Setup::default(), vec![script]);
    assert_eq!(
        outcome(ranks[0].replies.clone()),
        [
            ("ReleaseMemoryOccupationReqOutput", false),
            ("ReleaseMemoryOccupationReqOutput", true),
            ("ResumeMemoryOccupationReqOutput", true),
        ]
    );
    assert_scripted_tokens(&ranks[0], "still-awake", 2);
    for rank in ranks {
        assert_eq!(rank.outcome, Ok(()));
        assert_eq!(
            rank.device_ops,
            ["release kv_cache,weights", "resume weights,kv_cache"]
        );
    }
}

#[test]
fn dp_ascend_graph_release_matches_effective_tags_and_waits_for_kv_wake() {
    for wake_kv in [false, true] {
        let mut scripts = vec![vec![Vec::new(); 30]; 2];
        scripts[0][0].push(memory_transition(true, Some(&["acl_graph"])));
        scripts[1][2].push(memory_transition(true, Some(&["kv_cache", "acl_graph"])));
        for script in &mut scripts {
            script[6].push(memory_transition(false, Some(&["acl_graph"])));
            if wake_kv {
                script[18].push(memory_transition(false, Some(&["kv_cache"])));
            }
        }
        scripts[0][10].push(EngineRequest::Add(Box::new(request("awake", 4, budget(2)))));
        let ranks = run_scripted(
            Layout::TpDp { tp: 2, dp: 2 },
            0,
            Tiers::DeviceOnly,
            Setup {
                replica_memory_transitions: true,
                ascend_graph: true,
                ..Setup::default()
            },
            scripts,
            no_reaction,
        );
        let mut expected_ops = vec!["release kv_cache,acl_graph", "resume acl_graph"];
        let mut expected_replies = vec![
            ("ReleaseMemoryOccupationReqOutput", true),
            ("ResumeMemoryOccupationReqOutput", true),
        ];
        if wake_kv {
            expected_ops.push("resume kv_cache");
            expected_replies.push(("ResumeMemoryOccupationReqOutput", true));
            assert_scripted_tokens(&ranks[0], "awake", 2);
        } else {
            assert!(ranks[0].finished.is_empty());
            assert_eq!(ranks[0].forwards, 0);
        }
        for leader in [&ranks[0], &ranks[2]] {
            assert_eq!(outcome(leader.replies.clone()), expected_replies);
        }
        for rank in ranks {
            assert_eq!(rank.outcome, Ok(()));
            assert_eq!(rank.device_ops, expected_ops);
        }
    }
}

#[test]
fn dp_memory_release_rejects_mismatched_tags_or_ids_and_can_retry() {
    for (peer_tags, peer_id) in [
        (Some(&["weights"][..]), "release-attempt"),
        (Some(&["kv_cache"][..]), "different-attempt"),
    ] {
        let mut scripts = vec![vec![Vec::new(); 20]; 2];
        scripts[0][0].push(EngineRequest::Control(Box::new(memory_request(
            true,
            Some(&["kv_cache"]),
            Some("release-attempt"),
        ))));
        scripts[1][2].push(EngineRequest::Control(Box::new(memory_request(
            true,
            peer_tags,
            Some(peer_id),
        ))));
        for script in &mut scripts {
            script[6].push(memory_transition(true, None));
            script[9].push(memory_transition(false, None));
        }
        let ranks = run_memory_scripted(Layout::Dp(2), true, Setup::default(), scripts);
        for rank in ranks {
            assert_eq!(rank.outcome, Ok(()));
            assert_eq!(
                outcome(rank.replies),
                [
                    ("ReleaseMemoryOccupationReqOutput", false),
                    ("ReleaseMemoryOccupationReqOutput", true),
                    ("ResumeMemoryOccupationReqOutput", true),
                ]
            );
            assert_eq!(
                rank.device_ops,
                ["release kv_cache,weights", "resume weights,kv_cache"]
            );
        }
    }
}

#[test]
fn dp_memory_release_and_resume_cannot_match_one_another() {
    let mut scripts = vec![vec![Vec::new(); 20]; 2];
    for script in &mut scripts {
        script[0].push(memory_transition(true, Some(&["kv_cache"])));
        script[10].push(memory_transition(false, Some(&["kv_cache"])));
    }
    scripts[0][4].push(memory_transition(true, Some(&["weights"])));
    scripts[1][6].push(memory_transition(false, Some(&["kv_cache"])));
    let ranks = run_memory_scripted(Layout::Dp(2), true, Setup::default(), scripts);
    for (rank, observed) in ranks.into_iter().enumerate() {
        assert_eq!(observed.outcome, Ok(()));
        assert_eq!(
            outcome(observed.replies),
            [
                ("ReleaseMemoryOccupationReqOutput", true),
                (
                    if rank == 0 {
                        "ReleaseMemoryOccupationReqOutput"
                    } else {
                        "ResumeMemoryOccupationReqOutput"
                    },
                    false,
                ),
                ("ResumeMemoryOccupationReqOutput", true),
            ]
        );
        assert_eq!(observed.device_ops, ["release kv_cache", "resume kv_cache"]);
    }
}

#[test]
fn dp_memory_missing_release_participant_can_cancel_without_sleeping() {
    let mut scripts = vec![vec![Vec::new(); 30]; 2];
    scripts[0][0].push(memory_transition(true, None));
    scripts[1][3].push(EngineRequest::Add(Box::new(request("peer", 4, budget(3)))));
    scripts[1][10].push(EngineRequest::Control(Box::new(resume())));
    scripts[0][13].push(EngineRequest::Add(Box::new(request(
        "after-cancel",
        4,
        budget(2),
    ))));
    let ranks = run_memory_scripted(Layout::Dp(2), true, Setup::default(), scripts);
    assert_scripted_tokens(&ranks[0], "after-cancel", 2);
    assert_scripted_tokens(&ranks[1], "peer", 3);
    assert_eq!(
        outcome(ranks[0].replies.clone()),
        [("ReleaseMemoryOccupationReqOutput", false)]
    );
    assert_eq!(
        outcome(ranks[1].replies.clone()),
        [("ResumeSchedulerReqOutput", true)]
    );
    assert!(ranks.iter().all(|rank| rank.device_ops.is_empty()));
}

#[test]
fn dp_memory_timeout_refuses_the_observed_batch_and_a_late_peer_can_retry() {
    use std::sync::mpsc;

    fn wait_for(deadline: Instant, mut ready: impl FnMut() -> bool) {
        while !ready() {
            assert!(
                Instant::now() < deadline,
                "memory timeout test did not finish"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }
    let (inputs, receivers): (Vec<_>, Vec<_>) = (0..2).map(|_| mpsc::channel()).unzip();
    let deadline = Instant::now() + Duration::from_secs(120);
    let send = |ingress: usize, messages| {
        let (delivered, delivery) = mpsc::channel();
        inputs[ingress].send((messages, delivered)).unwrap();
        delivery
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .unwrap();
    };
    let calls = |ids: &[&str]| {
        ids.iter()
            .map(|id| EngineRequest::Control(Box::new(memory_request(true, None, Some(id)))))
            .collect()
    };
    let ranks = run_scripted_live(
        Layout::TpDp { tp: 2, dp: 2 },
        0,
        Tiers::DeviceOnly,
        Setup {
            replica_memory_transitions: true,
            ..Setup::default()
        },
        vec![Vec::new(); 2],
        no_reaction,
        LiveRun {
            inputs: receivers,
            drive: |observed: &[LiveObservation], stop: &AtomicBool| {
                let leaders = [&observed[0], &observed[2]];
                send(0, calls(&["old-head"]));
                // A genuinely younger follower must be refused with the expired
                // head, rather than wait for its own production 30-second deadline.
                thread::sleep(Duration::from_secs(15));
                send(0, calls(&["young"]));
                wait_for(Instant::now() + Duration::from_secs(25), || {
                    leaders[0].replies.lock().unwrap().len() == 2
                });
                assert_eq!(
                    outcome(leaders[0].replies.lock().unwrap().clone()),
                    [("ReleaseMemoryOccupationReqOutput", false); 2]
                );
                assert!(leaders[1].replies.lock().unwrap().is_empty());

                // The late delivery belongs to the original fanout. Its own real
                // expiry must drain both old replies before a fresh ID can retry.
                send(1, calls(&["old-head", "young"]));
                wait_for(deadline, || leaders[1].replies.lock().unwrap().len() == 2);
                assert_eq!(
                    outcome(leaders[1].replies.lock().unwrap().clone()),
                    [("ReleaseMemoryOccupationReqOutput", false); 2]
                );
                assert!(observed
                    .iter()
                    .all(|rank| rank.device_ops.lock().unwrap().is_empty()));
                for (release, id, count) in [(true, "fresh-release", 3), (false, "fresh-resume", 4)]
                {
                    for ingress in 0..2 {
                        send(
                            ingress,
                            vec![EngineRequest::Control(Box::new(memory_request(
                                release,
                                None,
                                Some(id),
                            )))],
                        );
                    }
                    wait_for(deadline, || {
                        leaders
                            .iter()
                            .all(|rank| rank.replies.lock().unwrap().len() == count)
                    });
                }
                for leader in &leaders {
                    assert_eq!(
                        outcome(leader.replies.lock().unwrap().clone()),
                        [
                            ("ReleaseMemoryOccupationReqOutput", false),
                            ("ReleaseMemoryOccupationReqOutput", false),
                            ("ReleaseMemoryOccupationReqOutput", true),
                            ("ResumeMemoryOccupationReqOutput", true),
                        ]
                    );
                }
                for ingress in 0..2 {
                    send(
                        ingress,
                        vec![EngineRequest::Add(Box::new(request("awake", 4, budget(2))))],
                    );
                }
                wait_for(deadline, || {
                    leaders.iter().all(|rank| {
                        rank.outputs
                            .lock()
                            .unwrap()
                            .iter()
                            .any(|row| row.finish_reason.is_some())
                    })
                });
                stop.store(true, Ordering::Relaxed);
            },
        },
    );
    for leader in [&ranks[0], &ranks[2]] {
        assert_scripted_tokens(leader, "awake", 2);
    }
    for rank in ranks {
        assert_eq!(rank.outcome, Ok(()));
        assert_eq!(
            rank.device_ops,
            ["release kv_cache,weights", "resume weights,kv_cache"]
        );
    }
}

#[test]
fn dp_memory_continue_cancels_same_batch_and_asymmetric_calls_then_accepts_retry() {
    for asymmetric in [false, true] {
        let mut scripts = vec![vec![Vec::new(); 20]; 2];
        for (ingress, script) in scripts.iter_mut().enumerate() {
            if !asymmetric || ingress == 0 {
                script[0].push(EngineRequest::Control(Box::new(resume())));
            }
            script[0].push(memory_transition(true, None));
            script[6].push(memory_transition(true, None));
            script[10].push(memory_transition(false, None));
        }
        scripts[0][13].push(EngineRequest::Add(Box::new(request("awake", 4, budget(2)))));
        let ranks = run_memory_scripted(
            Layout::TpDp { tp: 2, dp: 2 },
            true,
            Setup::default(),
            scripts,
        );
        assert_scripted_tokens(&ranks[0], "awake", 2);
        for (rank, observed) in ranks.into_iter().enumerate() {
            assert_eq!(observed.outcome, Ok(()));
            let mut expected = Vec::new();
            if rank % 2 == 0 {
                if !asymmetric || rank == 0 {
                    expected.push(("ResumeSchedulerReqOutput", true));
                }
                expected.extend([
                    ("ReleaseMemoryOccupationReqOutput", false),
                    ("ReleaseMemoryOccupationReqOutput", true),
                    ("ResumeMemoryOccupationReqOutput", true),
                ]);
            }
            assert_eq!(outcome(observed.replies), expected);
            assert_eq!(
                observed.device_ops,
                ["release kv_cache,weights", "resume weights,kv_cache"]
            );
        }
    }
}

#[test]
fn dp_memory_cancels_an_active_drain_without_clearing_peer_prefixes() {
    let mut scripts = vec![vec![Vec::new(); 45]; 2];
    scripts[0][0].push(EngineRequest::Add(Box::new(request("seed", 33, budget(1)))));
    scripts[1][3].push(EngineRequest::Add(Box::new(request(
        "drain",
        4,
        budget(28),
    ))));
    scripts[0][6].push(memory_transition(true, None));
    scripts[1][7].push(memory_transition(true, None));
    scripts[1][11].push(EngineRequest::Control(Box::new(resume())));
    scripts[0][13].push(EngineRequest::Add(Box::new(request(
        "probe",
        33,
        budget(1),
    ))));
    let ranks = run_memory_scripted(Layout::Dp(2), true, Setup::default(), scripts);
    assert_scripted_tokens(&ranks[0], "probe", 1);
    assert_scripted_tokens(&ranks[1], "drain", 28);
    assert!(ranks[0]
        .prefill_prefixes
        .contains(&("probe".to_owned(), 32)));
    assert_eq!(
        outcome(ranks[0].replies.clone()),
        [("ReleaseMemoryOccupationReqOutput", false)]
    );
    let peer_replies = outcome(ranks[1].replies.clone());
    assert_eq!(peer_replies.len(), 2);
    assert!(peer_replies.contains(&("ReleaseMemoryOccupationReqOutput", false)));
    assert!(peer_replies.contains(&("ResumeSchedulerReqOutput", true)));
    assert!(ranks.iter().all(|rank| rank.device_ops.is_empty()));
}

#[test]
fn dp_memory_cancelled_resume_leaves_every_replica_asleep() {
    let mut scripts = vec![vec![Vec::new(); 25]; 2];
    for script in &mut scripts {
        script[0].push(memory_transition(true, None));
    }
    scripts[0][4].push(memory_transition(false, None));
    scripts[1][10].push(EngineRequest::Control(Box::new(resume())));
    scripts[1][13].push(EngineRequest::Add(Box::new(request(
        "still-asleep",
        4,
        budget(2),
    ))));
    let ranks = run_memory_scripted(Layout::Dp(2), true, Setup::default(), scripts);
    assert_eq!(
        outcome(ranks[0].replies.clone()),
        [
            ("ReleaseMemoryOccupationReqOutput", true),
            ("ResumeMemoryOccupationReqOutput", false),
        ]
    );
    assert_eq!(
        outcome(ranks[1].replies.clone()),
        [
            ("ReleaseMemoryOccupationReqOutput", true),
            ("ResumeSchedulerReqOutput", false),
        ]
    );
    for rank in ranks {
        assert_eq!(rank.outcome, Ok(()));
        assert_eq!(rank.device_ops, ["release kv_cache,weights"]);
        assert_eq!(rank.forwards, 0);
    }
}
