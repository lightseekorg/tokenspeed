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

fn wake_regions(tags: Option<Vec<String>>) -> ControlRequest {
    ControlRequest::ResumeMemoryOccupation(ts_wire::control::ResumeMemoryOccupationReqInput {
        rid: None,
        http_worker_ipc: None,
        tags,
    })
}

#[test]
fn configured_memory_regions_normalize_and_round_trip_in_lifecycle_order() {
    for regions in [
        &["kv_cache", "weights", "acl_graph"][..],
        &["kv_cache", "weights", "cuda_graph"][..],
        &["kv_cache", "weights", "cuda_graph", "nccl"][..],
    ] {
        for omitted in [false, true] {
            let mut controls = controls(true);
            controls.config.memory_regions =
                regions.iter().rev().map(|tag| (*tag).to_owned()).collect();
            let requested = (!omitted).then(|| {
                let mut tags: Vec<_> = regions.iter().rev().copied().collect();
                tags.push("weights");
                tags
            });
            controls.handle(release(requested), None);
            let Some(Drain::Release(tags)) = controls.take_drain() else {
                panic!("configured regions drain");
            };
            assert_eq!(tags, regions);
            assert!(controls.take_replies().is_empty());
            controls.finish_release(tags);
            assert_eq!(
                successes(&controls.take_replies()),
                [("ReleaseMemoryOccupationReqOutput", true)]
            );
            let expected: Vec<_> = regions.iter().rev().copied().collect();
            let requested =
                (!omitted).then(|| regions.iter().map(|tag| (*tag).to_owned()).collect());
            let effect = controls.handle(wake_regions(requested), None);
            let tags = if regions.contains(&"nccl") {
                assert!(matches!(effect, Effect::None));
                assert!(controls.take_replies().is_empty());
                let Some(ControlRequest::ResumeMemoryOccupation(resume)) = controls.pop_weight_op()
                else {
                    panic!("NCCL wake waits for the operation gate");
                };
                assert!(controls.weight_head().is_none());
                resume.tags.unwrap()
            } else {
                let Effect::ResumeMemory(tags) = effect else {
                    panic!("regions resume");
                };
                tags
            };
            assert_eq!(tags, expected);
            controls.finish_resume(&tags);
            assert!(!controls.memory_released());
            assert_eq!(
                successes(&controls.take_replies()),
                [("ResumeMemoryOccupationReqOutput", true)]
            );
        }
    }
}

#[test]
fn memory_tag_refusals_use_the_backend_tag_set_and_configured_regions() {
    for (graph, unavailable) in [("acl_graph", "cuda_graph"), ("cuda_graph", "acl_graph")] {
        let mut controls = controls(true);
        controls.config.memory_regions.push(graph.to_owned());
        // Unknown, known-on-another-backend, and known-but-unavailable tags
        // must all refuse without starting a drain or changing sleeping state.
        for tag in ["unknown", unavailable, "nccl"] {
            for releasing in [true, false] {
                let request = if releasing {
                    release(Some(vec![tag]))
                } else {
                    wake_regions(Some(vec![tag.to_owned()]))
                };
                controls.handle(request, None);
                let replies = controls.take_replies();
                let (success, message) = match replies.as_slice() {
                    [ControlOutput::ReleaseMemoryOccupation(reply)] => {
                        (reply.success, &reply.message)
                    }
                    [ControlOutput::ResumeMemoryOccupation(reply)] => {
                        (reply.success, &reply.message)
                    }
                    _ => panic!("one refused reply"),
                };
                assert!(!success);
                assert!(message.contains(tag));
                assert!(message.contains(graph), "backend regions: {message}");
                if tag == "unknown" {
                    assert!(
                        !message.contains(unavailable),
                        "backend tag universe: {message}"
                    );
                }
                if tag != "nccl" || graph == "acl_graph" {
                    assert!(message.starts_with("invalid tags:"));
                } else {
                    assert!(message.contains("not releasable"));
                }
                assert!(controls.drain().is_none());
                assert!(!controls.memory_released());
            }
        }
    }
}

#[test]
fn ascend_graph_release_includes_kv_once_in_release_order() {
    for (coordinated, requested, expected) in [
        (false, &["acl_graph"][..], &["kv_cache", "acl_graph"][..]),
        (
            true,
            &["acl_graph", "acl_graph"][..],
            &["kv_cache", "acl_graph"][..],
        ),
        (
            false,
            &["acl_graph", "weights"][..],
            &["kv_cache", "weights", "acl_graph"][..],
        ),
        (
            true,
            &["acl_graph", "weights", "kv_cache", "kv_cache"][..],
            &["kv_cache", "weights", "acl_graph"][..],
        ),
    ] {
        let mut controls = controls(true);
        controls.config.replica_memory_transitions = coordinated;
        controls.config.memory_regions.push("acl_graph".to_owned());
        controls.handle(release(Some(requested.to_vec())), None);
        let Some(Drain::Release(tags)) = controls.take_drain() else {
            panic!("graph sleep drains KV");
        };
        assert_eq!(tags, expected);
        assert!(controls.take_replies().is_empty());
        controls.finish_release(tags);
        assert_eq!(
            successes(&controls.take_replies()),
            [("ReleaseMemoryOccupationReqOutput", true)]
        );
    }
}

#[test]
fn graph_wake_preserves_the_backend_kv_sleeping_state() {
    for (graph, coordinated, kv_first) in [
        ("acl_graph", false, false),
        ("acl_graph", true, true),
        ("cuda_graph", false, false),
        ("cuda_graph", true, false),
    ] {
        let mut controls = controls(true);
        controls.config.replica_memory_transitions = coordinated;
        controls.config.memory_regions.push(graph.to_owned());
        if kv_first {
            controls.handle(release(Some(vec!["kv_cache"])), None);
            let Some(Drain::Release(tags)) = controls.take_drain() else {
                panic!("KV drains");
            };
            assert_eq!(tags, ["kv_cache"]);
            controls.finish_release(tags);
            assert_eq!(
                successes(&controls.take_replies()),
                [("ReleaseMemoryOccupationReqOutput", true)]
            );
        }
        controls.handle(release(Some(vec![graph])), None);
        let Some(Drain::Release(tags)) = controls.take_drain() else {
            panic!("graph drains");
        };
        let expected = if graph == "acl_graph" && !kv_first {
            vec!["kv_cache", graph]
        } else {
            vec![graph]
        };
        assert_eq!(tags, expected);
        controls.finish_release(tags);
        assert_eq!(
            successes(&controls.take_replies()),
            [("ReleaseMemoryOccupationReqOutput", true)]
        );
        for tag in if graph == "acl_graph" {
            vec![graph, "kv_cache"]
        } else {
            vec![graph]
        } {
            let Effect::ResumeMemory(tags) =
                controls.handle(wake_regions(Some(vec![tag.to_owned()])), None)
            else {
                panic!("requested region resumes");
            };
            assert_eq!(tags, [tag]);
            controls.finish_resume(&tags);
            let kv_asleep = graph == "acl_graph" && tag == graph;
            assert_eq!(controls.memory_released(), kv_asleep);
            assert_eq!(controls.forward_blocked(), kv_asleep);
            assert_eq!(
                successes(&controls.take_replies()),
                [("ResumeMemoryOccupationReqOutput", true)]
            );
        }
    }
}
