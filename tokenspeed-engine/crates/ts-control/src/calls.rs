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

//! Control calls ([`ControlCall`]): the control operations a frontend names
//! by method, turned into the engine's own [`ControlRequest`]s. The
//! transport answers each call with one `ControlReplySlim`; the operation
//! itself runs as any other control operation does.
//!
//! The methods and their arguments are the ones SMG calls on every
//! scheduler it drives:
//!
//! * `flush_cache([timeout_s])`: drop the prefix caches. A flush the engine
//!   refuses (it is not idle) is retried until `max(timeout_s, 2 s)` after
//!   the call arrived.
//! * `start_profile([options])`: start the profiler; `options` is a map of
//!   [`ProfileReq`]'s fields, and a key outside them is refused, so a caller
//!   never reads success for an option that did nothing.
//! * `stop_profile()`: stop it and export the traces.
//!
//! The RL and weight operations take `[body]`, the JSON body of the
//! SGLang-compatible RL route of the same name (which SMG's `/v1/rl` routes
//! carry), as a map (`[]` for a route whose body is optional or absent), and
//! read it with that route's field names and defaults, so a trainer's request
//! means what it means to an SGLang engine; a key the route ignores is
//! ignored here too. The engine's control loop then applies its own rules
//! (the pause modes, the L3 weight-version checks):
//!
//! * `pause_generation([{mode = "wait"}])`, `continue_generation()`;
//! * `init_weights_update_group([{master_address, master_port, rank_offset,
//!   world_size, group_name, backend}])`,
//!   `destroy_weights_update_group([{group_name}])`;
//! * `update_weights_from_distributed([{names, dtypes, shapes, group_name,
//!   flush_cache = false, weight_version}])`,
//!   `update_weights_from_disk([{model_path, load_format, flush_cache = true,
//!   weight_version}])`, `update_weights_from_mooncake([{version,
//!   flush_cache = true, weight_version}])`;
//! * `release_memory_occupation([{tags}])`,
//!   `resume_memory_occupation([{tags}])`;
//! * `rebalance_experts()`.

use std::time::{Duration, Instant};

use ts_wire::{
    control::{
        ControlCall, DestroyWeightsUpdateGroupReqInput, FlushCacheReqInput,
        InitWeightsUpdateGroupReqInput, PauseSchedulerReqInput, ProfileReq,
        RebalanceExpertsReqInput, ReleaseMemoryOccupationReqInput, ResumeMemoryOccupationReqInput,
        ResumeSchedulerReqInput, UpdateWeightFromDiskReqInput,
        UpdateWeightsFromDistributedReqInput, UpdateWeightsFromMooncakeReqInput, PROFILE_START,
        PROFILE_STOP,
    },
    msgpack::Integer,
    ControlRequest, Value,
};

/// The weight-update group an RL route names when its body names none.
const DEFAULT_GROUP: &str = "weight_update_group";

/// How often a refused flush is retried.
pub const FLUSH_RETRY_INTERVAL: Duration = Duration::from_millis(50);
/// The shortest a flush call keeps retrying, whatever its timeout.
const MIN_FLUSH_WAIT: Duration = Duration::from_secs(2);

/// A call as the engine runs it.
#[derive(Debug, Clone, PartialEq)]
pub struct Translated {
    /// The operation.
    pub request: ControlRequest,
    /// For a flush, until when a refusal is retried.
    pub flush_until: Option<Instant>,
}

/// The operation `call` names, arrived at `now`, or why it cannot run.
pub fn translate(call: &ControlCall, now: Instant) -> Result<Translated, String> {
    let request = |request| Translated {
        request,
        flush_until: None,
    };
    match call.method.as_str() {
        "flush_cache" => {
            let timeout = match call.args.as_slice() {
                [] => 0.0,
                [timeout] => number(timeout).ok_or_else(|| {
                    format!("flush_cache takes [timeout_s], a number; got {timeout:?}")
                })?,
                args => return Err(format!("flush_cache takes [timeout_s], got {args:?}")),
            };
            let wait = Duration::try_from_secs_f64(timeout.max(0.0)).unwrap_or(MIN_FLUSH_WAIT);
            Ok(Translated {
                request: ControlRequest::FlushCache(FlushCacheReqInput {
                    rid: None,
                    http_worker_ipc: None,
                }),
                flush_until: Some(now + wait.max(MIN_FLUSH_WAIT)),
            })
        }
        "start_profile" => {
            let options = match call.args.as_slice() {
                [] => &[][..],
                [Value::Map(options)] => options.as_slice(),
                args => {
                    return Err(format!(
                        "start_profile takes [options], a map of the profiler's options; got \
                         {args:?}"
                    ))
                }
            };
            Ok(request(ControlRequest::Profile(profile_start(options)?)))
        }
        "stop_profile" => {
            if !call.args.is_empty() {
                return Err(format!(
                    "stop_profile takes no arguments, got {:?}",
                    call.args
                ));
            }
            Ok(request(ControlRequest::Profile(profile(PROFILE_STOP))))
        }
        method => rl_operation(method, &call.args).map(request),
    }
}

/// The RL or weight operation `method` names, its body read off `args` (see
/// the module docs).
fn rl_operation(method: &str, args: &[Value]) -> Result<ControlRequest, String> {
    let body = Body::of(method, args)?;
    Ok(match method {
        "pause_generation" => ControlRequest::PauseScheduler(PauseSchedulerReqInput {
            rid: None,
            http_worker_ipc: None,
            mode: body.str_or("mode", "wait")?,
        }),
        "continue_generation" => {
            body.empty()?;
            ControlRequest::ResumeScheduler(ResumeSchedulerReqInput {
                rid: None,
                http_worker_ipc: None,
            })
        }
        "init_weights_update_group" => {
            ControlRequest::InitWeightsUpdateGroup(InitWeightsUpdateGroupReqInput {
                rid: None,
                http_worker_ipc: None,
                master_address: body.str("master_address")?,
                master_port: body.int("master_port")?,
                rank_offset: body.int("rank_offset")?,
                world_size: body.int("world_size")?,
                group_name: body.str_or("group_name", DEFAULT_GROUP)?,
                backend: body.str_or("backend", "nccl")?,
            })
        }
        "destroy_weights_update_group" => {
            ControlRequest::DestroyWeightsUpdateGroup(DestroyWeightsUpdateGroupReqInput {
                rid: None,
                http_worker_ipc: None,
                group_name: body.str_or("group_name", DEFAULT_GROUP)?,
            })
        }
        "update_weights_from_distributed" => {
            let names = body.strs("names")?;
            let dtype_names = body.strs("dtypes")?;
            let shapes = body.shapes("shapes")?;
            if names.len() != dtype_names.len() || names.len() != shapes.len() {
                return Err(format!(
                    "{method}: names, dtypes, shapes must have equal length"
                ));
            }
            ControlRequest::UpdateWeightsFromDistributed(UpdateWeightsFromDistributedReqInput {
                rid: None,
                http_worker_ipc: None,
                names,
                dtype_names,
                shapes,
                group_name: body.str_or("group_name", DEFAULT_GROUP)?,
                flush_cache: body.bool_or("flush_cache", false)?,
                weight_version: body.str_opt("weight_version")?,
            })
        }
        "update_weights_from_disk" => {
            ControlRequest::UpdateWeightFromDisk(UpdateWeightFromDiskReqInput {
                rid: None,
                http_worker_ipc: None,
                model_path: body.str("model_path")?,
                load_format: body.str_opt("load_format")?,
                weight_version: body.str_opt("weight_version")?,
                flush_cache: body.bool_or("flush_cache", true)?,
            })
        }
        "update_weights_from_mooncake" => {
            let version = body.int("version")?;
            let flush_cache = body.bool_or("flush_cache", true)?;
            // A flushed load without a version publishes the committed
            // version's own identity (`mooncake_load_weight_version`).
            let weight_version = body
                .str_opt("weight_version")?
                .or_else(|| flush_cache.then(|| version.to_string()));
            ControlRequest::UpdateWeightsFromMooncake(UpdateWeightsFromMooncakeReqInput {
                rid: None,
                http_worker_ipc: None,
                version,
                flush_cache,
                weight_version,
            })
        }
        "release_memory_occupation" => {
            ControlRequest::ReleaseMemoryOccupation(ReleaseMemoryOccupationReqInput {
                rid: None,
                http_worker_ipc: None,
                tags: body.strs_opt("tags")?,
            })
        }
        "resume_memory_occupation" => {
            ControlRequest::ResumeMemoryOccupation(ResumeMemoryOccupationReqInput {
                rid: None,
                http_worker_ipc: None,
                tags: body.strs_opt("tags")?,
            })
        }
        "rebalance_experts" => {
            body.empty()?;
            ControlRequest::RebalanceExperts(RebalanceExpertsReqInput {
                rid: None,
                http_worker_ipc: None,
            })
        }
        other => unreachable!("Body::of admitted {other}"),
    })
}

/// The methods [`rl_operation`] serves.
const RL_METHODS: &[&str] = &[
    "pause_generation",
    "continue_generation",
    "init_weights_update_group",
    "destroy_weights_update_group",
    "update_weights_from_distributed",
    "update_weights_from_disk",
    "update_weights_from_mooncake",
    "release_memory_occupation",
    "resume_memory_occupation",
    "rebalance_experts",
];

/// An RL call's body: the route's JSON body as a map, read by field name. A
/// nil field is an absent one, as the route's `body.get` reads it.
struct Body<'a> {
    method: &'a str,
    fields: &'a [(Value, Value)],
}

impl<'a> Body<'a> {
    /// `[body]` or `[]` for an RL method; an unknown method is refused here,
    /// naming the ones served.
    fn of(method: &'a str, args: &'a [Value]) -> Result<Self, String> {
        if !RL_METHODS.contains(&method) {
            return Err(format!(
                "unknown control call {method:?}: this engine serves flush_cache, start_profile, \
                 stop_profile, {}",
                RL_METHODS.join(", ")
            ));
        }
        let fields = match args {
            [] => &[][..],
            [Value::Map(fields)] => fields.as_slice(),
            args => {
                return Err(format!(
                    "{method} takes [body], the JSON body of its HTTP route as a map; got {args:?}"
                ))
            }
        };
        Ok(Self { method, fields })
    }

    fn get(&self, name: &str) -> Option<&'a Value> {
        self.fields
            .iter()
            .find(|(key, _)| matches!(key, Value::Str(key) if key == name))
            .map(|(_, value)| value)
            .filter(|value| **value != Value::Nil)
    }

    fn wrong(&self, name: &str, expected: &str, value: &Value) -> String {
        format!("{} {name} takes {expected}, got {value:?}", self.method)
    }

    fn missing(&self, name: &str) -> String {
        format!("{}: missing {name:?} in the request body", self.method)
    }

    /// A route without a body: its call takes none.
    fn empty(&self) -> Result<(), String> {
        if self.fields.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{} takes no body, got {:?}",
                self.method, self.fields
            ))
        }
    }

    fn str_opt(&self, name: &str) -> Result<Option<String>, String> {
        self.get(name)
            .map(|value| string(value).ok_or_else(|| self.wrong(name, "a string", value)))
            .transpose()
    }

    fn str(&self, name: &str) -> Result<String, String> {
        self.str_opt(name)?.ok_or_else(|| self.missing(name))
    }

    fn str_or(&self, name: &str, default: &str) -> Result<String, String> {
        Ok(self.str_opt(name)?.unwrap_or_else(|| default.to_owned()))
    }

    fn int(&self, name: &str) -> Result<i64, String> {
        let value = self.get(name).ok_or_else(|| self.missing(name))?;
        integer(value).ok_or_else(|| self.wrong(name, "an integer", value))
    }

    fn bool_or(&self, name: &str, default: bool) -> Result<bool, String> {
        self.get(name).map_or(Ok(default), |value| {
            boolean(value).ok_or_else(|| self.wrong(name, "a boolean", value))
        })
    }

    fn strs_opt(&self, name: &str) -> Result<Option<Vec<String>>, String> {
        self.get(name)
            .map(|value| {
                let Value::Array(items) = value else {
                    return Err(self.wrong(name, "a list of strings", value));
                };
                items
                    .iter()
                    .map(string)
                    .collect::<Option<_>>()
                    .ok_or_else(|| self.wrong(name, "a list of strings", value))
            })
            .transpose()
    }

    fn strs(&self, name: &str) -> Result<Vec<String>, String> {
        self.strs_opt(name)?.ok_or_else(|| self.missing(name))
    }

    fn shapes(&self, name: &str) -> Result<Vec<Vec<i64>>, String> {
        let value = self.get(name).ok_or_else(|| self.missing(name))?;
        let wrong = || self.wrong(name, "a list of integer lists", value);
        let Value::Array(shapes) = value else {
            return Err(wrong());
        };
        shapes
            .iter()
            .map(|shape| match shape {
                Value::Array(dims) => dims.iter().map(integer).collect::<Option<_>>(),
                _ => None,
            })
            .collect::<Option<_>>()
            .ok_or_else(wrong)
    }
}

/// A refused flush's reply, once its retries ran out.
pub fn flush_refusal(waited: Duration) -> String {
    format!(
        "the caches were not flushed: the engine was not idle within {:.1} s",
        waited.as_secs_f64()
    )
}

fn profile(kind: i64) -> ProfileReq {
    ProfileReq {
        rid: None,
        http_worker_ipc: None,
        kind,
        output_dir: None,
        start_step: None,
        num_steps: None,
        activities: None,
        profile_by_stage: false,
        with_stack: None,
        record_shapes: None,
        profile_id: None,
    }
}

fn profile_start(options: &[(Value, Value)]) -> Result<ProfileReq, String> {
    let mut request = profile(PROFILE_START);
    for (key, value) in options {
        let Value::Str(key) = key else {
            return Err(format!(
                "start_profile option keys are strings, got {key:?}"
            ));
        };
        let wrong = |expected: &str| format!("start_profile {key} takes {expected}, got {value:?}");
        // A nil value leaves the option unset.
        if *value == Value::Nil {
            if matches!(key.as_str(), "profile_by_stage") {
                return Err(wrong("a boolean"));
            }
            continue;
        }
        match key.as_str() {
            "output_dir" => {
                request.output_dir = Some(string(value).ok_or_else(|| wrong("a string"))?)
            }
            "profile_id" => {
                request.profile_id = Some(string(value).ok_or_else(|| wrong("a string"))?)
            }
            "start_step" => {
                request.start_step = Some(integer(value).ok_or_else(|| wrong("an integer"))?)
            }
            "num_steps" => {
                request.num_steps = Some(integer(value).ok_or_else(|| wrong("an integer"))?)
            }
            "with_stack" => {
                request.with_stack = Some(boolean(value).ok_or_else(|| wrong("a boolean"))?)
            }
            "record_shapes" => {
                request.record_shapes = Some(boolean(value).ok_or_else(|| wrong("a boolean"))?)
            }
            "profile_by_stage" => {
                request.profile_by_stage = boolean(value).ok_or_else(|| wrong("a boolean"))?
            }
            "activities" => {
                let Value::Array(items) = value else {
                    return Err(wrong("a list of strings"));
                };
                request.activities = Some(
                    items
                        .iter()
                        .map(string)
                        .collect::<Option<_>>()
                        .ok_or_else(|| wrong("a list of strings"))?,
                );
            }
            other => {
                return Err(format!(
                    "start_profile has no option {other:?}; it takes output_dir, start_step, \
                     num_steps, activities, with_stack, record_shapes, profile_by_stage and \
                     profile_id"
                ))
            }
        }
    }
    Ok(request)
}

fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Float(value) => Some(*value),
        Value::Int(Integer::Signed(value)) => Some(*value as f64),
        Value::Int(Integer::Unsigned(value)) => Some(*value as f64),
        _ => None,
    }
}

fn integer(value: &Value) -> Option<i64> {
    match value {
        Value::Int(value) => value.narrow("an option").ok(),
        _ => None,
    }
}

fn boolean(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(value) => Some(*value),
        _ => None,
    }
}

fn string(value: &Value) -> Option<String> {
    match value {
        Value::Str(value) => Some(value.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(method: &str, args: Vec<Value>) -> ControlCall {
        ControlCall {
            call_id: 1,
            method: method.to_owned(),
            args,
        }
    }

    #[test]
    fn a_flush_keeps_trying_for_its_timeout_and_at_least_two_seconds() {
        let now = Instant::now();
        let flush = translate(&call("flush_cache", vec![Value::Float(5.0)]), now).unwrap();
        assert!(matches!(flush.request, ControlRequest::FlushCache(_)));
        assert_eq!(flush.flush_until, Some(now + Duration::from_secs(5)));
        let short = translate(&call("flush_cache", vec![Value::Float(0.5)]), now).unwrap();
        assert_eq!(short.flush_until, Some(now + MIN_FLUSH_WAIT));
        let bare = translate(&call("flush_cache", Vec::new()), now).unwrap();
        assert_eq!(bare.flush_until, Some(now + MIN_FLUSH_WAIT));
        assert!(translate(&call("flush_cache", vec!["soon".into()]), now).is_err());
    }

    #[test]
    fn profile_options_map_onto_the_profilers_request() {
        let options = Value::Map(vec![
            ("output_dir".into(), "/tmp/trace".into()),
            ("num_steps".into(), Value::Int(Integer::Unsigned(5))),
            (
                "activities".into(),
                Value::Array(vec!["CPU".into(), "GPU".into()]),
            ),
            ("with_stack".into(), Value::Bool(true)),
            ("profile_by_stage".into(), Value::Bool(false)),
            ("profile_id".into(), "run-1".into()),
            ("start_step".into(), Value::Nil),
        ]);
        let start = translate(&call("start_profile", vec![options]), Instant::now()).unwrap();
        let ControlRequest::Profile(request) = start.request else {
            panic!("a profile request");
        };
        assert_eq!(request.kind, PROFILE_START);
        assert_eq!(request.output_dir.as_deref(), Some("/tmp/trace"));
        assert_eq!(request.num_steps, Some(5));
        assert_eq!(
            request.activities,
            Some(vec!["CPU".to_owned(), "GPU".to_owned()])
        );
        assert_eq!(request.with_stack, Some(true));
        assert_eq!(request.start_step, None);
        assert_eq!(request.profile_id.as_deref(), Some("run-1"));
        assert_eq!(start.flush_until, None);

        let stop = translate(&call("stop_profile", Vec::new()), Instant::now()).unwrap();
        assert!(matches!(
            stop.request,
            ControlRequest::Profile(ProfileReq {
                kind: PROFILE_STOP,
                ..
            })
        ));
    }

    #[test]
    fn what_the_engine_cannot_do_is_refused_by_name() {
        let now = Instant::now();
        let unknown_option = Value::Map(vec![("merge_profiles".into(), Value::Bool(true))]);
        let error = translate(&call("start_profile", vec![unknown_option]), now).unwrap_err();
        assert!(error.contains("merge_profiles"), "{error}");
        let error = translate(&call("load_lora_adapter", Vec::new()), now).unwrap_err();
        assert!(error.contains("load_lora_adapter"), "{error}");
        assert!(error.contains("update_weights_from_distributed"), "{error}");
        assert!(translate(&call("stop_profile", vec![Value::Nil]), now).is_err());
    }

    fn body(fields: Vec<(&str, Value)>) -> Value {
        Value::Map(
            fields
                .into_iter()
                .map(|(key, value)| (key.into(), value))
                .collect(),
        )
    }

    fn int(value: i64) -> Value {
        Value::Int(Integer::Signed(value))
    }

    fn rl(method: &str, args: Vec<Value>) -> Result<ControlRequest, String> {
        translate(&call(method, args), Instant::now()).map(|translated| translated.request)
    }

    /// A route's body means the same over a call: its field names (a
    /// distributed update's `dtypes` are the request's `dtype_names`) and its
    /// defaults, which are not always the request's own (a pause waits, a
    /// distributed update keeps the cache, a disk or Mooncake load flushes it).
    #[test]
    fn rl_bodies_read_with_their_routes_names_and_defaults() {
        let ControlRequest::PauseScheduler(pause) = rl("pause_generation", Vec::new()).unwrap()
        else {
            panic!("a pause");
        };
        assert_eq!(pause.mode, "wait");
        let ControlRequest::PauseScheduler(pause) = rl(
            "pause_generation",
            vec![body(vec![("mode", "abort".into())])],
        )
        .unwrap() else {
            panic!("a pause");
        };
        assert_eq!(pause.mode, "abort");
        assert!(matches!(
            rl("continue_generation", Vec::new()).unwrap(),
            ControlRequest::ResumeScheduler(_)
        ));

        let ControlRequest::InitWeightsUpdateGroup(init) = rl(
            "init_weights_update_group",
            vec![body(vec![
                ("master_address", "10.0.0.1".into()),
                ("master_port", int(29500)),
                ("rank_offset", int(1)),
                ("world_size", int(9)),
            ])],
        )
        .unwrap() else {
            panic!("an init");
        };
        assert_eq!(
            (
                init.master_address.as_str(),
                init.master_port,
                init.rank_offset,
                init.world_size
            ),
            ("10.0.0.1", 29500, 1, 9)
        );
        assert_eq!(
            (init.group_name.as_str(), init.backend.as_str()),
            (DEFAULT_GROUP, "nccl")
        );

        let ControlRequest::UpdateWeightsFromDistributed(update) = rl(
            "update_weights_from_distributed",
            vec![body(vec![
                ("names", Value::Array(vec!["w".into()])),
                ("dtypes", Value::Array(vec!["bfloat16".into()])),
                (
                    "shapes",
                    Value::Array(vec![Value::Array(vec![int(2), int(3)])]),
                ),
                ("weight_version", "v2".into()),
                ("load_format", Value::Nil),
            ])],
        )
        .unwrap() else {
            panic!("a distributed update");
        };
        assert_eq!(update.dtype_names, ["bfloat16"]);
        assert_eq!(update.shapes, [vec![2, 3]]);
        assert!(!update.flush_cache);
        assert_eq!(update.weight_version.as_deref(), Some("v2"));
        assert_eq!(update.group_name, DEFAULT_GROUP);

        let ControlRequest::UpdateWeightFromDisk(disk) = rl(
            "update_weights_from_disk",
            vec![body(vec![("model_path", "/ckpt".into())])],
        )
        .unwrap() else {
            panic!("a disk load");
        };
        assert_eq!(disk.model_path, "/ckpt");
        assert!(disk.flush_cache);

        let ControlRequest::ReleaseMemoryOccupation(release) = rl(
            "release_memory_occupation",
            vec![body(vec![("tags", Value::Array(vec!["kv_cache".into()]))])],
        )
        .unwrap() else {
            panic!("a release");
        };
        assert_eq!(release.tags, Some(vec!["kv_cache".to_owned()]));
        let ControlRequest::ResumeMemoryOccupation(resume) =
            rl("resume_memory_occupation", Vec::new()).unwrap()
        else {
            panic!("a resume");
        };
        assert_eq!(resume.tags, None);
        assert!(matches!(
            rl("destroy_weights_update_group", Vec::new()).unwrap(),
            ControlRequest::DestroyWeightsUpdateGroup(DestroyWeightsUpdateGroupReqInput {
                ref group_name,
                ..
            }) if group_name == DEFAULT_GROUP
        ));
        assert!(matches!(
            rl("rebalance_experts", Vec::new()).unwrap(),
            ControlRequest::RebalanceExperts(_)
        ));
    }

    /// A flushed Mooncake load without a version publishes the committed
    /// version's identity; an unflushed one keeps the current namespace; an
    /// explicit version always wins.
    #[test]
    fn a_mooncake_load_names_its_version_as_the_route_does() {
        let load = |fields| match rl("update_weights_from_mooncake", vec![body(fields)]) {
            Ok(ControlRequest::UpdateWeightsFromMooncake(load)) => load,
            other => panic!("a Mooncake load, got {other:?}"),
        };
        let flushed = load(vec![("version", int(7))]);
        assert!(flushed.flush_cache);
        assert_eq!(flushed.weight_version.as_deref(), Some("7"));
        let kept = load(vec![
            ("version", int(7)),
            ("flush_cache", Value::Bool(false)),
        ]);
        assert_eq!(kept.weight_version, None);
        let named = load(vec![
            ("version", int(7)),
            ("weight_version", "step-7".into()),
        ]);
        assert_eq!(named.weight_version.as_deref(), Some("step-7"));
    }

    /// A body the route would refuse is refused by name: a missing or
    /// mistyped field, lists of different lengths, a body where the route
    /// takes none, and arguments that are not one map.
    #[test]
    fn a_body_the_route_refuses_is_refused_by_name() {
        let error = rl("init_weights_update_group", vec![body(vec![])]).unwrap_err();
        assert!(error.contains("master_address"), "{error}");
        let error = rl(
            "update_weights_from_mooncake",
            vec![body(vec![("version", Value::Bool(true))])],
        )
        .unwrap_err();
        assert!(error.contains("version takes an integer"), "{error}");
        let error = rl(
            "update_weights_from_distributed",
            vec![body(vec![
                ("names", Value::Array(vec!["a".into(), "b".into()])),
                ("dtypes", Value::Array(vec!["bfloat16".into()])),
                ("shapes", Value::Array(vec![Value::Array(vec![int(1)])])),
            ])],
        )
        .unwrap_err();
        assert!(error.contains("equal length"), "{error}");
        let error = rl(
            "rebalance_experts",
            vec![body(vec![("now", Value::Bool(true))])],
        )
        .unwrap_err();
        assert!(error.contains("takes no body"), "{error}");
        let error = rl("pause_generation", vec!["wait".into()]).unwrap_err();
        assert!(error.contains("takes [body]"), "{error}");
    }
}
