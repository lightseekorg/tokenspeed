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

//! Control operations (`engine/io_struct.py`): cache flush, pause, memory
//! release, weight updates, profiling and the like.
//!
//! A frontend sends one to every engine as a CONTROL message; each engine
//! answers with the matching reply on its output socket, beside the output
//! batches. Replies carry no request id: an engine answers each kind of
//! operation in the order it received them. Every struct is tagged and
//! starts with the two `BaseReq` fields.

use crate::{
    codec::{Dict, StrOrList, Wire},
    error::{Error, Result},
    msgpack::{Decoder, Encoder},
    value::Value,
};

/// Declares tagged control structs that start with the `BaseReq` fields.
macro_rules! control_struct {
    ($(
        $(#[$meta:meta])*
        $name:ident {
            $( $(#[$fmeta:meta])* $field:ident: $fty:ty $(= $fdefault:expr)? ),* $(,)?
        }
    )*) => {$(
        wire_struct! {
            $(#[$meta])*
            #[derive(Debug, Clone, PartialEq)]
            pub struct $name: array(stringify!($name)) {
                /// Request id; control operations rarely set it.
                pub rid: Option<StrOrList> = None,
                /// In-process HTTP worker address; unused across processes.
                pub http_worker_ipc: Option<String> = None,
                $( $(#[$fmeta])* pub $field: $fty $(= $fdefault)? ),*
            }
        }
    )*};
}

control_struct! {
    /// Drop the prefix caches.
    FlushCacheReqInput {}
    /// [`FlushCacheReqInput`]'s reply.
    FlushCacheReqOutput {
        /// Whether the caches were dropped.
        success: bool,
    }

    /// Stop scheduling: `abort` aborts what runs, `wait` lets it finish,
    /// `keep` freezes it in place.
    PauseSchedulerReqInput {
        /// `abort`, `wait` or `keep`.
        mode: String = "abort".to_owned(),
    }
    /// [`PauseSchedulerReqInput`]'s reply, once the pause took hold.
    PauseSchedulerReqOutput {
        /// Whether the engine paused.
        success: bool,
        /// Why not, if it did not.
        message: String = String::new(),
    }

    /// Resume scheduling after a pause.
    ResumeSchedulerReqInput {}
    /// [`ResumeSchedulerReqInput`]'s reply.
    ResumeSchedulerReqOutput {
        /// Whether the engine resumed.
        success: bool,
        /// Why not, if it did not.
        message: String = String::new(),
    }

    /// Whether scheduling is paused.
    IsSchedulerPausedReqInput {}
    /// [`IsSchedulerPausedReqInput`]'s reply.
    IsSchedulerPausedReqOutput {
        /// Paused.
        is_paused: bool,
    }

    /// Load new weights from a checkpoint on disk.
    UpdateWeightFromDiskReqInput {
        /// The checkpoint.
        model_path: String,
        /// Its format.
        load_format: Option<String> = None,
        /// The weight version after a successful load.
        weight_version: Option<String> = None,
        /// Drop the cached KV first: it was computed under the old weights.
        flush_cache: bool = true,
    }
    /// [`UpdateWeightFromDiskReqInput`]'s reply.
    UpdateWeightFromDiskReqOutput {
        /// Whether the weights were loaded.
        success: bool,
        /// Detail.
        message: String,
        /// Requests paused during the load.
        num_paused_requests: Option<i64> = Some(0),
    }

    /// Receive new weights from the trainer's NCCL group.
    UpdateWeightsFromDistributedReqInput {
        /// Parameter names, in broadcast order.
        names: Vec<String>,
        /// Their dtypes.
        dtype_names: Vec<String>,
        /// Their shapes.
        shapes: Vec<Vec<i64>>,
        /// The weight-update group.
        group_name: String = "weight_update_group".to_owned(),
        /// Drop the cached KV first.
        flush_cache: bool = true,
        /// The L3 namespace afterwards; `None` keeps the current one.
        weight_version: Option<String>,
    }
    /// [`UpdateWeightsFromDistributedReqInput`]'s reply.
    UpdateWeightsFromDistributedReqOutput {
        /// Whether the weights were received.
        success: bool,
        /// Detail.
        message: String,
    }

    /// Load weights from serialized tensors; the engines refuse it.
    UpdateWeightsFromTensorReqInput {
        /// One serialized tensor dict per world rank.
        serialized_named_tensors: Vec<Value>,
        /// Their format.
        load_format: Option<String>,
        /// Drop the cached KV first.
        flush_cache: bool,
        /// The weight version afterwards.
        weight_version: Option<String> = None,
    }
    /// [`UpdateWeightsFromTensorReqInput`]'s reply.
    UpdateWeightsFromTensorReqOutput {
        /// Whether the weights were loaded.
        success: bool,
        /// Detail.
        message: String,
    }

    /// Join the trainer's weight-update group.
    InitWeightsUpdateGroupReqInput {
        /// The group's rendezvous host.
        master_address: String,
        /// Its port.
        master_port: i64,
        /// This engine's first rank in the group.
        rank_offset: i64,
        /// The group's size.
        world_size: i64,
        /// The group's name.
        group_name: String = "weight_update_group".to_owned(),
        /// Its backend.
        backend: String = "nccl".to_owned(),
    }
    /// [`InitWeightsUpdateGroupReqInput`]'s reply.
    InitWeightsUpdateGroupReqOutput {
        /// Whether the engine joined.
        success: bool,
        /// Detail.
        message: String,
    }

    /// Leave a weight-update group.
    DestroyWeightsUpdateGroupReqInput {
        /// The group's name.
        group_name: String = "weight_update_group".to_owned(),
    }
    /// [`DestroyWeightsUpdateGroupReqInput`]'s reply.
    DestroyWeightsUpdateGroupReqOutput {
        /// Whether the engine left.
        success: bool,
        /// Detail.
        message: String,
    }

    /// Load one committed version from a Mooncake weight store.
    UpdateWeightsFromMooncakeReqInput {
        /// The version.
        version: i64,
        /// Drop the cached KV first.
        flush_cache: bool,
        /// The L3 namespace afterwards; `None` for the version's default.
        weight_version: Option<String>,
    }
    /// [`UpdateWeightsFromMooncakeReqInput`]'s reply.
    UpdateWeightsFromMooncakeReqOutput {
        /// Whether the weights were loaded.
        success: bool,
        /// Detail.
        message: String,
    }

    /// Start one online expert rebalance now.
    RebalanceExpertsReqInput {}
    /// [`RebalanceExpertsReqInput`]'s reply.
    RebalanceExpertsReqOutput {
        /// Whether the rebalance started.
        success: bool,
        /// Detail.
        message: String,
    }

    /// Release GPU memory (`weights`, `kv_cache`; all when `None`).
    ReleaseMemoryOccupationReqInput {
        /// The regions.
        tags: Option<Vec<String>> = None,
    }
    /// [`ReleaseMemoryOccupationReqInput`]'s reply.
    ReleaseMemoryOccupationReqOutput {
        /// Whether the memory was released.
        success: bool = true,
        /// Detail.
        message: String = String::new(),
    }

    /// Take released GPU memory back (all released regions when `None`).
    ResumeMemoryOccupationReqInput {
        /// The regions.
        tags: Option<Vec<String>> = None,
    }
    /// [`ResumeMemoryOccupationReqInput`]'s reply.
    ResumeMemoryOccupationReqOutput {
        /// Whether the memory was taken back.
        success: bool = true,
        /// Detail.
        message: String = String::new(),
    }

    /// Whether the engine released its memory.
    IsSleepingReqInput {}
    /// [`IsSleepingReqInput`]'s reply.
    IsSleepingReqOutput {
        /// Released.
        is_sleeping: bool,
    }

    /// The engine's internal state.
    GetInternalStateReq {}
    /// [`GetInternalStateReq`]'s reply.
    GetInternalStateReqOutput {
        /// The state.
        internal_state: Dict<Value, Value>,
    }

    /// Change server arguments at run time.
    SetInternalStateReq {
        /// The arguments to change.
        server_args: Dict<Value, Value>,
    }
    /// [`SetInternalStateReq`]'s reply.
    SetInternalStateReqOutput {
        /// Whether anything changed.
        updated: bool,
        /// The arguments now.
        server_args: Dict<Value, Value>,
    }

    /// Start (`kind` 1) or stop (`kind` 2) the profiler.
    ProfileReq {
        /// `ProfileReqType`: 1 starts, 2 stops.
        kind: i64,
        /// Where traces go.
        output_dir: Option<String> = None,
        /// The forward step to start at.
        start_step: Option<i64> = None,
        /// Steps to profile before stopping.
        num_steps: Option<i64> = None,
        /// What to record (`CPU`, `GPU`, `MEM`, ...).
        activities: Option<Vec<String>> = None,
        /// One trace per stage (prefill, decode).
        profile_by_stage: bool = false,
        /// Record Python stacks.
        with_stack: Option<bool> = None,
        /// Record tensor shapes.
        record_shapes: Option<bool> = None,
        /// Names the trace files.
        profile_id: Option<String> = None,
    }
    /// [`ProfileReq`]'s reply.
    ProfileReqOutput {
        /// Whether the profiler changed state.
        success: bool,
        /// Detail.
        message: String,
    }
}

/// `ProfileReq.kind` that starts the profiler.
pub const PROFILE_START: i64 = 1;
/// `ProfileReq.kind` that stops it.
pub const PROFILE_STOP: i64 = 2;

/// A control operation named by method, the other payload a CONTROL message
/// carries: the array `[call_id, method, args]`, as SMG calls a scheduler's
/// operations on every engine it drives. The engine turns it into its own
/// [`ControlRequest`] and answers with one [`ControlReplySlim`] under
/// `call_id`. It is told apart from a [`ControlRequest`] by its first
/// element: an integer here, the struct's tag string there.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlCall {
    /// The caller's id for the call; the reply carries it.
    pub call_id: i64,
    /// The operation, such as `flush_cache`.
    pub method: String,
    /// The operation's positional arguments.
    pub args: Vec<Value>,
}

impl Wire for ControlCall {
    fn encode(&self, encoder: &mut Encoder) {
        encoder.array_len(3);
        encoder.int(self.call_id);
        encoder.str(&self.method);
        self.args.encode(encoder);
    }

    fn decode(decoder: &mut Decoder<'_>) -> Result<Self> {
        let len = decoder.array_len()?;
        if len != 3 {
            return Err(Error::Invalid(format!(
                "a control call is [call_id, method, args], got {len} elements"
            )));
        }
        Ok(Self {
            call_id: decoder.integer()?.narrow("call_id")?,
            method: decoder.str()?.to_owned(),
            args: Vec::<Value>::decode(decoder)?,
        })
    }
}

wire_struct! {
    /// The answer to a [`ControlCall`], exactly one per call.
    #[derive(Debug, Clone, PartialEq)]
    pub struct ControlReplySlim: array("ControlReplySlim") {
        /// The call's id.
        pub call_id: i64,
        /// Whether the operation succeeded.
        pub success: bool,
        /// Why it failed, or the engine's words about it; `None` when there
        /// are none.
        pub message: Option<String> = None,
        /// The answering engine.
        pub engine_index: i64 = 0,
    }
}

/// Declares an enum over tagged structs, decoded by their tag.
macro_rules! tagged_union {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident { $( $(#[$vmeta:meta])* $variant:ident($ty:ident) ),* $(,)? }
    ) => {
        $(#[$meta])*
        $vis enum $name {
            $( $(#[$vmeta])* $variant($ty) ),*
        }

        impl $name {
            /// The variant's wire tag (its Python class name).
            pub fn tag(&self) -> &'static str {
                match self {
                    $( Self::$variant(_) => stringify!($ty) ),*
                }
            }

            /// Whether `tag` names a variant.
            pub fn knows(tag: &str) -> bool {
                matches!(tag, $( stringify!($ty) )|*)
            }
        }

        impl Wire for $name {
            fn encode(&self, encoder: &mut Encoder) {
                match self {
                    $( Self::$variant(value) => value.encode(encoder) ),*
                }
            }

            fn decode(decoder: &mut Decoder<'_>) -> Result<Self> {
                match decoder.peek_tag()? {
                    $( stringify!($ty) => Ok(Self::$variant($ty::decode(decoder)?)), )*
                    other => Err(Error::Invalid(format!(
                        concat!("unknown ", stringify!($name), " {}"),
                        other
                    ))),
                }
            }
        }
    };
}

tagged_union! {
    /// A control operation, the payload of a CONTROL message.
    #[derive(Debug, Clone, PartialEq)]
    pub enum ControlRequest {
        /// See [`FlushCacheReqInput`].
        FlushCache(FlushCacheReqInput),
        /// See [`PauseSchedulerReqInput`].
        PauseScheduler(PauseSchedulerReqInput),
        /// See [`ResumeSchedulerReqInput`].
        ResumeScheduler(ResumeSchedulerReqInput),
        /// See [`IsSchedulerPausedReqInput`].
        IsSchedulerPaused(IsSchedulerPausedReqInput),
        /// See [`UpdateWeightFromDiskReqInput`].
        UpdateWeightFromDisk(UpdateWeightFromDiskReqInput),
        /// See [`UpdateWeightsFromDistributedReqInput`].
        UpdateWeightsFromDistributed(UpdateWeightsFromDistributedReqInput),
        /// See [`UpdateWeightsFromTensorReqInput`].
        UpdateWeightsFromTensor(UpdateWeightsFromTensorReqInput),
        /// See [`InitWeightsUpdateGroupReqInput`].
        InitWeightsUpdateGroup(InitWeightsUpdateGroupReqInput),
        /// See [`DestroyWeightsUpdateGroupReqInput`].
        DestroyWeightsUpdateGroup(DestroyWeightsUpdateGroupReqInput),
        /// See [`UpdateWeightsFromMooncakeReqInput`].
        UpdateWeightsFromMooncake(UpdateWeightsFromMooncakeReqInput),
        /// See [`RebalanceExpertsReqInput`].
        RebalanceExperts(RebalanceExpertsReqInput),
        /// See [`ReleaseMemoryOccupationReqInput`].
        ReleaseMemoryOccupation(ReleaseMemoryOccupationReqInput),
        /// See [`ResumeMemoryOccupationReqInput`].
        ResumeMemoryOccupation(ResumeMemoryOccupationReqInput),
        /// See [`IsSleepingReqInput`].
        IsSleeping(IsSleepingReqInput),
        /// See [`GetInternalStateReq`].
        GetInternalState(GetInternalStateReq),
        /// See [`SetInternalStateReq`].
        SetInternalState(SetInternalStateReq),
        /// See [`ProfileReq`].
        Profile(ProfileReq),
    }
}

tagged_union! {
    /// A control operation's reply.
    #[derive(Debug, Clone, PartialEq)]
    pub enum ControlOutput {
        /// See [`FlushCacheReqOutput`].
        FlushCache(FlushCacheReqOutput),
        /// See [`PauseSchedulerReqOutput`].
        PauseScheduler(PauseSchedulerReqOutput),
        /// See [`ResumeSchedulerReqOutput`].
        ResumeScheduler(ResumeSchedulerReqOutput),
        /// See [`IsSchedulerPausedReqOutput`].
        IsSchedulerPaused(IsSchedulerPausedReqOutput),
        /// See [`UpdateWeightFromDiskReqOutput`].
        UpdateWeightFromDisk(UpdateWeightFromDiskReqOutput),
        /// See [`UpdateWeightsFromDistributedReqOutput`].
        UpdateWeightsFromDistributed(UpdateWeightsFromDistributedReqOutput),
        /// See [`UpdateWeightsFromTensorReqOutput`].
        UpdateWeightsFromTensor(UpdateWeightsFromTensorReqOutput),
        /// See [`InitWeightsUpdateGroupReqOutput`].
        InitWeightsUpdateGroup(InitWeightsUpdateGroupReqOutput),
        /// See [`DestroyWeightsUpdateGroupReqOutput`].
        DestroyWeightsUpdateGroup(DestroyWeightsUpdateGroupReqOutput),
        /// See [`UpdateWeightsFromMooncakeReqOutput`].
        UpdateWeightsFromMooncake(UpdateWeightsFromMooncakeReqOutput),
        /// See [`RebalanceExpertsReqOutput`].
        RebalanceExperts(RebalanceExpertsReqOutput),
        /// See [`ReleaseMemoryOccupationReqOutput`].
        ReleaseMemoryOccupation(ReleaseMemoryOccupationReqOutput),
        /// See [`ResumeMemoryOccupationReqOutput`].
        ResumeMemoryOccupation(ResumeMemoryOccupationReqOutput),
        /// See [`IsSleepingReqOutput`].
        IsSleeping(IsSleepingReqOutput),
        /// See [`GetInternalStateReqOutput`].
        GetInternalState(GetInternalStateReqOutput),
        /// See [`SetInternalStateReqOutput`].
        SetInternalState(SetInternalStateReqOutput),
        /// See [`ProfileReqOutput`].
        Profile(ProfileReqOutput),
    }
}

impl ControlRequest {
    /// The tag of the reply this operation gets.
    pub fn reply_tag(&self) -> &'static str {
        match self {
            Self::FlushCache(_) => "FlushCacheReqOutput",
            Self::PauseScheduler(_) => "PauseSchedulerReqOutput",
            Self::ResumeScheduler(_) => "ResumeSchedulerReqOutput",
            Self::IsSchedulerPaused(_) => "IsSchedulerPausedReqOutput",
            Self::UpdateWeightFromDisk(_) => "UpdateWeightFromDiskReqOutput",
            Self::UpdateWeightsFromDistributed(_) => "UpdateWeightsFromDistributedReqOutput",
            Self::UpdateWeightsFromTensor(_) => "UpdateWeightsFromTensorReqOutput",
            Self::InitWeightsUpdateGroup(_) => "InitWeightsUpdateGroupReqOutput",
            Self::DestroyWeightsUpdateGroup(_) => "DestroyWeightsUpdateGroupReqOutput",
            Self::UpdateWeightsFromMooncake(_) => "UpdateWeightsFromMooncakeReqOutput",
            Self::RebalanceExperts(_) => "RebalanceExpertsReqOutput",
            Self::ReleaseMemoryOccupation(_) => "ReleaseMemoryOccupationReqOutput",
            Self::ResumeMemoryOccupation(_) => "ResumeMemoryOccupationReqOutput",
            Self::IsSleeping(_) => "IsSleepingReqOutput",
            Self::GetInternalState(_) => "GetInternalStateReqOutput",
            Self::SetInternalState(_) => "SetInternalStateReqOutput",
            Self::Profile(_) => "ProfileReqOutput",
        }
    }
}

impl ControlOutput {
    /// Whether the operation succeeded, and the engine's words about it
    /// (`None` when it said nothing): what a [`ControlReplySlim`] carries.
    pub fn outcome(&self) -> (bool, Option<String>) {
        fn said(success: bool, message: &str) -> (bool, Option<String>) {
            (success, (!message.is_empty()).then(|| message.to_owned()))
        }
        match self {
            Self::FlushCache(output) => (output.success, None),
            Self::PauseScheduler(output) => said(output.success, &output.message),
            Self::ResumeScheduler(output) => said(output.success, &output.message),
            Self::UpdateWeightFromDisk(output) => said(output.success, &output.message),
            Self::UpdateWeightsFromDistributed(output) => said(output.success, &output.message),
            Self::UpdateWeightsFromTensor(output) => said(output.success, &output.message),
            Self::InitWeightsUpdateGroup(output) => said(output.success, &output.message),
            Self::DestroyWeightsUpdateGroup(output) => said(output.success, &output.message),
            Self::UpdateWeightsFromMooncake(output) => said(output.success, &output.message),
            Self::RebalanceExperts(output) => said(output.success, &output.message),
            Self::ReleaseMemoryOccupation(output) => said(output.success, &output.message),
            Self::ResumeMemoryOccupation(output) => said(output.success, &output.message),
            Self::Profile(output) => said(output.success, &output.message),
            Self::IsSchedulerPaused(_) | Self::IsSleeping(_) | Self::GetInternalState(_) => {
                (true, None)
            }
            Self::SetInternalState(output) => (output.updated, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{decode_frames, encode_frames};

    #[test]
    fn a_reply_tag_names_a_reply() {
        let requests = [
            ControlRequest::FlushCache(FlushCacheReqInput {
                rid: None,
                http_worker_ipc: None,
            }),
            ControlRequest::PauseScheduler(PauseSchedulerReqInput {
                rid: None,
                http_worker_ipc: None,
                mode: "wait".to_owned(),
            }),
        ];
        for request in requests {
            assert!(ControlOutput::knows(request.reply_tag()));
            assert!(ControlRequest::knows(request.tag()));
            let decoded: ControlRequest = decode_frames(&encode_frames(&request)).unwrap();
            assert_eq!(decoded, request);
        }
        assert!(!ControlOutput::knows("BatchTokenIDOutSlim"));
    }

    /// SMG's control call `(call_id, method, args)`, as its msgpack encoder
    /// writes `(7, "flush_cache", [5.0])`.
    const FLUSH_CALL: &[u8] = &[
        0x93, 0x07, 0xab, b'f', b'l', b'u', b's', b'h', b'_', b'c', b'a', b'c', b'h', b'e', 0x91,
        0xcb, 0x40, 0x14, 0, 0, 0, 0, 0, 0,
    ];

    #[test]
    fn a_control_call_reads_as_smg_writes_it() {
        let call: ControlCall = decode_frames(&[bytes::Bytes::from_static(FLUSH_CALL)]).unwrap();
        assert_eq!(
            call,
            ControlCall {
                call_id: 7,
                method: "flush_cache".to_owned(),
                args: vec![Value::Float(5.0)],
            }
        );
        assert_eq!(
            encode_frames(&call),
            [bytes::Bytes::from_static(FLUSH_CALL)]
        );
    }

    #[test]
    fn a_control_reply_is_the_tag_then_its_four_fields() {
        let reply = ControlReplySlim {
            call_id: 7,
            success: true,
            message: None,
            engine_index: 1,
        };
        let mut expected = vec![0x95, 0xb0];
        expected.extend_from_slice(b"ControlReplySlim");
        expected.extend_from_slice(&[0x07, 0xc3, 0xc0, 0x01]);
        assert_eq!(encode_frames(&reply), [bytes::Bytes::from(expected)]);
        assert_eq!(
            ControlReplySlim::FIELDS,
            ["call_id", "success", "message", "engine_index"]
        );
    }

    #[test]
    fn a_reply_names_its_outcome() {
        let refused = ControlOutput::FlushCache(FlushCacheReqOutput {
            rid: None,
            http_worker_ipc: None,
            success: false,
        });
        assert_eq!(refused.outcome(), (false, None));
        let started = ControlOutput::Profile(ProfileReqOutput {
            rid: None,
            http_worker_ipc: None,
            success: true,
            message: "Start profiling.".to_owned(),
        });
        assert_eq!(
            started.outcome(),
            (true, Some("Start profiling.".to_owned()))
        );
    }
}
