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

//! Startup adapter for the optional prepared L3 source broker.

use parking_lot::Mutex;
use pyo3::prelude::*;
use serde_json::Value;
use std::{collections::HashSet, path::Path, sync::Arc, time::Duration};
use ts_control::peer::{storage as source, storage_servicer, wire::Stamp, PeerSettings};
use ts_scheduler::Scheduler;

pub(super) struct Settings {
    knobs: Value,
    l3: ts_l3::Settings,
    grain: i32,
}
impl Settings {
    pub(super) fn from_python(value: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        if value.is_none() {
            return Ok(None);
        }
        let raw: String = value.getattr("config")?.extract()?;
        Ok(Some(Self {
            knobs: serde_json::from_str(&raw)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
            l3: ts_l3::Settings::from_python(&value.getattr("l3")?)?,
            grain: value.getattr("prefix_granularity")?.extract()?,
        }))
    }

    pub(super) fn start(
        self,
        peer: &PeerSettings,
        members: usize,
        scheduler: Arc<Mutex<Scheduler>>,
        registered: Option<&ts_control::peer::registered::Replica>,
    ) -> Result<storage_servicer::Service, String> {
        let number = |key: &str| {
            self.knobs[key]
                .as_u64()
                .ok_or_else(|| format!("invalid direct L3 {key}"))
        };
        let string = |key: &str| {
            self.knobs[key]
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("invalid direct L3 {key}"))
        };
        let master = string("master")?;
        let tenant = string("tenant")?;
        match &self.l3.store {
            ts_l3::settings::StoreSettings::Mooncake { setup, .. }
                if setup.master_server_address == master && setup.tenant_id == tenant => {}
            _ => {
                return Err(
                    "direct L3 reader and PUT executor must share master and tenant".into(),
                );
            }
        }
        if self.grain <= 0 {
            return Err("direct peer landing needs a positive grain".into());
        }
        if self.l3.layout.groups.iter().any(|g| g.shard_count != 1) && registered.is_none() {
            return Err("sharded direct L3 requires the registered cache layout".into());
        }
        if let Some(layout) = registered {
            if layout.members.len() != members
                || layout.contract.group_specs.len() != self.l3.layout.groups.len()
                || layout
                    .contract
                    .group_specs
                    .iter()
                    .zip(&self.l3.layout.groups)
                    .any(|(spec, group)| spec.shard_count != group.shard_count as i64)
            {
                return Err("direct L3 and registered Peer cache layouts disagree".into());
            }
        }
        let limits = source::Limits {
            max_query_keys: number("max_query_keys")? as usize,
            max_candidates: number("max_candidates")? as usize,
            max_prepare_bytes: number("max_prepare_bytes")?,
            max_source_batch_bytes: number("max_source_batch_bytes")?,
            max_source_batches: number("max_source_batches")? as usize,
        };
        let max_restore =
            i32::try_from(number("max_restore_tokens")?).map_err(|e| e.to_string())?;
        let layouts: Vec<_> = (0..members)
            .map(|rank| {
                let mut layout = self.l3.layout.clone();
                layout.tp_rank = rank;
                layout.owner_rank = rank
                    % layout
                        .groups
                        .iter()
                        .map(|g| g.shard_count)
                        .max()
                        .unwrap_or(1);
                layout
            })
            .collect();
        let namespace = self.l3.namespace.clone();
        let grain = self.grain;
        let local_scheduler = scheduler.clone();
        let local_prefix: storage_servicer::LocalPrefix = Arc::new(move |prompt| {
            local_scheduler
                .lock()
                .local_prefix_tokens_from_hashes(
                    &prompt.tokens,
                    &prompt.prefix_hashes,
                    prompt.max_tokens,
                )
                .map_err(|error| source::Error(error.to_string()))
        });
        let planner: storage_servicer::Planner = Arc::new(move |prompt, weight_version| {
            let prefix = ts_l3::keys::key_prefix(
                &namespace.deployment_id,
                weight_version,
                if namespace.has_draft {
                    weight_version
                } else {
                    ""
                },
            );
            let scheduler = scheduler.lock();
            let mut boundary = prompt.max_tokens.min(max_restore) / grain * grain;
            let mut candidates = Vec::new();
            let mut keys = HashSet::new();
            for _ in 0..limits.max_candidates {
                if boundary <= 0 {
                    break;
                }
                let plan = source::PrefixPlan::from_prompt(
                    &scheduler,
                    prompt,
                    boundary,
                    &prefix,
                    &layouts,
                    limits.max_query_keys,
                )?;
                if let Some(plan) = plan {
                    let extra: HashSet<_> = plan
                        .objects
                        .iter()
                        .map(|o| o.key.clone())
                        .filter(|k| !keys.contains(k))
                        .collect();
                    if keys.len() + extra.len() <= limits.max_query_keys {
                        keys.extend(extra);
                        candidates.push(plan);
                    }
                }
                boundary = if boundary % grain != 0 {
                    boundary / grain * grain
                } else {
                    boundary - grain
                };
            }
            Ok(candidates)
        });
        let name = format!("l3@{}", peer.name);
        let settings = storage_servicer::Settings {
            stamp: Stamp {
                name: name.clone(),
                incarnation: peer.incarnation,
                identity: format!("{}/{}", peer.deployment_id, peer.weight_version),
            },
            weight_version: self.l3.weight_version.clone(),
            members,
            limits,
            max_records: number("max_records")? as usize,
            min_extra_tokens: i32::try_from(number("min_fetch_tokens")?)
                .map_err(|e| e.to_string())?,
            record_lifetime: Duration::from_millis(number("record_lifetime_ms")?),
            read_budget: Duration::from_millis(number("read_budget_ms")?),
        };
        let backend = source::Mooncake {
            client: ts_mooncake::offload::Client::load(Path::new(&string("library")?))
                .map_err(|e| e.to_string())?,
            master,
            tenant,
            timeout: Duration::from_millis(number("rpc_timeout_ms")?),
        };
        let budget =
            source::SourceBudget::new(number("source_budget_bytes")?).map_err(|e| e.to_string())?;
        let service =
            storage_servicer::Service::start(settings, planner, local_prefix, backend, budget)
                .map_err(|e| e.to_string())?;
        tracing::info!(
            endpoint = name,
            max_restore_tokens = max_restore,
            "direct L3 source broker up"
        );
        Ok(service)
    }
}
