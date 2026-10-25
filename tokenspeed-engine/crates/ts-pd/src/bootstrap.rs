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

//! The bootstrap server: the HTTP rendezvous of a disaggregated deployment's
//! data-source ranks (a Prefill node's for PD, an encode node's for EPD) and
//! the ranks that read from them.
//!
//! * `PUT /route`: a source rank registers its control address under its
//!   attention DP group and TP rank, with its node's parallel sizes.
//! * `GET /route?engine_rank=-1&target_dp_group=-1`: the node's parallel
//!   sizes (and, for a cache-transfer node, its logical cache layout and
//!   per-stage field placement).
//! * `GET /route?engine_rank=R&target_dp_group=G`: one rank's address.
//! * `GET /health`: liveness.
//!
//! The routes and payloads are those of the Python server this replaced.
//! Plain HTTP/1.1, one request per connection.

use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

use serde_json::{json, Value};

use crate::layout::{contract::Contract, route::validate_stage_fields};

/// The largest request the server reads.
const MAX_REQUEST_BYTES: usize = 4 << 20;

/// How long one connection may take to send its request.
const READ_TIMEOUT: Duration = Duration::from_secs(10);

/// How often the accept loop checks for shutdown.
const ACCEPT_POLL: Duration = Duration::from_millis(50);

/// What a source node registers besides its ranks' addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Registry {
    /// Addresses and parallel sizes only (an EPD encode node).
    Addresses,
    /// Also the node's logical cache layout and per-stage field placement,
    /// identical on every rank (a PD Prefill node).
    CacheLayouts,
}

/// One registered source rank.
#[derive(Debug, Clone)]
struct RankAddress {
    ip: Value,
    port: i64,
}

/// What the source ranks registered.
#[derive(Debug, Default)]
struct Table {
    world_size: Option<Value>,
    dp_size: Option<Value>,
    tp_size_per_dp_rank: Option<i64>,
    pp_size: Option<i64>,
    /// DP group -> TP rank within it -> address.
    ranks: HashMap<i64, HashMap<i64, RankAddress>>,
    /// The first registered layout's wire form, and the layout.
    cache_layout: Option<(String, Contract)>,
    cache_fields_by_stage: Option<Vec<Vec<String>>>,
}

/// A request the server refuses, with its status and text.
struct Refusal(u16, String);

fn internal(message: impl Into<String>) -> Refusal {
    Refusal(500, message.into())
}

/// Python's `int(value)` of a JSON integer or numeric string.
fn integer(value: Option<&Value>, name: &str) -> Result<i64, Refusal> {
    let value = value.ok_or_else(|| internal(format!("missing {name}")))?;
    value
        .as_i64()
        .or_else(|| value.as_str().and_then(|text| text.trim().parse().ok()))
        .ok_or_else(|| internal(format!("{name} is not an integer")))
}

impl Table {
    /// Record a cache-transfer node's layout and stage placement
    /// (`MooncakeKVBootstrapServer._ingest_put_extra`): every rank registers
    /// the same.
    fn ingest_cache_layout(&mut self, data: &Value) -> Result<(), Refusal> {
        let wire = data
            .get("cache_layout")
            .and_then(Value::as_str)
            .filter(|wire| wire.is_ascii())
            .ok_or_else(|| internal("CachePD bootstrap layout must be a string"))?;
        let layout = Contract::from_wire(wire.as_bytes())
            .map_err(|_| internal("CachePD bootstrap layout is invalid"))?;
        let stages = data
            .get("cache_fields_by_stage")
            .and_then(Value::as_array)
            .ok_or_else(|| internal("CachePD requires explicit stage field placement"))?
            .iter()
            .map(|stage| {
                stage
                    .as_array()
                    .ok_or_else(|| internal("CachePD requires explicit stage field placement"))?
                    .iter()
                    .map(|field| {
                        field.as_str().map(str::to_owned).ok_or_else(|| {
                            internal(
                                "cache stage placement must cover every logical field exactly once",
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        validate_stage_fields(&layout, &stages).map_err(|error| internal(error.0))?;
        if stages.len() as i64 != integer(data.get("pp_size"), "pp_size")? {
            return Err(internal(
                "CachePD placement does not match pipeline stage count",
            ));
        }
        if self
            .cache_fields_by_stage
            .as_ref()
            .is_some_and(|known| *known != stages)
        {
            return Err(internal(
                "CachePD prefill ranks registered incompatible field placement",
            ));
        }
        if self
            .cache_layout
            .as_ref()
            .is_some_and(|(_, known)| *known != layout)
        {
            return Err(internal(
                "CachePD prefill ranks registered incompatible layouts",
            ));
        }
        if self.cache_layout.is_none() {
            self.cache_layout = Some((wire.to_owned(), layout));
        }
        self.cache_fields_by_stage = Some(stages);
        Ok(())
    }

    /// `PUT /route`: a source rank registers.
    fn register(&mut self, registry: Registry, body: &[u8]) -> Result<(), Refusal> {
        let data: Value =
            serde_json::from_slice(body).map_err(|_| internal("the registration is not JSON"))?;
        let field = |key: &str| {
            data.get(key)
                .cloned()
                .ok_or_else(|| internal(format!("missing {key}")))
        };
        let role = field("role")?;
        let world_size = field("world_size")?;
        let dp_size = field("dp_size")?;
        let ip = field("rank_ip")?;
        let port = integer(data.get("rank_port"), "rank_port")?;
        let engine_rank = integer(data.get("engine_rank"), "engine_rank")?;
        if registry == Registry::CacheLayouts {
            self.ingest_cache_layout(&data)?;
        }
        self.world_size.get_or_insert_with(|| world_size.clone());
        self.dp_size.get_or_insert_with(|| dp_size.clone());
        if let Some(pp_size) = data.get("pp_size") {
            self.pp_size = Some(integer(Some(pp_size), "pp_size")?);
        }
        // With PP the per-DP world spans the stages; the table keys by the
        // dense stage-major rank.
        let world = integer(Some(&world_size), "world_size")?;
        let dp = integer(Some(&dp_size), "dp_size")?;
        if dp == 0 {
            return Err(internal("dp_size must not be zero"));
        }
        let tp_size_per_dp_rank = world.div_euclid(dp);
        self.tp_size_per_dp_rank.get_or_insert(tp_size_per_dp_rank);
        if role.as_str() == Some("Prefill") {
            if tp_size_per_dp_rank == 0 {
                return Err(internal("world_size must not be below dp_size"));
            }
            let dp_group = engine_rank.div_euclid(tp_size_per_dp_rank);
            let tp_rank = engine_rank.rem_euclid(tp_size_per_dp_rank);
            self.ranks
                .entry(dp_group)
                .or_default()
                .insert(tp_rank, RankAddress { ip, port });
            tracing::debug!(engine_rank, port, "a source rank registered");
        }
        Ok(())
    }

    /// `GET /route`: the node's parallel sizes, or one rank's address.
    fn route(&self, registry: Registry, query: &HashMap<String, String>) -> Result<Value, Refusal> {
        let (Some(engine_rank), Some(dp_group)) = (
            query.get("engine_rank").filter(|value| !value.is_empty()),
            query
                .get("target_dp_group")
                .filter(|value| !value.is_empty()),
        ) else {
            return Err(Refusal(
                400,
                "Missing inputs for bootstrap server.".to_owned(),
            ));
        };
        let parse = |value: &str, name: &str| {
            value
                .trim()
                .parse::<i64>()
                .map_err(|_| internal(format!("{name} is not an integer")))
        };
        let engine_rank = parse(engine_rank, "engine_rank")?;
        let dp_group = parse(dp_group, "target_dp_group")?;
        if engine_rank == -1 && dp_group == -1 {
            let mut info = json!({
                "prefill_tp_size": self.world_size.clone().unwrap_or(Value::Null),
                "prefill_dp_size": self.dp_size.clone().unwrap_or(Value::Null),
                "prefill_pp_size": self.pp_size.unwrap_or(1),
            });
            if registry == Registry::CacheLayouts {
                info["cache_layout"] = self
                    .cache_layout
                    .as_ref()
                    .map_or(Value::Null, |(wire, _)| Value::from(wire.as_str()));
                info["cache_fields_by_stage"] = self
                    .cache_fields_by_stage
                    .as_ref()
                    .map_or(Value::Null, |stages| json!(stages));
            }
            return Ok(info);
        }
        let address = self
            .ranks
            .get(&dp_group)
            .and_then(|ranks| ranks.get(&engine_rank))
            .ok_or_else(|| internal("Bootstrap info not Found"))?;
        Ok(json!({"rank_ip": address.ip, "rank_port": address.port}))
    }
}

/// One parsed request.
struct Request {
    method: String,
    path: String,
    query: HashMap<String, String>,
    body: Vec<u8>,
}

/// `%XX` and `+` decoding of a query component.
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let escaped = (bytes[index] == b'%')
            .then(|| bytes.get(index + 1..index + 3))
            .flatten()
            .and_then(|hex| std::str::from_utf8(hex).ok())
            .and_then(|hex| u8::from_str_radix(hex, 16).ok());
        match (escaped, bytes[index]) {
            (Some(byte), _) => {
                out.push(byte);
                index += 3;
                continue;
            }
            (None, b'+') => out.push(b' '),
            (None, byte) => out.push(byte),
        }
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn read_request(stream: &TcpStream) -> Option<Request> {
    let mut reader = BufReader::new(stream.take(MAX_REQUEST_BYTES as u64));
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_owned();
    let target = parts.next()?.to_owned();
    let mut length = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).ok()? == 0 {
            break;
        }
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                length = value.trim().parse().ok()?;
            }
        }
    }
    if length > MAX_REQUEST_BYTES {
        return None;
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).ok()?;
    let (path, query) = target.split_once('?').unwrap_or((target.as_str(), ""));
    let query = query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            (percent_decode(key), percent_decode(value))
        })
        .collect();
    Some(Request {
        method,
        path: path.to_owned(),
        query,
        body,
    })
}

fn reply(mut stream: &TcpStream, status: u16, content_type: &str, body: &str) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "Internal Server Error",
    };
    let _ = write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.flush();
}

fn serve(stream: TcpStream, registry: Registry, table: &Mutex<Table>) {
    let _ = stream.set_read_timeout(Some(READ_TIMEOUT));
    let Some(request) = read_request(&stream) else {
        reply(&stream, 400, "text/plain", "Bad Request");
        return;
    };
    let answer = match (request.path.as_str(), request.method.as_str()) {
        ("/health", "GET") => Ok(None),
        ("/route", "PUT") => table
            .lock()
            .expect("the bootstrap table lock is poisoned")
            .register(registry, &request.body)
            .map(|()| None),
        ("/route", "GET") => table
            .lock()
            .expect("the bootstrap table lock is poisoned")
            .route(registry, &request.query)
            .map(Some),
        ("/route" | "/health", _) => Err(Refusal(405, "Method not allowed".to_owned())),
        _ => Err(Refusal(404, "Not Found".to_owned())),
    };
    match answer {
        Ok(None) => reply(&stream, 200, "text/plain", "OK"),
        Ok(Some(body)) => reply(&stream, 200, "application/json", &body.to_string()),
        Err(Refusal(status, message)) => {
            if status == 500 {
                tracing::error!(
                    path = request.path,
                    message,
                    "the bootstrap server refused a request"
                );
            }
            reply(&stream, status, "text/plain", &message);
        }
    }
}

/// Bind `port` on every interface: IPv6 and IPv4 where the host has IPv6,
/// else IPv4.
fn bind(port: u16) -> std::io::Result<TcpListener> {
    TcpListener::bind(SocketAddr::from(([0u16; 8], port)))
        .or_else(|_| TcpListener::bind(SocketAddr::from(([0u8; 4], port))))
}

/// A running bootstrap server.
pub struct BootstrapServer {
    port: u16,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl BootstrapServer {
    /// Bind `port` (0 picks one) and serve `registry` until closed.
    pub fn start(port: u16, registry: Registry) -> std::io::Result<Self> {
        let listener = bind(port)?;
        let port = listener.local_addr()?.port();
        listener.set_nonblocking(true)?;
        let stop = Arc::new(AtomicBool::new(false));
        let table = Arc::new(Mutex::new(Table::default()));
        let thread = {
            let stop = stop.clone();
            thread::Builder::new()
                .name("ts-bootstrap".to_owned())
                .spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        match listener.accept() {
                            Ok((stream, _)) => {
                                let table = table.clone();
                                let _ = stream.set_nonblocking(false);
                                let spawned = thread::Builder::new()
                                    .name("ts-bootstrap-conn".to_owned())
                                    .spawn(move || serve(stream, registry, &table));
                                if let Err(error) = spawned {
                                    tracing::error!(%error, "cannot serve a bootstrap connection");
                                }
                            }
                            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                                thread::sleep(ACCEPT_POLL);
                            }
                            Err(error) => {
                                tracing::warn!(%error, "the bootstrap server failed to accept");
                                thread::sleep(ACCEPT_POLL);
                            }
                        }
                    }
                })?
        };
        tracing::info!(port, ?registry, "bootstrap server up");
        Ok(Self {
            port,
            stop,
            thread: Some(thread),
        })
    }

    /// The bound port.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Stop serving and release the port; idempotent.
    pub fn close(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for BootstrapServer {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http;

    const TIMEOUT: Duration = Duration::from_secs(5);

    fn url(server: &BootstrapServer, target: &str) -> String {
        format!("http://127.0.0.1:{}{target}", server.port())
    }

    fn registration(engine_rank: i64, port: i64) -> Value {
        json!({
            "role": "Prefill",
            "world_size": 4,
            "dp_size": 2,
            "rank_ip": "10.0.0.1",
            "rank_port": port,
            "engine_rank": engine_rank,
        })
    }

    #[test]
    fn source_ranks_register_and_are_looked_up() {
        let mut server = BootstrapServer::start(0, Registry::Addresses).unwrap();
        assert!(http::healthy(&url(&server, "/health"), TIMEOUT, TIMEOUT));
        for (rank, port) in [(0, 7000), (3, 7003)] {
            http::put_json(&url(&server, "/route"), &registration(rank, port), TIMEOUT).unwrap();
        }
        let info: Value = serde_json::from_str(
            &http::get_text(
                &url(&server, "/route?engine_rank=-1&target_dp_group=-1"),
                TIMEOUT,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            info,
            json!({"prefill_tp_size": 4, "prefill_dp_size": 2, "prefill_pp_size": 1})
        );
        // Rank 3 is TP rank 1 of DP group 1.
        let rank: Value = serde_json::from_str(
            &http::get_text(
                &url(&server, "/route?engine_rank=1&target_dp_group=1"),
                TIMEOUT,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(rank, json!({"rank_ip": "10.0.0.1", "rank_port": 7003}));
        let missing = http::get_text(
            &url(&server, "/route?engine_rank=1&target_dp_group=0"),
            TIMEOUT,
        );
        assert!(missing.unwrap_err().starts_with("500"));
        let incomplete = http::get_text(&url(&server, "/route?engine_rank=1"), TIMEOUT);
        assert!(incomplete.unwrap_err().starts_with("400"));
        server.close();
        assert!(!http::healthy(&url(&server, "/health"), TIMEOUT, TIMEOUT));
    }

    fn layout(page_count: i64) -> String {
        json!({
            "plan": {
                "prefix_granularity": 16,
                "lcm_block_bytes": 64,
                "num_lcm_blocks": page_count - 1,
                "groups": [{"group_id": "kv", "cache_blocks_per_lcm_block": 1,
                            "page_count": page_count}],
                "planes": [{"plane_id": "p0", "bytes_per_lcm_block": 64, "arena_offset_bytes": 0}],
                "fields": [
                    {"group_id": "kv", "field_id": "layer.0.kv", "plane_id": "p0", "shape": [16],
                     "dtype": "bfloat16", "field_offset_bytes": 0, "page_stride_bytes": 64},
                    {"group_id": "kv", "field_id": "layer.1.kv", "plane_id": "p0", "shape": [16],
                     "dtype": "bfloat16", "field_offset_bytes": 32, "page_stride_bytes": 64}
                ],
                "resident_block_bytes": null
            },
            "group_specs": [{"group_id": "kv", "attn_kind": "full_history", "rows_per_page": 16,
                             "entry_stride_tokens": 1, "sliding_window_tokens": null,
                             "replayable": false, "transfer_policy": "full_suffix",
                             "shard_count": 1}],
            "transfer_schema": {"fields": []},
            "field_addresses": null
        })
        .to_string()
    }

    fn cache_registration(engine_rank: i64, layout: &str, stages: Value) -> Value {
        let mut data = registration(engine_rank, 7000 + engine_rank);
        data["pp_size"] = json!(stages.as_array().unwrap().len());
        data["cache_layout"] = json!(layout);
        data["cache_fields_by_stage"] = stages;
        data
    }

    #[test]
    fn a_cache_transfer_node_registers_one_layout_and_placement() {
        let server = BootstrapServer::start(0, Registry::CacheLayouts).unwrap();
        let route = url(&server, "/route");
        let stages = json!([["layer.0.kv"], ["layer.1.kv"]]);
        http::put_json(
            &route,
            &cache_registration(0, &layout(8), stages.clone()),
            TIMEOUT,
        )
        .unwrap();
        // Another layout, a placement missing a field, or one whose stage
        // count is not the node's: refused, and nothing recorded.
        for refused in [
            cache_registration(1, &layout(9), stages.clone()),
            cache_registration(1, &layout(8), json!([["layer.0.kv"]])),
            cache_registration(1, &layout(8), json!([["layer.0.kv", "layer.1.kv"]])),
        ] {
            let error = http::put_json(&route, &refused, TIMEOUT).unwrap_err();
            assert!(error.starts_with("500"), "{error}");
        }
        let mut wrong_pp = cache_registration(1, &layout(8), stages.clone());
        wrong_pp["pp_size"] = json!(1);
        assert!(http::put_json(&route, &wrong_pp, TIMEOUT).is_err());
        let lookup = url(&server, "/route?engine_rank=1&target_dp_group=0");
        assert!(http::get_text(&lookup, TIMEOUT).is_err());

        http::put_json(
            &route,
            &cache_registration(1, &layout(8), stages.clone()),
            TIMEOUT,
        )
        .unwrap();
        let info: Value = serde_json::from_str(
            &http::get_text(
                &url(&server, "/route?engine_rank=-1&target_dp_group=-1"),
                TIMEOUT,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(info["prefill_pp_size"], json!(2));
        assert_eq!(info["cache_fields_by_stage"], stages);
        let served =
            Contract::from_wire(info["cache_layout"].as_str().unwrap().as_bytes()).unwrap();
        assert_eq!(served, Contract::from_wire(layout(8).as_bytes()).unwrap());
        assert!(http::get_text(&lookup, TIMEOUT).is_ok());
    }

    #[test]
    fn a_taken_port_is_refused() {
        let server = BootstrapServer::start(0, Registry::Addresses).unwrap();
        assert!(BootstrapServer::start(server.port(), Registry::Addresses).is_err());
    }
}
