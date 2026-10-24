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

//! Grammar-constrained decoding's control half: compiling each
//! request's grammar off the round, the compiled-grammar cache with its
//! timeouts and retention, and the matchers requests run with. The data
//! plane only fills token bitmasks from those matchers (`ts_grammar::table`).

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{mpsc, Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use ts_grammar::{CompiledGrammar, Compiler, GrammarKind, Matcher, TokenizerInfo};
use ts_wire::SamplingParams;

use crate::codec::{Reader, Writer};

/// Tokens a matcher can undo (`MAX_ROLLBACK_TOKENS`).
const MAX_ROLLBACK_TOKENS: i32 = 200;

/// The compiled grammars kept for reuse, in bytes at most.
const RETAINED_LIMIT_BYTES: usize = 1 << 30;

/// Threads xgrammar may use for one compile (its compiler's default).
const COMPILE_THREADS: i32 = 8;

/// Compiles running at once.
const COMPILE_WORKERS: usize = 4;

/// The grammar settings (`--grammar-backend xgrammar`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarConfig {
    /// The vocabulary as xgrammar sees it (`TokenizerInfo.serialize_json`).
    pub tokenizer_info: String,
    /// JSON schemas allow any whitespace (not `--disable-any-whitespace`).
    pub any_whitespace: bool,
    /// How long a request may wait for its grammar
    /// (`--grammar-compile-timeout-secs`).
    pub compile_timeout: Duration,
    /// Timeouts a grammar is retried after before it is given up on
    /// (`--grammar-compile-max-retries`).
    pub max_retries: u32,
}

/// What a request constrains its output to: the kind and its source.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GrammarKey {
    /// `json`, `regex`, `ebnf` or `structural_tag`.
    pub kind: &'static str,
    /// The schema, expression, grammar or tag.
    pub source: String,
}

impl GrammarKey {
    /// The request's grammar, if it has one.
    pub fn of(params: &SamplingParams) -> Option<Self> {
        let (kind, source) = if let Some(schema) = &params.json_schema {
            ("json", schema)
        } else if let Some(regex) = &params.regex {
            ("regex", regex)
        } else if let Some(ebnf) = &params.ebnf {
            ("ebnf", ebnf)
        } else {
            ("structural_tag", params.structural_tag.as_ref()?)
        };
        Some(Self {
            kind,
            source: source.clone(),
        })
    }
}

/// A grammar's state in the cache.
enum Entry {
    /// A worker compiles it; `started_at` once it began.
    Compiling {
        started_at: Option<Instant>,
    },
    Ready(Arc<CompiledGrammar>),
    /// It does not compile, or timed out: until `expires_at`, or for good.
    Invalid {
        message: String,
        expires_at: Option<Instant>,
    },
}

/// What the cache says of a grammar.
#[derive(Clone)]
pub enum Lookup {
    /// Compiled.
    Ready(Arc<CompiledGrammar>),
    /// It cannot be used.
    Invalid(InvalidGrammar),
    /// Still compiling.
    Compiling,
}

/// Why a grammar cannot be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidGrammar {
    /// The reason.
    pub message: String,
    /// No retry can fix it: it does not compile, or it timed out more
    /// often than the retry limit. The request's own error, so a client
    /// error (400) rather than an internal one; a timeout marker that
    /// expires lets a later request compile again.
    pub permanent: bool,
}

/// Every queued request's [`InvalidGrammar`] on this rank, as one rank
/// sends them to the others.
pub(crate) fn encode_invalid(entries: &[Option<InvalidGrammar>]) -> Vec<u8> {
    let mut out = Writer::new();
    for entry in entries {
        out.option(entry.as_ref(), |out, invalid| {
            out.flag(invalid.permanent);
            out.bytes(invalid.message.as_bytes());
        });
    }
    out.finish()
}

/// What [`encode_invalid`] wrote for `count` requests, or `None` if `bytes`
/// is not that.
pub(crate) fn decode_invalid(bytes: &[u8], count: usize) -> Option<Vec<Option<InvalidGrammar>>> {
    let mut reader = Reader::new(bytes);
    let entries = (0..count)
        .map(|_| {
            reader.option(|reader| {
                let permanent = reader.flag()?;
                Some(InvalidGrammar {
                    message: String::from_utf8(reader.bytes()?.to_vec()).ok()?,
                    permanent,
                })
            })
        })
        .collect::<Option<Vec<_>>>()?;
    reader.finish(entries)
}

impl Entry {
    /// What a request sees of an entry.
    fn lookup(&self) -> Lookup {
        match self {
            Entry::Compiling { .. } => Lookup::Compiling,
            Entry::Ready(grammar) => Lookup::Ready(grammar.clone()),
            Entry::Invalid {
                message,
                expires_at,
            } => Lookup::Invalid(InvalidGrammar {
                message: message.clone(),
                permanent: expires_at.is_none(),
            }),
        }
    }
}

#[derive(Default)]
struct Shared {
    cache: HashMap<GrammarKey, Entry>,
    /// Timeouts per grammar since its last successful compile.
    timeout_history: HashMap<GrammarKey, u32>,
    /// Grammars a worker compiled into `cache`; dropped from it once no
    /// queued request waits for them.
    compiled_keys: HashSet<GrammarKey>,
}

/// The compiled-grammar cache and its compile workers
/// (`XGrammarGrammarBackend`).
pub struct Grammars {
    config: GrammarConfig,
    shared: Arc<Mutex<Shared>>,
    jobs: mpsc::Sender<GrammarKey>,
    /// Compiled grammars requests were admitted with, least recently used
    /// first, and their sizes. Changed only on the control thread, in the
    /// order every replica rank sees, so every rank keeps the same ones.
    retained: VecDeque<GrammarKey>,
    retained_grammars: HashMap<GrammarKey, (Arc<CompiledGrammar>, usize)>,
    retained_bytes: usize,
    /// The bound on `retained_bytes` (`RETAINED_LIMIT_BYTES`).
    retained_limit_bytes: usize,
}

/// xgrammar's source for a grammar, by kind.
fn compile(
    compiler: &Compiler,
    key: &GrammarKey,
    any_whitespace: bool,
) -> Result<CompiledGrammar, String> {
    let failed = |error: ts_grammar::Error| format!("RuntimeError: {error}");
    match key.kind {
        "json" => {
            let schema = if key.source == "$$ANY$$" {
                r#"{"type": "object"}"#
            } else {
                &key.source
            };
            compiler
                .compile(GrammarKind::JsonSchema, schema, any_whitespace)
                .map_err(failed)
        }
        "regex" => compiler
            .compile(GrammarKind::Regex, &key.source, any_whitespace)
            .map_err(failed),
        "ebnf" => compiler
            .compile(GrammarKind::Ebnf, &key.source, any_whitespace)
            .map_err(failed),
        "structural_tag" => {
            let tag = structural_tag(&key.source)?;
            compiler
                .compile(GrammarKind::StructuralTag, &tag, any_whitespace)
                .map_err(failed)
        }
        other => Err(format!("ValueError: Invalid key_type: {other}")),
    }
}

/// A structural tag in xgrammar's JSON form: as given when it names its
/// `format`, else the legacy `structures` / `triggers` form converted as
/// xgrammar's `StructuralTag.from_legacy_structural_tag` does.
fn structural_tag(source: &str) -> Result<String, String> {
    let tag: serde_json::Value =
        serde_json::from_str(source).map_err(|error| format!("JSONDecodeError: {error}"))?;
    if tag.get("format").is_some() {
        return Ok(source.to_owned());
    }
    let missing = |field: &str| format!("KeyError: '{field}'");
    let structures = tag
        .get("structures")
        .and_then(|s| s.as_array())
        .ok_or_else(|| missing("structures"))?;
    let triggers = tag.get("triggers").ok_or_else(|| missing("triggers"))?;
    let tags = structures
        .iter()
        .map(|structure| {
            let field = |name: &str| structure.get(name).cloned().ok_or_else(|| missing(name));
            Ok(serde_json::json!({
                "type": "tag",
                "begin": field("begin")?,
                "content": {
                    "type": "json_schema",
                    "json_schema": field("schema")?,
                    "style": "json",
                    "any_order": false,
                    "max_whitespace_cnt": null,
                    "excludes": [],
                },
                "end": field("end")?,
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(serde_json::json!({
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": triggers,
            "tags": tags,
            "at_least_one": false,
            "stop_after_first": false,
            "excludes": [],
        },
    })
    .to_string())
}

impl Grammars {
    /// The cache, with its workers started.
    pub fn new(config: GrammarConfig) -> Result<Self, String> {
        let info = TokenizerInfo::from_json(&config.tokenizer_info).map_err(|e| e.to_string())?;
        let compiler = Arc::new(Compiler::new(&info, COMPILE_THREADS).map_err(|e| e.to_string())?);
        let shared = Arc::new(Mutex::new(Shared::default()));
        let (jobs, queue) = mpsc::channel::<GrammarKey>();
        let queue = Arc::new(Mutex::new(queue));
        for worker in 0..COMPILE_WORKERS {
            let (compiler, shared, queue) = (compiler.clone(), shared.clone(), queue.clone());
            let any_whitespace = config.any_whitespace;
            thread::Builder::new()
                .name(format!("ts-grammar-{worker}"))
                .spawn(move || loop {
                    let Ok(key) = queue.lock().expect("compile queue lock").recv() else {
                        return;
                    };
                    if let Some(Entry::Compiling { started_at }) = shared
                        .lock()
                        .expect("grammar cache lock")
                        .cache
                        .get_mut(&key)
                    {
                        *started_at = Some(Instant::now());
                    }
                    let compiled = compile(&compiler, &key, any_whitespace);
                    let mut state = shared.lock().expect("grammar cache lock");
                    // A timeout may have replaced the entry meanwhile; a
                    // finished compile still wins, as `record_timeout`
                    // leaves a valid grammar be.
                    match compiled {
                        Ok(grammar) => {
                            state.timeout_history.remove(&key);
                            state.compiled_keys.insert(key.clone());
                            state.cache.insert(key, Entry::Ready(Arc::new(grammar)));
                        }
                        Err(message) => {
                            tracing::warn!(kind = key.kind, %message, "grammar does not compile");
                            state.cache.insert(
                                key,
                                Entry::Invalid {
                                    message,
                                    expires_at: None,
                                },
                            );
                        }
                    }
                })
                .map_err(|error| error.to_string())?;
        }
        Ok(Self {
            config,
            shared,
            jobs,
            retained: VecDeque::new(),
            retained_grammars: HashMap::new(),
            retained_bytes: 0,
            retained_limit_bytes: RETAINED_LIMIT_BYTES,
        })
    }

    /// How long a request may wait for its grammar.
    pub fn compile_timeout(&self) -> Duration {
        self.config.compile_timeout
    }

    /// The grammar for `key`, starting its compile on a miss; whether it
    /// was a cache hit.
    pub fn lookup(&mut self, key: &GrammarKey) -> (Lookup, bool) {
        if let Some((grammar, _)) = self.retained_grammars.get(key) {
            let grammar = grammar.clone();
            self.retained.retain(|retained| retained != key);
            self.retained.push_back(key.clone());
            return (Lookup::Ready(grammar), true);
        }
        let mut state = self.shared.lock().expect("grammar cache lock");
        if let Some(Entry::Invalid {
            expires_at: Some(expires_at),
            ..
        }) = state.cache.get(key)
        {
            if Instant::now() >= *expires_at {
                state.cache.remove(key);
            }
        }
        match state.cache.get(key) {
            Some(entry @ (Entry::Ready(_) | Entry::Invalid { .. })) => (entry.lookup(), true),
            Some(Entry::Compiling { .. }) => (Lookup::Compiling, false),
            None => {
                state
                    .cache
                    .insert(key.clone(), Entry::Compiling { started_at: None });
                drop(state);
                let _ = self.jobs.send(key.clone());
                (Lookup::Compiling, false)
            }
        }
    }

    /// The grammar's state now, without starting anything.
    pub fn status(&self, key: &GrammarKey) -> Lookup {
        if let Some((grammar, _)) = self.retained_grammars.get(key) {
            return Lookup::Ready(grammar.clone());
        }
        match self
            .shared
            .lock()
            .expect("grammar cache lock")
            .cache
            .get(key)
        {
            Some(entry) => entry.lookup(),
            None => Lookup::Compiling,
        }
    }

    /// When a worker began compiling `key`, if one did.
    pub fn started_at(&self, key: &GrammarKey) -> Option<Instant> {
        match self
            .shared
            .lock()
            .expect("grammar cache lock")
            .cache
            .get(key)
        {
            Some(Entry::Compiling { started_at }) => *started_at,
            _ => None,
        }
    }

    /// A request timed out waiting for `key`:
    /// later requests fail fast for one timeout, after which it is retried,
    /// up to the retry limit; then it is given up on for good. A grammar
    /// that compiled meanwhile is left be.
    pub fn record_timeout(&mut self, key: &GrammarKey, message: &str) {
        if self.retained_grammars.contains_key(key) {
            return;
        }
        let mut state = self.shared.lock().expect("grammar cache lock");
        if matches!(state.cache.get(key), Some(Entry::Ready(_))) {
            return;
        }
        let count = {
            let count = state.timeout_history.entry(key.clone()).or_insert(0);
            *count += 1;
            *count
        };
        let (message, expires_at) = if count > self.config.max_retries {
            (
                format!("{message} (gave up after {} retries)", count - 1),
                None,
            )
        } else {
            (
                message.to_owned(),
                Some(Instant::now() + self.config.compile_timeout),
            )
        };
        state.cache.insert(
            key.clone(),
            Entry::Invalid {
                message,
                expires_at,
            },
        );
    }

    /// Keep the grammars requests were just admitted with, most recently
    /// used last, within the byte bound; drop compiled grammars no request
    /// in `waiting` still needs.
    pub fn retain(
        &mut self,
        admitted: &[(GrammarKey, Arc<CompiledGrammar>)],
        waiting: &HashSet<GrammarKey>,
    ) {
        for (key, grammar) in admitted {
            if self.retained_grammars.contains_key(key) {
                self.retained.retain(|retained| retained != key);
                self.retained.push_back(key.clone());
                continue;
            }
            let per_char = if key.source.is_ascii() { 1 } else { 4 };
            let size = key.source.len() * per_char + grammar.memory_size_bytes();
            if size > self.retained_limit_bytes {
                continue;
            }
            self.retained.push_back(key.clone());
            self.retained_grammars
                .insert(key.clone(), (grammar.clone(), size));
            self.retained_bytes += size;
        }
        while self.retained_bytes > self.retained_limit_bytes {
            let Some(key) = self.retained.pop_front() else {
                break;
            };
            if let Some((_, size)) = self.retained_grammars.remove(&key) {
                self.retained_bytes -= size;
            }
        }
        let mut state = self.shared.lock().expect("grammar cache lock");
        let done: Vec<GrammarKey> = state
            .compiled_keys
            .iter()
            .filter(|key| !waiting.contains(*key))
            .cloned()
            .collect();
        for key in done {
            state.compiled_keys.remove(&key);
            if matches!(state.cache.get(&key), Some(Entry::Ready(_))) {
                state.cache.remove(&key);
            }
        }
    }

    /// Whether any compiled grammar awaits `retain`'s cleanup.
    pub fn has_compiled(&self) -> bool {
        !self
            .shared
            .lock()
            .expect("grammar cache lock")
            .compiled_keys
            .is_empty()
    }

    /// A fresh matcher for a request.
    pub fn matcher(grammar: &CompiledGrammar) -> Result<Matcher, String> {
        grammar
            .matcher(MAX_ROLLBACK_TOKENS)
            .map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cache over the nine-token vocabulary of `ts-grammar`'s tests.
    fn grammars(compile_timeout: Duration, max_retries: u32) -> Grammars {
        Grammars::new(GrammarConfig {
            tokenizer_info: include_str!("../../ts-grammar/tests/fixtures/toy_tokenizer_info.json")
                .to_owned(),
            any_whitespace: true,
            compile_timeout,
            max_retries,
        })
        .unwrap()
    }

    fn regex(source: &str) -> GrammarKey {
        GrammarKey {
            kind: "regex",
            source: source.to_owned(),
        }
    }

    /// `key`'s state once its compile finished.
    fn compiled(grammars: &Grammars, key: &GrammarKey) -> Lookup {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match grammars.status(key) {
                Lookup::Compiling if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(1));
                }
                lookup => return lookup,
            }
        }
    }

    fn invalid(lookup: Lookup) -> Option<InvalidGrammar> {
        match lookup {
            Lookup::Invalid(invalid) => Some(invalid),
            _ => None,
        }
    }

    #[test]
    fn invalid_grammars_survive_the_replica_exchange() {
        let entries = vec![
            None,
            Some(InvalidGrammar {
                message: "timed out".to_owned(),
                permanent: false,
            }),
            Some(InvalidGrammar {
                message: "does not compile: ü".to_owned(),
                permanent: true,
            }),
        ];
        let bytes = encode_invalid(&entries);
        assert_eq!(decode_invalid(&bytes, 3), Some(entries));
        assert_eq!(decode_invalid(&bytes[..bytes.len() - 1], 3), None);
        assert_eq!(decode_invalid(&bytes, 2), None);
    }

    #[test]
    fn a_grammar_that_does_not_compile_stays_invalid() {
        let mut grammars = grammars(Duration::from_millis(20), 2);
        let key = regex("(");
        assert!(matches!(grammars.lookup(&key), (Lookup::Compiling, false)));
        assert!(invalid(compiled(&grammars, &key)).unwrap().permanent);
        thread::sleep(Duration::from_millis(40));
        let (lookup, hit) = grammars.lookup(&key);
        assert!(hit && invalid(lookup).unwrap().permanent);
    }

    #[test]
    fn a_timed_out_grammar_fails_fast_for_one_timeout_then_compiles_again() {
        let mut grammars = grammars(Duration::from_millis(50), 2);
        let key = regex("a");
        grammars.record_timeout(&key, "timed out");
        let (lookup, hit) = grammars.lookup(&key);
        assert!(hit);
        assert_eq!(
            invalid(lookup),
            Some(InvalidGrammar {
                message: "timed out".to_owned(),
                permanent: false,
            })
        );
        thread::sleep(Duration::from_millis(60));
        assert!(matches!(grammars.lookup(&key), (Lookup::Compiling, false)));
        assert!(matches!(compiled(&grammars, &key), Lookup::Ready(_)));
    }

    #[test]
    fn repeated_timeouts_give_a_grammar_up_for_good() {
        let mut grammars = grammars(Duration::from_millis(20), 1);
        let key = regex("a");
        grammars.record_timeout(&key, "timed out");
        grammars.record_timeout(&key, "timed out");
        thread::sleep(Duration::from_millis(30));
        let given_up = invalid(grammars.lookup(&key).0).unwrap();
        assert!(given_up.permanent);
        assert!(
            given_up.message.contains("gave up after 1 retries"),
            "{given_up:?}"
        );
    }

    #[test]
    fn a_compile_clears_the_timeout_history_and_wins_over_a_late_timeout() {
        let mut grammars = grammars(Duration::from_millis(20), 1);
        let key = regex("a");
        grammars.record_timeout(&key, "timed out");
        thread::sleep(Duration::from_millis(30));
        grammars.lookup(&key);
        assert!(matches!(compiled(&grammars, &key), Lookup::Ready(_)));
        // A timeout the compile finished before changes nothing.
        grammars.record_timeout(&key, "timed out");
        assert!(matches!(grammars.status(&key), Lookup::Ready(_)));
        // Once no request waits for it, the cache lets it go; a later
        // timeout counts from zero again, so it is retried.
        grammars.retain(&[], &HashSet::new());
        grammars.record_timeout(&key, "timed out");
        assert!(!invalid(grammars.status(&key)).unwrap().permanent);
    }

    #[test]
    fn a_compiled_grammar_stays_while_its_request_waits() {
        let mut grammars = grammars(Duration::from_secs(30), 2);
        let key = regex("a");
        grammars.lookup(&key);
        assert!(matches!(compiled(&grammars, &key), Lookup::Ready(_)));
        let waiting: HashSet<GrammarKey> = [key.clone()].into();
        grammars.retain(&[], &waiting);
        assert!(grammars.has_compiled());
        assert!(matches!(grammars.status(&key), Lookup::Ready(_)));
        grammars.retain(&[], &HashSet::new());
        assert!(!grammars.has_compiled());
        assert!(matches!(grammars.status(&key), Lookup::Compiling));
    }

    #[test]
    fn admitted_grammars_are_kept_most_recently_used_within_the_bound() {
        let mut grammars = grammars(Duration::from_secs(30), 2);
        let keys: Vec<GrammarKey> = ["a", "b", "x"].into_iter().map(regex).collect();
        let mut admitted = Vec::new();
        for key in &keys {
            grammars.lookup(key);
            let Lookup::Ready(grammar) = compiled(&grammars, key) else {
                panic!("{key:?} did not compile");
            };
            admitted.push((key.clone(), grammar));
        }
        // Room for two: the size counts the source and the compiled grammar.
        let size = |(key, grammar): &(GrammarKey, Arc<CompiledGrammar>)| {
            key.source.len() + grammar.memory_size_bytes()
        };
        grammars.retained_limit_bytes = size(&admitted[0]) + size(&admitted[1]);
        grammars.retain(&admitted[..2], &HashSet::new());
        // A hit on the first makes the second the least recently used.
        assert!(matches!(
            grammars.lookup(&keys[0]),
            (Lookup::Ready(_), true)
        ));
        grammars.retain(&admitted[2..], &HashSet::new());
        assert!(matches!(grammars.status(&keys[0]), Lookup::Ready(_)));
        assert!(matches!(grammars.status(&keys[1]), Lookup::Compiling));
        assert!(matches!(grammars.status(&keys[2]), Lookup::Ready(_)));
        assert!(grammars.retained_bytes <= grammars.retained_limit_bytes);
    }

    #[test]
    fn a_legacy_structural_tag_becomes_triggered_tags() {
        let legacy = r#"{"structures": [{"begin": "<f>", "schema": {"type": "object"}, "end": "</f>"}], "triggers": ["<f"]}"#;
        let tag: serde_json::Value =
            serde_json::from_str(&structural_tag(legacy).unwrap()).unwrap();
        // The JSON `StructuralTag.from_legacy_structural_tag` serializes.
        let expected: serde_json::Value = serde_json::from_str(
            r#"{"type":"structural_tag","format":{"type":"triggered_tags","triggers":["<f"],"tags":[{"type":"tag","begin":"<f>","content":{"type":"json_schema","json_schema":{"type":"object"},"style":"json","any_order":false,"max_whitespace_cnt":null,"excludes":[]},"end":"</f>"}],"at_least_one":false,"stop_after_first":false,"excludes":[]}}"#,
        )
        .unwrap();
        assert_eq!(tag, expected);
        let modern =
            r#"{"type": "structural_tag", "format": {"type": "const_string", "value": "x"}}"#;
        assert_eq!(structural_tag(modern).unwrap(), modern);
        assert!(structural_tag("{")
            .unwrap_err()
            .starts_with("JSONDecodeError"));
    }
}
