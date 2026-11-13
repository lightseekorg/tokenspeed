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

//! Inkling2's native JSON answers: how a vocabulary frames them in protocol
//! tokens (`JsonFraming`), and where a prompt may end before one
//! (`MessageBoundary`).

use std::sync::Arc;

use ts_grammar::JsonStringGuard;

/// A vocabulary whose messages are framed by protocol tokens (Inkling2):
/// a structural tag whose last element is a JSON answer framed by
/// `answer_begin` and `answer_end` has its block delimiters matched as those
/// protocol tokens rather than as any text that spells them, and its
/// matchers keep protocol tokens out of the answer's JSON strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonFraming {
    /// The answer block's opener, as structural tags spell it.
    pub answer_begin: String,
    /// The closer of every block, as structural tags spell it.
    pub answer_end: String,
    /// Each block delimiter structural tags spell (the empty one included)
    /// and the protocol tokens it stands for.
    pub delimiters: Vec<(String, Vec<i32>)>,
    /// The guard the answer's matchers keep.
    pub guard: Arc<JsonStringGuard>,
}

impl JsonFraming {
    /// The framing of a vocabulary of `string_steps.len()` tokens:
    /// `text_token` opens an answer's payload, `end_token` closes a
    /// message, `reserved_tokens` may not stand inside a JSON string, and
    /// `string_steps` holds each token's `ts_grammar::string_steps_of`.
    pub fn new(
        answer_begin: String,
        answer_end: String,
        delimiters: Vec<(String, Vec<i32>)>,
        text_token: i32,
        end_token: i32,
        reserved_tokens: &[i32],
        string_steps: Vec<u8>,
    ) -> Result<Self, String> {
        let guard = JsonStringGuard::new(text_token, end_token, reserved_tokens, string_steps)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            answer_begin,
            answer_end,
            delimiters,
            guard: Arc::new(guard),
        })
    }

    /// For a structural tag that ends in a framed JSON answer: the tag with
    /// its block delimiters lowered to protocol tokens, and whether the
    /// answer starts in its payload (an empty opener: the prompt opened it).
    /// `None` for any other tag.
    pub fn frame(&self, tag: &str) -> Result<Option<(String, bool)>, String> {
        let mut tag: serde_json::Value =
            serde_json::from_str(tag).map_err(|error| format!("JSONDecodeError: {error}"))?;
        let Some(format) = tag.get_mut("format") else {
            return Ok(None);
        };
        let Some(answer) = format
            .get("elements")
            .and_then(|elements| elements.as_array())
            .and_then(|elements| elements.last())
        else {
            return Ok(None);
        };
        let field = |value: &serde_json::Value, name: &str| {
            value.get(name).and_then(|v| v.as_str()).map(str::to_owned)
        };
        let begin = field(answer, "begin");
        let content = answer.get("content");
        let framed = field(answer, "type").as_deref() == Some("tag")
            && (begin.as_deref() == Some(self.answer_begin.as_str())
                || begin.as_deref() == Some(""))
            && field(answer, "end").as_deref() == Some(self.answer_end.as_str())
            && content.and_then(|c| field(c, "type")).as_deref() == Some("json_schema")
            && content
                .and_then(|c| field(c, "style"))
                .is_none_or(|style| style == "json");
        if !framed {
            return Ok(None);
        }
        let starts_in_json = begin.as_deref() == Some("");
        self.lower(format);
        Ok(Some((tag.to_string(), starts_in_json)))
    }

    /// Replace each block whose delimiters are protocol spellings by the
    /// sequence of those tokens around its content.
    fn lower(&self, format: &mut serde_json::Value) {
        let tokens = |spelling: Option<&serde_json::Value>| {
            let spelling = spelling?.as_str()?;
            self.delimiters
                .iter()
                .find(|(text, _)| text == spelling)
                .map(|(_, ids)| ids.clone())
        };
        let kind = format
            .get("type")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        match kind.as_deref() {
            Some("tag") => {
                let Some(object) = format.as_object() else {
                    return;
                };
                let keys_match = object.len() == 4
                    && ["type", "begin", "content", "end"]
                        .iter()
                        .all(|key| object.contains_key(*key));
                let end_matches =
                    object.get("end").and_then(|v| v.as_str()) == Some(self.answer_end.as_str());
                if !keys_match || !end_matches {
                    return;
                }
                let (Some(begin), Some(end)) =
                    (tokens(object.get("begin")), tokens(object.get("end")))
                else {
                    return;
                };
                let token = |id: i32| serde_json::json!({"type": "token", "token": id});
                let mut elements: Vec<serde_json::Value> = begin.into_iter().map(token).collect();
                elements.push(object["content"].clone());
                elements.extend(end.into_iter().map(token));
                *format = serde_json::json!({"type": "sequence", "elements": elements});
            }
            Some("sequence") => {
                if let Some(elements) = format.get_mut("elements").and_then(|v| v.as_array_mut()) {
                    for element in elements {
                        self.lower(element);
                    }
                }
            }
            Some("star") => {
                if let Some(content) = format.get_mut("content") {
                    self.lower(content);
                }
            }
            _ => {}
        }
    }
}

/// Where a prompt may end before a JSON answer when the model writes every
/// block's header itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageBoundary {
    /// An empty answer header: the reply continues it with the payload
    /// (`continues`).
    pub open_answer: Vec<i32>,
    /// The endings of a closed message: the reply writes its own blocks
    /// (`may_open`).
    pub closed: Vec<Vec<i32>>,
}

impl MessageBoundary {
    /// Whether the reply continues an empty answer header `prompt` ends
    /// with (`continues`) rather than following a closed message or no
    /// prompt (`may_open`); any other ending is the request's error.
    pub fn opens_answer(&self, prompt: &[i32]) -> Result<bool, String> {
        if prompt.ends_with(&self.open_answer) {
            return Ok(true);
        }
        if prompt.is_empty() || self.closed.iter().any(|end| prompt.ends_with(end)) {
            return Ok(false);
        }
        Err("JSON responses require a closed message or an empty assistant text header".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        thread,
        time::{Duration, Instant},
    };

    use super::*;
    use crate::grammar::{GrammarConfig, GrammarKey, Grammars, Lookup};

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

    /// The framing of `ts-grammar`'s framed vocabulary: `"` 4, `\` 8, and
    /// the protocol tokens model 9, text 10, thinking 11, end 12, tool 13.
    fn framing() -> JsonFraming {
        let decoded: [&[u8]; 15] = [
            b"a",
            b"b",
            b"{",
            b"}",
            b"\"",
            b"x",
            b":",
            b",",
            b"\\",
            b"<|message_model|>",
            b"<|content_text|>",
            b"<|content_thinking|>",
            b"<|end_message|>",
            b"<|message_tool|>",
            b"</s>",
        ];
        let steps = decoded
            .iter()
            .map(|bytes| ts_grammar::string_steps_of(bytes))
            .collect();
        JsonFraming {
            answer_begin: "<|message_model|><|content_text|>".to_owned(),
            answer_end: "<|end_message|>".to_owned(),
            delimiters: vec![
                (String::new(), vec![]),
                ("<|message_model|><|content_text|>".to_owned(), vec![9, 10]),
                (
                    "<|message_model|><|content_thinking|>".to_owned(),
                    vec![9, 11],
                ),
                ("<|end_message|>".to_owned(), vec![12]),
            ],
            guard: Arc::new(JsonStringGuard::new(10, 12, &[9, 10, 11, 12, 13], steps).unwrap()),
        }
    }

    const SCHEMA_CONTENT: &str = r#"{"type":"json_schema","json_schema":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false},"style":"json","any_order":false}"#;

    /// A reply that may reason in blocks, then answers in JSON; its answer
    /// opens with `begin`.
    fn reasoning_reply(begin: &str, thinking: bool) -> String {
        let thinking = if thinking {
            r#"{"type":"star","content":{"type":"tag","begin":"<|message_model|><|content_thinking|>","content":{"type":"any_text","excludes":["<|message_tool|>"]},"end":"<|end_message|>"}},"#
        } else {
            ""
        };
        format!(
            r#"{{"type":"structural_tag","format":{{"type":"sequence","elements":[{thinking}{{"type":"tag","begin":"{begin}","content":{SCHEMA_CONTENT},"end":"<|end_message|>"}}]}}}}"#
        )
    }

    #[test]
    fn a_framed_json_answer_matches_its_delimiters_as_protocol_tokens() {
        let framing = framing();
        let token = |id: i32| serde_json::json!({"type": "token", "token": id});
        let content: serde_json::Value = serde_json::from_str(SCHEMA_CONTENT).unwrap();

        let (framed, starts_in_json) = framing
            .frame(&reasoning_reply("<|message_model|><|content_text|>", true))
            .unwrap()
            .unwrap();
        assert!(!starts_in_json);
        let framed: serde_json::Value = serde_json::from_str(&framed).unwrap();
        let elements = &framed["format"]["elements"];
        assert_eq!(
            elements[1],
            serde_json::json!({"type": "sequence", "elements": [token(9), token(10), content, token(12)]})
        );
        let block = &elements[0]["content"]["elements"];
        assert_eq!(block[0], token(9));
        assert_eq!(block[1], token(11));
        assert_eq!(block[2]["type"], "any_text");
        assert_eq!(block[3], token(12));

        // An answer the prompt opened starts in its payload.
        let (framed, starts_in_json) = framing.frame(&reasoning_reply("", false)).unwrap().unwrap();
        assert!(starts_in_json);
        let framed: serde_json::Value = serde_json::from_str(&framed).unwrap();
        assert_eq!(
            framed["format"]["elements"][0],
            serde_json::json!({"type": "sequence", "elements": [content, token(12)]})
        );

        // Other answers keep their tags and get no guard.
        for tag in [
            reasoning_reply("<answer>", true),
            reasoning_reply("<|message_model|><|content_text|>", true)
                .replace(SCHEMA_CONTENT, r#"{"type":"regex","pattern":"a"}"#),
            r#"{"type":"structural_tag","format":{"type":"const_string","value":"x"}}"#.to_owned(),
        ] {
            assert_eq!(framing.frame(&tag).unwrap(), None, "{tag}");
        }
        assert!(framing.frame("{").is_err());
    }

    #[test]
    fn a_framed_answer_s_matchers_keep_protocol_tokens_out_of_strings() {
        let mut grammars = Grammars::new(GrammarConfig {
            tokenizer_info: include_str!(
                "../../ts-grammar/tests/fixtures/framed_tokenizer_info.json"
            )
            .to_owned(),
            any_whitespace: false,
            compile_timeout: Duration::from_secs(30),
            max_retries: 0,
            json_framing: Some(framing()),
        })
        .unwrap();
        let key = GrammarKey {
            kind: "structural_tag",
            source: reasoning_reply("<|message_model|><|content_text|>", true),
        };
        let _ = grammars.lookup(&key);
        let Lookup::Ready(grammar) = compiled(&grammars, &key) else {
            panic!("the framed answer compiles");
        };
        let mut matcher = Grammars::matcher(&grammar).unwrap();
        // The header is the protocol tokens, not text that spells it.
        assert!(!matcher.accept_token(0));
        // {"a":" then a protocol token inside the string.
        for token in [9, 10, 2, 4, 0, 4, 6, 4] {
            assert!(matcher.accept_token(token), "{token}");
        }
        assert!(!matcher.accept_token(13));
        for token in [1, 4, 3, 12, 14] {
            assert!(matcher.accept_token(token), "{token}");
        }
        assert!(matcher.is_terminated());
    }

    #[test]
    fn a_framing_over_another_vocabulary_is_refused() {
        let mut grammars = Grammars::new(GrammarConfig {
            tokenizer_info: include_str!("../../ts-grammar/tests/fixtures/toy_tokenizer_info.json")
                .to_owned(),
            any_whitespace: false,
            compile_timeout: Duration::from_secs(30),
            max_retries: 0,
            json_framing: Some(framing()),
        })
        .unwrap();
        let key = GrammarKey {
            kind: "structural_tag",
            source: reasoning_reply("<|message_model|><|content_text|>", true),
        };
        let _ = grammars.lookup(&key);
        let Lookup::Invalid(invalid) = compiled(&grammars, &key) else {
            panic!("a 15-token framing pairs with a 9-token grammar");
        };
        assert!(
            invalid.message.contains("tokenizer info"),
            "{}",
            invalid.message
        );
    }
}
