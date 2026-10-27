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

//! Drive a headless TokenSpeed engine with greedy requests.
//!
//! ```text
//! generate <handshake-address> <num-engines> <max-new-tokens> <prompts-file>
//! ```
//!
//! Start the engine (one per attention-DP rank is started for you) with
//! `python -m tokenspeed.runtime.entrypoints.engine --model <m>
//! --data-parallel-rpc-port <port>`. Each line of `<prompts-file>` is a
//! comma-separated list of prompt token ids. All prompts are submitted at
//! once; each result is printed as one JSON line in prompt order.

use std::time::Duration;

use ts_client::{ClientConfig, EngineClient, LoadBalance, Routing};
use ts_wire::{SamplingParams, TokenizedGenerateReqInput};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, handshake, num_engines, max_new_tokens, prompts] = args.as_slice() else {
        return Err(
            "usage: generate <handshake-address> <num-engines> <max-new-tokens> <prompts-file>"
                .into(),
        );
    };
    let prompts: Vec<Vec<i32>> = std::fs::read_to_string(prompts)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.split(',').map(|id| id.trim().parse()).collect())
        .collect::<Result<_, _>>()?;

    let client = EngineClient::connect(ClientConfig {
        handshake_address: handshake.clone(),
        input_address: "tcp://127.0.0.1:0".to_owned(),
        output_address: "tcp://127.0.0.1:0".to_owned(),
        first_engine_index: 0,
        num_engines: num_engines.parse()?,
        routing: Routing::Aggregated(LoadBalance::LeastInFlight),
        // Model loading and CUDA-graph capture happen before the handshake.
        handshake_timeout: Duration::from_secs(1800),
        send_timeout: Duration::from_secs(10),
    })
    .await?;
    eprintln!("connected to {} engine(s)", client.engines().len());

    let max_new_tokens: i64 = max_new_tokens.parse()?;
    let mut streams = Vec::new();
    for (i, input_ids) in prompts.into_iter().enumerate() {
        let sampling_params = SamplingParams {
            max_new_tokens: Some(max_new_tokens),
            temperature: 0.0,
            ..Default::default()
        };
        let request = TokenizedGenerateReqInput {
            input_ids: Some(input_ids),
            stream: true,
            ..TokenizedGenerateReqInput::new(format!("rust-{i}"), sampling_params)
        };
        streams.push(client.generate(request).await?);
    }

    let mut results = Vec::new();
    for mut stream in streams {
        let mut output_ids = Vec::new();
        let mut last = None;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            output_ids.extend_from_slice(&chunk.output_ids);
            last = Some(chunk);
        }
        let last = last.ok_or("stream ended without output")?;
        results.push(format!(
            r#"{{"rid": "{}", "output_ids": {:?}, "finish_reason": "{}", "prompt_tokens": {}, "completion_tokens": {}, "cached_tokens": {}}}"#,
            stream.rid(),
            output_ids,
            last.finish_reason.unwrap_or_default(),
            last.prompt_tokens,
            last.completion_tokens,
            last.cached_tokens,
        ));
    }
    for line in results {
        println!("{line}");
    }
    eprintln!("engine loads: {:?}", client.loads());
    Ok(())
}
