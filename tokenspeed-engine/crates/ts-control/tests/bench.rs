// Sustained microbenchmarks of the control plane, opt-in with
// `cargo test --release -p tokenspeed-control --test bench -- --ignored
// --nocapture`. Numbers are wall-clock on one machine: A/B two checkouts,
// several runs each, and compare within a machine.

use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use ts_control::output::{FinishLog, ForwardResult, OutputProcessor, Role};
use ts_control::request::RequestState;
use ts_scheduler::ForwardBatchView;

const NO_FINISH_LOG: FinishLog = FinishLog {
    enabled: false,
    request_stats: false,
};

/// (rid, input_length, extend_prefix_len, prefill_length)
fn batch(rows: &[(&str, i32, i32, i32)], num_extends: usize) -> ForwardBatchView {
    ForwardBatchView {
        request_ids: rows.iter().map(|r| r.0.to_owned()).collect(),
        request_pool_indices: (1..=rows.len() as i32).collect(),
        input_lengths: rows.iter().map(|r| r.1).collect(),
        prefill_lengths: rows.iter().map(|r| r.3).collect(),
        extend_prefix_lens: rows[..num_extends].iter().map(|r| r.2).collect(),
        extend_replay_lens: vec![0; num_extends],
        num_extends,
    }
}

fn result(tokens: &[i32], lengths: &[i32]) -> ForwardResult {
    ForwardResult {
        output_tokens: tokens.to_vec(),
        output_lengths: lengths.to_vec(),
        output_logprobs: None,
        nan_flags: None,
        next_input_ids: None,
        serving_compiles: None,
        grammar_terminated_at: None,
        input_token_logprobs: None,
        input_top_logprobs: None,
        output_top_logprobs: None,
    }
}

/// `OutputProcessor::commit`'s steady decode cost: one prebuilt batch of
/// never-finishing requests, every token streamed, so the wall time is
/// commit's own work. The batch and result are built once outside the
/// loop; it prints the microseconds per commit.
#[test]
#[ignore = "a sustained load that A/Bs the commit path"]
fn bench_commit_decode() {
    let bs = 256;
    let run = Duration::from_secs(2);
    let mut p = OutputProcessor::new(None, 1, Some(1 << 24), Role::Plain, NO_FINISH_LOG);
    let mut rids: Vec<String> = Vec::with_capacity(bs);
    for i in 0..bs {
        let rid = format!("r{i}");
        let state = RequestState::new(vec![0; 16], 1 << 30, BTreeSet::new(), true, None, false);
        p.register(rid.clone(), state);
        let _ = p
            .commit(
                &batch(&[(rid.as_str(), 16, 0, 16)], 1),
                None,
                &result(&[7], &[1]),
            )
            .unwrap();
        rids.push(rid);
    }
    let rows: Vec<(&str, i32, i32, i32)> =
        rids.iter().map(|rid| (rid.as_str(), 1, 0, 16)).collect();
    let one = batch(&rows, 0);
    let res = result(&vec![9; bs], &vec![1; bs]);
    let start = Instant::now();
    let mut commits = 0usize;
    while start.elapsed() < run {
        let _ = p.commit(&one, None, &res).unwrap();
        commits += 1;
    }
    println!(
        "commit: {:.0} us/commit over {commits} commits",
        start.elapsed().as_secs_f64() * 1e6 / commits as f64
    );
}
