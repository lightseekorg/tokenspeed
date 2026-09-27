# Code Review Guidelines

Read [AGENTS.md](AGENTS.md) and any local overrides first. Map changed files to
the relevant TokenSpeed design documents below before reviewing.

## Severity

Prefix **every inline comment** with one of these markers:

| Marker | Severity | Meaning |
| ------ | -------- | ------- |
| 🔴 | **Important** | A bug that should be fixed before merging |
| 🟡 | **Nit** | A minor issue, worth fixing but not blocking |
| 🟣 | **Pre-existing** | A bug in the codebase not introduced by this PR |

For example:

> 🔴 **Important**: The buffer is sized to the capture ladder rather than the
> maximum decode batch. A larger eager batch will write past its capacity.

Explain the triggering condition, observable impact, and a concrete correction.
Verify findings against the surrounding code and callers; avoid speculative
bugs or performance claims without evidence. Check existing review threads and
do not repeat an unresolved finding. After posting inline comments, write a
brief summary with counts for each severity, including zero counts. Distinguish
pre-existing issues from regressions introduced by the PR.

## Focus on

- Production correctness, security vulnerabilities, resource lifetime errors,
  races, deadlocks, and silent failures or swallowed exceptions.
- Explicit execution modes and backend choices, preserved wrapper arguments,
  and validation that rejects unsupported configurations.
- Scheduler state transitions, admission/retraction/recovery, KV ownership,
  cancellation, and ordering of asynchronous forward and transfer work.
- Consistent behavior across eager/CUDA-graph execution, speculation,
  prefill/decode disaggregation, and supported hardware configurations.
- Kernel indexing, shapes, dtypes, strides, padding, numerical accuracy, and
  capability-based selection; vendor libraries must remain behind the kernel
  package boundary described in AGENTS.md.
- Tests that exercise the changed behavior, especially failure paths and
  boundary cases. Identify the missing regression case rather than demanding
  broad coverage without a concrete reason.
- Broken code/documentation references and configuration documentation that
  disagrees with the implementation.

## Domain references

Treat the design documents as the source of truth. Deliberate deviations must
be justified and documented in the same change.

| Changed area | Read |
| ------------ | ---- |
| Event loop and execution coordination | [Event loop](docs/design/event-loop.md): control/data plane separation, centralized feedback, in-flight depth, hooks |
| C++ scheduler | [Scheduler](docs/design/scheduler.md): chunk admission, retraction, engine roles, recovery invariants |
| Cache allocation, prefix reuse, transfer | [Cache concepts](docs/design/cache-concepts.md): logical/physical units, ownership, geometry, layering |
| Attention metadata and decode execution | [Unified decode path](docs/design/unified_path.md): refresh-in-place metadata, padding, buffer capacity, graph mechanics |
| KDA prefill graphs | [KDA prefill subgraphs](docs/design/kda-prefill-subgraphs.md) |
| Kernel registration and backends | [Kernel design](tokenspeed-kernel/README.md) and the affected operation's README |
| CI task declarations and validation | [CI task specs](test/ci/README.md) |
| Public serving configuration | [Server parameters](docs/configuration/server.md) and [parallelism](docs/serving/parallelism.md) |

## Skip

- Formatting-only changes and style comments already enforced by tooling.
- Documentation-only PRs under `docs/**`.
- Dependency version bumps with no code changes.

## Claude workflow setup and operation

[The workflow](.github/workflows/claude-code-review.yml) runs both automatic
reviews and `@claude` replies on the self-managed runner scale set
`org-k8s-runner-cpu`. Grant this repository access to that runner set and install
the [Claude GitHub App](https://github.com/apps/claude) for this repository.
The workflow uses the app's short-lived token for reviews and replies.

As in the SMG workflow, provide `ANTHROPIC_API_KEY` through the runner pod's
environment, typically from a Kubernetes Secret; the workflow passes it to the
Claude action. Do not commit the key or print it in logs. The runner needs Git,
`jq`, `curl`, `tar`, and network access to GitHub and the Anthropic API, plus the
prerequisites for `anthropics/claude-code-action@v1`. The workflow installs the
Linux AMD64 GitHub CLI (`gh`) if it is missing.

Automatic reviews run for same-repository pull requests on `opened`,
`synchronize` (new commits), and `reopened`, including drafts. Fork PRs and runs
triggered by `dependabot[bot]` are skipped. PR events are also skipped when all
changed paths match `*.md`, `docs/**`, or `*.lock`.

Mention `@claude` in a new issue/PR conversation comment or inline PR review
comment to request a reply. Both jobs load the official `pr-review-toolkit`
plugin. This CPU runner reviews source and CI evidence; GPU tests remain in
the repository's existing CI workflows.
