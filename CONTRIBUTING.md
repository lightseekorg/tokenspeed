# TokenSpeed project ethos and external contributions

TokenSpeed is intentionally built as a lean, engineering-focused infrastructure project.

In spirit, TokenSpeed's development style is closer to projects such as OpenAI Triton:
a lean and small core team, move fast, high technical ownership, and an extremely high quality bar.

As a result, the project is intentionally selective about external contributions — not out of disregard for community involvement, but because maintaining long-term simplicity, performance, and reliability matters deeply for infrastructure software.

TokenSpeed is built by contributors from different companies and open-source communities under the LightSeek Foundation ecosystem. The mission is simple:

The future of AI should be transparent, collaborative, and inclusive. LightSeek exists to accelerate the development of open, efficient, and trustworthy AI systems—so that progress in AI benefits everyone, not just a few.

## External contributions

These contributions are especially welcome:

- obvious bug fixes
- small and verifiable production-needed features
- performance optimizations that fit the existing codebase style and do not introduce unnecessary complexity
- documentation, tooling, and benchmarking improvements

For larger features or architectural changes, consider starting with an RFC or design discussion before implementation.

## Commit attribution

The `Commit trailers` CI check requires every PR commit author and
`Co-authored-by` email to resolve to a GitHub account. It reports the commit
and email when attribution is missing; matching a display name is not enough.

GitHub bot accounts, whose resolved logins end in `[bot]`, cannot be commit
authors or co-authors. A bot-authored commit can become a bot co-author when
the PR is squash merged. Amend the author to the responsible contributor and
remove bot `Co-authored-by` trailers. Regular user accounts with `bot` in
their names are allowed.

To fix an unlinked email, add and verify the address in GitHub **Settings > Emails**
and rerun the check, or use the exact noreply address shown in those settings.
If you change an address, amend or rebase the affected commits, update any
`Co-authored-by` trailers, and push the updated branch.

`Signed-off-by` is a DCO attestation and remains subject to the DCO check.
GitHub's attribution API resolves authors and co-authors, not sign-off-only
addresses; this check does not require contributors to publish their profile email.

To enforce this policy before merging, configure `Commit trailers` as a
required status check in the repository's branch rules.

## Engineering principles

TokenSpeed values:

- reproducible performance
- correctness and stability
- production-oriented engineering
- maintainable implementations

Open-source infrastructure advances through open collaboration, healthy technical discussion, and shared engineering progress.

Thanks to everyone contributing code, ideas, reviews, benchmarks, and feedback.

— TokenSpeed Team
