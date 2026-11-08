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

module.exports = async function cleanupBranches({ github, core, context, dryRun, now }) {
  const { owner, repo } = context.repo;
  const { data: repository } = await github.rest.repos.get({ owner, repo });
  if (context.ref !== `refs/heads/${repository.default_branch}`) {
    throw new Error('Branch cleanup must run from the default branch.');
  }
  const cutoff = now - 7 * 24 * 60 * 60 * 1000;
  const branches = await github.paginate(github.rest.repos.listBranches, {
    owner, repo, per_page: 100,
  });
  let eligible = 0;
  let deleted = 0;
  let failures = 0;

  for (const branch of branches) {
    if (branch.name === repository.default_branch || branch.protected) continue;

    // Query by branch name, not commit association: unrelated branches can
    // share a commit. Only PR heads from this repository count as its history.
    const pulls = (await github.paginate(github.rest.pulls.list, {
      owner, repo, head: `${owner}:${branch.name}`, state: 'all', per_page: 100,
    })).filter(pr => pr.head.repo?.id === repository.id && pr.head.ref === branch.name);
    if (pulls.some(pr => pr.state === 'open')) continue;

    // A branch may also be the base of an open (including stacked) PR.
    const openBases = await github.paginate(github.rest.pulls.list, {
      owner, repo, base: branch.name, state: 'open', per_page: 100,
    });
    if (openBases.length) continue;

    // Recheck the ref after the PR queries. GitHub's delete-ref API does not
    // support an expected SHA, so this is a best-effort concurrent-push guard.
    let current;
    try {
      ({ data: current } = await github.rest.repos.getBranch({
        owner, repo, branch: branch.name,
      }));
    } catch (error) {
      if (error.status === 404) continue; // Already deleted since enumeration.
      throw error;
    }
    if (current.protected || current.commit.sha !== branch.commit.sha) continue;

    let reason;
    if (pulls.some(pr => pr.state === 'closed')) {
      reason = 'all associated PRs are closed';
    } else {
      const committedAt = Date.parse(current.commit.commit.committer?.date);
      if (!Number.isFinite(committedAt) || committedAt >= cutoff) continue;
      reason = 'no PR history and tip commit is older than seven days';
    }

    eligible++;
    if (dryRun) {
      core.info(`Would delete ${branch.name}: ${reason}`);
    } else {
      try {
        await github.rest.git.deleteRef({ owner, repo, ref: `heads/${branch.name}` });
        deleted++;
        core.info(`Deleted ${branch.name}: ${reason}`);
      } catch (error) {
        failures++;
        core.warning(`Could not delete ${branch.name}: ${error.message}`);
      }
    }
  }
  core.info(`${dryRun ? 'Eligible' : 'Deleted'} branches: ${dryRun ? eligible : deleted}`);
  if (failures) core.setFailed(`Could not delete ${failures} branch(es). See warnings above.`);
};
