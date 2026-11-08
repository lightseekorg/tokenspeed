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

const assert = require('node:assert/strict');
const test = require('node:test');
const cleanup = require('./cleanup-branches.js');

const now = Date.parse('2026-01-15T00:00:00Z');
const day = 24 * 60 * 60 * 1000;

function branch(name, ageDays, protectedBranch = false) {
  return {
    name, protected: protectedBranch,
    commit: { sha: name, commit: { committer: { date: new Date(now - ageDays * day).toISOString() } } },
  };
}

function pr(name, state, repositoryId = 1) {
  return { state, head: { ref: name, repo: { id: repositoryId } }, base: { ref: 'main' } };
}

function fixture(branches, pulls) {
  const deleted = [];
  const messages = [];
  const failures = [];
  const github = {
    rest: {
      repos: {
        get: async () => ({ data: { id: 1, default_branch: 'main' } }),
        listBranches: async () => branches,
        getBranch: async ({ branch: name }) => ({ data: branches.find(b => b.name === name) }),
      },
      pulls: {
        list: async ({ head, base, state }) => pulls.filter(p =>
          (!head || head === `owner:${p.head.ref}`) &&
          (!base || base === p.base.ref) &&
          (state === 'all' || state === p.state)
        ),
      },
      git: { deleteRef: async ({ ref }) => deleted.push(ref) },
    },
    paginate: async (method, parameters) => method(parameters),
  };
  return {
    github, deleted, messages, failures,
    run: dryRun => cleanup({
      github, core: {
        info: message => messages.push(message),
        warning: message => messages.push(message),
        setFailed: message => failures.push(message),
      },
      context: { repo: { owner: 'owner', repo: 'repo' }, ref: 'refs/heads/main' },
      dryRun, now,
    }),
  };
}

test('deletes closed/merged PR branches immediately and no-PR branches strictly after seven days', async () => {
  const f = fixture([
    branch('closed', 0), branch('merged', 0), branch('stale', 8),
    branch('recent', 6), branch('boundary', 7), branch('fork-name', 0),
  ], [pr('closed', 'closed'), { ...pr('merged', 'closed'), merged_at: '2026-01-14T00:00:00Z' },
    pr('fork-name', 'closed', 2)]);
  await f.run(false);
  assert.deepEqual(f.deleted, ['heads/closed', 'heads/merged', 'heads/stale']);
});

test('keeps default/protected branches and open PR heads or bases', async () => {
  const f = fixture([
    branch('main', 30), branch('protected', 30, true),
    branch('reused', 30), branch('stack-base', 30),
  ], [pr('reused', 'closed'), pr('reused', 'open'),
    { ...pr('stack-head', 'open', 2), base: { ref: 'stack-base' } }]);
  await f.run(false);
  assert.deepEqual(f.deleted, []);
});

test('keeps branches whose head or protection changed since enumeration', async () => {
  const f = fixture([branch('advanced', 30), branch('protected-later', 30)], []);
  f.github.rest.repos.getBranch = async ({ branch: name }) => ({
    data: name === 'advanced'
      ? { ...branch(name, 0), commit: { sha: 'new-head' } }
      : branch(name, 30, true),
  });
  await f.run(false);
  assert.deepEqual(f.deleted, []);
});

test('dry run reports candidates without deleting refs', async () => {
  const f = fixture([branch('stale', 8)], []);
  await f.run(true);
  assert.deepEqual(f.deleted, []);
  assert.ok(f.messages.some(message => message.startsWith('Would delete stale:')));
});

test('does not delete when PR lookup fails or the commit date is unavailable', async () => {
  const stale = branch('stale', 8);
  stale.commit.commit.committer = null;
  const f = fixture([stale], []);
  await f.run(false);
  assert.deepEqual(f.deleted, []);
  f.github.rest.pulls.list = async () => { throw new Error('API unavailable'); };
  await assert.rejects(f.run(false), /API unavailable/);
  assert.deepEqual(f.deleted, []);
});

test('continues after a rejected deletion while reporting the run as failed', async () => {
  const f = fixture([branch('rejected', 8), branch('stale', 8)], []);
  f.github.rest.git.deleteRef = async ({ ref }) => {
    if (ref === 'heads/rejected') throw new Error('Deletion rejected');
    f.deleted.push(ref);
  };
  await f.run(false);
  assert.deepEqual(f.deleted, ['heads/stale']);
  assert.equal(f.failures.length, 1);
  assert.ok(f.messages.includes('Deleted branches: 1'));
});
