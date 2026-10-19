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
const check = require('./check-commit-attribution.js');

const head = 'a'.repeat(40);
const linked = { email: 'contributor@example.invalid', user: { login: 'contributor' } };
const unlinked = { email: 'unlinked@example.invalid', user: null };
const bot = { email: 'contributor@example.invalid', user: { login: 'github-actions[bot]' } };

function page(nodes, next) {
  return { nodes, pageInfo: { hasNextPage: next !== null, endCursor: next } };
}

function commit(oid, authors, next) {
  return { commit: { oid, authors: page(authors, next) } };
}

function pr(commits, next, headRefOid) {
  return { repository: { pullRequest: { headRefOid, commits: page(commits, next) } } };
}

async function run(graphql) {
  const errors = [];
  const failures = [];
  await check({
    github: { graphql },
    context: {
      repo: { owner: 'example', repo: 'project' },
      payload: { pull_request: { number: 1, head: { sha: head } } },
    },
    core: {
      error: message => errors.push(message),
      setFailed: message => failures.push(message),
      info: () => {},
    },
  });
  return { errors, failures };
}

test('accepts linked authors and co-authors without requiring public profile emails', async () => {
  const botNamedUser = {
    email: '243258330+lightseek-bot@users.noreply.github.com',
    user: { login: 'lightseek-bot' },
  };
  const result = await run(async () => pr([commit(head, [botNamedUser, linked], null)], null, head));
  assert.deepEqual(result, { errors: [], failures: [] });
});

test('rejects a bot author based on its resolved account, not its raw email', async () => {
  const result = await run(async () => pr([commit(head, [bot], null)], null, head));
  assert.equal(result.errors.length, 1);
  assert.match(result.errors[0], /author resolves to GitHub bot github-actions\[bot\]/);
  assert.equal(result.failures.length, 1);
  assert.match(result.failures[0], /Replace bot authors/);
  assert.match(result.failures[0], /remove bot Co-authored-by trailers/);
});

test('reports unlinked authors and co-authors with repair instructions', async () => {
  const result = await run(async () => pr([
    commit('b'.repeat(40), [unlinked], null),
    commit(head, [linked, unlinked], null),
  ], null, head));
  assert.equal(result.errors.length, 2);
  assert.match(result.errors[0], /Commit b{40}: author email unlinked@example\.invalid/);
  assert.match(result.errors[1], /Commit a{40}: Co-authored-by email unlinked@example\.invalid/);
  assert.equal(result.failures.length, 1);
  assert.match(result.failures[0], /add and verify/);
  assert.match(result.failures[0], /noreply address/);
});

test('checks later pages of both commits and co-authors', async () => {
  const result = await run(async (_query, variables) => {
    if (variables.oid === head) {
      return { repository: { object: { authors: page([unlinked, bot], null) } } };
    }
    if (variables.cursor === 'next-commit') {
      return pr([commit(head, [linked], 'next-author')], null, head);
    }
    return pr([commit('b'.repeat(40), [linked], null)], 'next-commit', head);
  });
  assert.equal(result.errors.length, 2);
  assert.match(result.errors[0], /Co-authored-by email unlinked@example\.invalid/);
  assert.match(result.errors[1], /Co-authored-by resolves to GitHub bot github-actions\[bot\]/);
  assert.equal(result.failures.length, 1);
});

test('does not pass when the PR head changed or GitHub is unavailable', async () => {
  await assert.rejects(run(async () => pr([], null, 'b'.repeat(40))), /PR head changed/);
  await assert.rejects(run(async () => { throw new Error('API unavailable'); }), /API unavailable/);
});
