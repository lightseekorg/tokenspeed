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

const authorsFields = `
  nodes { email user { login } }
  pageInfo { hasNextPage endCursor }
`;

module.exports = async function checkCommitAttribution({ github, core, context }) {
  const pullRequest = context.payload.pull_request;
  let cursor = null;
  let failures = 0;

  do {
    const { repository } = await github.graphql(`
      query($owner: String!, $repo: String!, $number: Int!, $cursor: String) {
        repository(owner: $owner, name: $repo) {
          pullRequest(number: $number) {
            headRefOid
            commits(first: 100, after: $cursor) {
              nodes {
                commit {
                  oid
                  authors(first: 100) { ${authorsFields} }
                }
              }
              pageInfo { hasNextPage endCursor }
            }
          }
        }
      }
    `, { ...context.repo, number: pullRequest.number, cursor });

    const current = repository.pullRequest;
    if (current.headRefOid !== pullRequest.head.sha) {
      throw new Error('PR head changed. Run this check on the latest commit.');
    }

    for (const { commit } of current.commits.nodes) {
      let authors = commit.authors;
      let authorIndex = 0;
      do {
        for (const author of authors.nodes) {
          const role = authorIndex++ === 0 ? 'author' : 'Co-authored-by';
          if (!author.user) {
            core.error(
              `Commit ${commit.oid}: ${role} email ` +
              `${author.email || '(missing email)'} is not linked to a GitHub account.`
            );
            failures++;
          } else if (author.user.login.endsWith('[bot]')) {
            core.error(
              `Commit ${commit.oid}: ${role} resolves to GitHub bot ${author.user.login}. ` +
              'GitHub bots cannot be commit authors or co-authors.'
            );
            failures++;
          }
        }
        if (!authors.pageInfo.hasNextPage) break;
        const result = await github.graphql(`
          query($owner: String!, $repo: String!, $oid: GitObjectID!, $cursor: String!) {
            repository(owner: $owner, name: $repo) {
              object(oid: $oid) {
                ... on Commit {
                  authors(first: 100, after: $cursor) { ${authorsFields} }
                }
              }
            }
          }
        `, { ...context.repo, oid: commit.oid, cursor: authors.pageInfo.endCursor });
        authors = result.repository.object.authors;
      } while (true);
      if (authorIndex === 0) {
        throw new Error(`GitHub returned no authors for commit ${commit.oid}. Retry this check.`);
      }
    }
    cursor = current.commits.pageInfo.hasNextPage
      ? current.commits.pageInfo.endCursor : null;
  } while (cursor);

  if (failures) {
    core.setFailed(
      'Fix the commit attribution errors listed above. Replace bot authors with the ' +
      'responsible contributor and remove bot Co-authored-by trailers. ' +
      'For unlinked emails, each contributor must add and verify ' +
      'their email in GitHub Settings > Emails, then rerun this check; or replace it with ' +
      'their exact GitHub-provided noreply address, amend/rebase the affected commits ' +
      '(including Co-authored-by trailers), and push the updated branch. ' +
      'A matching display name alone does not link an email to an account.'
    );
  } else {
    core.info('All commit authors and co-authors are linked to non-bot GitHub accounts.');
  }
};
