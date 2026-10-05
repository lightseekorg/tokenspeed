# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Release stable packages in dependency order, retaining state for manual recovery."""

import argparse
import base64
import hashlib
import json
import os
import re
import runpy
import subprocess
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from html import escape
from pathlib import Path

from cryptography import x509
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO = "lightseekorg/tokenspeed"
WHL = "lightseekorg/whl"
IDENTITY = "243258330+lightseek-bot@users.noreply.github.com"
STAGES = ("plan", "amd", "kernel", "tokenspeed", "index", "docker", "release")
PACKAGES = {
    "amd": "tokenspeed-kernel-amd",
    "kernel": "tokenspeed-kernel",
    "tokenspeed": "tokenspeed",
}
PROJECTS = {
    "tokenspeed-mla": "tokenspeed-mla/pyproject.toml",
    "tokenspeed-scheduler": "tokenspeed-scheduler/pyproject.toml",
    "tokenspeed-kernel-amd": "tokenspeed-kernel-amd/pyproject.toml",
    "tokenspeed": "python/pyproject.toml",
}
PR_FILES = {
    "amd": [PROJECTS[PACKAGES["amd"]]],
    "kernel": [
        "tokenspeed-kernel/python/setup.py",
        "tokenspeed-kernel/python/requirements/rocm-thirdparty.txt",
    ],
    "tokenspeed": [PROJECTS["tokenspeed"], "python/tokenspeed/version.py"],
}


def command(*args, cwd=None, env=None):
    return subprocess.check_output(args, text=True, cwd=cwd, env=env).strip()


def request(url, *, github, data=None):
    headers = {"Accept": "application/vnd.github+json"} if github else {}
    if github:
        if not url.startswith("https://api.github.com/"):
            raise ValueError("Unexpected GitHub API destination")
        headers.update(
            {
                "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                "X-GitHub-Api-Version": "2026-03-10",
            }
        )
    if data is not None:
        headers["Content-Type"] = "application/json"
    body = json.dumps(data).encode() if data is not None else None
    with urllib.request.urlopen(
        urllib.request.Request(url, data=body, headers=headers), timeout=30
    ) as response:
        content = response.read()
        return json.loads(content) if content else None


def api(path, *, data=None):
    return request(
        f"https://api.github.com/repos/{REPO}/{path}", github=True, data=data
    )


def pypi(package, version=None):
    suffix = f"/{version}" if version else ""
    try:
        return request(f"https://pypi.org/pypi/{package}{suffix}/json", github=False)
    except urllib.error.HTTPError as error:
        if error.code == 404 and version:
            return None
        raise


def latest_version(package):
    releases = pypi(package)["releases"]
    versions = [
        Version(v)
        for v, files in releases.items()
        if files
        and any(not f["yanked"] for f in files)
        and not Version(v).is_prerelease
        and not Version(v).is_devrelease
    ]
    return str(max(versions))


def source_sha(package, version, workflow):
    """Read source claims served by PyPI; this is not signature verification."""
    release = pypi(package, version)
    if (
        release is None
        or not release["urls"]
        or any(f["yanked"] for f in release["urls"])
    ):
        raise RuntimeError(f"{package} {version} is missing or has yanked files")
    sources = set()
    for file in release["urls"]:
        filename = urllib.parse.quote(file["filename"], safe="")
        provenance = request(
            f"https://pypi.org/integrity/{package}/{version}/{filename}/provenance",
            github=False,
        )
        claims = set()
        for bundle in provenance["attestation_bundles"]:
            publisher = bundle["publisher"]
            if (
                publisher.get("repository") != REPO
                or publisher.get("workflow") != workflow
            ):
                continue
            for attestation in bundle["attestations"]:
                statement = json.loads(
                    base64.b64decode(attestation["envelope"]["statement"])
                )
                if statement["subject"] != [
                    {
                        "name": file["filename"],
                        "digest": {"sha256": file["digests"]["sha256"]},
                    }
                ]:
                    raise RuntimeError(
                        f"Provenance digest mismatch for {file['filename']}"
                    )
                certificate = x509.load_der_x509_certificate(
                    base64.b64decode(
                        attestation["verification_material"]["certificate"]
                    )
                )

                def claim(number):
                    return certificate.extensions.get_extension_for_oid(
                        x509.ObjectIdentifier(f"1.3.6.1.4.1.57264.1.{number}")
                    ).value.value.decode()

                if (
                    claim(5) != REPO
                    or claim(1) != "https://token.actions.githubusercontent.com"
                ):
                    raise RuntimeError("Unexpected provenance repository or issuer")
                sha = claim(3)
                if not re.fullmatch(r"[0-9a-f]{40}", sha):
                    raise RuntimeError("Invalid provenance source SHA")
                claims.add(sha)
        if len(claims) != 1:
            raise RuntimeError(
                f"Missing or ambiguous source provenance for {file['filename']}"
            )
        sources.update(claims)
    if len(sources) != 1:
        raise RuntimeError(f"{package} {version} contains files from different commits")
    return sources.pop()


def read_version(package):
    if package == "tokenspeed-kernel":
        return re.search(
            r'^BASE_VERSION = "([^"]+)"$',
            Path("tokenspeed-kernel/python/setup.py").read_text(),
            re.M,
        )[1]
    return tomllib.loads(Path(PROJECTS[package]).read_text())["project"]["version"]


def requirements(path):
    text = Path(path).read_text()
    lines = (
        tomllib.loads(text)["project"]["dependencies"]
        if path.endswith(".toml")
        else text.splitlines()
    )
    return {
        canonicalize_name(r.name): r
        for line in lines
        if line.strip() and not line.startswith("#")
        for r in [Requirement(line)]
    }


def check_tree(sha, directory):
    command("git", "fetch", "--no-tags", "origin", sha)
    command("git", "merge-base", "--is-ancestor", sha, "HEAD")
    if (
        subprocess.run(
            ["git", "diff", "--quiet", sha, "HEAD", "--", directory]
        ).returncode
        != 0
    ):
        raise RuntimeError(
            f"Unreleased changes under {directory}; manual upstream release required"
        )


def preflight():
    versions = {}
    for package, path in (
        ("tokenspeed-mla", "tokenspeed-kernel/python/requirements/cuda-thirdparty.txt"),
        ("tokenspeed-scheduler", "python/pyproject.toml"),
    ):
        version = latest_version(package)
        if read_version(package) != version:
            raise RuntimeError(
                f"{package} main version is not the latest published version {version}"
            )
        expected = f"=={version}" if package == "tokenspeed-mla" else f">={version}"
        if str(requirements(path)[package].specifier) != expected:
            raise RuntimeError(f"{path} must depend on {package}{expected}")
        sha = source_sha(package, version, f"release-{package}.yml")
        check_tree(sha, package)
        versions[package] = version
    if runpy.run_path("python/tokenspeed/version.py")["__version__"] != read_version(
        "tokenspeed"
    ):
        raise RuntimeError("TokenSpeed's two version declarations disagree")
    return versions


def next_version(current, published, requested):
    for value in (current, published, requested):
        if value and not re.fullmatch(r"\d+\.\d+\.\d+", value):
            raise ValueError(f"Expected a stable major.minor.patch version: {value}")
    if requested:
        if Version(requested) <= Version(published) or Version(requested) <= Version(
            current
        ):
            raise ValueError("Requested version would reuse or downgrade a release")
        return requested
    major, minor, patch = max(Version(current), Version(published)).release
    return f"{major}.{minor}.{patch + 1}"


def replace(path, pattern, replacement):
    text, count = re.subn(pattern, replacement, Path(path).read_text(), flags=re.M)
    if count != 1:
        raise RuntimeError(f"Expected one version declaration in {path}")
    Path(path).write_text(text)


def update_metadata(stage, versions):
    version = versions[PACKAGES[stage]]
    current = read_version(PACKAGES[stage])
    if Version(current) > Version(version):
        raise RuntimeError(
            "Main has moved past the reserved version; manual intervention required"
        )
    if stage in ("amd", "tokenspeed"):
        replace(
            PROJECTS[PACKAGES[stage]], r'^version = "[^"]+"$', f'version = "{version}"'
        )
    if stage == "kernel":
        replace(
            PR_FILES[stage][0],
            r'^BASE_VERSION = "[^"]+"$',
            f'BASE_VERSION = "{version}"',
        )
        replace(
            PR_FILES[stage][1],
            r"^tokenspeed-kernel-amd>=[^\n]+$",
            f'tokenspeed-kernel-amd>={versions["tokenspeed-kernel-amd"]}',
        )
    if stage == "tokenspeed":
        replace(
            PR_FILES[stage][1], r'^__version__ = "[^"]+"$', f'__version__ = "{version}"'
        )
        replace(
            PROJECTS["tokenspeed"],
            r'"tokenspeed-kernel>=[^"]+"',
            f'"tokenspeed-kernel>={versions["tokenspeed-kernel"]}"',
        )


def checks_ready(pr, *, required_checks, bypass_reviews):
    checks = pr["statusCheckRollup"]
    for check in checks:
        result = check.get("conclusion") or check.get("state")
        if result in (
            "FAILURE",
            "ERROR",
            "CANCELLED",
            "TIMED_OUT",
            "ACTION_REQUIRED",
            "STARTUP_FAILURE",
            "STALE",
        ):
            raise RuntimeError(
                f"PR check failed: {check.get('name', check.get('context'))}"
            )
    lint = any(
        c.get("name") == "lint" and c.get("conclusion") == "SUCCESS" for c in checks
    )
    complete = all(
        c.get("conclusion") in ("SUCCESS", "SKIPPED", "NEUTRAL")
        or c.get("state") == "SUCCESS"
        for c in checks
    )
    required = all(
        any(
            (c.get("name") or c.get("context")) == name
            and (c.get("conclusion") == "SUCCESS" or c.get("state") == "SUCCESS")
            for c in checks
        )
        for name in required_checks
    )
    review_only_block = (
        bypass_reviews
        and pr["mergeStateStatus"] == "BLOCKED"
        and pr["reviewDecision"] == "REVIEW_REQUIRED"
        and pr["mergeable"] == "MERGEABLE"
    )
    return (
        lint
        and complete
        and required
        and pr["reviewDecision"] != "CHANGES_REQUESTED"
        and (pr["mergeStateStatus"] in ("CLEAN", "HAS_HOOKS") or review_only_block)
    )


def merge_policy():
    """Use an existing, explicit bot exemption; never change repository rules."""
    user_id = int(command("gh", "api", "user", "--jq", ".id"))
    rules = api("rules/branches/main")
    required = {
        check["context"]
        for rule in rules
        if rule["type"] == "required_status_checks"
        for check in rule["parameters"]["required_status_checks"]
    }
    approvals = [rule for rule in rules if rule["type"] == "pull_request"]
    bypass = bool(approvals)
    for rule in approvals:
        if (
            rule["ruleset_source"] != REPO
            or rule["ruleset_source_type"] != "Repository"
        ):
            bypass = False
            break
        actors = api(f"rulesets/{rule['ruleset_id']}")["bypass_actors"]
        if not any(
            actor["actor_type"] == "User"
            and actor["actor_id"] == user_id
            and actor["bypass_mode"] == "always"
            for actor in actors
        ):
            bypass = False
            break
    return required, bypass


def index_release(root, variant, package, release):
    index = root / variant / package / "index.html"
    text = index.read_text() if index.exists() else "<!DOCTYPE html>\n"
    wheels = [a for a in release["assets"] if a["name"].endswith(".whl")]
    if not wheels:
        raise RuntimeError("Stable release has no wheels")
    for asset in wheels:
        url, digest, name = (
            asset["browser_download_url"],
            asset["digest"],
            asset["name"],
        )
        if not url.startswith(
            f"https://github.com/{WHL}/releases/download/"
        ) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest or ""):
            raise RuntimeError("Invalid public wheel URL or digest")
        entry = f'<a href="{escape(url)}#sha256={digest[7:]}">{escape(name)}</a><br>\n'
        existing = re.search(
            rf'<a href="([^"]+)">{re.escape(escape(name))}</a><br>\n', text
        )
        if existing:
            if not existing[1].endswith(f"#sha256={digest[7:]}"):
                raise RuntimeError(f"Refusing to replace an indexed wheel: {name}")
        else:
            text += entry
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(text)
    for parent, label in ((root / variant, package), (root, variant)):
        path = parent / "index.html"
        text = path.read_text() if path.exists() else "<!DOCTYPE html>\n"
        link = f'<a href="{label}/">{label}</a><br>\n'
        if link not in text:
            path.write_text(text + link)


class Release:
    def __init__(self, state_path, stage):
        self.path = state_path
        self.stage = stage
        self.state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "run_id": os.environ["GITHUB_RUN_ID"],
                "stages": {},
                "runs": {},
                "versions": {},
            }
        )
        if self.state["run_id"] != os.environ["GITHUB_RUN_ID"]:
            raise RuntimeError("Recovery state belongs to a different weekly run")
        self.deadline = time.monotonic() + 340 * 60
        self.phase = self.state["stages"].setdefault(stage, {})

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2) + "\n")
        temporary.replace(self.path)

    def pause(self):
        if time.monotonic() >= self.deadline:
            raise RuntimeError(
                "Timed out; inspect recorded child runs and resume after manual intervention"
            )
        time.sleep(30)

    def guard(self):
        if (
            os.environ["GITHUB_REPOSITORY"] != REPO
            or os.environ["GITHUB_REF"] != "refs/heads/main"
        ):
            raise RuntimeError(
                "Weekly releases must run from the source repository's main branch"
            )
        command("gh", "auth", "status")
        if command("gh", "api", "user", "--jq", ".login") != "lightseek-bot":
            raise RuntimeError("LIGHTSEEK_BOT_TOKEN must authenticate lightseek-bot")
        for repo in (REPO, WHL):
            if (
                command(
                    "gh",
                    "repo",
                    "view",
                    repo,
                    "--json",
                    "visibility",
                    "--jq",
                    ".visibility",
                )
                != "PUBLIC"
            ):
                raise RuntimeError(f"Expected a public release destination: {repo}")
        remote = command("git", "remote", "get-url", "--push", "origin").removesuffix(
            ".git"
        )
        if remote != f"https://github.com/{REPO}":
            raise RuntimeError("Unexpected source push destination")
        command("git", "config", "user.name", "lightseek-bot")
        command("git", "config", "user.email", IDENTITY)
        command("gh", "auth", "setup-git", "--hostname", "github.com")

    def checkout_main(self):
        command("git", "fetch", "--no-tags", "origin", "main")
        command("git", "checkout", "--detach", "origin/main")

    def gate(self):
        versions = preflight()
        if self.state["versions"] and any(
            self.state["versions"][p] != v for p, v in versions.items()
        ):
            raise RuntimeError(
                "Upstream releases changed during this weekly run; manual intervention required"
            )
        return versions

    def plan(self, requested):
        self.checkout_main()
        upstream = self.gate()
        if self.state["versions"]:
            return
        if os.environ["DOCKER_READY"] != "true":
            raise RuntimeError("DOCKERHUB_USERNAME and DOCKERHUB_TOKEN are required")
        versions = dict(upstream)
        for stage, package in PACKAGES.items():
            versions[package] = next_version(
                read_version(package),
                latest_version(package),
                requested if stage == "tokenspeed" else "",
            )
            if pypi(package, versions[package]) is not None:
                raise RuntimeError(
                    f"Reserved version already exists: {package} {versions[package]}"
                )
            ref = self.branch(stage, versions[package])
            if command("git", "ls-remote", "origin", f"refs/heads/{ref}"):
                raise RuntimeError(f"Release branch already exists: {ref}")
            tags = (
                [
                    f"tokenspeed-kernel-v{versions[package]}-{variant}"
                    for variant in ("cu129", "cu130", "rocm72")
                ]
                if stage == "kernel"
                else [f"{package}-v{versions[package]}"]
            )
            if any(self.release_exists(WHL, tag) for tag in tags):
                raise RuntimeError(
                    "Reserved wheelhouse version already exists; inspect the previous release"
                )
        self.state["versions"] = versions
        self.state["initial_versions"] = {
            package: read_version(package) for package in PACKAGES.values()
        }
        self.save()

    @staticmethod
    def branch(stage, version):
        return f"release/{version}" if stage == "amd" else f"release/{stage}-{version}"

    def pr(self, stage):
        phase = self.state["stages"][stage]
        if "sha" in phase:
            command("git", "fetch", "--no-tags", "origin", phase["sha"])
            command("git", "checkout", "--detach", phase["sha"])
            self.gate()
            update_metadata(stage, self.state["versions"])
            if command("git", "diff", "--name-only"):
                raise RuntimeError("Recorded release source has unexpected metadata")
            return phase["sha"]
        self.checkout_main()
        self.gate()
        package = PACKAGES[stage]
        version = self.state["versions"][package]
        branch = f"bot/weekly-{package}-{version}"
        prs = json.loads(
            command(
                "gh",
                "pr",
                "list",
                "--repo",
                REPO,
                "--head",
                branch,
                "--state",
                "all",
                "--json",
                "number",
            )
        )
        if not prs:
            current = read_version(package)
            if current not in (self.state["initial_versions"][package], version):
                raise RuntimeError("Main package version changed outside this release")
            title = f"build: release {package} {version}"
            existing = command("git", "ls-remote", "origin", f"refs/heads/{branch}")
            if existing:
                if existing.split()[0] != phase.get("head"):
                    raise RuntimeError("Version branch was created outside this run")
                command("git", "fetch", "--no-tags", "origin", branch)
                command("git", "checkout", "--detach", "FETCH_HEAD")
            else:
                command("git", "switch", "-c", branch)
                update_metadata(stage, self.state["versions"])
                if not command("git", "diff", "--name-only"):
                    raise RuntimeError(
                        "Reserved metadata already on main without this run's PR; inspect manually"
                    )
                hook_env = dict(os.environ, SKIP="clang-format")
                hook_env.pop("GH_TOKEN", None)
                result = subprocess.run(
                    ["pre-commit", "run", "--all-files"], env=hook_env
                )
                if result.returncode:
                    command("pre-commit", "run", "--all-files", env=hook_env)
                changed = set(command("git", "diff", "--name-only").splitlines())
                if not changed <= set(PR_FILES[stage]):
                    raise RuntimeError(
                        "Pre-commit changed files outside the version update"
                    )
                command("git", "add", "--", *PR_FILES[stage])
                command("git", "diff", "--cached", "--check")
                command("git", "diff", "--cached")
                command("git", "commit", "-s", "-m", title)
                phase["head"] = command("git", "rev-parse", "HEAD")
                self.save()
                # Only metadata from the confirmed public source repository is outbound.
                command(
                    "git",
                    "push",
                    f"--force-with-lease=refs/heads/{branch}:",
                    "origin",
                    f"HEAD:refs/heads/{branch}",
                )
                if (
                    command(
                        "git", "ls-remote", "origin", f"refs/heads/{branch}"
                    ).split()[0]
                    != phase["head"]
                ):
                    raise RuntimeError("Version branch remote readback mismatch")
            body = self.path.parent / "pr-body.txt"
            body.write_text(
                "Keep the weekly release ordered so downstream packages require already published component versions.\n"
            )
            command(
                "gh",
                "pr",
                "create",
                "--repo",
                REPO,
                "--head",
                branch,
                "--base",
                "main",
                "--title",
                title,
                "--body-file",
                str(body),
            )
            prs = json.loads(
                command(
                    "gh",
                    "pr",
                    "list",
                    "--repo",
                    REPO,
                    "--head",
                    branch,
                    "--state",
                    "all",
                    "--json",
                    "number",
                )
            )
            live = json.loads(
                command(
                    "gh",
                    "pr",
                    "view",
                    str(prs[0]["number"]),
                    "--repo",
                    REPO,
                    "--json",
                    "body,title",
                )
            )
            if (
                live["body"].strip() != body.read_text().strip()
                or live["title"] != title
            ):
                raise RuntimeError("Version PR readback mismatch")
        if len(prs) != 1:
            raise RuntimeError("Expected exactly one version PR")
        phase["pr"] = prs[0]["number"]
        self.save()
        while True:
            pr = json.loads(
                command(
                    "gh",
                    "pr",
                    "view",
                    str(phase["pr"]),
                    "--repo",
                    REPO,
                    "--json",
                    "state,headRefOid,mergeCommit,statusCheckRollup,mergeStateStatus,reviewDecision,mergeable",
                )
            )
            if pr["state"] == "MERGED":
                phase["sha"] = pr["mergeCommit"]["oid"]
                self.save()
                command("git", "fetch", "--no-tags", "origin", phase["sha"])
                command("git", "checkout", "--detach", phase["sha"])
                self.gate()
                # Check the merged metadata, including both TokenSpeed version sources.
                update_metadata(stage, self.state["versions"])
                if command("git", "diff", "--name-only"):
                    raise RuntimeError("Merged release PR has unexpected metadata")
                return phase["sha"]
            if pr["state"] != "OPEN":
                raise RuntimeError("Version PR was closed without merging")
            if pr["headRefOid"] != phase.get("head"):
                raise RuntimeError("Version PR head changed outside this run")
            required, bypass = merge_policy()
            if checks_ready(pr, required_checks=required, bypass_reviews=bypass):
                # The bot may already be explicitly exempt from review requirements.
                # Still require every registered and required CI check to succeed.
                merge_args = ["--admin"] if pr["mergeStateStatus"] == "BLOCKED" else []
                command(
                    "gh",
                    "pr",
                    "merge",
                    str(phase["pr"]),
                    "--repo",
                    REPO,
                    "--squash",
                    "--match-head-commit",
                    pr["headRefOid"],
                    *merge_args,
                )
            else:
                self.pause()

    def immutable_ref(self, stage, sha):
        ref = self.branch(stage, self.state["versions"][PACKAGES[stage]])
        command("git", "fetch", "--no-tags", "origin", "main")
        command("git", "merge-base", "--is-ancestor", sha, "origin/main")
        existing = command("git", "ls-remote", "origin", f"refs/heads/{ref}")
        if existing:
            if existing.split()[0] != sha:
                raise RuntimeError("Existing release branch points to another commit")
        else:
            command(
                "git",
                "push",
                f"--force-with-lease=refs/heads/{ref}:",
                "origin",
                f"{sha}:refs/heads/{ref}",
            )
        if command("git", "ls-remote", "origin", f"refs/heads/{ref}").split()[0] != sha:
            raise RuntimeError("Release branch readback mismatch")
        return ref

    def find_run(self, workflow, sha, ref, event):
        query = urllib.parse.urlencode(
            {"head_sha": sha, "branch": ref, "event": event, "per_page": 100}
        )
        runs = api(f"actions/workflows/{workflow}/runs?{query}")["workflow_runs"]
        matching = [
            r
            for r in runs
            if r["head_sha"] == sha
            and r["head_branch"] == ref
            and r["event"] == event
            and r["actor"]["login"] == "lightseek-bot"
        ]
        if len(matching) > 1:
            raise RuntimeError(
                "Ambiguous publication runs; choose the correct run manually"
            )
        return matching[0]["id"] if matching else None

    def child(self, workflow, sha, ref, inputs, *, event):
        record = self.state["runs"].setdefault(
            workflow, {"sha": sha, "ref": ref, "event": event}
        )
        if (record["sha"], record["ref"], record["event"]) != (sha, ref, event):
            raise RuntimeError(
                "Recorded workflow source differs from the reserved source"
            )
        if "id" not in record:
            if event == "workflow_dispatch" and not record.get("dispatch_started"):
                record["dispatch_started"] = True
                self.save()  # An ambiguous response must not lead to a second dispatch.
                response = api(
                    f"actions/workflows/{workflow}/dispatches",
                    data={"ref": ref, "inputs": inputs},
                )
                if response:
                    record["id"] = response["workflow_run_id"]
                    self.save()
            while "id" not in record:
                run_id = self.find_run(workflow, sha, ref, event)
                if run_id:
                    record["id"] = run_id
                    self.save()
                else:
                    self.pause()
        while True:
            run = api(f"actions/runs/{record['id']}")
            if (
                run["head_sha"] != sha
                or run["head_branch"] != ref
                or run["event"] != event
                or run["path"].split("@")[0] != f".github/workflows/{workflow}"
            ):
                raise RuntimeError("Child workflow source mismatch")
            print(f"Waiting for {workflow}: {run['html_url']}", flush=True)
            if run["status"] == "completed":
                if run["conclusion"] != "success":
                    raise RuntimeError(
                        f"Child workflow failed; repair and rerun {run['html_url']} before resuming"
                    )
                record["complete"] = True
                self.save()
                return record["id"]
            self.pause()

    def published(self, package, workflow, sha):
        version = self.state["versions"][package]
        while pypi(package, version) is None:
            self.pause()
        if source_sha(package, version, workflow) != sha:
            raise RuntimeError(f"{package} {version} was published from another source")

    def wheelhouse(self, tag, sha, wheel_count):
        release = json.loads(
            command(
                "gh",
                "release",
                "view",
                tag,
                "--repo",
                WHL,
                "--json",
                "body,assets,isDraft,isPrerelease",
            )
        )
        if (
            release["isDraft"]
            or release["isPrerelease"]
            or f"{REPO}@{sha}" not in release["body"]
        ):
            raise RuntimeError("Wheelhouse release has unexpected source or visibility")
        if sum(a["name"].endswith(".whl") for a in release["assets"]) != wheel_count:
            raise RuntimeError("Wheelhouse release is missing expected wheels")
        # REST assets include SHA256 digests, unlike older gh release JSON fields.
        return request(
            f"https://api.github.com/repos/{WHL}/releases/tags/{tag}", github=True
        )

    def packages(self, stage):
        self.checkout_main()
        self.gate()
        if stage == "kernel":
            check_tree(self.state["stages"]["amd"]["sha"], "tokenspeed-kernel-amd")
        if stage == "tokenspeed":
            check_tree(self.state["stages"]["kernel"]["sha"], "tokenspeed-kernel")
        sha = self.pr(stage)
        ref = self.immutable_ref(stage, sha)
        version = self.state["versions"][PACKAGES[stage]]
        if stage == "amd":
            workflow = "release-tokenspeed-kernel-amd.yml"
            self.child(
                workflow,
                sha,
                ref,
                {
                    "create_release_branch": False,
                    "prerelease": False,
                    "publish_github": True,
                    "publish_pypi": True,
                },
                event="workflow_dispatch",
            )
            self.published(PACKAGES[stage], workflow, sha)
            self.wheelhouse(f"tokenspeed-kernel-amd-v{version}", sha, 1)
        elif stage == "kernel":
            workflow = "release-tokenspeed-kernel.yml"
            self.child(
                workflow,
                sha,
                ref,
                {
                    "nightly": False,
                    "cuda_variant": "all",
                    "pypi_cuda_variant": "cu130",
                    "publish_github": True,
                    "publish_pypi": True,
                    "prerelease": False,
                },
                event="workflow_dispatch",
            )
            self.published(PACKAGES[stage], workflow, sha)
            self.child(
                "release-tokenspeed-kernel-rocm.yml",
                sha,
                ref,
                {"nightly": False, "publish_github": True, "prerelease": False},
                event="workflow_dispatch",
            )
            for variant, count in (("cu129", 8), ("cu130", 8), ("rocm72", 4)):
                self.wheelhouse(f"tokenspeed-kernel-v{version}-{variant}", sha, count)
        else:
            workflow = "release-pypi.yml"
            run_id = self.child(workflow, sha, "main", {}, event="push")
            self.published("tokenspeed", workflow, sha)
            tag = f"tokenspeed-v{version}"
            release = (
                request(
                    f"https://api.github.com/repos/{WHL}/releases/tags/{tag}",
                    github=True,
                )
                if self.release_exists(WHL, tag)
                else None
            )
            if release is None:
                dist = self.path.parent / "tokenspeed-dist"
                command(
                    "gh",
                    "run",
                    "download",
                    str(run_id),
                    "--repo",
                    REPO,
                    "--name",
                    "tokenspeed-dist",
                    "--dir",
                    str(dist),
                )
                files = list(dist.iterdir())
                expected = {
                    f["filename"]: f["digests"]["sha256"]
                    for f in pypi("tokenspeed", version)["urls"]
                }
                if {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files
                } != expected:
                    raise RuntimeError(
                        "TokenSpeed artifact differs from published PyPI files"
                    )
                notes = self.path.parent / "whl-notes.txt"
                notes.write_text(f"tokenspeed {version} built from {REPO}@{sha}\n")
                command(
                    "gh",
                    "release",
                    "create",
                    tag,
                    *map(str, files),
                    "--repo",
                    WHL,
                    "--title",
                    f"tokenspeed {version}",
                    "--notes-file",
                    str(notes),
                )
            self.wheelhouse(tag, sha, 1)

    @staticmethod
    def release_exists(repo, tag):
        try:
            request(
                f"https://api.github.com/repos/{repo}/releases/tags/{tag}", github=True
            )
            return True
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return False
            raise

    def index(self):
        root = self.path.parent / "wheelhouse"
        command(
            "git",
            "clone",
            "--branch",
            "gh-pages",
            "--single-branch",
            f"https://github.com/{WHL}.git",
            str(root),
        )
        if (
            command("git", "remote", "get-url", "--push", "origin", cwd=root)
            != f"https://github.com/{WHL}.git"
        ):
            raise RuntimeError("Unexpected wheel index push destination")
        command("git", "config", "user.name", "lightseek-bot", cwd=root)
        command("git", "config", "user.email", IDENTITY, cwd=root)
        versions = self.state["versions"]
        for attempt in range(3):
            command("git", "fetch", "origin", "gh-pages", cwd=root)
            command("git", "reset", "--hard", "origin/gh-pages", cwd=root)
            paths = {"index.html"}
            for variant, suffix, count in (
                ("cu129", "cu129", 8),
                ("cu130", "cu130", 8),
                ("rocm7.2", "rocm72", 4),
            ):
                paths.add(f"{variant}/index.html")
                for stage, tag, wheel_count in (
                    (
                        "kernel",
                        f"tokenspeed-kernel-v{versions['tokenspeed-kernel']}-{suffix}",
                        count,
                    ),
                    ("tokenspeed", f"tokenspeed-v{versions['tokenspeed']}", 1),
                ):
                    release = self.wheelhouse(
                        tag, self.state["stages"][stage]["sha"], wheel_count
                    )
                    index_release(root, variant, PACKAGES[stage], release)
                    paths.add(f"{variant}/{PACKAGES[stage]}/index.html")
                if variant == "rocm7.2":
                    release = self.wheelhouse(
                        f"tokenspeed-kernel-amd-v{versions['tokenspeed-kernel-amd']}",
                        self.state["stages"]["amd"]["sha"],
                        1,
                    )
                    index_release(root, variant, "tokenspeed-kernel-amd", release)
                    paths.add(f"{variant}/tokenspeed-kernel-amd/index.html")
            command("git", "add", "index.html", "cu129", "cu130", "rocm7.2", cwd=root)
            if command("git", "diff", "--cached", "--name-only", cwd=root):
                command("git", "diff", "--cached", "--check", cwd=root)
                command("git", "diff", "--cached", cwd=root)
                if (root / ".pre-commit-config.yaml").exists():
                    command(
                        "pre-commit",
                        "run",
                        "--all-files",
                        cwd=root,
                        env={k: v for k, v in os.environ.items() if k != "GH_TOKEN"},
                    )
                command(
                    "git",
                    "commit",
                    "-s",
                    "-m",
                    "Publish weekly stable wheel index",
                    cwd=root,
                )
                push = subprocess.run(
                    ["git", "push", "origin", "HEAD:gh-pages"], cwd=root
                )
                if push.returncode:
                    if attempt == 2:
                        raise RuntimeError(
                            "Wheel index push failed after three attempts"
                        )
                    continue
            break
        for path in sorted(paths):
            encoded = command(
                "gh",
                "api",
                f"repos/{WHL}/contents/{path}?ref=gh-pages",
                "--jq",
                ".content",
            )
            if base64.b64decode(encoded).decode() != (root / path).read_text():
                raise RuntimeError("Wheel index remote readback mismatch")

    def docker(self):
        sha = self.state["stages"]["tokenspeed"]["sha"]
        ref = self.branch("tokenspeed", self.state["versions"]["tokenspeed"])
        self.child(
            "publish-release-docker.yml", sha, ref, {}, event="workflow_dispatch"
        )
        image = f"lightseekorg/tokenspeed:{self.state['versions']['tokenspeed']}"
        manifest = json.loads(
            command("docker", "buildx", "imagetools", "inspect", image, "--raw")
        )
        platforms = {
            (m["platform"]["os"], m["platform"]["architecture"])
            for m in manifest["manifests"]
        }
        if not {("linux", "amd64"), ("linux", "arm64")} <= platforms:
            raise RuntimeError("Docker release is missing a supported platform")
        self.phase["image"] = image

    def release(self):
        if any(
            not self.state["stages"][stage].get("complete") for stage in STAGES[:-1]
        ):
            raise RuntimeError(
                "Cannot publish release notes before all destinations succeed"
            )
        version = self.state["versions"]["tokenspeed"]
        sha = self.state["stages"]["tokenspeed"]["sha"]
        tag = f"v{version}"
        existing = command("git", "ls-remote", "origin", f"refs/tags/{tag}")
        if existing and existing.split()[0] != sha:
            raise RuntimeError("Existing TokenSpeed tag belongs to another source")
        if self.release_exists(REPO, tag):
            release = request(
                f"https://api.github.com/repos/{REPO}/releases/tags/{tag}", github=True
            )
            if "Weekly component versions" not in release["body"]:
                raise RuntimeError(
                    "Existing release page does not belong to this weekly release"
                )
            return
        notes = self.path.parent / "release-notes.md"
        text = "Weekly component versions\n\n| Package | Version |\n| --- | --- |\n"
        text += "".join(f"| {p} | {v} |\n" for p, v in self.state["versions"].items())
        text += f"\nDocker: `{self.state['stages']['docker']['image']}` (linux/amd64, linux/arm64).\n\n"
        text += "".join(
            f"- [{workflow}](https://github.com/{REPO}/actions/runs/{run['id']})\n"
            for workflow, run in self.state["runs"].items()
        )
        notes.write_text(text)
        command(
            "gh",
            "release",
            "create",
            tag,
            "--repo",
            REPO,
            "--target",
            sha,
            "--title",
            f"TokenSpeed {version}",
            "--generate-notes",
            "--notes-file",
            str(notes),
        )
        live = request(
            f"https://api.github.com/repos/{REPO}/releases/tags/{tag}", github=True
        )
        if (
            text.strip() not in live["body"]
            or command("git", "ls-remote", "origin", f"refs/tags/{tag}").split()[0]
            != sha
        ):
            raise RuntimeError("Release page readback mismatch")

    def run(self, requested):
        self.guard()
        if self.stage != "plan":
            previous = STAGES[STAGES.index(self.stage) - 1]
            if not self.state["stages"].get(previous, {}).get("complete"):
                raise RuntimeError("Previous stage did not complete")
        if self.stage == "plan":
            self.plan(requested)
        elif self.stage in PACKAGES:
            self.packages(self.stage)
        else:
            {"index": self.index, "docker": self.docker, "release": self.release}[
                self.stage
            ]()
        self.phase["complete"] = True
        self.phase.pop("error", None)
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--version", default="")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(preflight(), indent=2))
        return
    release = Release(args.state, args.stage)
    try:
        release.run(args.version)
    except Exception as error:
        release.phase["error"] = str(error)
        release.save()
        raise
    finally:
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with Path(summary).open("a") as output:
                print(
                    f"{args.stage}: {'success' if release.phase.get('complete') else 'manual intervention required'}",
                    file=output,
                )
                print(
                    f"\n```json\n{json.dumps(release.state, indent=2)}\n```",
                    file=output,
                )


if __name__ == "__main__":
    main()
