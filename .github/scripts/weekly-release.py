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
from fnmatch import fnmatchcase
from html import escape
from pathlib import Path

import yaml
from cryptography import x509
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO = "lightseekorg/tokenspeed"
WHL = "lightseekorg/whl"
IDENTITY = "243258330+lightseek-bot@users.noreply.github.com"
STAGES = ("plan", "amd", "kernel", "tokenspeed", "index", "docker", "release")
SCHEDULER_STAGES = ("scheduler-version", "scheduler-publish", "scheduler-dependency")
MLA_STAGES = ("mla-version", "mla-publish", "mla-dependency")
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


def request(url, *, github, data=None, accept="application/json"):
    headers = {"Accept": "application/vnd.github+json" if github else accept}
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


def pages(path, key):
    result = json.loads(
        command("gh", "api", "--paginate", "--slurp", f"repos/{REPO}/{path}")
    )
    return [item for page in result for item in page[key]]


def main_ci_sources(sha):
    sources = {}
    for workflow in pages("actions/workflows?per_page=100", "workflows"):
        path = Path(workflow["path"])
        if (
            workflow["state"] != "active"
            or not path.is_file()
            or path.name.startswith(("release-", "publish-"))
        ):
            continue
        triggers = yaml.load(path.read_text(), Loader=yaml.BaseLoader)["on"]
        if "push" not in triggers:
            continue
        push = triggers["push"] or {}
        if not any(fnmatchcase("main", p) for p in push.get("branches", ["main"])):
            continue
        if any(fnmatchcase("main", p) for p in push.get("branches-ignore", [])):
            continue
        patterns = push.get("paths", [])
        pathspecs = [
            f":(exclude,glob){p[1:]}" if p.startswith("!") else f":(glob){p}"
            for p in patterns
        ]
        pathspecs += [f":(exclude,glob){p}" for p in push.get("paths-ignore", [])]
        source = (
            command(
                "git",
                "log",
                "--first-parent",
                "-1",
                "--format=%H",
                sha,
                "--",
                *pathspecs,
            )
            if pathspecs
            else sha
        )
        if not source or command(
            "git", "diff", "--name-only", source, sha, "--", *pathspecs
        ):
            raise RuntimeError(f"Cannot establish unchanged CI inputs for {path}")
        sources[str(path)] = {"sha": source, "workflow_id": workflow["id"]}
    if ".github/workflows/lint.yml" not in sources:
        raise RuntimeError("The main CI gate requires an active Lint workflow")
    return sources


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
            f'"tokenspeed-kernel>={versions["tokenspeed-kernel"]}.dev0"',
        )


def version_pr_ready(pr):
    if pr["isDraft"] or pr["reviewDecision"] == "CHANGES_REQUESTED":
        raise RuntimeError("Version PR requires manual review")
    if pr["mergeable"] == "CONFLICTING":
        raise RuntimeError("Version PR has merge conflicts")
    return pr["mergeable"] == "MERGEABLE"


def merge_policy():
    """Use an existing, explicit bot exemption; never change repository rules."""
    user_id = int(command("gh", "api", "user", "--jq", ".id"))
    rules = api("rules/branches/main")
    verified = set()
    for rule in rules:
        if (
            rule["ruleset_source"] != REPO
            or rule["ruleset_source_type"] != "Repository"
        ):
            return False
        if rule["ruleset_id"] in verified:
            continue
        actors = api(f"rulesets/{rule['ruleset_id']}")["bypass_actors"]
        if not any(
            actor["actor_type"] == "User"
            and actor["actor_id"] == user_id
            and actor["bypass_mode"] == "always"
            for actor in actors
        ):
            return False
        verified.add(rule["ruleset_id"])
    return True


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
    packages_by_stage = PACKAGES
    pr_files = PR_FILES
    pr_body = "Keep the weekly release ordered so downstream packages require already published component versions.\n"

    def __init__(self, state_path, stage, *, resume_run_id=""):
        self.path = state_path
        self.stage = stage
        self.resume_run_id = resume_run_id
        self.publications_verified = False
        self.page_verified = False
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
        if resume_run_id and (
            not re.fullmatch(r"[1-9][0-9]*", resume_run_id) or stage != "release"
        ):
            raise RuntimeError(
                "Recovery requires an original run ID and the release stage"
            )
        expected_run = resume_run_id or os.environ["GITHUB_RUN_ID"]
        if self.state["run_id"] != expected_run:
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

    def require_main(self, sha):
        remote = command("git", "ls-remote", "origin", "refs/heads/main")
        if not remote or remote.split()[0] != sha:
            raise RuntimeError(
                "Main changed during release; manual intervention required"
            )

    def check_main_ci(self):
        sha = command("git", "rev-parse", "HEAD")
        ci = self.state.get("main_ci")
        if ci is None:
            ci = {"sha": sha, "workflows": main_ci_sources(sha), "validated": False}
            self.state["main_ci"] = ci
            self.save()
        if ci["sha"] != sha:
            raise RuntimeError("Main changed while validating CI; start a new release")
        ci["validated"] = False
        self.save()
        while True:
            self.require_main(sha)
            ready = True
            for path, record in ci["workflows"].items():
                if "id" not in record:
                    runs = pages(
                        f"actions/workflows/{record['workflow_id']}/runs?branch=main&event=push&head_sha={record['sha']}&per_page=100",
                        "workflow_runs",
                    )
                    if not runs:
                        if record["sha"] != sha:
                            raise RuntimeError(
                                f"Missing source CI run for {path}; inspect manually"
                            )
                        ready = False
                        continue
                    record["id"] = max(runs, key=lambda r: r["id"])["id"]
                    self.save()
                run = api(f"actions/runs/{record['id']}")
                if (
                    run["head_sha"] != record["sha"]
                    or run["head_branch"] != "main"
                    or run["event"] != "push"
                    or run["path"].split("@")[0] != path
                ):
                    raise RuntimeError("Main CI run does not match its recorded source")
                if run["status"] != "completed":
                    ready = False
                    continue
                if run["conclusion"] == "success":
                    record["successful_attempt"] = run["run_attempt"]
                    continue
                if run["run_attempt"] >= 2:
                    raise RuntimeError(
                        f"Main CI failed after retry: {path}; inspect manually"
                    )
                retry = record.get("retry")
                if retry is None:
                    retry = {"attempt": run["run_attempt"], "requested": False}
                    record["retry"] = retry
                    self.save()
                    try:
                        command(
                            "gh",
                            "run",
                            "rerun",
                            str(record["id"]),
                            "--repo",
                            REPO,
                            "--failed",
                        )
                    except subprocess.CalledProcessError:
                        current = api(f"actions/runs/{record['id']}")
                        if (
                            current["run_attempt"] <= retry["attempt"]
                            and current["status"] == "completed"
                        ):
                            raise RuntimeError(
                                "CI retry outcome is unknown; inspect the recorded run"
                            )
                    retry["requested"] = True
                    self.save()
                elif not retry["requested"]:
                    raise RuntimeError(
                        "CI retry outcome is unknown; inspect the recorded run"
                    )
                ready = False
            if ready:
                self.require_main(sha)
                ci["validated"] = True
                self.save()
                return
            self.pause()

    def version_base(self, stage):
        ci = self.state.get("main_ci", {})
        if not ci.get("validated"):
            raise RuntimeError("Version merging requires successful main CI")
        previous = {"kernel": "amd", "tokenspeed": "kernel"}.get(stage)
        return self.state["stages"][previous]["sha"] if previous else ci["sha"]

    def update_metadata(self, stage):
        update_metadata(stage, self.state["versions"])

    def pr_branch(self, stage, version):
        return f"bot/weekly-{self.packages_by_stage[stage]}-{version}"

    def pr_title(self, stage, version):
        return f"build: release {self.packages_by_stage[stage]} {version}"

    def verify_version_diff(self, stage, base, head):
        parents = command("git", "rev-list", "--parents", "-n", "1", head).split()
        if parents != [head, base]:
            raise RuntimeError(
                "Version PR must be one commit on the validated main chain"
            )
        command("git", "checkout", "--detach", base)
        options = ("--binary", "--full-index", "--no-ext-diff", "--no-renames")
        try:
            self.update_metadata(stage)
            expected = command("git", "diff", *options, base)
        finally:
            command("git", "reset", "--hard", base)
        actual = command("git", "diff", *options, base, head)
        if not expected or actual != expected:
            raise RuntimeError(
                "Version PR contains changes outside the expected metadata update"
            )

    def fast_forward_version(self, stage, pr):
        phase = self.state["stages"][stage]
        base = self.version_base(stage)
        if phase["base"] != base or pr["headRefOid"] != phase["head"]:
            raise RuntimeError("Version PR source changed outside this release")
        command("git", "fetch", "--no-tags", "origin", phase["head"])
        self.verify_version_diff(stage, base, phase["head"])
        remote = command("git", "ls-remote", "origin", "refs/heads/main").split()[0]
        if remote == phase["head"]:
            # A successful push may precede GitHub's asynchronous merged-state update.
            self.pause()
            return
        self.require_main(base)
        if pr["baseRefOid"] != base:
            raise RuntimeError("Version PR base changed outside this release")
        if not merge_policy():
            raise RuntimeError(
                "Version merging requires an existing explicit bot exemption"
            )
        live = json.loads(
            command(
                "gh",
                "pr",
                "view",
                str(phase["pr"]),
                "--repo",
                REPO,
                "--json",
                "state,headRefOid,baseRefOid,isDraft,reviewDecision,mergeable",
            )
        )
        if live["headRefOid"] != phase["head"] or live["baseRefOid"] != base:
            raise RuntimeError("Version PR source changed before merging")
        if live["state"] != "OPEN" or not version_pr_ready(live):
            self.pause()
            return
        # The single commit is a fast-forward; the explicit lease atomically pins main.
        command(
            "git",
            "push",
            f"--force-with-lease=refs/heads/main:{base}",
            "origin",
            f"{phase['head']}:refs/heads/main",
        )
        self.require_main(phase["head"])

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
        self.check_main_ci()
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
            self.update_metadata(stage)
            if command("git", "diff", "--name-only"):
                raise RuntimeError("Recorded release source has unexpected metadata")
            return phase["sha"]
        self.checkout_main()
        self.gate()
        base = self.version_base(stage)
        main_head = command("git", "rev-parse", "HEAD")
        if main_head not in (base, phase.get("head")):
            raise RuntimeError("Version source differs from the validated main chain")
        self.require_main(main_head)
        if phase.setdefault("base", base) != base:
            raise RuntimeError("Recorded version PR base changed")
        self.save()
        package = self.packages_by_stage[stage]
        version = self.state["versions"][package]
        branch = self.pr_branch(stage, version)
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
            title = self.pr_title(stage, version)
            existing = command("git", "ls-remote", "origin", f"refs/heads/{branch}")
            if existing:
                if existing.split()[0] != phase.get("head"):
                    raise RuntimeError("Version branch was created outside this run")
                command("git", "fetch", "--no-tags", "origin", branch)
                command("git", "checkout", "--detach", "FETCH_HEAD")
            else:
                command("git", "switch", "-c", branch)
                self.update_metadata(stage)
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
                if not changed <= set(self.pr_files[stage]):
                    raise RuntimeError(
                        "Pre-commit changed files outside the version update"
                    )
                command("git", "add", "--", *self.pr_files[stage])
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
            body.write_text(self.pr_body)
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
                    "state,headRefOid,baseRefOid,mergeCommit,isDraft,reviewDecision,mergeable",
                )
            )
            if pr["state"] == "MERGED":
                if (
                    pr["headRefOid"] != phase.get("head")
                    or pr["mergeCommit"]["oid"] != phase["head"]
                ):
                    raise RuntimeError(
                        "Merged version PR differs from its recorded commit"
                    )
                phase["sha"] = pr["mergeCommit"]["oid"]
                self.save()
                command("git", "fetch", "--no-tags", "origin", phase["sha"])
                command("git", "checkout", "--detach", phase["sha"])
                self.gate()
                # Check the merged metadata, including both TokenSpeed version sources.
                self.update_metadata(stage)
                if command("git", "diff", "--name-only"):
                    raise RuntimeError("Merged release PR has unexpected metadata")
                return phase["sha"]
            if pr["state"] != "OPEN":
                raise RuntimeError("Version PR was closed without merging")
            if pr["headRefOid"] != phase.get("head"):
                raise RuntimeError("Version PR head changed outside this run")
            if version_pr_ready(pr):
                self.fast_forward_version(stage, pr)
            else:
                self.pause()

    def immutable_ref(self, stage, sha):
        ref = self.branch(stage, self.state["versions"][self.packages_by_stage[stage]])
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
        self.state.setdefault("release_refs", {})[stage] = {"ref": ref, "sha": sha}
        self.save()
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

    def validate_recovery(self):
        run = api(f"actions/runs/{self.resume_run_id}")
        if (
            run["status"] != "completed"
            or run["path"].split("@")[0] != ".github/workflows/weekly-release.yml"
            or run["head_branch"] != "main"
            or run["head_repository"]["full_name"] != REPO
            or run["event"] not in ("schedule", "workflow_dispatch")
        ):
            raise RuntimeError(
                "Recovery source must be a completed release run on main"
            )
        jobs = json.loads(
            command(
                "gh",
                "api",
                "--paginate",
                "--slurp",
                f"repos/{REPO}/actions/runs/{self.resume_run_id}/jobs?filter=latest&per_page=100",
            )
        )
        results = {
            job["name"]: job["conclusion"] for page in jobs for job in page["jobs"]
        }
        if any(results.get(f"{stage} / stage") != "success" for stage in STAGES[:-1]):
            raise RuntimeError(
                "Release-only recovery requires all publication stages to have succeeded"
            )
        self.state["resumed_by_run_id"] = os.environ["GITHUB_RUN_ID"]

    def verify_publications(self):
        if any(
            not self.state["stages"].get(stage, {}).get("complete")
            for stage in STAGES[:-1]
        ):
            raise RuntimeError(
                "Cannot publish release notes before all destinations succeed"
            )
        expected = (
            ("release-tokenspeed-kernel-amd.yml", "amd", "workflow_dispatch"),
            ("release-tokenspeed-kernel.yml", "kernel", "workflow_dispatch"),
            ("release-tokenspeed-kernel-rocm.yml", "kernel", "workflow_dispatch"),
            ("release-pypi.yml", "tokenspeed", "push"),
            ("publish-release-docker.yml", "tokenspeed", "workflow_dispatch"),
        )
        if set(self.state["runs"]) != {workflow for workflow, _, _ in expected} or set(
            self.state["versions"]
        ) != set(PROJECTS) | {"tokenspeed-kernel"}:
            raise RuntimeError("Recovery state has unexpected packages or publishers")
        for workflow, stage, event in expected:
            version = self.state["versions"][PACKAGES[stage]]
            sha = self.state["stages"][stage]["sha"]
            if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) or not re.fullmatch(
                r"[0-9a-f]{40}", sha
            ):
                raise RuntimeError("Invalid recorded release version or source")
            ref = "main" if event == "push" else self.branch(stage, version)
            record = self.state["runs"][workflow]
            if not isinstance(record.get("id"), int) or record["id"] <= 0:
                raise RuntimeError("Invalid recorded publisher ID")
            if (record["sha"], record["ref"], record["event"]) != (
                sha,
                ref,
                event,
            ) or not record.get("complete"):
                raise RuntimeError("Recorded publication source mismatch")
            run = api(f"actions/runs/{record['id']}")
            if (
                run["status"] != "completed"
                or run["conclusion"] != "success"
                or run["head_sha"] != sha
                or run["head_branch"] != ref
                or run["event"] != event
                or run["actor"]["login"] != "lightseek-bot"
                or run["path"].split("@")[0] != f".github/workflows/{workflow}"
                or run["head_repository"]["full_name"] != REPO
            ):
                raise RuntimeError(
                    "Recorded publisher is incomplete or has another source"
                )
        for stage, workflow in (
            ("amd", expected[0][0]),
            ("kernel", expected[1][0]),
            ("tokenspeed", expected[3][0]),
        ):
            package = PACKAGES[stage]
            version = self.state["versions"][package]
            sha = self.state["stages"][stage]["sha"]
            if source_sha(package, version, workflow) != sha:
                raise RuntimeError("Published PyPI source mismatch")
        versions = self.state["versions"]
        self.wheelhouse(
            f"tokenspeed-kernel-amd-v{versions['tokenspeed-kernel-amd']}",
            self.state["stages"]["amd"]["sha"],
            1,
        )
        self.wheelhouse(
            f"tokenspeed-v{versions['tokenspeed']}",
            self.state["stages"]["tokenspeed"]["sha"],
            1,
        )
        for variant, count in (("cu129", 8), ("cu130", 8), ("rocm72", 4)):
            self.wheelhouse(
                f"tokenspeed-kernel-v{versions['tokenspeed-kernel']}-{variant}",
                self.state["stages"]["kernel"]["sha"],
                count,
            )
        # Reuse the recorded successful Docker run; never dispatch a replacement.
        self.docker()
        self.publications_verified = True

    def notes(self, generated, previous):
        versions = self.state["versions"]
        tag = f"v{versions['tokenspeed']}"
        text = "Biweekly component versions\n\n| Package | Version |\n| --- | --- |\n"
        text += "".join(
            f"| {p} | [{v}](https://pypi.org/project/{p}/{v}/) |\n"
            for p, v in versions.items()
        )
        text += f"\nDocker: [{self.state['stages']['docker']['image']}](https://hub.docker.com/r/lightseekorg/tokenspeed/tags?name={versions['tokenspeed']}) (linux/amd64, linux/arm64).\n\n"
        text += (
            "Stable pip indexes: "
            + ", ".join(
                f"[{v}](https://lightseek.org/whl/{v}/)"
                for v in ("cu129", "cu130", "rocm7.2")
            )
            + ".\n\n"
        )
        tags = (
            f"tokenspeed-kernel-amd-v{versions['tokenspeed-kernel-amd']}",
            f"tokenspeed-v{versions['tokenspeed']}",
            *(
                f"tokenspeed-kernel-v{versions['tokenspeed-kernel']}-{v}"
                for v in ("cu129", "cu130", "rocm72")
            ),
        )
        text += "".join(
            f"- [{t}](https://github.com/{WHL}/releases/tag/{t})\n" for t in tags
        )
        text += "\nPublication runs:\n\n" + "".join(
            f"- [{workflow}](https://github.com/{REPO}/actions/runs/{run['id']})\n"
            for workflow, run in self.state["runs"].items()
        )
        compare = (
            f"https://github.com/{REPO}/compare/{previous}...{tag}"
            if previous
            else f"https://github.com/{REPO}/commits/{tag}"
        )
        footer = f"\n\n**Full Changelog**: {compare}\n"
        omitted = "\n\nRelease notes shortened; see the full changelog for all changes."
        budget = 100000 - len((text + footer + omitted).encode())
        if budget < 0:
            raise RuntimeError("Component release notes exceed the page limit")
        lines = []
        for line in generated.splitlines(keepends=True):
            size = len(line.encode())
            if size > budget:
                break
            lines.append(line)
            budget -= size
        excerpt = "".join(lines)
        return text + excerpt + (omitted if excerpt != generated else "") + footer

    def release(self):
        if any(
            not self.state["stages"].get(stage, {}).get("complete")
            for stage in STAGES[:-1]
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
            if (
                release["draft"]
                or release["prerelease"]
                or not existing
                or (
                    "Biweekly component versions" not in release["body"]
                    and "Weekly component versions" not in release["body"]
                )
            ):
                raise RuntimeError(
                    "Existing release page does not belong to this weekly release"
                )
            self.page_verified = True
            return
        notes = self.path.parent / "release-notes.md"
        candidates = [
            r["tag_name"]
            for r in api("releases?per_page=100")
            if not r["draft"]
            and not r["prerelease"]
            and re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", r["tag_name"])
            and Version(r["tag_name"][1:]) < Version(version)
        ]
        previous = max(candidates, key=lambda t: Version(t[1:])) if candidates else None
        data = {"tag_name": tag, "target_commitish": sha}
        if previous:
            if not command("git", "ls-remote", "origin", f"refs/tags/{previous}"):
                raise RuntimeError("Previous release tag is missing")
            data["previous_tag_name"] = previous
        generated = api("releases/generate-notes", data=data)["body"]
        text = self.notes(generated, previous)
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
            "--notes-file",
            str(notes),
        )
        live = request(
            f"https://api.github.com/repos/{REPO}/releases/tags/{tag}", github=True
        )
        if (
            text.strip() != live["body"].strip()
            or live["draft"]
            or live["prerelease"]
            or command("git", "ls-remote", "origin", f"refs/tags/{tag}").split()[0]
            != sha
        ):
            raise RuntimeError("Release page readback mismatch")
        self.page_verified = True

    def cleanup(self):
        if not self.publications_verified or not self.page_verified:
            raise RuntimeError(
                "Cleanup requires verified publications and release page"
            )
        # Legacy runs did not record refs separately; derive only their exact three refs.
        refs = self.state.setdefault("release_refs", {})
        pending = []
        for stage, package in PACKAGES.items():
            ref = self.branch(stage, self.state["versions"][package])
            sha = self.state["stages"][stage]["sha"]
            record = refs.setdefault(stage, {"ref": ref, "sha": sha})
            if record != {"ref": ref, "sha": sha}:
                raise RuntimeError(
                    "Cleanup ref differs from the recorded release source"
                )
            current = command("git", "ls-remote", "origin", f"refs/heads/{ref}")
            if current and current.split()[0] != sha:
                raise RuntimeError("Cleanup refuses a release branch that moved")
            if current:
                pending.append((ref, sha))
        self.save()
        for ref, sha in pending:
            command(
                "git",
                "push",
                f"--force-with-lease=refs/heads/{ref}:{sha}",
                "origin",
                f":refs/heads/{ref}",
            )
            if command("git", "ls-remote", "origin", f"refs/heads/{ref}"):
                raise RuntimeError("Release branch deletion readback failed")
        self.phase["branches_cleaned"] = True

    def run(self, requested):
        if self.stage == "release":
            self.phase.pop("complete", None)
        self.guard()
        if self.resume_run_id:
            self.validate_recovery()
        if self.stage != "plan":
            previous = STAGES[STAGES.index(self.stage) - 1]
            if not self.state["stages"].get(previous, {}).get("complete"):
                raise RuntimeError("Previous stage did not complete")
        if self.stage == "plan":
            self.plan(requested)
        elif self.stage in PACKAGES:
            self.packages(self.stage)
        elif self.stage == "release":
            self.verify_publications()
            self.release()
            self.cleanup()
        else:
            {"index": self.index, "docker": self.docker}[self.stage]()
        self.phase["complete"] = True
        self.phase.pop("error", None)
        self.save()


class ComponentRelease(Release):
    """Release one component before merging its downstream requirement."""

    package: str
    stages: tuple[str, ...]

    def branch(self, stage, version):
        return f"bot/{self.package.removeprefix('tokenspeed-')}-release-{version}"

    def pr_branch(self, stage, version):
        return f"bot/{stage}-{version}"

    def pr_title(self, stage, version):
        if stage == self.stages[2]:
            return f"build: require {self.package} {version}"
        return super().pr_title(stage, version)

    def version_base(self, stage):
        return self.state["stages"][stage]["base"]

    def update_metadata(self, stage):
        version = self.state["versions"][self.package]
        replace(
            PROJECTS[self.package],
            r'^version = "[^"]+"$',
            f'version = "{version}"',
        )

    def reserve(self, requested):
        if self.state["versions"]:
            return
        self.checkout_main()
        current = read_version(self.package)
        version = next_version(current, latest_version(self.package), requested)
        self.available(version)
        if command(
            "git",
            "ls-remote",
            "origin",
            f"refs/heads/{self.branch(self.stage, version)}",
        ):
            raise RuntimeError(
                f"{self.package} release branch already exists; inspect manually"
            )
        self.phase["base"] = command("git", "rev-parse", "HEAD")
        self.state["initial_versions"] = {self.package: current}
        self.state["versions"] = {self.package: version}
        self.state["initial_dependency"] = str(
            requirements(self.pr_files[self.stages[2]][0])[self.package].specifier
        )
        self.save()

    def wait_index(self, files):
        while True:
            index = request(
                f"https://pypi.org/simple/{self.package}/",
                github=False,
                accept="application/vnd.pypi.simple.v1+json",
            )
            indexed = {f["filename"]: f for f in index["files"]}
            for name, digest in files.items():
                if name in indexed and (
                    indexed[name]["yanked"]
                    or indexed[name]["hashes"].get("sha256") != digest
                ):
                    raise RuntimeError(
                        f"{self.package} index files differ from the verified publication"
                    )
            if files.keys() <= indexed.keys():
                return
            self.pause()

    def gate(self):
        current = read_version(self.package)
        version = self.state["versions"][self.package]
        if self.stage == self.stages[2]:
            if (
                not self.state["stages"].get(self.stages[1], {}).get("complete")
                or current != version
            ):
                raise RuntimeError(
                    f"{self.package} dependency requires a completed publication of the current version"
                )
            check_tree(self.state["stages"][self.stages[0]]["sha"], self.package)
            if self.files() != self.state["publication_files"]:
                raise RuntimeError(
                    f"{self.package} publication changed before the dependency merge"
                )
            self.wait_index(self.state["publication_files"])
        elif current not in (
            self.state["initial_versions"][self.package],
            version,
        ):
            raise RuntimeError(f"{self.package} version changed outside this release")
        return {self.package: current}

    def fast_forward_version(self, stage, pr):
        self.gate()
        super().fast_forward_version(stage, pr)

    def run(self, requested):
        self.phase.pop("complete", None)
        self.guard()
        index = self.stages.index(self.stage)
        if index and not self.state["stages"].get(self.stages[index - 1], {}).get(
            "complete"
        ):
            raise RuntimeError(
                f"Previous {self.package.removeprefix('tokenspeed-')} stage did not complete"
            )
        if self.stage == self.stages[0]:
            self.reserve(requested)
            self.pr(self.stage)
        elif self.stage == self.stages[1]:
            self.publish()
        else:
            self.checkout_main()
            self.gate()
            self.phase.setdefault("base", command("git", "rev-parse", "HEAD"))
            self.save()
            self.pr(self.stage)
        self.phase["complete"] = True
        self.phase.pop("error", None)
        self.save()


class SchedulerRelease(ComponentRelease):
    package = "tokenspeed-scheduler"
    stages = SCHEDULER_STAGES
    packages_by_stage = {
        "scheduler-version": "tokenspeed-scheduler",
        "scheduler-dependency": "tokenspeed-scheduler",
    }
    pr_files = {
        "scheduler-version": [PROJECTS["tokenspeed-scheduler"]],
        "scheduler-dependency": [PROJECTS["tokenspeed"]],
    }
    pr_body = "Keep the scheduler release ordered so TokenSpeed requires an already published version.\n"

    def update_metadata(self, stage):
        if stage == "scheduler-version":
            return super().update_metadata(stage)
        version = self.state["versions"]["tokenspeed-scheduler"]
        specifier = str(
            requirements(PROJECTS["tokenspeed"])["tokenspeed-scheduler"].specifier
        )
        if not re.fullmatch(r">=\d+\.\d+\.\d+", specifier) or Version(
            specifier[2:]
        ) > Version(version):
            raise RuntimeError("Scheduler dependency changed outside this release")
        replace(
            PROJECTS["tokenspeed"],
            r'"tokenspeed-scheduler>=[^"]+"',
            f'"tokenspeed-scheduler>={version}"',
        )

    def available(self, version):
        tag = f"tokenspeed-scheduler-v{version}"
        if pypi("tokenspeed-scheduler", version) is not None or self.release_exists(
            WHL, tag
        ):
            raise RuntimeError(
                "Reserved scheduler version already published; inspect manually"
            )
        try:
            request(
                f"https://api.github.com/repos/{WHL}/git/ref/tags/{tag}", github=True
            )
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        else:
            raise RuntimeError(
                "Reserved scheduler tag already exists; inspect manually"
            )

    def files(self):
        version = self.state["versions"]["tokenspeed-scheduler"]
        release = pypi("tokenspeed-scheduler", version)
        if (
            not release
            or not release["urls"]
            or any(f["yanked"] for f in release["urls"])
        ):
            raise RuntimeError("Scheduler publication is missing or yanked")
        files = {f["filename"]: f["digests"]["sha256"] for f in release["urls"]}
        if (
            len(files) != 9
            or sum(name.endswith(".whl") for name in files) != 8
            or sum(name.endswith(".tar.gz") for name in files) != 1
        ):
            raise RuntimeError(
                "Scheduler publication requires eight wheels and one source distribution"
            )
        return files

    def publish(self):
        workflow = "release-tokenspeed-scheduler.yml"
        version = self.state["versions"]["tokenspeed-scheduler"]
        sha = self.state["stages"]["scheduler-version"]["sha"]
        ref = self.immutable_ref("scheduler-version", sha)
        if not self.state["runs"].get(workflow, {}).get("dispatch_started"):
            self.available(version)
        self.child(
            workflow,
            sha,
            ref,
            {"tag_name": f"tokenspeed-scheduler-v{version}", "prerelease": False},
            event="workflow_dispatch",
        )
        self.published("tokenspeed-scheduler", workflow, sha)
        files = self.files()
        release = request(
            f"https://api.github.com/repos/{WHL}/releases/tags/tokenspeed-scheduler-v{version}",
            github=True,
        )
        if (
            release["draft"]
            or release["prerelease"]
            or {a["name"]: a["digest"] for a in release["assets"]}
            != {name: f"sha256:{digest}" for name, digest in files.items()}
        ):
            raise RuntimeError(
                "Scheduler wheelhouse files differ from the verified PyPI publication"
            )
        self.wait_index(files)
        self.state["publication_files"] = files
        self.save()


class MLARelease(ComponentRelease):
    package = "tokenspeed-mla"
    stages = MLA_STAGES
    packages_by_stage = {
        "mla-version": "tokenspeed-mla",
        "mla-dependency": "tokenspeed-mla",
    }
    pr_files = {
        "mla-version": [PROJECTS["tokenspeed-mla"]],
        "mla-dependency": ["tokenspeed-kernel/python/requirements/cuda-thirdparty.txt"],
    }
    pr_body = (
        "Publish the MLA version before updating the kernel's exact requirement.\n"
    )

    def update_metadata(self, stage):
        if stage == "mla-version":
            return super().update_metadata(stage)
        version = self.state["versions"][self.package]
        path = self.pr_files[stage][0]
        specifier = str(requirements(path)[self.package].specifier)
        if specifier not in (self.state["initial_dependency"], f"=={version}"):
            raise RuntimeError("MLA dependency changed outside this release")
        replace(path, r"^tokenspeed-mla==[^\n]+$", f"tokenspeed-mla=={version}")

    def available(self, version):
        if pypi(self.package, version) is not None:
            raise RuntimeError(
                "Reserved MLA version already published; inspect manually"
            )

    def files(self):
        version = self.state["versions"][self.package]
        release = pypi(self.package, version)
        if not release or any(f["yanked"] for f in release["urls"]):
            raise RuntimeError("MLA publication is missing or yanked")
        files = {f["filename"]: f["digests"]["sha256"] for f in release["urls"]}
        if set(files) != {f"tokenspeed_mla-{version}-py3-none-any.whl"}:
            raise RuntimeError("MLA publication requires exactly one universal wheel")
        return files

    def publish(self):
        workflow = "release-tokenspeed-mla.yml"
        version = self.state["versions"][self.package]
        sha = self.state["stages"]["mla-version"]["sha"]
        ref = self.immutable_ref("mla-version", sha)
        if not self.state["runs"].get(workflow, {}).get("dispatch_started"):
            self.available(version)
        run_id = self.child(workflow, sha, ref, {}, event="workflow_dispatch")
        self.published(self.package, workflow, sha)
        files = self.files()
        if "publication_files" in self.state:
            if files != self.state["publication_files"]:
                raise RuntimeError(
                    "MLA publication changed after artifact verification"
                )
        else:
            dist = self.path.parent / "tokenspeed-mla-dist"
            command(
                "gh",
                "run",
                "download",
                str(run_id),
                "--repo",
                REPO,
                "--name",
                "tokenspeed-mla-dist",
                "--dir",
                str(dist),
            )
            if {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in dist.iterdir()
            } != files:
                raise RuntimeError("MLA artifact differs from published PyPI files")
            self.state["publication_files"] = files
            self.save()
        self.wait_index(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=(*STAGES, *SCHEDULER_STAGES, *MLA_STAGES), required=True
    )
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--version", default="")
    parser.add_argument("--resume-run-id", default="")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(preflight(), indent=2))
        return
    if args.stage in SCHEDULER_STAGES:
        controller = SchedulerRelease
    elif args.stage in MLA_STAGES:
        controller = MLARelease
    else:
        controller = Release
    release = controller(args.state, args.stage, resume_run_id=args.resume_run_id)
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
