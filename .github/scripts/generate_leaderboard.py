"""
generate_leaderboard.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generates a contributor leaderboard for a GitHub organisation
and writes it into README.md between marker comments.

Requirements: pip install requests tenacity rich
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
ORG: str = os.environ["ORG_NAME"]
TOKEN: str = os.environ["ORG_TOKEN"]
TOP_N: int = int(os.environ.get("TOP_N", "20"))
MAX_WORKERS: int = int(os.environ.get("MAX_WORKERS", "8"))
README_PATH: str = os.environ.get("README_PATH", "README.md")
MARKER_START: str = "<!-- LEADERBOARD_START -->"
MARKER_END: str = "<!-- LEADERBOARD_END -->"

HEADERS: dict[str, str] = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class Contributor:
    login: str
    total_commits: int = 0
    repos: list[str] = field(default_factory=list)
    avatar_url: str = ""
    profile_url: str = ""

    @property
    def repo_count(self) -> int:
        return len(self.repos)


# ── GitHub API helpers ────────────────────────────────────────────────────────
def _check_rate_limit(response: requests.Response) -> None:
    """Pause if we are close to hitting the GitHub rate limit."""
    remaining = int(response.headers.get("X-RateLimit-Remaining", 9999))
    reset_at = int(response.headers.get("X-RateLimit-Reset", 0))

    if remaining < 50:
        wait = max(0, reset_at - int(time.time())) + 5
        log.warning(
            "Rate limit low (%d remaining). Sleeping %ds until reset.", remaining, wait
        )
        time.sleep(wait)


@retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def _get(url: str, params: Optional[dict] = None) -> requests.Response:
    """GET with automatic retries on network errors."""
    response = requests.get(url, headers=HEADERS, params=params, timeout=20)
    _check_rate_limit(response)
    response.raise_for_status()
    return response


def paginate(url: str, params: Optional[dict] = None) -> list[dict]:
    """Fetch all pages from a GitHub list endpoint."""
    results: list[dict] = []
    page = 1
    base_params = {**(params or {}), "per_page": 100}

    while True:
        try:
            resp = _get(url, {**base_params, "page": page})
            data = resp.json()
        except requests.HTTPError as exc:
            log.warning("HTTP %s on %s — skipping.", exc.response.status_code, url)
            break

        if not data:
            break

        results.extend(data)

        # Use GitHub's Link header for reliable pagination detection
        if "next" not in resp.headers.get("Link", ""):
            break

        page += 1

    return results


# ── Core logic ────────────────────────────────────────────────────────────────
def fetch_org_repos() -> list[str]:
    """Return all non-forked repo names in the org."""
    log.info("Fetching repositories for org: %s", ORG)
    repos = paginate(
        f"https://api.github.com/orgs/{ORG}/repos",
        {"type": "all"},
    )
    names = [r["name"] for r in repos if not r.get("fork")]
    log.info("Found %d repositories (forks excluded).", len(names))
    return names


def fetch_repo_contributors(repo: str) -> dict[str, dict]:
    """
    Return contributor data for one repo.
    Returns { login: { commits, avatar_url, profile_url } }
    """
    data = paginate(
        f"https://api.github.com/repos/{ORG}/{repo}/contributors",
        {"anon": "false"},
    )
    contributors: dict[str, dict] = {}
    for c in data:
        if c.get("type") != "User":   # skip bots / anonymous
            continue
        contributors[c["login"]] = {
            "commits": c["contributions"],
            "avatar_url": c.get("avatar_url", ""),
            "html_url": c.get("html_url", f"https://github.com/{c['login']}"),
        }
    return contributors


def aggregate_contributions(repos: list[str]) -> dict[str, Contributor]:
    """
    Fetch contributor data concurrently and merge across all repos.
    """
    totals: dict[str, Contributor] = {}

    def process(repo: str) -> tuple[str, dict[str, dict]]:
        log.info("  → %s", repo)
        return repo, fetch_repo_contributors(repo)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process, r): r for r in repos}

        for future in as_completed(futures):
            repo = futures[future]
            try:
                _, contributors = future.result()
            except Exception as exc:  # noqa: BLE001
                log.error("Failed to process repo '%s': %s", repo, exc)
                continue

            for login, info in contributors.items():
                if login not in totals:
                    totals[login] = Contributor(
                        login=login,
                        avatar_url=info["avatar_url"],
                        profile_url=info["html_url"],
                    )
                totals[login].total_commits += info["commits"]
                totals[login].repos.append(repo)

    return totals


# ── Leaderboard rendering ─────────────────────────────────────────────────────
MEDALS = {0: "🥇", 1: "🥈", 2: "🥉"}


def render_leaderboard(contributors: dict[str, Contributor]) -> str:
    """Build the Markdown leaderboard block."""
    ranked = sorted(
        contributors.values(), key=lambda c: c.total_commits, reverse=True
    )[:TOP_N]

    now = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")

    rows: list[str] = []
    for i, c in enumerate(ranked):
        medal = MEDALS.get(i, f"`#{i + 1}`")
        avatar = (
            f'<img src="{c.avatar_url}&s=20" width="20" height="20" '
            f'style="border-radius:50%" alt="{c.login}"/>'
            if c.avatar_url
            else ""
        )
        user_cell = f'{avatar} [@{c.login}]({c.profile_url})'
        rows.append(
            f"| {medal} | {user_cell} | {c.total_commits:,} | {c.repo_count} |"
        )

    table_rows = "\n".join(rows)

    return f"""\
## 🏆 Contributor Leaderboard

> 🤖 Auto-generated by GitHub Actions &nbsp;|&nbsp; 🕒 Last updated: **{now}**  
> Showing top **{len(ranked)}** contributors by total commits across all org repositories.

| Rank | Contributor | Total Commits | Repos |
|:----:|:-----------|:-------------:|:-----:|
{table_rows}

<sub>Only human contributors are counted. Forked repositories are excluded.</sub>
"""


# ── README patching ───────────────────────────────────────────────────────────
def update_readme(leaderboard_md: str) -> None:
    """Inject the leaderboard into README.md between marker comments."""
    try:
        with open(README_PATH, encoding="utf-8") as fh:
            content = fh.read()
    except FileNotFoundError:
        log.warning("%s not found — creating a new one.", README_PATH)
        content = f"# {ORG}\n\n"

    block = f"{MARKER_START}\n{leaderboard_md}\n{MARKER_END}"

    if MARKER_START in content and MARKER_END in content:
        content = re.sub(
            rf"{re.escape(MARKER_START)}.*?{re.escape(MARKER_END)}",
            block,
            content,
            flags=re.DOTALL,
        )
        log.info("Replaced existing leaderboard block in %s.", README_PATH)
    else:
        content += f"\n{block}\n"
        log.info("Appended leaderboard block to %s.", README_PATH)

    with open(README_PATH, "w", encoding="utf-8") as fh:
        fh.write(content)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    log.info("━━━ Contributor Leaderboard Generator ━━━")
    log.info("Org: %s | Top N: %d | Workers: %d", ORG, TOP_N, MAX_WORKERS)

    repos = fetch_org_repos()

    if not repos:
        log.error("No repositories found. Check token permissions.")
        raise SystemExit(1)

    log.info("Aggregating contributions across %d repos...", len(repos))
    contributors = aggregate_contributions(repos)

    if not contributors:
        log.error("No contributor data returned. Check token permissions.")
        raise SystemExit(1)

    log.info("Total unique contributors found: %d", len(contributors))

    leaderboard_md = render_leaderboard(contributors)
    update_readme(leaderboard_md)

    log.info("✅ Done! README.md updated successfully.")


if __name__ == "__main__":
    main()
