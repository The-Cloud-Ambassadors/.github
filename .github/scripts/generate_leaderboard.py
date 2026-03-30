"""
generate_leaderboard.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generates a modern SVG contributor leaderboard with real profile pictures,
center-aligned layout, and embeds it in profile/README.md.

Requirements: pip install requests tenacity
"""

from __future__ import annotations

import logging
import os
import re
import time
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
ORG: str         = os.environ["ORG_NAME"]
TOKEN: str       = os.environ["ORG_TOKEN"]
TOP_N: int       = int(os.environ.get("TOP_N", "20"))
MAX_WORKERS: int = int(os.environ.get("MAX_WORKERS", "8"))
README_PATH: str = os.environ.get("README_PATH", "profile/README.md")
SVG_PATH: str    = os.environ.get("SVG_PATH",    "profile/leaderboard.svg")
MARKER_START     = "<!-- LEADERBOARD_START -->"
MARKER_END       = "<!-- LEADERBOARD_END -->"

HEADERS: dict[str, str] = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# ── Colours ───────────────────────────────────────────────────────────────────
# (bg, stroke, name_fill, sub_fill, badge_fill, bar_fill)
RANK_STYLES = [
    ("#EEEDFE", "#AFA9EC", "#26215C", "#534AB7", "#7F77DD", "#7F77DD"),  # 🥇 purple
    ("#E1F5EE", "#5DCAA5", "#04342C", "#0F6E56", "#1D9E75", "#1D9E75"),  # 🥈 teal
    ("#FAECE7", "#F0997B", "#4A1B0C", "#993C1D", "#D85A30", "#D85A30"),  # 🥉 coral
]
DEFAULT_STYLE = ("#F0EEE9", "#D3D1C7", "#2C2C2A", "#888780", "#B4B2A9", "#B4B2A9")
MEDALS = ["🥇", "🥈", "🥉"]

SVG_W = 760   # total canvas width
COL_X = 30    # left edge of content column
COL_W = 700   # content column width


# ── Data model ────────────────────────────────────────────────────────────────
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

    @property
    def initials(self) -> str:
        parts = re.split(r"[-_]", self.login)
        if len(parts) >= 2:
            return (parts[0][0] + parts[1][0]).upper()
        return self.login[:2].upper()


# ── GitHub API ────────────────────────────────────────────────────────────────
def _check_rate_limit(resp: requests.Response) -> None:
    remaining = int(resp.headers.get("X-RateLimit-Remaining", 9999))
    reset_at  = int(resp.headers.get("X-RateLimit-Reset", 0))
    if remaining < 50:
        wait = max(0, reset_at - int(time.time())) + 5
        log.warning("Rate limit low (%d left). Sleeping %ds.", remaining, wait)
        time.sleep(wait)


@retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def _get(url: str, params: Optional[dict] = None) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
    _check_rate_limit(resp)
    resp.raise_for_status()
    return resp


def paginate(url: str, params: Optional[dict] = None) -> list[dict]:
    results, page = [], 1
    base = {**(params or {}), "per_page": 100}
    while True:
        try:
            resp = _get(url, {**base, "page": page})
            data = resp.json()
        except requests.HTTPError as exc:
            log.warning("HTTP %s on %s — skipping.", exc.response.status_code, url)
            break
        if not data:
            break
        results.extend(data)
        if "next" not in resp.headers.get("Link", ""):
            break
        page += 1
    return results


def fetch_org_repos() -> list[str]:
    log.info("Fetching repos for org: %s", ORG)
    repos = paginate(f"https://api.github.com/orgs/{ORG}/repos", {"type": "all"})
    names = [r["name"] for r in repos if not r.get("fork")]
    log.info("Found %d repos (forks excluded).", len(names))
    return names


def fetch_repo_contributors(repo: str) -> dict[str, dict]:
    data = paginate(
        f"https://api.github.com/repos/{ORG}/{repo}/contributors",
        {"anon": "false"},
    )
    out: dict[str, dict] = {}
    for c in data:
        if c.get("type") != "User":
            continue
        out[c["login"]] = {
            "commits":    c["contributions"],
            "avatar_url": c.get("avatar_url", ""),
            "html_url":   c.get("html_url", f"https://github.com/{c['login']}"),
        }
    return out


def aggregate(repos: list[str]) -> dict[str, Contributor]:
    totals: dict[str, Contributor] = {}

    def _process(repo: str) -> tuple[str, dict]:
        log.info("  → %s", repo)
        return repo, fetch_repo_contributors(repo)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_process, r): r for r in repos}
        for future in as_completed(futures):
            repo = futures[future]
            try:
                _, contributors = future.result()
            except Exception as exc:
                if "Expecting value" in str(exc):
                    log.debug("Repo '%s' empty/archived — skipped.", repo)
                else:
                    log.error("Repo '%s' failed: %s", repo, exc)
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


# ── Avatar fetching (base64 embed) ────────────────────────────────────────────
def fetch_avatar_b64(url: str, size: int = 80) -> str:
    """
    Download a GitHub avatar and return a base64 data URI string.
    Falls back to empty string on any error so the SVG still renders.
    """
    if not url:
        return ""
    try:
        resp = requests.get(f"{url}&s={size}", timeout=10, headers={"User-Agent": "leaderboard-bot"})
        resp.raise_for_status()
        mime = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        import base64
        b64 = base64.b64encode(resp.content).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception as exc:
        log.warning("Avatar fetch failed for %s: %s", url, exc)
        return ""


def prefetch_avatars(ranked: list["Contributor"]) -> dict[str, str]:
    """Fetch all avatars concurrently. Returns { avatar_url: data_uri }."""
    log.info("Prefetching %d avatars...", len(ranked))
    results: dict[str, str] = {}

    def _fetch(c: "Contributor") -> tuple[str, str]:
        return c.avatar_url, fetch_avatar_b64(c.avatar_url)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch, c): c for c in ranked}
        for future in as_completed(futures):
            try:
                url, data_uri = future.result()
                results[url] = data_uri
            except Exception as exc:
                log.warning("Avatar prefetch error: %s", exc)

    log.info("Avatars fetched: %d/%d succeeded.", sum(1 for v in results.values() if v), len(ranked))
    return results


# ── SVG builder ───────────────────────────────────────────────────────────────
def _x(text: str) -> str:
    """XML-escape a string."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _progress_bar(x: int, y: int, w: int, pct: float, color: str) -> str:
    filled = max(4, int(w * pct))
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="5" rx="2.5" fill="#E0DEDB"/>'
        f'<rect x="{x}" y="{y}" width="{filled}" height="5" rx="2.5" '
        f'fill="{color}" opacity="0.75"/>'
    )


def build_svg(ranked: list[Contributor], total_contributors: int, total_repos: int, avatar_map: Optional[dict] = None) -> str:
    now       = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    max_c     = ranked[0].total_commits if ranked else 1
    av        = avatar_map or {}   # { avatar_url: base64_data_uri }

    # ── Layout ────────────────────────────────────────────────────────────────
    HEADER_H  = 76
    TOP3_H    = 68
    TOP3_GAP  = 8
    DIV_H     = 28
    ROW_H     = 48
    ROW_GAP   = 5
    FOOTER_H  = 48
    PAD_V     = 20

    n_top3 = min(3, len(ranked))
    n_rest = max(0, len(ranked) - 3)

    total_h = (
        PAD_V
        + HEADER_H
        + n_top3 * TOP3_H + max(0, n_top3 - 1) * TOP3_GAP
        + (DIV_H if n_rest > 0 else 0)
        + n_rest * (ROW_H + ROW_GAP)
        + FOOTER_H
        + PAD_V
    )

    L: list[str] = []

    # ── Open SVG ──────────────────────────────────────────────────────────────
    L.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{SVG_W}" height="{total_h}" viewBox="0 0 {SVG_W} {total_h}">'
    )

    # Canvas background
    L.append(
        f'<rect width="{SVG_W}" height="{total_h}" rx="16" '
        f'fill="#FAFAF8" stroke="#E8E6E1" stroke-width="1"/>'
    )

    # ── Clip paths for circular avatars ───────────────────────────────────────
    L.append("<defs>")
    yc = PAD_V + HEADER_H
    for i in range(n_top3):
        cy = yc + TOP3_H // 2
        L.append(f'<clipPath id="av{i}"><circle cx="{COL_X + 54}" cy="{cy}" r="22"/></clipPath>')
        yc += TOP3_H + TOP3_GAP
    if n_rest > 0:
        yc += DIV_H
    for i in range(n_rest):
        cy = yc + ROW_H // 2
        L.append(f'<clipPath id="av{3+i}"><circle cx="{COL_X + 47}" cy="{cy}" r="15"/></clipPath>')
        yc += ROW_H + ROW_GAP
    L.append("</defs>")

    # ── Header ────────────────────────────────────────────────────────────────
    hy = PAD_V
    L.append(
        f'<rect x="{COL_X}" y="{hy}" width="{COL_W}" height="{HEADER_H - 8}" rx="12" '
        f'fill="#7F77DD" opacity="0.09"/>'
    )
    cx = SVG_W // 2
    L.append(
        f'<text x="{cx}" y="{hy + 30}" text-anchor="middle" '
        f'font-family="system-ui,-apple-system,sans-serif" font-size="18" '
        f'font-weight="700" fill="#3C3489">🏆  Contributor Leaderboard</text>'
    )
    L.append(
        f'<text x="{cx}" y="{hy + 50}" text-anchor="middle" '
        f'font-family="system-ui,-apple-system,sans-serif" font-size="11" fill="#7F77DD">'
        f'🤖 Auto-generated by GitHub Actions  ·  🕒 Last updated: {_x(now)}</text>'
    )
    L.append(
        f'<text x="{cx}" y="{hy + 66}" text-anchor="middle" '
        f'font-family="system-ui,-apple-system,sans-serif" font-size="11" fill="#7F77DD">'
        f'Showing top {len(ranked)} contributors  ·  {total_repos} repos  ·  '
        f'{total_contributors} total contributors</text>'
    )

    # ── Top 3 podium cards ────────────────────────────────────────────────────
    yc = PAD_V + HEADER_H
    for i in range(n_top3):
        c  = ranked[i]
        bg, stroke, name_col, sub_col, badge_col, bar_col = RANK_STYLES[i]
        cy = yc + TOP3_H // 2
        pct = c.total_commits / max_c

        # Card background
        L.append(
            f'<rect x="{COL_X}" y="{yc}" width="{COL_W}" height="{TOP3_H}" rx="12" '
            f'fill="{bg}" stroke="{stroke}" stroke-width="1.2"/>'
        )

        # Medal emoji
        L.append(
            f'<text x="{COL_X + 16}" y="{cy + 8}" '
            f'font-family="system-ui,-apple-system,sans-serif" font-size="22">'
            f'{MEDALS[i]}</text>'
        )

        # Avatar circle shadow
        L.append(
            f'<circle cx="{COL_X + 54}" cy="{cy}" r="23" '
            f'fill="{badge_col}" opacity="0.2"/>'
        )
        # Real GitHub profile picture (base64 embedded — GitHub strips external hrefs in SVGs)
        data_uri = av.get(c.avatar_url, "")
        if data_uri:
            L.append(
                f'<image href="{data_uri}" '
                f'x="{COL_X + 32}" y="{cy - 22}" width="44" height="44" '
                f'clip-path="url(#av{i})" preserveAspectRatio="xMidYMid slice"/>'
            )
        else:
            L.append(
                f'<text x="{COL_X + 54}" y="{cy + 1}" text-anchor="middle" '
                f'dominant-baseline="middle" font-family="system-ui,sans-serif" '
                f'font-size="14" font-weight="700" fill="{badge_col}">'
                f'{_x(c.initials)}</text>'
            )

        # Username (clickable link)
        L.append(
            f'<a href="{_x(c.profile_url)}" target="_blank">'
            f'<text x="{COL_X + 90}" y="{cy - 10}" '
            f'font-family="system-ui,-apple-system,sans-serif" '
            f'font-size="15" font-weight="700" fill="{name_col}">'
            f'@{_x(c.login)}</text></a>'
        )
        # Repos contributed label
        L.append(
            f'<text x="{COL_X + 90}" y="{cy + 9}" '
            f'font-family="system-ui,-apple-system,sans-serif" '
            f'font-size="11" fill="{sub_col}">'
            f'{c.repo_count} repos contributed</text>'
        )
        # Progress bar
        L.append(_progress_bar(COL_X + 90, cy + 20, 230, pct, bar_col))

        # Commit count badge (right side)
        bx = COL_X + COL_W - 105
        L.append(
            f'<rect x="{bx}" y="{cy - 20}" width="92" height="40" rx="10" '
            f'fill="{badge_col}" opacity="0.14"/>'
        )
        L.append(
            f'<text x="{bx + 46}" y="{cy - 2}" text-anchor="middle" '
            f'font-family="system-ui,-apple-system,sans-serif" '
            f'font-size="20" font-weight="800" fill="{name_col}">'
            f'{c.total_commits:,}</text>'
        )
        L.append(
            f'<text x="{bx + 46}" y="{cy + 15}" text-anchor="middle" '
            f'font-family="system-ui,-apple-system,sans-serif" '
            f'font-size="10" fill="{sub_col}">commits</text>'
        )

        yc += TOP3_H + TOP3_GAP

    # ── Divider ───────────────────────────────────────────────────────────────
    if n_rest > 0:
        dy = yc + DIV_H // 2
        L.append(
            f'<line x1="{COL_X + 16}" y1="{dy}" x2="{COL_X + COL_W - 16}" y2="{dy}" '
            f'stroke="#D3D1C7" stroke-width="1"/>'
        )
        yc += DIV_H

    # ── Rank rows #4 onward ───────────────────────────────────────────────────
    for i, c in enumerate(ranked[3:], start=3):
        bg, stroke, name_col, sub_col, badge_col, bar_col = DEFAULT_STYLE
        row_bg = "#F3F1ED" if i % 2 == 0 else "#FAFAF8"
        cy     = yc + ROW_H // 2
        pct    = c.total_commits / max_c

        # Row background
        L.append(
            f'<rect x="{COL_X}" y="{yc}" width="{COL_W}" height="{ROW_H}" '
            f'rx="8" fill="{row_bg}"/>'
        )

        # Rank number
        L.append(
            f'<text x="{COL_X + 20}" y="{cy + 1}" text-anchor="middle" '
            f'dominant-baseline="middle" font-family="system-ui,sans-serif" '
            f'font-size="11" font-weight="600" fill="#888780">#{i + 1}</text>'
        )

        # Avatar circle
        L.append(
            f'<circle cx="{COL_X + 47}" cy="{cy}" r="16" fill="#D3D1C7" opacity="0.45"/>'
        )
        data_uri = av.get(c.avatar_url, "")
        if data_uri:
            L.append(
                f'<image href="{data_uri}" '
                f'x="{COL_X + 32}" y="{cy - 15}" width="30" height="30" '
                f'clip-path="url(#av{i})" preserveAspectRatio="xMidYMid slice"/>'
            )
        else:
            L.append(
                f'<text x="{COL_X + 47}" y="{cy + 1}" text-anchor="middle" '
                f'dominant-baseline="middle" font-family="system-ui,sans-serif" '
                f'font-size="10" font-weight="600" fill="#888780">'
                f'{_x(c.initials)}</text>'
            )

        # Username + repos
        L.append(
            f'<a href="{_x(c.profile_url)}" target="_blank">'
            f'<text x="{COL_X + 72}" y="{cy - 6}" '
            f'font-family="system-ui,-apple-system,sans-serif" '
            f'font-size="13" font-weight="600" fill="{name_col}">'
            f'@{_x(c.login)}</text></a>'
        )
        L.append(
            f'<text x="{COL_X + 72}" y="{cy + 10}" '
            f'font-family="system-ui,-apple-system,sans-serif" '
            f'font-size="10" fill="{sub_col}">'
            f'{c.repo_count} repos</text>'
        )

        # Progress bar
        L.append(_progress_bar(COL_X + 72, cy + 20, 190, pct, bar_col))

        # Commit count
        L.append(
            f'<text x="{COL_X + COL_W - 12}" y="{cy - 5}" text-anchor="end" '
            f'font-family="system-ui,-apple-system,sans-serif" '
            f'font-size="15" font-weight="700" fill="{name_col}">'
            f'{c.total_commits:,}</text>'
        )
        L.append(
            f'<text x="{COL_X + COL_W - 12}" y="{cy + 12}" text-anchor="end" '
            f'font-family="system-ui,-apple-system,sans-serif" '
            f'font-size="10" fill="{sub_col}">commits</text>'
        )

        yc += ROW_H + ROW_GAP

    # ── Footer ────────────────────────────────────────────────────────────────
    fy = yc + 16
    L.append(
        f'<text x="{SVG_W // 2}" y="{fy}" text-anchor="middle" '
        f'font-family="system-ui,-apple-system,sans-serif" font-size="10" fill="#C4C2BC">'
        f'Only human contributors counted  ·  Forks excluded  ·  Bots filtered</text>'
    )
    L.append(
        f'<text x="{SVG_W // 2}" y="{fy + 18}" text-anchor="middle" '
        f'font-family="system-ui,-apple-system,sans-serif" font-size="10" fill="#C4C2BC">'
        f'Refreshes every Monday at midnight UTC via GitHub Actions</text>'
    )

    L.append("</svg>")
    return "\n".join(L)


# ── README patch ──────────────────────────────────────────────────────────────
def update_readme(svg_filename: str) -> None:
    try:
        with open(README_PATH, encoding="utf-8") as fh:
            content = fh.read()
    except FileNotFoundError:
        log.warning("%s not found — creating new.", README_PATH)
        content = f"# {ORG}\n\n"

    image_block = (
        f'<p align="center">\n'
        f'  <img src="{svg_filename}" alt="Contributor Leaderboard" width="760"/>\n'
        f'</p>'
    )
    block = f"{MARKER_START}\n{image_block}\n{MARKER_END}"

    if MARKER_START in content and MARKER_END in content:
        content = re.sub(
            rf"{re.escape(MARKER_START)}.*?{re.escape(MARKER_END)}",
            block, content, flags=re.DOTALL,
        )
        log.info("Replaced leaderboard block in %s.", README_PATH)
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
    contributors = aggregate(repos)
    if not contributors:
        log.error("No contributors found. Check token permissions.")
        raise SystemExit(1)

    log.info("Total unique contributors: %d", len(contributors))
    ranked = sorted(contributors.values(), key=lambda c: c.total_commits, reverse=True)[:TOP_N]

    avatar_map = prefetch_avatars(ranked)
    svg_content = build_svg(ranked, len(contributors), len(repos), avatar_map)

    os.makedirs(os.path.dirname(SVG_PATH) or ".", exist_ok=True)
    with open(SVG_PATH, "w", encoding="utf-8") as fh:
        fh.write(svg_content)
    log.info("SVG written → %s", SVG_PATH)

    update_readme(os.path.basename(SVG_PATH))
    log.info("✅ Done! leaderboard.svg and README.md updated.")


if __name__ == "__main__":
    main()
