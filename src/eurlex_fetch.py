"""
Fetch EUR-Lex documents via CELLAR (publications.europa.eu),
with on-disk HTML caching.

Why CELLAR and not eur-lex.europa.eu:
    eur-lex.europa.eu is fronted by AWS WAF, which serves a JavaScript
    challenge page to non-browser clients (cannot be solved without a
    real browser executing JS). CELLAR is the underlying EU Publications
    Office content repository, exposed specifically for programmatic
    access — no WAF, no JS execution, content negotiation supported.

URL pattern:
    http://publications.europa.eu/resource/celex/<celex>
    Accept:          text/html;notice=branch   → full HTML rendering
    Accept-Language: en                        → English manifestation

Why HTML and not PDF/XML: HTML preserves the original character spans we
need for RapidFuzz `source_quote` traceback, and exposes annex tables
as structured `<table>` nodes.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

CELLAR_URL = "http://publications.europa.eu/resource/celex/{celex}"

DEFAULT_USER_AGENT = (
    "MasterThesis-research/0.1 "
    "(University of Konstanz master's thesis)"
)
DEFAULT_TIMEOUT = 30.0
DEFAULT_RATE_LIMIT_S = 1.0  # be polite — CELLAR doesn't publish a rate cap

# Markers we use to recognise a bad response (WAF challenge / interstitial).
_WAF_MARKERS = ("awswaf", "AwsWafIntegration", "challenge-container")
# Marker we use to recognise a real EUR-Lex/CELLAR document rendering.
# CELLAR HTML usually contains either "oj-doc-ti" (rendered legal text)
# or "FORMEX" structure markers, plus the literal "CELEX" appears in
# metadata blocks.
_DOC_MARKERS = ("oj-doc-ti", "FORMEX", "celex/")


class WAFChallenge(RuntimeError):
    """Raised when the upstream returned an AWS WAF challenge page."""


class NotADocument(RuntimeError):
    """Raised when the upstream returned a non-document page (404, search, etc.)."""


def _cache_path(cache_dir: Path, celex: str) -> Path:
    # Filename = celex + short hash of URL, so URL changes invalidate cache.
    h = hashlib.sha1(CELLAR_URL.format(celex=celex).encode()).hexdigest()[:8]
    return cache_dir / f"{celex}.{h}.html"


def _looks_like_waf(html: str) -> bool:
    return any(m in html for m in _WAF_MARKERS)


def _looks_like_document(html: str) -> bool:
    # Be permissive: any of the markers is enough, and we also accept
    # responses that are simply large enough to plausibly be a document.
    return any(m in html for m in _DOC_MARKERS) or len(html) > 20_000


def fetch_celex(
    celex: str,
    cache_dir: Optional[Path] = None,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = DEFAULT_TIMEOUT,
    rate_limit_s: float = DEFAULT_RATE_LIMIT_S,
    force_refresh: bool = False,
) -> str:
    """
    Fetch the HTML for a single CELEX via CELLAR. Returns the HTML string.

    Caches successful fetches under `cache_dir` (defaults to ./data/raw_html).
    Raises `WAFChallenge` if the upstream returned an anti-bot challenge,
    `NotADocument` if it returned a page that doesn't look like a document.
    """
    if cache_dir is None:
        cache_dir = Path("data/raw_html")
    cache_dir.mkdir(parents=True, exist_ok=True)

    path = _cache_path(cache_dir, celex)
    if path.exists() and not force_refresh:
        logger.debug("Cache hit: %s", path)
        return path.read_text(encoding="utf-8")

    url = CELLAR_URL.format(celex=celex)
    logger.info("Fetching %s", url)

    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html;notice=branch",
        "Accept-Language": "en",
    }
    # CELLAR uses 303 See Other to redirect to the manifestation; let
    # requests follow it automatically.
    resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    body = resp.text

    if _looks_like_waf(body):
        raise WAFChallenge(
            f"CELEX {celex}: upstream returned an AWS WAF challenge page "
            f"(size={len(body)}). The URL or endpoint needs to change."
        )
    if not _looks_like_document(body):
        # Don't cache junk responses — that hides the bug on retry.
        raise NotADocument(
            f"CELEX {celex}: upstream returned a non-document page "
            f"(size={len(body)}). Check that the CELEX number is valid."
        )

    path.write_text(body, encoding="utf-8")
    logger.debug("Cached: %s (%d bytes)", path, len(body))

    if rate_limit_s > 0:
        time.sleep(rate_limit_s)
    return body


def fetch_many(
    celex_list: list[str],
    cache_dir: Optional[Path] = None,
    **kwargs,
) -> dict[str, str]:
    """Fetch a batch; failures are logged and excluded from the result map."""
    out: dict[str, str] = {}
    for celex in celex_list:
        try:
            out[celex] = fetch_celex(celex, cache_dir=cache_dir, **kwargs)
        except Exception as e:
            logger.error("Failed to fetch %s: %s", celex, e)
    return out
