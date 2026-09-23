"""Model-provider routing: LOCAL Ollama vs the DBvis group's HOSTED Ollama.

Background (2026-06-24): the group set up a shared Ollama behind OpenWebUI
(URL in `.env` as OLLAMA_CLOUD_URL; not kept in code). It proxies the *entire* Ollama API under `/ollama`
and authenticates with a per-user API key (Authorization: Bearer <key>). This
lifts the local RTX-3060M / 6 GB ceiling — we can run bigger models remotely.

A model identifier used anywhere in the pipeline/UI may carry an optional
provider prefix:

    qwen3.5:4b            -> LOCAL daemon (no prefix == local; back-compat)
    local:qwen3.5:4b      -> LOCAL daemon (explicit; prefix stripped from id)
    cloud:gemma4:latest   -> CLOUD (DBvis hosted), Bearer-authenticated

`resolve()` turns an identifier into everything the dispatcher needs: the bare
Ollama model tag to put in the request body, the base URL, the auth headers,
and whether the caller may MANAGE (unload) VRAM on that endpoint — we must
never evict models on the shared cloud box, it serves the whole group.

Design notes
------------
* LOCAL ids are returned with NO prefix (an explicit ``local:`` is stripped),
  so on-disk fact dirs, ``annotator`` fields and fact-id hashes stay
  byte-identical to everything produced before this module existed.
* CLOUD ids keep the ``cloud:`` prefix as their stored identity, so a cloud
  run is a *distinct annotator* and writes to its own facts dir
  (``cloud_gemma4_latest``) — no collision with a same-named local model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLOUD_PREFIX = "cloud:"
LOCAL_PREFIX = "local:"


# ---------------------------------------------------------------------------
# .env loading (zero-dependency; does not override already-set env vars)
# ---------------------------------------------------------------------------

def load_dotenv(path: Path | None = None) -> None:
    """Populate os.environ from a KEY=VALUE .env file if present.

    Intentionally tiny (no python-dotenv dependency). Existing environment
    variables win, so an explicit `OLLAMA_CLOUD_API_KEY=… python -m ui.app`
    still overrides the file. Called once at import time below.
    """
    path = path or (_PROJECT_ROOT / ".env")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


load_dotenv()


# ---------------------------------------------------------------------------
# Endpoint configuration (read AFTER load_dotenv so .env is honoured)
# ---------------------------------------------------------------------------

def _local_base_url() -> str:
    return os.environ.get("OLLAMA_URL", "http://localhost:11434")


def _cloud_base_url() -> str:
    # OpenWebUI proxies the Ollama API under /ollama (2026-06-24). The host is
    # deployment config, not code: it lives in .env, never as a default here.
    url = os.environ.get("OLLAMA_CLOUD_URL", "").strip().rstrip("/")
    if not url:
        raise RuntimeError("OLLAMA_CLOUD_URL is not set; copy .env.example to .env "
                           "and fill it in")
    return url


def _cloud_api_key() -> str:
    return os.environ.get("OLLAMA_CLOUD_API_KEY", "").strip()


def cloud_enabled() -> bool:
    """True when a cloud API key is configured (else cloud options are hidden)."""
    return bool(_cloud_api_key())


def local_enabled() -> bool:
    """False on cloud-only deployments. Set OLLAMA_LOCAL_DISABLED=1 (e.g. on the
    lab server) to hide local models from the UI; default keeps them on for dev."""
    return os.environ.get("OLLAMA_LOCAL_DISABLED", "").strip().lower() not in ("1", "true", "yes")


def _cloud_headers() -> dict:
    key = _cloud_api_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


# Static fallbacks used when an endpoint's /api/tags is unreachable at the
# moment the UI asks for the model list. The pipeline still works against any
# tag the user selects; these just seed the dropdowns.
LOCAL_FALLBACK = ("qwen3.5:4b", "gemma3:4b", "nemotron-3-nano:4b")
# The frozen annotator set (scripts/run_phase2.ANNOTATORS_OF_RECORD, without the
# `cloud:` prefix; a test keeps the two equal). The shared box answers /api/tags
# slowly under load, and this list used to be `gemma4:latest` — the 8B model the
# 31B annotator must not be confused with — so a slow answer left the pickers
# without a single model of record to default to (BUG-2026-09-23-G).
CLOUD_FALLBACK = (
    "qwen3.6:35b",
    "hf.co/unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL",
    "hf.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF:UD-Q4_K_M",
)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedModel:
    provider: str                 # "local" | "cloud"
    model_id: str                 # stored identity (cloud keeps prefix; local bare)
    model_tag: str                # bare tag sent to Ollama ("gemma4:latest")
    base_url: str                 # endpoint root, no trailing slash
    headers: dict = field(default_factory=dict)
    manages_vram: bool = True     # may we unload other models on this endpoint?


def resolve(model_id: str, *, default_base_url: str | None = None) -> ResolvedModel:
    """Turn a (possibly prefixed) model identifier into dispatch parameters."""
    raw = (model_id or "").strip()

    if raw.startswith(CLOUD_PREFIX):
        tag = raw[len(CLOUD_PREFIX):].strip()
        return ResolvedModel(
            provider="cloud",
            model_id=raw,                 # keep prefix as the stored identity
            model_tag=tag,
            base_url=_cloud_base_url(),
            headers=_cloud_headers(),
            manages_vram=False,           # NEVER evict on shared group infra
        )

    # Local (default). Strip an explicit local: prefix so the stored identity
    # stays byte-identical to pre-existing on-disk data.
    tag = raw[len(LOCAL_PREFIX):].strip() if raw.startswith(LOCAL_PREFIX) else raw
    return ResolvedModel(
        provider="local",
        model_id=tag,
        model_tag=tag,
        base_url=(default_base_url or _local_base_url()),
        headers={},
        manages_vram=True,
    )


def is_cloud(model_id: str) -> bool:
    return (model_id or "").strip().startswith(CLOUD_PREFIX)


# ---------------------------------------------------------------------------
# Model discovery (for the UI dropdowns)
# ---------------------------------------------------------------------------

def _tags(base_url: str, headers: dict, *, timeout: float) -> list[str]:
    """Names from GET {base_url}/api/tags, or raise on any failure."""
    resp = requests.get(f"{base_url.rstrip('/')}/api/tags", headers=headers, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    return [m["name"] for m in body.get("models", []) if m.get("name")]


def _entries(tags, *, provider: str, prefix: str) -> list[dict]:
    out = []
    for t in tags:
        out.append({"id": f"{prefix}{t}", "tag": t, "provider": provider})
    return out


def list_available_models(*, timeout: float = 2.5) -> dict:
    """Return {"local": [...], "cloud": [...]} for the model pickers.

    Each entry is {id, tag, provider}. `id` is what the UI sends back and what
    the pipeline stores (cloud carries the ``cloud:`` prefix). Live /api/tags is
    attempted per endpoint with a short timeout; on failure we fall back to the
    static lists so the UI is never empty. Cloud is omitted entirely when no API
    key is configured.
    """
    # Local (skipped entirely on cloud-only deployments)
    local: list[dict] = []
    if local_enabled():
        try:
            local_tags = _tags(_local_base_url(), {}, timeout=timeout)
        except Exception:
            local_tags = list(LOCAL_FALLBACK)
        local = _entries(local_tags, provider="local", prefix="")

    # Cloud (only if a key is set)
    cloud: list[dict] = []
    if cloud_enabled():
        try:
            cloud_tags = _tags(_cloud_base_url(), _cloud_headers(), timeout=timeout)
        except Exception:
            cloud_tags = list(CLOUD_FALLBACK)
        cloud = _entries(cloud_tags, provider="cloud", prefix=CLOUD_PREFIX)

    return {"local": local, "cloud": cloud}
