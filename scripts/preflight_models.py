"""Check that a model can actually do the job, before a run depends on it.

Reachable is not the same as usable. `cloud:qwen3.6:35b` was reachable on
2026-07-02 and returned an empty `response` to every `/api/generate` with
`format=json` — 7/7 on arbitration, and the same failure ended a Phase-2 run
with 25/25 escalated pairs unresolved. That cost a run to discover. This script
discovers it in one call per model.

Four probes per model, in the order they can fail:

    tags        the model is installed on the endpoint at all
    generate    a plain call returns a non-empty body
    json        `format=json` + `think=False` returns non-empty, parseable JSON
                — the exact regime `extractor._generate_json` and
                `conflict_layer2._chat_once` use on their first attempt
    fallback    free-text + tolerant extraction, the retry path Layer-2 takes
                when `json` comes back empty

A model that passes `tags` and `generate` but fails `json` is the qwen3.6
failure mode: usable only through the fallback path, and worth reporting to
whoever administers the box, because it usually means the Ollama runtime
predates the `think` field.

    python -m scripts.preflight_models                      # everything visible
    python -m scripts.preflight_models --models cloud:gemma4:latest,qwen3.5:4b
    python -m scripts.preflight_models --require cloud:gemma4:latest

Exit code is non-zero if any `--require`d model fails a probe, so this can gate
a batch run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

from src import providers  # noqa: E402

# Short, extraction-shaped: a model that can only echo will still fail it.
PROBE_PROMPT = (
    "Read this clause and return ONE fact as JSON.\n\n"
    "Clause: \"The Commission shall fix the maximum export refund by 30 June 2004.\"\n\n"
    'Return strictly: {"subject": "...", "predicate": "...", "object": "..."}'
)
REQUIRED_KEYS = {"subject", "predicate", "object"}


def _generate(r: providers.ResolvedModel, *, format_json: bool,
              timeout: float) -> tuple[str, float, str]:
    """Returns (text, seconds, channel); channel is "response" or "thinking".

    Mirrors the guard in `conflict_layer2._chat_once`: on a reasoning model the
    answer goes to the THINKING channel and `response` comes back "". Reading
    only `response` here reported a working model as broken — that was the
    pre-2026-08-09 diagnosis, and this gate would otherwise still block a run
    on a fault the pipeline already handles.
    """
    body = {"model": r.model_tag, "prompt": PROBE_PROMPT, "stream": False,
            "think": False, "options": {"temperature": 0.1, "num_ctx": 4096}}
    if format_json:
        body["format"] = "json"
    t0 = time.monotonic()
    resp = requests.post(f"{r.base_url}/api/generate", json=body,
                         timeout=timeout, headers=r.headers)
    resp.raise_for_status()
    payload = resp.json()
    text = payload.get("response") or ""
    channel = "response"
    if not text.strip():
        text = payload.get("thinking") or ""
        if text.strip():
            channel = "thinking"
    return text, time.monotonic() - t0, channel


def _parse_ok(raw: str) -> bool:
    from src.conflict_layer2 import _extract_json_obj
    obj = _extract_json_obj(raw)
    return isinstance(obj, dict) and bool(REQUIRED_KEYS & set(obj))


def probe(model_id: str, *, timeout: float = 120.0, installed: set[str] | None = None) -> dict:
    r = providers.resolve(model_id)
    out = {"model_id": model_id, "provider": r.provider, "endpoint": r.base_url,
           "tags": None, "generate": None, "json": None, "fallback": None,
           "seconds": None, "json_channel": None, "note": ""}

    if installed is not None:
        out["tags"] = r.model_tag in installed
        if not out["tags"]:
            out["note"] = "not installed on this endpoint"
            return out

    try:
        raw, dt, _ = _generate(r, format_json=False, timeout=timeout)
        out["generate"] = bool(raw.strip())
        out["seconds"] = round(dt, 1)
        out["fallback"] = _parse_ok(raw)
    except Exception as exc:
        out["generate"] = False
        out["note"] = f"{type(exc).__name__}: {exc}"[:160]
        return out

    try:
        raw, _, channel = _generate(r, format_json=True, timeout=timeout)
        out["json"] = bool(raw.strip()) and _parse_ok(raw)
        out["json_channel"] = channel
        if not raw.strip():
            out["note"] = ("empty under format=json on BOTH the response and "
                           "thinking channels — this is NOT the 2026-08-09 routing "
                           "bug; usable only via the Layer-2 fallback path")
        elif channel == "thinking":
            out["note"] = ("answer arrived on `thinking`, not `response` — "
                           "conflict_layer2._chat_once recovers this, so the "
                           "model is usable as-is")
    except Exception as exc:
        out["json"] = False
        out["note"] = out["note"] or f"{type(exc).__name__}: {exc}"[:160]
    return out


def installed_tags(provider: str, *, timeout: float = 10.0) -> tuple[set[str] | None, str]:
    """Tags on one endpoint, or (None, why) when the endpoint cannot be listed."""
    if provider == "cloud":
        if not providers.cloud_enabled():
            return None, "no OLLAMA_CLOUD_API_KEY set"
        r = providers.resolve("cloud:_")
    else:
        r = providers.resolve("_")
    try:
        resp = requests.get(f"{r.base_url}/api/tags", timeout=timeout, headers=r.headers)
        if resp.status_code == 401:
            return None, "401 Unauthorized — the API key is expired or invalid"
        resp.raise_for_status()
        return {m["name"] for m in resp.json().get("models", [])}, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:120]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=None,
                    help="comma-separated model ids; default is everything both "
                         "endpoints report")
    ap.add_argument("--require", default=None,
                    help="comma-separated model ids that must pass; exit 1 otherwise")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", type=Path, default=None, help="write the result as JSON")
    args = ap.parse_args()

    providers.load_dotenv()
    endpoints: dict[str, set[str] | None] = {}
    for provider in ("local", "cloud"):
        tags, why = installed_tags(provider, timeout=args.timeout)
        endpoints[provider] = tags
        label = f"{provider:5s}"
        if tags is None:
            print(f"  {label} UNREACHABLE — {why}")
        else:
            print(f"  {label} {len(tags)} model(s): {', '.join(sorted(tags)) or '(none)'}")
    print()

    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        wanted = [t for t in sorted(endpoints["local"] or ())]
        wanted += [f"cloud:{t}" for t in sorted(endpoints["cloud"] or ())]
    required = {m.strip() for m in (args.require or "").split(",") if m.strip()}
    wanted = sorted(set(wanted) | required)

    if not wanted:
        raise SystemExit("no models to probe: both endpoints are unreachable and "
                         "--models was not given")

    rows = []
    print(f"{'model':28s} {'tags':>5s} {'gen':>5s} {'json':>5s} {'fallbk':>7s} {'sec':>6s}  note")
    for model_id in wanted:
        provider = "cloud" if providers.is_cloud(model_id) else "local"
        row = probe(model_id, timeout=args.timeout, installed=endpoints[provider])
        rows.append(row)
        mark = lambda v: "-" if v is None else ("ok" if v else "FAIL")  # noqa: E731
        print(f"{model_id:28s} {mark(row['tags']):>5s} {mark(row['generate']):>5s} "
              f"{mark(row['json']):>5s} {mark(row['fallback']):>7s} "
              f"{(row['seconds'] if row['seconds'] is not None else ''):>6} "
              f" {row['note']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    if required:
        bad = [r["model_id"] for r in rows
               if r["model_id"] in required and not (r["generate"] and r["json"])]
        if bad:
            print(f"\nREQUIRED MODELS NOT USABLE: {', '.join(bad)}")
            raise SystemExit(1)
        print(f"\nall required models usable: {', '.join(sorted(required))}")


if __name__ == "__main__":
    main()
