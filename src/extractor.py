"""
Ollama dispatcher — Phase-1 atomic-fact extraction.

Public surface:

    extract(model_name, parsed_doc, *, guideline_version="v1", ...) -> list[AtomicFact]
    unload(model_name, *, base_url="http://localhost:11434") -> None
    load_parsed_doc(path) -> ParsedDocument
    run_doc(model_name, parsed_doc, output_dir, **kwargs) -> Path

CLI:

    python -m src.extractor extract --model qwen3.5:4b \
        --doc data/parsed/train-000000.json \
        --out data/facts

Design constraints (see CLAUDE.md and thesis/chapters/method.tex §Models
 and Inference):

* RTX 3060M / 6 GB VRAM ⇒ only ONE 4B-class model resident at a time. Before
  each run we POST `keep_alive=0` to any other model currently loaded
  (`/api/ps`). At the end of `extract()` we also unload, so the next call
  with a different model starts from a clean GPU.
* Per-section chunking (each recital / each article as its own LLM call)
  keeps the input under ~1K tokens. This is the empirical sweet-spot for
  4B-class models established in the preceding benchmarking project; see
  proposal §Phase 2 architectural justification.
* The extraction prompt is the file `prompts/extract_<version>.md` and its
  `<guideline>…</guideline>` block is the **v1 baseline** for the Phase-4
  v1→v2 distribution-shift experiment. Do not bake guideline text into
  this dispatcher; always read it from disk so analysts can edit it
  through the VA tool (Phase-3) without touching Python.
* Output of every LLM call is validated against the AtomicFact pydantic
  schema. Schema/quote failures trigger a bounded retry loop with the
  parse error fed back to the model.
* Each fact's `source_quote` is anchored back to character offsets in the
  appropriate section container (`preamble_text` for recitals,
  `enacting_text` for articles) via RapidFuzz, producing a `ProseLocator`.
"""

from __future__ import annotations
import os

import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import requests
from pydantic import ValidationError
from rapidfuzz import fuzz

# Resolve project root so `python src/extractor.py` and `python -m src.extractor`
# both work without a hard dependency on cwd.
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.schema import (  # noqa: E402
    AtomicFact,
    ParsedDocument,
    ProseLocator,
    SourceType,
)
from src.providers import resolve, ResolvedModel  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_KEEP_ALIVE = "5m"  # while a run is in flight
UNLOAD_KEEP_ALIVE = 0  # sentinel: unload immediately
DEFAULT_TIMEOUT_S = 600  # legal recitals are slow on a 4B Q4_K_M

#: Context window requested per call. 16384 was inherited from the 6 GB-local-4B
#: era and is OUR limit, not the models': all three cloud annotators report
#: >=262 144 (`/api/show`, checked 2026-09-07). It was small enough to bite —
#: `train-000125` under `cloud:qwen3.6:35b` came back with
#: prompt_eval 6203 + eval 10179 = 16 382 tokens and `done_reason: length`,
#: i.e. JSON truncated mid-string by the window, which the parser then reported
#: as `Unterminated string` and blamed on the model.
DEFAULT_NUM_CTX = 32768

#: Hard bound on ONE answer, so a runaway generation stops at a named boundary
#: that says `done_reason: length` instead of silently colliding with the
#: context edge. Most answers run 200-1500 tokens, but section size is very
#: unevenly distributed and the tail is what gets lost: of 433 dev sections
#: 11 exceed 2000 characters and one (`train-000124 preamble.recitals[0]`, a
#: 9715-character state-aid narrative) is 12x the median. Measured 2026-09-07,
#: in the documents the coverage audit flags, EVERY section over 2000
#: characters is missing and no section over 808 characters survived — a
#: too-small output budget was silently eating the big sections, which are
#: also the fact-dense ones.
DEFAULT_NUM_PREDICT = 16384

#: Part of the harness, therefore FROZEN with everything else once the run of
#: record starts: changing either of the two above changes what the annotators
#: can emit, which is not allowed to vary between guideline versions
#: (CLAUDE.md invariant 3). Record any change in doc/CHANGELOG.md.
OFFSET_MIN_SCORE = 70.0  # below this we drop the fact rather than mis-locate it

# Group-level fields (one atomic annotation = one human-readable statement + span).
GROUP_REQUIRED_FIELDS = ("natural_language", "source_quote")
# Triple-level fields (each SPO inside a group / each legacy element).
TRIPLE_REQUIRED_FIELDS = ("subject", "predicate", "object")
# Back-compat alias (kept for any external import).
REQUIRED_FIELDS = ("subject", "predicate", "object", "natural_language", "source_quote")


# ---------------------------------------------------------------------------
# Section enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Section:
    """One LLM-sized chunk of a ParsedDocument."""

    section_path: str           # e.g. "preamble.recitals[3]" or "enacting.article_1"
    text: str                   # the chunk fed to the model (verbatim slice)
    container_text: str         # the parent stream the offsets refer to
    base_offset: int            # text == container_text[base_offset : base_offset + len(text)]


def enumerate_sections(doc: ParsedDocument) -> Iterator[Section]:
    """Yield the units of extraction for v1: each recital, each article.

    We skip the citation block ("Having regard to …") because guideline §5
    instructs the model not to extract from it. We also skip the closing
    formula for the same reason. If a future guideline wants to revisit
    those, add them here — the prompt will still apply guideline §5.
    """
    for i, recital in enumerate(doc.recitals):
        # Slice rather than rebuilding the string so character offsets line
        # up exactly with `preamble_text`.
        text = doc.preamble_text[recital.char_start:recital.char_end]
        if text.strip():
            yield Section(
                section_path=f"preamble.recitals[{i}]",
                text=text,
                container_text=doc.preamble_text,
                base_offset=recital.char_start,
            )

    for article in doc.articles:
        text = doc.enacting_text[article.char_start:article.char_end]
        if text.strip():
            yield Section(
                section_path=f"enacting.article_{article.number}",
                text=text,
                container_text=doc.enacting_text,
                base_offset=article.char_start,
            )


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_prompt_template(guideline_version: str = "v1",
                         prompt_path: Path | None = None,
                         schema_version: str = "v1") -> str:
    """Assemble the extraction prompt from a SCHEMA frame + a GUIDELINE body.

    Separation of concerns (2026-06-19): the prompt is split across two files so
    the output contract and the annotation policy vary independently.

      * prompts/schema_<schema_version>.md    - the I/O CONTRACT: output JSON
        shape, field requirements, and the input template (the <<DOC_TITLE>> /
        <<SECTION_PATH>> / <<SECTION_TEXT>> sentinels). Carries one
        `<<GUIDELINE>>` slot into which the annotation policy is injected.
      * prompts/extract_<guideline_version>.md - PURE GUIDELINE: the annotation
        policy only (coverage mandate + <guideline> rules + examples); no output
        schema and no sentinels.

    The schema is selectable independently of the guideline, so one guideline
    can run against different output schemas (and vice-versa). Sentinels use
    plain string replacement (not str.format) so legal text with literal
    `{ }` does not break substitution.

    Back-compat: a legacy SELF-CONTAINED guideline file (one that still carries
    the <<SECTION_TEXT>> sentinel, i.e. pre-split) is returned verbatim with no
    schema frame applied.
    """
    if prompt_path is None:
        prompt_path = _PROJECT_ROOT / "prompts" / f"extract_{guideline_version}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {prompt_path}. "
            f"Did you write prompts/extract_{guideline_version}.md ?"
        )
    guideline = prompt_path.read_text(encoding="utf-8")

    # Legacy self-contained prompt (pre-split): use verbatim.
    if "<<SECTION_TEXT>>" in guideline:
        return guideline

    schema_path = _PROJECT_ROOT / "prompts" / f"schema_{schema_version}.md"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema frame not found: {schema_path}. "
            f"Did you write prompts/schema_{schema_version}.md ?"
        )
    schema = schema_path.read_text(encoding="utf-8")
    if "<<GUIDELINE>>" not in schema:
        raise ValueError(
            f"Schema frame {schema_path} is missing the <<GUIDELINE>> slot."
        )
    return schema.replace("<<GUIDELINE>>", guideline)


def render_prompt(template: str, *, doc_title: str, section: Section,
                  suppress_thinking: bool = True) -> str:
    """Substitute the three sentinels in the template.

    `suppress_thinking` appends `/no_think` — the Qwen3 inline switch that
    forces non-reasoning mode. This is a dispatcher-level adapter, kept
    OUT of the markdown template so the guideline stays model-agnostic
    (Gemma3 / Nemotron-3-nano ignore the token harmlessly).
    """
    out = template
    out = out.replace("<<DOC_TITLE>>", doc_title)
    out = out.replace("<<SECTION_PATH>>", section.section_path)
    out = out.replace("<<SECTION_TEXT>>", section.text)
    if suppress_thinking:
        out = out + "\n\n/no_think"
    return out


# ---------------------------------------------------------------------------
# Ollama HTTP wrappers
# ---------------------------------------------------------------------------


def _post(base_url: str, path: str, payload: dict, *, timeout: float = DEFAULT_TIMEOUT_S,
          transient_retries: int = 2, headers: dict | None = None) -> dict:
    """POST with retry on *transient* failures (HTTP 5xx, connection refused).

    On the 6 GB card, Ollama can 500 transiently while a model is being
    evicted/loaded (observed 2026-06-10: first qwen3.5:4b call of a pipeline
    run 500'd; the identical request succeeded seconds later). Treating that
    as fatal kills a whole batch cell, so retry with backoff before raising.
    Client errors (4xx) still raise immediately — those are real bugs.
    """
    url = f"{base_url.rstrip('/')}{path}"
    delay = 3.0
    for attempt in range(transient_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout, headers=headers)
            if resp.status_code >= 500 and attempt < transient_retries:
                logger.warning("Ollama %s -> HTTP %s (attempt %d/%d) — retrying in %.0fs",
                               path, resp.status_code, attempt + 1, transient_retries + 1, delay)
                time.sleep(delay); delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.ConnectionError:
            if attempt < transient_retries:
                logger.warning("Ollama connection error on %s (attempt %d/%d) — retrying in %.0fs",
                               path, attempt + 1, transient_retries + 1, delay)
                time.sleep(delay); delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


#: Stream failures worth another attempt. NOT just `ConnectionError`:
#: `ChunkedEncodingError` is a sibling of it under `RequestException`, not a
#: subclass, so catching only `ConnectionError` let a stream that died
#: mid-transfer through as a fatal error — `train-000124
#: preamble.recitals[0]` was skipped on "Response ended prematurely" after ONE
#: attempt, with two retries unused (BUG-2026-09-07-E). A long answer streams
#: for minutes on the shared box, which is plenty of time for the peer to cut
#: it, and the right response to a cut is to try again.
#: `HTTPError` is deliberately absent: a 4xx is our bug and retrying it just
#: repeats it (5xx is handled separately, by status code, above).
_STREAM_TRANSIENT = (requests.exceptions.ConnectionError,
                     requests.exceptions.ChunkedEncodingError,
                     requests.exceptions.Timeout)


def _post_stream(base_url: str, path: str, payload: dict, *,
                 timeout: float = DEFAULT_TIMEOUT_S,
                 transient_retries: int = 2,
                 headers: dict | None = None) -> dict:
    """POST a `stream: true` generate call and fold the NDJSON back into one body.

    Why streaming is not optional on the DBvis box: a gateway in front of it
    closes any connection that has sent **no bytes for ~50s**
    (BUG-2026-08-31-D). With `stream: false` Ollama sends nothing until the
    whole answer is ready, so every generation longer than 50s died with
    `RemoteDisconnected` at 50.1s — measured 2026-09-07: a 400-line count-up
    was cut at 50.1s unstreamed and completed in 84.7s streamed, first chunk
    at 22.7s. Prompt size is irrelevant (3061 prompt tokens evaluate in 1.5s);
    it is generation wall-clock that crosses the wall, so keeping the socket
    warm is the whole fix.

    Returns a dict shaped like the non-streamed body: `response` and `thinking`
    concatenated across chunks, plus the final chunk's counters.
    """
    url = f"{base_url.rstrip('/')}{path}"
    payload = {**payload, "stream": True}
    delay = 3.0
    for attempt in range(transient_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout,
                                 headers=headers, stream=True)
            if resp.status_code >= 500 and attempt < transient_retries:
                logger.warning("Ollama %s -> HTTP %s (attempt %d/%d) — retrying in %.0fs",
                               path, resp.status_code, attempt + 1,
                               transient_retries + 1, delay)
                resp.close(); time.sleep(delay); delay *= 2
                continue
            resp.raise_for_status()
            out: dict = {}
            parts: list[str] = []
            think: list[str] = []
            chunks = 0
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON chunk from %s: %.120s", path, line)
                    continue
                chunks += 1
                if isinstance(chunk.get("error"), str):
                    raise RuntimeError(f"Ollama error mid-stream: {chunk['error']}")
                # /api/generate puts the text on `response`; /api/chat puts it
                # on `message.content`. Fold both into `response` so callers
                # never care which endpoint served them.
                if isinstance(chunk.get("response"), str):
                    parts.append(chunk["response"])
                if isinstance(chunk.get("thinking"), str):
                    think.append(chunk["thinking"])
                msg = chunk.get("message")
                if isinstance(msg, dict):
                    if isinstance(msg.get("content"), str):
                        parts.append(msg["content"])
                    if isinstance(msg.get("thinking"), str):
                        think.append(msg["thinking"])
                if chunk.get("done"):
                    out = chunk
            out = dict(out)
            out["response"] = "".join(parts)
            out["thinking"] = "".join(think)
            out["_stream_chunks"] = chunks
            # A stream that ended without a `done` chunk was truncated by the
            # peer. Say so loudly: a partial JSON body parsed as "no facts"
            # is exactly the contamination of BUG-2026-09-04-A.
            if not out.get("done"):
                raise requests.ConnectionError(
                    f"stream ended after {chunks} chunks without a done marker "
                    f"({len(out['response'])} chars received) — treating as a "
                    "failed call, not an empty answer"
                )
            return out
        except _STREAM_TRANSIENT as exc:
            # Log the exception, not just "connection error": on a streamed
            # call the two failure modes look identical in a bare message and
            # mean different things. "Remote end closed" = no first byte
            # within the gateway's ~50s idle window (the request was still
            # queued behind other load — queueing counts as idle, so
            # streaming cannot help there). "stream ended ... without a done
            # marker" = we were receiving output and the peer cut it off.
            logger.warning("Ollama connection error on %s (attempt %d/%d) — "
                           "retrying in %.0fs: %s",
                           path, attempt + 1, transient_retries + 1, delay, exc)
            if attempt < transient_retries:
                time.sleep(delay); delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def _get(base_url: str, path: str, *, timeout: float = 10,
         headers: dict | None = None) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    return resp.json()


def list_loaded_models(base_url: str = DEFAULT_BASE_URL,
                       headers: dict | None = None) -> list[str]:
    """Names of models currently resident in VRAM, per Ollama `/api/ps`."""
    try:
        body = _get(base_url, "/api/ps", headers=headers)
    except requests.RequestException as exc:
        logger.warning("Could not reach Ollama at %s (/api/ps): %s", base_url, exc)
        return []
    return [m["name"] for m in body.get("models", [])]


def unload(model_name: str, *, base_url: str = DEFAULT_BASE_URL,
           headers: dict | None = None) -> None:
    """Evict a model from VRAM by issuing a no-op generate with keep_alive=0."""
    try:
        _post(base_url, "/api/generate",
              {"model": model_name, "prompt": "", "keep_alive": UNLOAD_KEEP_ALIVE},
              timeout=30, headers=headers)
        logger.info("Unloaded model %s", model_name)
    except requests.RequestException as exc:
        logger.warning("Failed to unload %s: %s", model_name, exc)


def _unload_others(keep: str, base_url: str, headers: dict | None = None) -> None:
    """Free VRAM by evicting every loaded model that is not `keep`."""
    for name in list_loaded_models(base_url, headers=headers):
        # Ollama returns names like "qwen2.5:4b-instruct-q4_K_M" — match exactly.
        if name != keep:
            unload(name, base_url=base_url, headers=headers)


def _generate_json(model_name: str,
                   prompt: str,
                   *,
                   base_url: str,
                   keep_alive: str | int = DEFAULT_KEEP_ALIVE,
                   temperature: float = 0.0,
                   num_ctx: int = DEFAULT_NUM_CTX,
                   num_predict: int = DEFAULT_NUM_PREDICT,
                   headers: dict | None = None) -> tuple[str, str]:
    """Call /api/generate with format=json.

    Returns `(raw_json_string, done_reason)`. The caller needs the reason:
    `"length"` means the text was CUT, so the right correction is "be shorter",
    not "fix your JSON".
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        # streamed by _post_stream — see its docstring (BUG-2026-08-31-D)
        "stream": True,
        "format": "json",
        "keep_alive": keep_alive,
        # Qwen3 / DeepSeek-style hybrid-reasoning models default to thinking
        # mode, which sinks tokens into <think>…</think> and leaves
        # `response` empty under format=json. We're a constrained-extraction
        # task — no chain-of-thought benefit — so turn it off.
        "think": False,
        "options": {
            # Deterministic-ish: phase-4 distribution comparison needs reproducibility.
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    body = _post_stream(base_url, "/api/generate", payload, headers=headers)
    raw = body.get("response") or ""
    # `done_reason: "length"` means the answer was CUT at a token limit, not
    # malformed by the model — the two are indistinguishable downstream, where
    # both surface as `Unterminated string` from json.loads, and they need
    # opposite fixes (raise the limit vs. correct the prompt). Say which.
    if body.get("done_reason") == "length":
        logger.warning(
            "%s hit the generation LIMIT (done_reason=length, eval_count=%s, "
            "prompt_eval_count=%s) — the JSON is truncated, not malformed. "
            "Raise num_predict/num_ctx rather than retrying the prompt.",
            model_name, body.get("eval_count"), body.get("prompt_eval_count"),
        )
    if not raw.strip():
        # On a reasoning model `think: False` is not always honoured — the
        # answer lands on the THINKING channel and `response` comes back "".
        # conflict_layer2._chat_once recovers this the same way
        # (BUG-2026-08-09-A); this dispatcher never got the same fix, which
        # silently turned every cloud:qwen3.6:35b extraction call into a
        # well-formed answer discarded for an empty string (BUG-2026-08-31-C).
        thinking = body.get("thinking") or ""
        if thinking.strip():
            logger.info(
                "%s answered on the THINKING channel, not response "
                "(prompt_eval_count=%s, eval_count=%s) — recovered.",
                model_name, body.get("prompt_eval_count"), body.get("eval_count"),
            )
            return thinking, str(body.get("done_reason") or "")
        # Surface what Ollama actually sent back so the user can tell
        # "empty response" from "model is paraphrasing instead of JSON".
        logger.warning(
            "Empty response from %s on BOTH channels (prompt_eval_count=%s, "
            "eval_count=%s). If this persists, check that `think=False` is "
            "honoured by your Ollama version.",
            model_name, body.get("prompt_eval_count"), body.get("eval_count"),
        )
    return raw


def _chat_json(model_name: str,
               messages: list[dict],
               *,
               base_url: str,
               keep_alive: str | int = DEFAULT_KEEP_ALIVE,
               temperature: float = 0.0,
               num_ctx: int = DEFAULT_NUM_CTX,
               num_predict: int = DEFAULT_NUM_PREDICT,
               headers: dict | None = None) -> tuple[str, str]:
    """Multi-turn `/api/chat` with `format=json`. Returns the raw JSON string.

    Why chat rather than `/api/generate`: a rejected extraction is corrected by
    *continuing the conversation* — prompt, the model's answer, the validator's
    complaint, the model's next answer — instead of re-sending the whole
    section prompt with the error appended. The model then sees its own
    previous JSON as its own turn, which is what makes "fix field X" a local
    edit rather than a fresh attempt at the whole task, and the section text is
    sent once instead of three times.

    Same two adapters as `_generate_json`: streamed (the gateway kills idle
    sockets, see `_post_stream`) and `think: False` with a fallback to the
    thinking channel, since a reasoning model does not always honour it
    (BUG-2026-08-09-A / BUG-2026-08-31-C).
    """
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "format": "json",
        "keep_alive": keep_alive,
        "think": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx,
                    "num_predict": num_predict},
    }
    body = _post_stream(base_url, "/api/chat", payload, headers=headers)
    if body.get("done_reason") == "length":
        logger.warning(
            "%s hit the generation LIMIT (done_reason=length, eval_count=%s, "
            "prompt_eval_count=%s) — the JSON is truncated, not malformed.",
            model_name, body.get("eval_count"), body.get("prompt_eval_count"),
        )
    raw = body.get("response") or ""
    if not raw.strip():
        thinking = body.get("thinking") or ""
        if thinking.strip():
            logger.info("%s answered on the THINKING channel, not content "
                        "(eval_count=%s) — recovered.",
                        model_name, body.get("eval_count"))
            return thinking, str(body.get("done_reason") or "")
        logger.warning("Empty chat response from %s on BOTH channels "
                       "(eval_count=%s).", model_name, body.get("eval_count"))
    return raw, str(body.get("done_reason") or "")


#: `/api/chat` support, probed once per process. The DBvis deployment sits
#: behind an OpenWebUI proxy which does not necessarily forward every Ollama
#: route, and a route it does not forward fails the same way a busy box does
#: (the connection is closed) — so probe once, remember, and fall back to
#: `/api/generate` with the conversation flattened into the prompt rather than
#: losing the section.
_CHAT_SUPPORTED: dict[str, bool] = {}


def _flatten_conversation(messages: list[dict]) -> str:
    """Render a chat conversation as a single `/api/generate` prompt.

    The fallback path when `/api/chat` is not available. It is NOT equivalent —
    the model no longer sees its previous answer as its own turn — so it is a
    degraded mode worth logging, not a design choice.
    """
    out: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        if role == "user":
            out.append(m.get("content", ""))
        elif role == "assistant":
            out.append("Your previous response was:\n" + m.get("content", ""))
    return "\n\n".join(out)


def _converse_json(resolved,
                   messages: list[dict],
                   *,
                   headers: dict | None = None) -> tuple[str, str]:
    """Send `messages` via `/api/chat`, falling back to `/api/generate` once
    per (process, base_url) if the endpoint is not served.

    Returns `(raw_json_string, done_reason)`.
    """
    key = resolved.base_url
    if _CHAT_SUPPORTED.get(key, True):
        try:
            return _chat_json(resolved.model_tag, messages,
                              base_url=resolved.base_url, headers=headers)
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status not in (400, 404, 405, 501):
                raise
            _CHAT_SUPPORTED[key] = False
            logger.warning(
                "/api/chat is not served by %s (HTTP %s) — falling back to "
                "/api/generate with the conversation flattened into the "
                "prompt. Error feedback still works, but the model no longer "
                "sees its own answer as its own turn.", key, status,
            )
    return _generate_json(resolved.model_tag, _flatten_conversation(messages),
                          base_url=resolved.base_url, headers=headers)


# ---------------------------------------------------------------------------
# Output parsing + offset recovery
# ---------------------------------------------------------------------------


class ExtractionError(Exception):
    """Raised when the model's response cannot be coerced into facts."""


class SectionUnparsed(RuntimeError):
    """A section exhausted every retry without one parseable response.

    A `RuntimeError` on purpose: `extract()` already treats that as "this
    section produced nothing, record the document as INCOMPLETE", which is
    exactly the accounting this case needs. Before 2026-09-07 exhaustion just
    `return []`-ed, so a document that lost a section to a malformed answer was
    written and recorded `status: ok` — the same silent under-count that
    BUG-2026-09-04-A had to repair for the network path, reached by the other
    door.
    """


def _parse_facts_response(raw: str) -> list[dict]:
    """Strict parse: every group must be valid or nothing is returned.

    Kept for callers that want all-or-nothing. The extraction loop uses
    `_parse_groups_tolerant`, because all-or-nothing is what threw away 64
    good groups over a missing field in the 65th (BUG-2026-09-07-D).
    """
    groups, defects = _parse_groups_tolerant(raw)
    if defects:
        raise ExtractionError(defects[0])
    return groups


def _parse_groups_tolerant(raw: str) -> tuple[list[dict], list[str]]:
    """Parse the model's raw response into atomic-fact GROUPS, per group.

    Returns `(groups, defects)`: the groups that are well-formed, and one
    message per group that is not. A malformed group costs that group and a
    line of feedback, not the answer — `train-000124 preamble.recitals[0]`
    returned 65 groups of which the 65th lacked an `object`, and the strict
    parser discarded all 65 (BUG-2026-09-07-D).

    Raises `ExtractionError` only when the response as a WHOLE is unusable:
    not JSON, no `facts` key, `facts` not an array, or every group defective.

    Each group is ONE annotation (a single natural_language + source_quote span)
    carrying one or more SPO triples. Two accepted element shapes:

      * multi-SPO:  {"natural_language", "source_quote",
                     "triples": [{"subject","predicate","object",
                                  "condition?","temporal_context?"}, ...]}
      * legacy:     {"subject","predicate","object","natural_language",
                     "source_quote","condition?","temporal_context?"}
                    (treated as a 1-triple group)

    Returns: list of {"natural_language", "source_quote",
    "triples": [{subject,predicate,object,condition,temporal_context}, ...]}.
    On any violation raises ExtractionError with a corrective message.
    """
    try:
        # strict=False tolerates a raw control character inside a string value
        # — a literal newline in a quoted passage, most often. Strict parsing
        # rejects the whole answer for it, and on 2026-09-09 that cost a
        # section whose annotations were otherwise fine (BUG-2026-09-09-B).
        # This cannot change any answer that already parsed: the two modes
        # differ ONLY on control characters inside strings, so where strict
        # succeeded there were none and both return the same object. What it
        # lets through is still checked downstream — a control character that
        # lands in `source_quote` breaks the character-for-character anchor
        # test, which is the guard that actually matters.
        obj = json.loads(raw, strict=False)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"Response was not valid JSON: {exc.msg}") from exc

    if not isinstance(obj, dict) or "facts" not in obj:
        raise ExtractionError(
            'Response JSON must be an object with a top-level "facts" key.'
        )
    facts = obj["facts"]
    if not isinstance(facts, list):
        raise ExtractionError('"facts" must be a JSON array.')

    groups: list[dict] = []
    defects: list[str] = []
    for i, f in enumerate(facts):
        try:
            groups.append(_parse_one_group(i, f))
        except ExtractionError as exc:
            defects.append(str(exc))
    if not groups and defects:
        # Nothing survived: this is a whole-response failure, and the caller
        # needs an exception so the section is not recorded as "no facts".
        raise ExtractionError(
            defects[0] if len(defects) == 1
            else f"{len(defects)} defective fact group(s); first: {defects[0]}"
        )
    return groups, defects


#: What to say when a triple arrives with an empty or missing `object`. It was
#: 9 of the 11 defects on 2026-09-07 and the model never fixed one of them in
#: three turns, because "object must be a non-empty string" states the schema
#: and teaches nothing: the clause it is annotating ("The security referred to
#: in Article 4(1)(a) shall be released") genuinely has no object, so the model
#: can only invent one or repeat itself.
#:
#: The convention below is not invented here — it is read off the gold, where
#: 0 of 556 triples have an empty object and passives are handled exactly this
#: way ("shall enter into force on" / "14 November 2003"; "should be" /
#: "abrogated"). `prompts/extract_v1.md` never states it, which is a GUIDELINE
#: GAP and belongs in the g_v1 -> g_v2 diagnosis, not in a quiet edit to v1:
#: the guideline is the independent variable and v1 is fixed. This text is
#: HARNESS — the same for every annotator and every guideline version, and
#: frozen with the rest of the harness once the run of record starts. It makes
#: the schema contract explicit at the point of rejection; it does not change
#: which clauses become facts.
EMPTY_OBJECT_HINT = (
    "Every triple needs a non-empty object. A passive or intransitive "
    "predication still has one: either absorb the preposition into the "
    "predicate and put its complement in the object "
    "(\"shall enter into force on\" / \"14 November 2003\"; "
    "\"shall apply from\" / \"31 March 2006\"), or split the verb so the "
    "state or participle becomes the object (\"should be\" / \"abrogated\"; "
    "\"shall be\" / \"released\"). Do not invent an object the text does not "
    "contain, and do not move a condition into it — conditions belong in "
    "`condition`."
)


def _parse_one_group(i: int, f: object) -> dict:
    """Validate ONE element of `facts`. Raises `ExtractionError` on any defect."""

    def _req_str(d, field, where):
        hint = " " + EMPTY_OBJECT_HINT if field == "object" else ""
        if field not in d:
            raise ExtractionError(
                f'{where} is missing required field "{field}".{hint}')
        if not isinstance(d[field], str) or not d[field].strip():
            raise ExtractionError(
                f'{where}."{field}" must be a non-empty string.{hint}')
        return d[field].strip()

    def _opt_str(d, field, fallback=""):
        val = d.get(field, fallback)
        return val.strip() if isinstance(val, str) else fallback

    if not isinstance(f, dict):
        raise ExtractionError(f"facts[{i}] is not a JSON object.")
    nl = _req_str(f, "natural_language", f"facts[{i}]")
    sq = _req_str(f, "source_quote", f"facts[{i}]")
    # group-level condition/temporal are inherited by triples that omit them
    g_cond = _opt_str(f, "condition")
    g_temp = _opt_str(f, "temporal_context")
    g_logic_group = _opt_str(f, "logic_group")
    g_logic_op = _opt_str(f, "logic_op").upper()
    if g_logic_op not in ("", "AND", "OR", "XOR"):
        g_logic_op = ""

    raw_triples = f.get("triples")
    if isinstance(raw_triples, list) and raw_triples:
        triple_dicts, legacy = raw_triples, False
    elif any(fld in f for fld in TRIPLE_REQUIRED_FIELDS):
        triple_dicts, legacy = [f], True   # legacy: element itself is the triple
    else:
        # Neither shape: an annotation with no triples at all — commonly
        # `"triples": []` or a mis-named key. Falling through to the legacy
        # branch here reported `missing required field "subject"`, which
        # tells the model to flatten the group rather than to fill in the
        # triples it dropped, and the retry then walks further away from
        # the contract (observed 2026-09-07 on cloud:qwen3.6:35b,
        # train-000004 enacting.article_1, 3/3 attempts). Name the actual
        # defect instead.
        raise ExtractionError(
            f'facts[{i}] carries no triples: "triples" must be a non-empty '
            'array of {"subject", "predicate", "object"} objects (or, for a '
            'single triple, those three fields directly on facts[' + str(i) + ']).'
        )

    triples: list[dict] = []
    for j, t in enumerate(triple_dicts):
        if not isinstance(t, dict):
            raise ExtractionError(f"facts[{i}].triples[{j}] is not a JSON object.")
        where = f"facts[{i}]" if legacy else f"facts[{i}].triples[{j}]"
        triple = {fld: _req_str(t, fld, where) for fld in TRIPLE_REQUIRED_FIELDS}
        triple["condition"] = _opt_str(t, "condition", g_cond)
        triple["temporal_context"] = _opt_str(t, "temporal_context", g_temp)
        triples.append(triple)

    return {"natural_language": nl, "source_quote": sq, "triples": triples,
            "logic_group": g_logic_group, "logic_op": g_logic_op}


def _recover_offsets(quote: str, section: Section) -> tuple[int, int, float, str]:
    """RapidFuzz-anchor `quote` inside `section.text`, then translate to
    container offsets.

    Returns (char_start, char_end, score, matched_substring).
    char_start/char_end are absolute positions inside section.container_text.
    """
    # Cheap exact match first — avoids fuzzy-matching jitter when the model
    # quoted verbatim, which it should ~95 % of the time given guideline §4.
    idx = section.text.find(quote)
    if idx >= 0:
        cs = section.base_offset + idx
        ce = cs + len(quote)
        return cs, ce, 100.0, quote

    # Fuzzy fallback for whitespace/punctuation drift.
    align = fuzz.partial_ratio_alignment(quote, section.text)
    if align is None:
        return -1, -1, 0.0, ""
    score = float(align.score)
    if score < OFFSET_MIN_SCORE:
        return -1, -1, score, ""
    ds, de = align.dest_start, align.dest_end
    cs = section.base_offset + ds
    ce = section.base_offset + de
    matched = section.container_text[cs:ce]
    return cs, ce, score, matched


def _group_id(model: str, doc_id: str, section_path: str, group_index: int) -> str:
    h = hashlib.sha1()
    h.update(f"grp|{model}|{doc_id}|{section_path}|{group_index}".encode("utf-8"))
    return h.hexdigest()[:16]


def _fact_id(model: str, doc_id: str, section_path: str,
             group_index: int, triple_index: int = 0) -> str:
    h = hashlib.sha1()
    h.update(f"{model}|{doc_id}|{section_path}|{group_index}|{triple_index}".encode("utf-8"))
    return h.hexdigest()[:16]


def _build_atomic_fact(triple: dict,
                       *,
                       model_name: str,
                       doc: ParsedDocument,
                       guideline_version: str,
                       section: Section,
                       locator: ProseLocator,
                       natural_language: str,
                       source_quote: str,
                       anchor_score: float,
                       group_id: str,
                       group_index: int,
                       triple_index: int,
                       logic_group: str = "",
                       logic_op: str = "") -> AtomicFact:
    """Build one AtomicFact for a single SPO triple of an atomic-annotation group.

    The group's source_quote was already anchored to `locator` by the caller, so
    every triple in the group shares the same provenance + group_id.
    """
    return AtomicFact(
        fact_id=_fact_id(model_name, doc.celex, section.section_path,
                         group_index, triple_index),
        doc_id=doc.celex,
        annotator=model_name,
        guideline_version=guideline_version,
        subject=triple["subject"],
        predicate=triple["predicate"],
        object=triple["object"],
        natural_language=natural_language,
        condition=triple.get("condition", ""),
        temporal_context=triple.get("temporal_context", ""),
        source_locator=locator,
        group_id=group_id,
        logic_group=logic_group,
        logic_op=logic_op,
        extra={
            "anchor_score": anchor_score,
            "model_quote": source_quote,
            "triple_index": triple_index,
        },
    )


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------


#: Size of the head and of the tail kept when a rejected answer is too large to
#: replay whole. Replaying the whole thing is the default, because a model
#: repairs its own JSON by looking at it — but past a few thousand characters
#: that stops being true and starts being harmful. Both failure modes were
#: measured on `train-000124 preamble.recitals[0]`: echoing a truncated
#: 8192-token fragment took turn 2 from prompt_eval 6205 to 14455 and it
#: truncated again in the same place; echoing a complete 57 871-character
#: answer (65 fact groups, rejected for one missing field in the 65th) took
#: turn 2 to prompt_eval 20 909 and the model rewrote all 65 from scratch
#: instead of fixing one, overrunning the window. With per-group tolerance the
#: siblings are already banked as facts, so the correction turn does not need
#: the whole answer in front of it.
ECHO_HEAD_TAIL_CHARS = 600

#: Answers longer than this are echoed as head + tail with the middle elided.
#: Below it, echo in full.
MAX_ECHO_CHARS = 6000

#: How many defects to quote back in one correction turn. All of them are
#: logged; feeding back a wall of complaints just invites the model to rewrite
#: everything, including the groups that were fine.
MAX_DEFECTS_FED_BACK = 4

#: Defects that are NOT the model's fault and must never cost a correction
#: turn. A detector that fires on a mechanism artefact will fire again on the
#: corrected answer, so retrying burns turns, and telling a model it erred
#: when it did not teaches it to distort a correct answer. Matched
#: case-insensitively against the defect message.
#:
#: Add a pattern here only with a reason: the test is "would the best possible
#: annotator hit this too?". If yes it belongs here AND the detector itself
#: probably needs revisiting — record that in doc/bugfix_log.md rather than
#: leaving the exclusion as the whole story.
MECHANISM_DEFECT_PATTERNS: tuple[str, ...] = (
    # (empty — see the note above before adding)
)


def is_mechanism_defect(message: str) -> bool:
    """True if `message` names a known mechanism artefact, not a model error."""
    low = message.lower()
    return any(pat.lower() in low for pat in MECHANISM_DEFECT_PATTERNS)


def _echo_answer(raw: str, done_reason: str) -> str:
    """The assistant turn to replay when asking for a correction.

    Full text while the answer is small enough for that to help. A truncated
    answer is always elided (there is no defect inside it to look at), and so
    is any answer over `MAX_ECHO_CHARS` — see that constant for the two
    measured failures this avoids.
    """
    elide = done_reason == "length" or len(raw) > MAX_ECHO_CHARS
    if not elide or len(raw) <= ECHO_HEAD_TAIL_CHARS * 2:
        return raw
    head = raw[:ECHO_HEAD_TAIL_CHARS]
    tail = raw[-ECHO_HEAD_TAIL_CHARS:]
    why = ("your answer was cut off by the output limit"
           if done_reason == "length" else "middle elided for length")
    return (f"{head}\n\n[... {len(raw) - 2 * ECHO_HEAD_TAIL_CHARS} characters "
            f"elided; {why} ...]\n\n{tail}")


def _summarise_validation_error(exc: ValidationError) -> str:
    """Render a pydantic error as a short, model-actionable instruction.

    The raw `str(exc)` is a multi-line report with a docs URL — worth having in
    the log, useless inside a correction turn, where it costs tokens and buries
    the one thing the model has to change.
    """
    parts: list[str] = []
    for err in exc.errors()[:4]:
        loc = ".".join(str(x) for x in err.get("loc", ())) or "(root)"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts)


def _extract_one_section(resolved: ResolvedModel,
                         doc: ParsedDocument,
                         section: Section,
                         template: str,
                         *,
                         guideline_version: str,
                         retries: int) -> list[AtomicFact]:
    """Run the prompt -> JSON -> validate -> anchor pipeline for one section.

    Correction is a CONVERSATION, capped at `retries + 1` turns and scoped to
    THIS section: the prompt is sent once, and each defect the detectors find
    is appended to the same `/api/chat` exchange — prompt, the model's JSON,
    the diagnosis, its next JSON. The conversation is built fresh for every
    section and thrown away with it, so nothing accumulates across sections or
    documents. Until 2026-09-07 a retry re-sent the whole section prompt with
    the error appended, which made every attempt a fresh attempt at the whole
    task instead of a local edit to an answer the model can see is its own —
    and re-paid for the section text each time.

    EVERY defect is fed back, not only the ones that make the response
    unparseable. Two classes used to be discarded in silence, with no second
    chance and no record:

      * a `source_quote` that cannot be anchored in the section text — the
        whole group was dropped on a `logger.warning`;
      * a triple the schema rejects — the triple was skipped, likewise.

    Both are the kind of mistake a model fixes when told, so both now cost a
    turn rather than the data. A truncated answer (`done_reason == "length"`)
    is fed back as what it is — "your answer was cut off, send fewer groups" —
    because telling a model its JSON is malformed when it was actually cut
    mid-string sends it chasing a bug it did not make.
    """
    prompt = render_prompt(template, doc_title=doc.title, section=section)
    messages: list[dict] = [{"role": "user", "content": prompt}]
    last_error: str | None = None
    raw = ""
    done_reason = ""
    best_effort: list[AtomicFact] = []

    for attempt in range(retries + 1):
        if last_error is not None:
            # Continue the SAME conversation: its own answer, then the
            # complaint about it.
            messages.append({"role": "assistant",
                             "content": _echo_answer(raw, done_reason)})
            messages.append({"role": "user", "content": (
                "Your previous response was rejected: " + last_error
                + "\nReturn the corrected JSON object only."
            )})
        try:
            raw, done_reason = _converse_json(resolved, messages,
                                              headers=resolved.headers)
            cleaned, parse_defects = _parse_groups_tolerant(raw)
        except ExtractionError as exc:
            if done_reason == "length":
                # Mechanism, not a modelling mistake: the text was cut, so
                # "Unterminated string" names the symptom and would send the
                # model looking for a syntax error it did not make.
                last_error = (
                    "your response was cut off by the output limit before the "
                    "JSON was complete. Return FEWER fact groups this time and "
                    "keep each source_quote as short as the rule allows."
                )
            else:
                last_error = str(exc)
            logger.info(
                "[%s] %s - attempt %d/%d rejected: %s (raw %d chars, "
                "done_reason=%s, ends %r)",
                resolved.model_id, section.section_path, attempt + 1, retries + 1,
                exc, len(raw or ""), done_reason or "-", (raw or "")[-60:],
            )
            continue
        except requests.RequestException as exc:
            # Network-level failure: do not waste retries on this.
            raise RuntimeError(
                f"Ollama request failed on section {section.section_path}: {exc}"
            ) from exc

        # Success path — anchor each GROUP's quote once, then build one
        # AtomicFact per SPO triple in the group (shared locator + group_id).
        facts: list[AtomicFact] = []
        # A group the parser rejected is one group lost, not the answer: it is
        # fed back like any other defect while its 64 well-formed siblings
        # become facts (BUG-2026-09-07-D).
        defects: list[str] = list(parse_defects)
        for gi, group in enumerate(cleaned):
            cs, ce, score, matched = _recover_offsets(group["source_quote"], section)
            if cs < 0:
                logger.warning(
                    "Group %d unanchored (quote not located in %s, fuzzy "
                    "score=%.1f): %r",
                    gi, section.section_path, score, group["source_quote"][:80],
                )
                defects.append(
                    "facts[{}].source_quote could not be found in the section "
                    "text (closest match scored {:.0f}/100): {!r}. It must be "
                    "copied CHARACTER-FOR-CHARACTER from the text above — do "
                    "not normalise spacing, quotes, dashes or capitalisation, "
                    "and do not join text from two places.".format(
                        gi, score, group["source_quote"][:120])
                )
                continue
            locator = ProseLocator(
                section_path=section.section_path,
                char_start=cs, char_end=ce, quote=matched,
            )
            gid = _group_id(resolved.model_id, doc.celex, section.section_path, gi)
            for ti, triple in enumerate(group["triples"]):
                try:
                    f = _build_atomic_fact(
                        triple,
                        model_name=resolved.model_id, doc=doc,
                        guideline_version=guideline_version, section=section,
                        locator=locator, natural_language=group["natural_language"],
                        source_quote=group["source_quote"], anchor_score=score,
                        group_id=gid, group_index=gi, triple_index=ti,
                        logic_group=group.get("logic_group", ""),
                        logic_op=group.get("logic_op", ""),
                    )
                except ValidationError as exc:
                    logger.warning("Rejected triple %d.%d in %s: %s",
                                   gi, ti, section.section_path, exc)
                    defects.append(
                        "facts[{}].triples[{}] was rejected by the schema: {}"
                        .format(gi, ti, _summarise_validation_error(exc))
                    )
                    continue
                facts.append(f)

        retryable = [d for d in defects if not is_mechanism_defect(d)]
        if retryable and attempt < retries:
            # Keep the partial result: never trade good facts for a retry that
            # might come back worse or die on the network.
            if len(facts) > len(best_effort):
                best_effort = facts
            last_error = " ; ".join(retryable[:MAX_DEFECTS_FED_BACK])
            logger.info(
                "[%s] %s - attempt %d/%d produced %d fact(s) but %d defect(s); "
                "asking the model to correct them in-conversation.",
                resolved.model_id, section.section_path, attempt + 1, retries + 1,
                len(facts), len(retryable),
            )
            continue
        if defects:
            # Out of turns, or nothing worth another turn: the facts stand, but
            # say how many were lost and why. A section that quietly
            # under-reports is the failure mode of BUG-2026-09-07-B.
            logger.warning(
                "[%s] %s - keeping %d fact(s) with %d UNCORRECTED defect(s) "
                "after %d attempt(s): %s",
                resolved.model_id, section.section_path, len(facts), len(defects),
                attempt + 1, " ; ".join(d[:160] for d in defects[:3]),
            )
        if len(best_effort) > len(facts):
            logger.info(
                "[%s] %s - the last turn returned fewer facts (%d) than an "
                "earlier one (%d); keeping the earlier result.",
                resolved.model_id, section.section_path, len(facts),
                len(best_effort),
            )
            facts = best_effort
        logger.info(
            "[%s] %s — %d facts from %d group(s) (attempt %d).",
            resolved.model_id, section.section_path, len(facts), len(cleaned), attempt + 1,
        )
        return facts

    if best_effort:
        # An earlier turn HAD parsed, and only the correction turns broke. The
        # loop used to fall straight through to SectionUnparsed from the
        # `except ExtractionError` branch's `continue`, which never consults
        # best_effort — so 71 already-parsed facts of
        # `train-000124 preamble.recitals[0]` were discarded because turn 3
        # returned malformed JSON (BUG-2026-09-09-C). Keeping good facts over a
        # failed retry is the same rule the success path already follows; the
        # defect that prompted the correction stays on the record, uncorrected.
        logger.warning(
            "[%s] %s - every correction turn failed to parse (last error: %s), "
            "but an earlier turn had already returned %d parseable fact(s) - "
            "keeping those rather than discarding the section. They carry "
            "whatever defect prompted the correction, UNCORRECTED.",
            resolved.model_id, section.section_path, last_error,
            len(best_effort),
        )
        return best_effort

    logger.error(
        "[%s] %s — all %d attempts failed (last error: %s).",
        resolved.model_id, section.section_path, retries + 1, last_error,
    )
    raise SectionUnparsed(
        f"section {section.section_path}: no parseable response in "
        f"{retries + 1} attempts (last error: {last_error})"
    )


def extract(model_name: str,
            parsed_doc: ParsedDocument,
            *,
            guideline_version: str = "v1",
            base_url: str = DEFAULT_BASE_URL,
            retries: int = 1,  # fail-fast: a section that can't yield valid JSON
                               # should not loop (v4 + dense sections on 6GB GPU).
            ensure_solo: bool = True,
            unload_after: bool = False,
            prompt_path: Path | None = None,
            on_progress=None,
            incomplete_out: list[str] | None = None) -> list[AtomicFact]:
    """Extract atomic facts from one parsed document using one Ollama model.

    Parameters
    ----------
    model_name
        Ollama model tag exactly as `ollama list` shows it
        (e.g. "qwen2.5:4b-instruct-q4_K_M").
    parsed_doc
        A `ParsedDocument` (typically loaded from `data/parsed/<id>.json`).
    guideline_version
        Selects `prompts/extract_<version>.md`. Stored on every fact so
        the Phase-4 v1→v2 comparison can group by guideline.
    base_url
        Ollama API root. Default is local.
    retries
        Per-section retry budget for schema failures.
    ensure_solo
        If True, unload every other model currently in VRAM before
        starting. Required on 6 GB hardware when switching between the
        three 4B models.
    unload_after
        If True, also unload `model_name` at the end. Set this when the
        next pipeline step (e.g. another model's extraction or the Phase-2
        Layer-2 arbitrator) needs the VRAM.
    prompt_path
        Override the default `prompts/extract_<version>.md` location.
    incomplete_out
        If given, every section skipped on a persistent network failure is
        appended to this list, so the CALLER can tell a document that lost
        sections to the proxy from one the model genuinely had nothing to say
        about. Without it, both look like a plain `list[AtomicFact]` and a
        materially incomplete document is recorded as a clean result
        (BUG-2026-09-04). Also: when every section was skipped, this function
        now raises instead of returning `[]`, because there is no partial
        result worth preserving in that case.
        Entries are `"<section_path> (unparsed)"` (every correction turn
        failed) or `"<section_path> (network)"`; unparsed sections are
        appended too, not only network ones. The format is parsed by
        `segment_population.failed_section_records` and, for manifests
        written before 2026-09-23, out of the manifest's `error` string, so
        change it only together with both.
    """
    resolved = resolve(model_name, default_base_url=base_url)
    template = load_prompt_template(guideline_version, prompt_path)

    if ensure_solo and resolved.manages_vram:
        _unload_others(keep=resolved.model_tag, base_url=resolved.base_url,
                       headers=resolved.headers)

    all_facts: list[AtomicFact] = []
    skipped: list[str] = []
    sections = list(enumerate_sections(parsed_doc))
    for i, section in enumerate(sections):
        if on_progress is not None:
            try:
                on_progress(i, len(sections), section.section_path)
            except Exception:                       # never let UI hooks kill a run
                pass
        try:
            all_facts.extend(_extract_one_section(
                resolved, parsed_doc, section, template,
                guideline_version=guideline_version,
                retries=retries,
            ))
        except SectionUnparsed as exc:
            # Distinct from the network case below only in its cause: the box
            # answered, the answer was unusable. Same consequence — the
            # section contributed nothing, so the document under-reports and
            # must not be recorded as a clean result.
            logger.error(
                "[%s] %s — SKIPPED, no parseable response, document will be "
                "INCOMPLETE for this section: %s",
                resolved.model_id, section.section_path, exc,
            )
            if incomplete_out is not None:
                incomplete_out.append(f"{section.section_path} (unparsed)")
            skipped.append(section.section_path)
        except RuntimeError as exc:
            # _extract_one_section wraps a persistent network failure (proxy
            # timeout under shared-box load, observed 2026-08-31 on the
            # DBvis cloud endpoint — see BUG-2026-08-31-D) as a RuntimeError,
            # not the requests.RequestException it came from — catch THAT,
            # not the original type, or this except clause never fires. It
            # used to abort the WHOLE document here, so one flaky section
            # threw away every other section's already-good facts and the
            # caller had to redo the entire doc from scratch.
            # Skip just this section instead; the document is recorded as
            # INCOMPLETE (not silently as if nothing were missing) so a
            # later pass can be targeted at exactly the gap.
            logger.error(
                "[%s] %s — SKIPPED after a persistent network failure, "
                "document will be INCOMPLETE for this section: %s",
                resolved.model_id, section.section_path, exc,
            )
            if incomplete_out is not None:
                incomplete_out.append(f"{section.section_path} (network)")
            skipped.append(section.section_path)

    # A document that lost EVERY section to the network produced no data at
    # all. Returning [] here is what let 14 such documents land in
    # data/facts/manifest.json as `status: ok, fact_count: 0` on 2026-09-03 —
    # indistinguishable from "the model found nothing" without reading the
    # logs (BUG-2026-09-04). There is no partial result to protect in this
    # case, so fail loudly and let the manifest record it as retryable.
    if skipped and not all_facts:
        raise RuntimeError(
            f"every section ({len(skipped)}/{len(sections)}) of "
            f"{parsed_doc.celex} was skipped (network failure or unparseable "
            f"response); no usable model output was received at all"
        )

    if skipped:
        logger.warning(
            "[%s] %s is INCOMPLETE: %d of %d sections were skipped (network "
            "failure or unparseable response: %s) — its fact count "
            "under-reports and must not be used "
            "as a clean result.",
            resolved.model_id, parsed_doc.celex, len(skipped), len(sections),
            ", ".join(skipped),
        )

    if unload_after and resolved.manages_vram:
        unload(resolved.model_tag, base_url=resolved.base_url, headers=resolved.headers)

    logger.info(
        "Done: %s on %s → %d facts (guideline=%s).",
        resolved.model_id, parsed_doc.celex, len(all_facts), guideline_version,
    )
    return all_facts


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_parsed_doc(path: Path | str) -> ParsedDocument:
    """Load a ParsedDocument from one of `data/parsed/*.json`.

    The file format produced by `scripts.run_dry_run` wraps the doc as
    `{"document": {...}, "linearized_table_rows": [...]}`. We accept both
    shapes for robustness.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "document" in raw:
        doc_payload = raw["document"]
    else:
        doc_payload = raw
    return ParsedDocument.model_validate(doc_payload)


def write_facts(facts: list[AtomicFact],
                *,
                model_name: str,
                doc_id: str,
                guideline_version: str,
                out_root: Path | str,
                failed_sections: list[dict] | None = None) -> Path:
    """Persist facts to `<out_root>/<model_safe>/<doc_id>.json`.

    `failed_sections` (added 2026-09-23) is the run's own record of the
    sections that returned NO validated output, as
    `[{"section_path", "cause": "unparsed" | "network"}]`. It is written
    beside `facts`, never into a fact, so the fact records and the extraction
    output contract are unchanged. `[]` means "recorded, none failed"; the key
    is omitted when the caller passes None ("not recorded"), so a writer that
    does not know (an interactive re-extraction) cannot claim a clean
    document. RQ1's segment population reads it (`src/segment_population.py`):
    a section with zero facts is a legitimate empty answer unless it is here.
    """
    safe_model = model_name.replace(":", "_").replace("/", "_")
    out_dir = Path(out_root) / safe_model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{doc_id}.json"
    payload = {
        "doc_id": doc_id,
        "annotator": model_name,
        "guideline_version": guideline_version,
        "fact_count": len(facts),
        "facts": [f.model_dump(mode="json") for f in facts],
    }
    if failed_sections is not None:
        payload["failed_sections"] = list(failed_sections)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return out_path


def run_doc(model_name: str,
            parsed_path: Path | str,
            *,
            out_root: Path | str = None,
            **kwargs: Any) -> Path:
    """Convenience: load → extract → write. Returns the output path."""
    if out_root is None:
        out_root = _PROJECT_ROOT / "data" / "facts"
    doc = load_parsed_doc(parsed_path)
    guideline_version = kwargs.get("guideline_version", "v1")
    facts = extract(model_name, doc, **kwargs)
    return write_facts(
        facts,
        model_name=model_name,
        doc_id=doc.celex,
        guideline_version=guideline_version,
        out_root=out_root,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli():
    import typer
    app = typer.Typer(add_completion=False, help="Phase-1 atomic fact extractor (Ollama).")

    @app.command("extract")
    def cli_extract(
        model: str = typer.Option(..., "--model", "-m",
                                  help="Ollama model tag, e.g. qwen2.5:4b-instruct-q4_K_M."),
        doc: Path = typer.Option(..., "--doc", "-d",
                                 help="Path to a ParsedDocument JSON in data/parsed/."),
        out: Path = typer.Option(_PROJECT_ROOT / "data" / "facts", "--out", "-o",
                                 help="Root directory for fact JSON output."),
        guideline_version: str = typer.Option("v1", "--guideline", "-g"),
        retries: int = typer.Option(2, "--retries"),
        base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
        no_solo: bool = typer.Option(False, "--no-solo",
                                     help="Do NOT evict other models first (use if you know VRAM is free)."),
        unload_after: bool = typer.Option(False, "--unload-after",
                                          help="Unload the model after the run finishes."),
    ):
        """Extract atomic facts from one parsed document."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        out_path = run_doc(
            model_name=model,
            parsed_path=doc,
            out_root=out,
            guideline_version=guideline_version,
            retries=retries,
            base_url=base_url,
            ensure_solo=not no_solo,
            unload_after=unload_after,
        )
        typer.echo(f"Wrote {out_path}")

    @app.command("unload")
    def cli_unload(
        model: str = typer.Option(..., "--model", "-m"),
        base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
    ):
        """Evict a model from VRAM."""
        unload(model, base_url=base_url)

    @app.command("ps")
    def cli_ps(base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url")):
        """List models currently resident in VRAM."""
        for name in list_loaded_models(base_url):
            typer.echo(name)

    return app


if __name__ == "__main__":
    _build_cli()()
