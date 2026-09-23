"""
Phase-2 — Step 2b: LLM arbitrator for escalated pairs.

We re-use the Ollama dispatcher pattern from `src/extractor.py` (sequential
model loading, JSON output) but turn *on* the model's thinking mode here —
the proposal explicitly separates these two regimes:

    Phase-1 extract:    think = OFF  (load-bearing; OFF prevents empty JSON)
    Phase-2 arbitrate:  think = ON   (we want the model to reason)

NOTE (implementation reality): empirically, `think=True` under `format=json`
brings back EMPTY responses on local qwen3.5 / gemma3 (2026-06-20: 25/25 empty;
2026-07-02: 7/7 empty on cloud:qwen3.6:35b). We therefore send `think=False`
and, on a second attempt, drop `format=json` and recover the label from
free text (`_extract_json_obj`). This deviates from the proposal's "think on"
wording; the Method chapter must reflect whichever regime we finally report.
RESOLVED 2026-08-09 — the empty body was never a generation failure: the answer
was being routed to the response's `thinking` channel while `response` came back
"". See `_chat_once` for the measured per-model table and the two guards.

Public surface:

    arbitrate(pair, *, model_name, doc_title, base_url) -> AlignedPair
    arbitrate_all(pairs, *, model_name, doc_title, base_url) -> dict[str, int]
    arbitrate_group(group, ...) -> FactGroup
    arbitrate_groups(groups, ..., shapes=("one_to_one",)) -> dict[str, int]

Groups, not pairs, are the unit of analysis (method.tex). `arbitrate_groups` is
the group-layer counterpart of `arbitrate_all`; both accept a shared `cache` so
a one-to-one group and its pair cost ONE call, not two.

The dispatcher writes back into the pair's `.conflict_label` and
`.layer2_reason` fields. If Layer-2 fails (model error, schema fail
after retries), the pair is left with `conflict_label = "escalate"` and a
diagnostic appended to `.layer2_reason` — never crash the batch.

This module **does not require Ollama to be live to be imported**. The
network call only happens inside `arbitrate()`. Unit tests can stub
`_chat_once` to exercise the logic without a model running.
"""

from __future__ import annotations
import os

import json
import logging
import re
from pathlib import Path

import requests

from .alignment import AlignedPair
from .providers import resolve

logger = logging.getLogger(__name__)


# Labels the LLM is allowed to output. Anything else gets coerced to
# "escalate" with a diagnostic; we'd rather surface the mistake than
# silently mis-label.
_VALID_LABELS = {"CONTRADICTION", "GRANULARITY", "REDUNDANCY", "NO_CONFLICT"}

_DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_DEFAULT_MODEL = "qwen3.5:4b"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def load_arbitrate_template(version: str = "v1") -> str:
    """Read prompts/arbitrate_<version>.md, stripping the leading HTML comment."""
    path = Path(__file__).resolve().parent.parent / "prompts" / f"arbitrate_{version}.md"
    text = path.read_text(encoding="utf-8")
    # Drop the leading `<!-- ... -->` block (developer notes, not for the LLM).
    text = re.sub(r"^<!--.*?-->\s*", "", text, count=1, flags=re.DOTALL)
    return text


def _render_prompt(template: str, *, doc_title: str, source_quote: str, fact_a: str, fact_b: str) -> str:
    return (
        template.replace("<<DOC_TITLE>>", doc_title)
        .replace("<<SOURCE_QUOTE>>", source_quote)
        .replace("<<FACT_A>>", fact_a)
        .replace("<<FACT_B>>", fact_b)
    )


# --- Rendering one side of a comparison ------------------------------------
# Shared by the pair path and the group path so the two cannot drift apart:
# a one-to-one group and the pair it came from must produce the SAME prompt,
# which is what makes the call cache below sound.


def _fact_nl(fact: dict) -> str:
    """Natural-language form of one fact, falling back to its slots."""
    nl = (fact.get("natural_language") or "").strip()
    if nl:
        return nl
    return " ".join(str(fact.get(k) or "").strip()
                    for k in ("subject", "predicate", "object")).strip()


def _side_text(facts: list[dict]) -> str:
    """One side of a comparison; a MANY side becomes a numbered list.

    The arbitrator cannot judge a granularity split unless it can see that this
    side is N separate facts rather than one long one, so the enumeration is
    load-bearing, not cosmetic.

    It also cannot judge a split whose members look identical. The 4B annotators
    routinely write the SAME `natural_language` for facts whose slots differ —
    on train-000004, `nemotron-3-nano:4b` did it for 10 of 25 facts, and 29 of
    36 one-to-many groups had every partner textually identical. Rendering NL
    only showed the model three copies of one sentence, and it answered
    (correctly, for what it was shown) "B repeats the same fact three times",
    converting a real granularity split into redundancy. So when the NL lines do
    not distinguish the members of a side, fall back to the slots — which is
    where the difference actually lives."""
    facts = [f for f in (facts or []) if f]
    if len(facts) == 1:
        return _fact_nl(facts[0])
    lines = [_fact_nl(f) for f in facts]
    if len(set(lines)) < len(lines):
        lines = [f"{nl}  [{f.get('subject') or ''} | {f.get('predicate') or ''} "
                 f"| {f.get('object') or ''}]"
                 for nl, f in zip(lines, facts)]
    return "\n".join(f"({i}) {ln}" for i, ln in enumerate(lines, 1))


def _quotes_text(*sides: list[dict]) -> str:
    """The distinct source spans behind a comparison, in first-seen order.

    Passing only ONE side's quote fed the arbitrator the WRONG evidence when
    that quote was mis-anchored to a near-duplicate line (legal lists: "- for
    Spain: ECU 45,85 / - for Portugal: ECU 42,53"): the model then hallucinated
    a Spain-vs-Portugal contradiction between two facts that BOTH said
    Portugal. Showing every distinct span cuts those false contradictions.
    (confirmed live 2026-07-08)"""
    seen: list[str] = []
    for facts in sides:
        for f in facts or []:
            q = ((f.get("source_locator", {}) or {}).get("quote", "") or "").strip()
            if q and q not in seen:
                seen.append(q)
    return "\n".join(seen)


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------


def _chat_once(
    prompt: str,
    *,
    model_name: str,
    base_url: str,
    timeout: float = 90.0,
    format_json: bool = True,
) -> str:
    """One /api/generate call. think=OFF via the API FIELD, not the /no_think
    prompt suffix: the suffix is a qwen-chat-template directive that newer
    qwen builds (and every non-qwen model) ignore, which brought back the
    empty-response failure under format=json. `"think": False` is what Phase-1
    extraction (_generate_json) has always sent - mirror it exactly.

    Why the field is sent explicitly, and never as True (probed live against
    Ollama 0.24.0 on 2026-08-09, all three local 4B models, real arbitrate_v1
    prompt — `response` / `thinking` is which channel carried the JSON):

        model               think=False   field omitted    think=True
        qwen3.5:4b          response      THINKING only    THINKING only
        gemma3:4b           response      response         400 Bad Request
        nemotron-3-nano:4b  response      THINKING only    THINKING only

    That is the whole of the "Layer-2 returns empty bodies" blocker: on a
    reasoning model the answer goes to the thinking channel and `response` is
    "". Hence (a) always send the field — omitting it is NOT the same as False,
    (b) never send True, gemma3 rejects the request outright, and (c) read
    `thinking` when `response` is empty, so a runtime that stops honouring the
    field degrades to a slow answer instead of a silently blank batch.

    `format_json` forces Ollama's grammar-constrained JSON. It is ON by default
    but the retry path in `arbitrate()` turns it OFF on a second attempt: some
    local builds (qwen3.5, gemma3) intermittently return an EMPTY response under
    `format=json`; free-text generation plus tolerant JSON extraction
    (`_extract_json_obj`) recovers the label."""
    r = resolve(model_name, default_base_url=base_url)
    body = {
        "model": r.model_tag,
        "prompt": prompt,
        # Streamed: the DBvis gateway closes any socket idle ~50s, so an
        # unstreamed answer that takes longer dies with RemoteDisconnected
        # (BUG-2026-08-31-D). See extractor._post_stream.
        "stream": True,
        "think": False,
        # Temperature 0, not 0.1. methodology_v2.tex says "temperature 0 and
        # schema-constrained JSON throughout" and rests the reproducibility
        # claim on it, but this call site had always sent 0.1 — so the conflict
        # distribution, which IS the RQ1 measure, was not reproducible. Measured
        # 2026-09-10 on `train-000004`, same facts, same thresholds, two runs
        # minutes apart: group labels went no_conflict 22 / redundancy 66 in one
        # and 16 / 72 in the other. Part of any observed g_v1 -> g_v2 shift would
        # have been arbitrator noise (BUG-2026-09-10-B).
        "options": {"temperature": 0.0, "num_ctx": 4096},
    }
    if format_json:
        body["format"] = "json"
    from .extractor import _post_stream
    body_json = _post_stream(r.base_url, "/api/generate", body,
                             timeout=timeout, headers=r.headers)
    text = body_json.get("response") or ""
    if not text.strip():
        # The answer went to the reasoning channel (see the table above).
        # Recover it instead of reporting an empty body.
        text = body_json.get("thinking") or ""
    return text


# ---------------------------------------------------------------------------
# Tolerant response parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_LABEL_TOKEN_RE = re.compile(
    r"\b(CONTRADICTION|GRANULARITY|REDUNDANCY|NO[_\s]?CONFLICT)\b", re.IGNORECASE
)


def _extract_json_obj(raw: str) -> dict | None:
    """Best-effort recovery of the JSON object from a model response that may be
    fenced in ```code```, prefixed with prose, or otherwise not strictly JSON.
    Returns the first dict we can parse, else None."""
    if not raw:
        return None
    # 1) direct parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # 2) strip a markdown code fence
    m = _FENCE_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # 3) first balanced {...} span
    start = raw.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(raw)):
            c = raw[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(raw[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = raw.find("{", start + 1)
    return None


def _norm_label(label: str) -> str:
    return label.strip().upper().replace(" ", "_")


def _parse_response(raw: str) -> tuple[str | None, str | None]:
    """Returns (label, reason). Returns (None, diagnostic) on failure.

    Tolerant of fenced / prose-wrapped JSON (common on local gemma3 / nemotron
    that don't perfectly honour format=json) and, as a last resort, of a single
    bare label word. Ambiguous responses mentioning several labels fail rather
    than guess."""
    if not raw or not raw.strip():
        return None, "empty_response"

    obj = _extract_json_obj(raw)
    if obj is not None and isinstance(obj.get("label"), str):
        label = _norm_label(obj["label"])
        if label in _VALID_LABELS:
            reason = obj.get("reason", "")
            return label, reason if isinstance(reason, str) else ""
        # structured but out-of-vocab label -> fall through to token scan

    # Last resort: a single unambiguous bare label token anywhere in the text.
    tokens = {_norm_label(m.group(1)) for m in _LABEL_TOKEN_RE.finditer(raw)}
    tokens = {t for t in tokens if t in _VALID_LABELS}
    if len(tokens) == 1:
        return next(iter(tokens)), "recovered from unstructured response"
    if len(tokens) > 1:
        return None, "ambiguous_labels:" + ",".join(sorted(tokens))
    return None, "no_label_found"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def arbitrate_once(*, doc_title: str = "", source_quote: str = "",
                   fact_a: str = "", fact_b: str = "",
                   model_name: str = _DEFAULT_MODEL,
                   base_url: str = _DEFAULT_OLLAMA_URL,
                   template: str | None = None, version: str = "v1") -> dict:
    """One-shot arbitration for the conflict-prompt debug UI.

    Lenient on purpose: returns whatever `label` the model emits (no
    `_VALID_LABELS` gate), so you can prototype NEW conflict types just by
    editing the prompt. Returns {label, reason, raw}."""
    tmpl = template if (template and template.strip()) else load_arbitrate_template(version)
    prompt = _render_prompt(tmpl, doc_title=doc_title, source_quote=source_quote,
                            fact_a=fact_a, fact_b=fact_b)
    # Local qwen3.5 / gemma3 INTERMITTENTLY return an empty response under
    # format=json (confirmed live 2026-07-08: same pair empty once, labelled on
    # retry). Retry once without format=json so the debug panel isn't stuck on a
    # blank result; tolerant extraction handles the free-text / fenced JSON.
    raw = ""
    for attempt in (1, 2):
        raw = _chat_once(prompt, model_name=model_name, base_url=base_url,
                         format_json=(attempt == 1))
        if raw and raw.strip():
            break
    label, reason = None, None
    obj = _extract_json_obj(raw)
    if isinstance(obj, dict):
        label = obj.get("label")
        reason = obj.get("reason")
    return {"label": label, "reason": reason, "raw": raw}


def _cache_key(facts_a: list[dict], facts_b: list[dict]) -> tuple:
    """Identity of one arbitration question, order-independent between sides.

    A one-to-one group asks exactly the question its pair asks, so keying on the
    fact ids lets the group pass and the pair pass share a single LLM call
    instead of paying twice for the same comparison."""
    def ids(facts):
        return tuple(sorted(str(f.get("fact_id") or _fact_nl(f))
                            for f in (facts or []) if f))
    a, b = ids(facts_a), ids(facts_b)
    return (a, b) if a <= b else (b, a)


def _decide(
    facts_a: list[dict],
    facts_b: list[dict],
    *,
    doc_title: str,
    model_name: str,
    base_url: str,
    template: str,
    chat_fn=None,
    cache: dict | None = None,
) -> tuple[str | None, str]:
    """Ask the arbitrator about one comparison → (label | None, reason).

    The call policy (bounded retry, format=json then free text), the parsing and
    the cache all live here, so the pair path and the group path cannot drift.
    Never raises: a failed comparison comes back as (None, diagnostic) and the
    caller keeps its pre-Layer-2 label."""
    key = _cache_key(facts_a, facts_b)
    if cache is not None and key in cache:
        return cache[key]

    prompt = _render_prompt(
        template,
        doc_title=doc_title,
        source_quote=_quotes_text(facts_a, facts_b) or "(quote missing)",
        fact_a=_side_text(facts_a),
        fact_b=_side_text(facts_b),
    )

    call = chat_fn or _chat_once
    use_default = chat_fn is None

    result: tuple[str | None, str] = (None, "no_attempt")
    for attempt in (1, 2):  # bounded retry — Layer-2 cost adds up across a doc
        try:
            if use_default:
                # Attempt 2 drops format=json — recovers the qwen/gemma
                # "empty response under format=json" failure mode.
                raw = call(prompt, model_name=model_name, base_url=base_url,
                           format_json=(attempt == 1))
            else:
                raw = call(prompt, model_name=model_name, base_url=base_url)
        except Exception as exc:  # network / server / timeout
            result = (None, f"call_error_attempt_{attempt}:{exc.__class__.__name__}")
            logger.warning("Layer-2 call failed: %s", result[1])
            continue

        label, reason = _parse_response(raw)
        if label is not None:
            result = (label, reason or f"layer2 {label}")
            break
        result = (None, f"parse_fail_attempt_{attempt}:{reason}")

    if cache is not None:
        cache[key] = result

    _note_call_outcome(result)
    return result


class Layer2Unavailable(RuntimeError):
    """The arbitrator stopped answering; the batch must stop, not degrade.

    Raised by `_note_call_outcome` after CONSECUTIVE_FAILURE_LIMIT comparisons
    in a row come back as transport failures. Without it a dead arbitrator is
    survivable in exactly the wrong way: every call returns (None, call_error),
    every group keeps `escalate` (which is correct per call — the label is never
    guessed), and the batch runs to completion and prints `ok=38`. Ollama died
    mid-run on 2026-09-10 and the run kept going for 26 more documents' worth of
    work before it was noticed by hand. `escalate > 0` is visible in the output,
    but nobody reads 38 documents of console output, and the files would have
    looked like a finished Phase-2 pass.

    A single failed comparison is still tolerated and still degrades to
    `escalate` — this only fires on a sustained outage.
    """


#: Consecutive transport failures before the batch aborts. Ten, not one:
#: a lone timeout under load is normal and must not kill a three-hour run.
CONSECUTIVE_FAILURE_LIMIT = 10

_consecutive_failures = 0


def reset_call_health() -> None:
    """Clear the failure streak (tests, and a deliberate resume)."""
    global _consecutive_failures
    _consecutive_failures = 0


def _note_call_outcome(result: tuple) -> None:
    global _consecutive_failures
    reason = result[1] if len(result) > 1 else ""
    if str(reason).startswith("call_error"):
        _consecutive_failures += 1
        if _consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
            n = _consecutive_failures
            reset_call_health()
            raise Layer2Unavailable(
                f"{n} consecutive Layer-2 comparisons failed to reach the "
                f"arbitrator (last: {reason}). Aborting rather than writing "
                f"conflict files whose groups are all `escalate`. Check that "
                f"the arbitrator is running, then resume — documents already "
                f"written are complete and are not re-run.")
    else:
        # Any answer at all — including a parse failure — means the transport
        # is alive, which is the only thing this counter is about.
        _consecutive_failures = 0


def arbitrate(
    pair: AlignedPair,
    *,
    model_name: str = _DEFAULT_MODEL,
    doc_title: str = "",
    base_url: str = _DEFAULT_OLLAMA_URL,
    template: str | None = None,
    version: str = "v1",
    chat_fn=None,            # injectable for unit tests
    cache: dict | None = None,
) -> AlignedPair:
    """Resolve a single escalated pair via LLM. Mutates and returns `pair`."""
    if pair.status != "matched":
        # Orphans don't have two facts to compare; nothing for Layer-2 here.
        return pair
    if pair.conflict_label not in ("escalate", "unlabeled"):
        # Already labelled by Layer-1 (e.g. REDUNDANCY).
        return pair

    fa, fb = pair.fact_a, pair.fact_b
    assert fa is not None and fb is not None

    template = template or load_arbitrate_template(version)
    label, reason = _decide(
        [fa], [fb], doc_title=doc_title, model_name=model_name,
        base_url=base_url, template=template, chat_fn=chat_fn, cache=cache,
    )
    if label is not None:
        pair.conflict_label = label.lower()
    # On failure the label stays 'escalate' so the miss is visible, never guessed.
    pair.layer2_reason = reason
    return pair


def arbitrate_all(
    pairs: list[AlignedPair],
    *,
    model_name: str = _DEFAULT_MODEL,
    doc_title: str = "",
    base_url: str = _DEFAULT_OLLAMA_URL,
    version: str = "v1",
    chat_fn=None,
    cache: dict | None = None,
) -> dict[str, int]:
    """Apply `arbitrate` to every escalated pair; return final label counts.

    Reads the prompt template once and shares it across calls. Pass the same
    `cache` dict as `arbitrate_groups` to bill each distinct comparison once.
    """
    template = load_arbitrate_template(version)
    counts: dict[str, int] = {}
    n_arbitrated = 0
    for p in pairs:
        if p.status == "matched" and p.conflict_label in ("escalate", "unlabeled"):
            arbitrate(
                p,
                model_name=model_name,
                doc_title=doc_title,
                base_url=base_url,
                template=template,
                chat_fn=chat_fn,
                cache=cache,
            )
            n_arbitrated += 1
        counts[p.conflict_label] = counts.get(p.conflict_label, 0) + 1
    counts["_layer2_calls"] = n_arbitrated
    return counts


# ---------------------------------------------------------------------------
# Group layer — the unit of analysis (method.tex)
#
# Nothing walked the group layer before 2026-08-09: `run_phase2` built the
# groups, labelled them by shape / Layer-1, and then arbitrated the PAIR list.
# So a one-to-one group whose Layer-1 rules did not fire kept `escalate`
# permanently — 39 of 132 groups on train-000004, i.e. every group the LLM
# arbitrator exists to resolve. `arbitrate_groups` is that missing pass.
# ---------------------------------------------------------------------------


# Which group shapes may be sent to the arbitrator. One-to-one only, by default.
ARBITRABLE_SHAPES_DEFAULT = ("one_to_one",)


def arbitrate_group(
    group,
    *,
    model_name: str = _DEFAULT_MODEL,
    doc_title: str = "",
    base_url: str = _DEFAULT_OLLAMA_URL,
    template: str | None = None,
    version: str = "v1",
    chat_fn=None,
    cache: dict | None = None,
):
    """Resolve one FactGroup via the LLM. Mutates and returns `group`.

    The many side is rendered as a numbered list (`_side_text`), so a 1:n group
    reaches the model as "one fact vs these three" — the shape the GRANULARITY
    label was written for.
    """
    template = template or load_arbitrate_template(version)
    prior = group.conflict_label
    label, reason = _decide(
        group.facts_a, group.facts_b, doc_title=doc_title, model_name=model_name,
        base_url=base_url, template=template, chat_fn=chat_fn, cache=cache,
    )
    if label is not None:
        group.prior_label = prior
        group.conflict_label = label.lower()
        group.label_source = "layer2"
        group.reason = reason
    else:
        # Keep the shape / Layer-1 verdict and append why Layer-2 could not help.
        group.reason = f"{group.reason} | layer2: {reason}".strip(" |")
    return group


def _needs_arbitration(group, shapes) -> bool:
    if group.shape not in shapes:
        return False
    if group.shape == "orphan":
        # No partner to compare against, and the orphan rate is coverage rather
        # than a conflict type (method.tex). Never billed to the arbitrator.
        return False
    if group.shape == "one_to_many":
        # Already `granularity` from its shape, so this is not gap-filling: it is
        # the opt-in CHECK of whether the shape rule was right on this group.
        return True
    return group.conflict_label in ("escalate", "unlabeled")


def arbitrate_groups(
    groups: list,
    *,
    model_name: str = _DEFAULT_MODEL,
    doc_title: str = "",
    base_url: str = _DEFAULT_OLLAMA_URL,
    version: str = "v1",
    chat_fn=None,
    cache: dict | None = None,
    shapes: tuple[str, ...] = ARBITRABLE_SHAPES_DEFAULT,
) -> dict[str, int]:
    """Arbitrate the groups `shapes` selects; return final group label counts.

    `shapes=("one_to_one",)` — the default — is pure gap-filling and changes no
    method: a one-to-one group on which no Layer-1 rule fired is exactly what
    Layer-2 is for, and it previously stayed on `escalate` forever.

    `shapes=("one_to_one", "one_to_many")` additionally asks the model to CHECK
    the shape rule on one-to-many groups. **That is not the run of record** —
    method.tex labels a one-to-many group `granularity` from its shape, so
    enabling it is a method change. It exists because it is the experiment behind
    the open linking-criterion question (CHANGELOG 2026-08-08: at tau=0.78, 40%
    of dev groups come out one-to-many, mostly shared institutional vocabulary
    rather than real splits, and symmetric cosine cannot tell the two apart).
    A re-labelled group keeps its shape verdict in `prior_label`, so the
    disagreement rate between shape and model is recoverable from the output.
    """
    template = load_arbitrate_template(version)
    counts: dict[str, int] = {}
    n_arbitrated = 0
    for g in groups:
        if _needs_arbitration(g, shapes):
            arbitrate_group(
                g,
                model_name=model_name,
                doc_title=doc_title,
                base_url=base_url,
                template=template,
                chat_fn=chat_fn,
                cache=cache,
            )
            n_arbitrated += 1
        counts[g.conflict_label] = counts.get(g.conflict_label, 0) + 1
    counts["_layer2_calls"] = n_arbitrated
    return counts
