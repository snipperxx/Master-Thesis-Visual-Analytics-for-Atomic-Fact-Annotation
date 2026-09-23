"""Storage, validation and loading for the hand-annotated EUR-Lex gold sample.

The gold is annotated *guideline-invariantly* (Section~\\ref{sec:gold-invariance}):
every clause carrying an independently checkable proposition is its own gold
fact, coordinated constructions are fully expanded with a logic tag, and no
merging decision is baked in. Gold facts use the same slots as model output so
that the two are directly comparable objects.

On-disk layout
    data/gold/segment_sample.json        the frozen 51-segment sample
    data/gold/facts/<doc_id>.json        gold facts, one file per document
    data/gold/decision_log.json          ambiguous cases + the choice made

Gold fact record
    gold_id            <doc>::<section_path>::f<NN>
    natural_language   self-contained restatement
    source_quote       VERBATIM contiguous substring of the segment text
    condition          precondition (if/when/under), or ""
    temporal_context   ordering/deadline, or ""
    logic_group        shared id for branches of ONE and/or/xor construction
    logic_op           "" | AND | OR | XOR
    triples            [{subject, predicate, object}]
    boilerplate        true for formulaic closing clauses (see decision log D-03)
    status             "unreviewed" | "accepted" | "edited" | "rejected"
    provenance         who produced this record

Nothing here reads model output: the gold is independent by construction.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARSED = ROOT / "data" / "parsed"
GOLD = ROOT / "data" / "gold"
GOLD_FACTS = GOLD / "facts"
# The resolution of record since D-18: 312 facts / 556 triples, human-adjudicated
# 2026-08-31. `GOLD_FACTS` above is the coarse root it superseded, still
# machine-adjudicated. Both describe the same 59 segments with the same
# gold_ids, so a cross-wired root does not fail, it lies (BUG-2026-08-17-A).
GOLD_FACTS_HIRES = GOLD / "facts_hires"

VALID_OPS = {"", "AND", "OR", "XOR"}
VALID_STATUS = {"unreviewed", "accepted", "edited", "rejected"}
REQUIRED_FACT_KEYS = {"gold_id", "natural_language", "source_quote", "triples"}


# ---------------------------------------------------------------------------
# Segment text
# ---------------------------------------------------------------------------

def segment_text(doc_id: str, section_path: str, char_start: int, char_end: int,
                 parsed_root: Path = PARSED) -> str:
    doc = json.loads((parsed_root / f"{doc_id}.json").read_text(encoding="utf-8"))["document"]
    container = doc["preamble_text"] if section_path.startswith("preamble") else doc["enacting_text"]
    return container[char_start:char_end]


def remap_sample_offsets(sample: dict, parsed_root: Path = PARSED,
                         doc_ids: set[str] | None = None) -> list[str]:
    """Slide the sample's stored offsets onto the current `data/parsed` text.

    **Any migration that changes the length of `preamble_text` or
    `enacting_text` must call this in the same pass.** `segment_text` above
    slices the container with the offsets stored in `segment_sample.json`, not
    with the ones in `data/parsed`; leaving them behind slides every segment
    window of the affected document, which surfaces as "not a verbatim
    substring" on quotes nobody touched. That cost 7 spurious errors during the
    2026-08-06 table migration — see doc/bugfix_log.md BUG-2026-08-06-A.

    Segments are matched on `section_path` (recital index / article number), so
    this is a pure remap: nothing is added, removed or reassigned. Returns a
    human-readable note per segment whose offsets moved.
    """
    notes: list[str] = []
    cache: dict[str, dict] = {}

    for seg in sample.get("segments", []):
        doc_id = seg["doc_id"]
        if doc_ids is not None and doc_id not in doc_ids:
            continue
        if doc_id not in cache:
            cache[doc_id] = json.loads(
                (parsed_root / f"{doc_id}.json").read_text(encoding="utf-8"))["document"]
        doc = cache[doc_id]
        path = seg["section_path"]

        if path.startswith("preamble.recitals["):
            i = int(path[len("preamble.recitals["):-1])
            items = doc.get("recitals") or []
            src = items[i] if i < len(items) else None
            container = doc.get("preamble_text") or ""
        else:
            num = path.split("article_", 1)[1]
            src = next((a for a in doc.get("articles") or [] if a["number"] == num), None)
            container = doc.get("enacting_text") or ""

        if src is None:
            notes.append(f"{seg['segment_id']}: {path} no longer exists in the parsed doc")
            continue

        old = (seg["char_start"], seg["char_end"])
        seg["char_start"], seg["char_end"] = src["char_start"], src["char_end"]
        seg["n_words"] = len(container[src["char_start"]:src["char_end"]].split())
        if old != (seg["char_start"], seg["char_end"]):
            notes.append(f"{seg['segment_id']}: {old[0]}..{old[1]} -> "
                         f"{seg['char_start']}..{seg['char_end']} (n_words={seg['n_words']})")
    return notes


def load_sample(gold_root: Path = GOLD) -> dict:
    return json.loads((gold_root / "segment_sample.json").read_text(encoding="utf-8"))


def doc_fingerprint(doc_id: str, gold_root: Path = GOLD) -> str:
    """Hash of what an adjudicator is shown for one document.

    An external pass takes days and the gold can move under it. A reply naming a
    fact that no longer exists is caught by the importer, but a reply that
    *accepts* a fact whose slots were edited after it was sent is not: the quote
    still checks out, and a verdict about text nobody showed the adjudicator gets
    recorded as agreement. The packet carries this hash and the reply carries it
    back, so that case becomes an error rather than a silent one.

    Covers the fact content only. Mechanical flags are excluded on purpose ---
    re-running `machine_review_gold` changes what is worth looking at, not what
    is being judged.
    """
    payload = json.loads((gold_root / "facts" / f"{doc_id}.json").read_text(encoding="utf-8"))
    rows = []
    for seg in payload["segments"]:
        for f in seg["facts"]:
            rows.append([seg["segment_id"], f["gold_id"], f["natural_language"],
                         f["source_quote"], f.get("condition", ""),
                         f.get("temporal_context", ""), f.get("logic_op", ""),
                         [[t.get("subject", ""), t.get("predicate", ""), t.get("object", "")]
                          for t in f["triples"]]])
    blob = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def sample_index(gold_root: Path = GOLD) -> dict[str, dict]:
    return {s["segment_id"]: s for s in load_sample(gold_root)["segments"]}


# ---------------------------------------------------------------------------
# Normalisation used when checking that a quote is verbatim
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")
# The EUR-Lex text mixes straight and curly quotes/dashes; a quote that differs
# only in those is still verbatim for our purposes.
_FOLD = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
}


def fold(s: str) -> str:
    for a, b in _FOLD.items():
        s = s.replace(a, b)
    return _WS.sub(" ", s).strip()


def is_verbatim(quote: str, text: str) -> bool:
    return fold(quote) in fold(text)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_docs: int = 0
    n_segments: int = 0
    n_facts: int = 0
    n_triples: int = 0
    n_boilerplate: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_doc(payload: dict, rep: Report, expect_segments: dict[str, dict]) -> None:
    doc_id = payload.get("doc_id", "?")
    seen_ids: set[str] = set()
    groups: dict[str, set[str]] = {}

    for seg in payload.get("segments", []):
        sid = seg.get("segment_id", "?")
        rep.n_segments += 1
        meta = expect_segments.get(sid)
        if meta is None:
            rep.errors.append(f"{sid}: not part of the frozen sample")
            continue
        text = segment_text(doc_id, meta["section_path"], meta["char_start"], meta["char_end"])

        if not seg.get("facts"):
            rep.warnings.append(f"{sid}: no facts annotated")

        for f in seg.get("facts", []):
            rep.n_facts += 1
            missing = REQUIRED_FACT_KEYS - set(f)
            if missing:
                rep.errors.append(f"{sid}: fact missing keys {sorted(missing)}")
                continue
            gid = f["gold_id"]
            if gid in seen_ids:
                rep.errors.append(f"{doc_id}: duplicate gold_id {gid}")
            seen_ids.add(gid)
            if not gid.startswith(f"{sid}::"):
                rep.errors.append(f"{gid}: gold_id does not match its segment {sid}")

            if not is_verbatim(f["source_quote"], text):
                rep.errors.append(
                    f"{gid}: source_quote is not a verbatim substring -> "
                    f"{f['source_quote'][:70]!r}")
            if not f["natural_language"].strip():
                rep.errors.append(f"{gid}: empty natural_language")
            if not f["triples"]:
                rep.errors.append(f"{gid}: no triples")
            for t in f["triples"]:
                rep.n_triples += 1
                for slot in ("subject", "predicate", "object"):
                    v = (t.get(slot) or "").strip()
                    if not v:
                        rep.errors.append(f"{gid}: empty {slot}")
                    elif v.lower() == "null":
                        rep.errors.append(f"{gid}: literal 'null' in {slot}")

            op = f.get("logic_op", "")
            grp = f.get("logic_group", "")
            if op not in VALID_OPS:
                rep.errors.append(f"{gid}: bad logic_op {op!r}")
            if bool(op) != bool(grp):
                rep.errors.append(f"{gid}: logic_op and logic_group must be set together")
            if grp:
                groups.setdefault(grp, set()).add(op)
            if f.get("status", "unreviewed") not in VALID_STATUS:
                rep.errors.append(f"{gid}: bad status {f.get('status')!r}")
            if f.get("boilerplate"):
                rep.n_boilerplate += 1

    for grp, ops in groups.items():
        if len(ops) > 1:
            rep.errors.append(f"{doc_id}: logic_group {grp} carries mixed ops {sorted(ops)}")
        members = sum(1 for seg in payload.get("segments", [])
                      for f in seg.get("facts", []) if f.get("logic_group") == grp)
        if members < 2:
            rep.errors.append(f"{doc_id}: logic_group {grp} has {members} member(s); needs >= 2")


def validate_all(gold_root: Path = GOLD) -> Report:
    rep = Report()
    expect = sample_index(gold_root)
    covered: set[str] = set()
    for p in sorted((gold_root / "facts").glob("*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        rep.n_docs += 1
        validate_doc(payload, rep, expect)
        covered |= {s["segment_id"] for s in payload.get("segments", [])}
    for sid in expect:
        if sid not in covered:
            rep.errors.append(f"{sid}: sampled but not annotated")
    return rep


# ---------------------------------------------------------------------------
# Loading for evaluation
# ---------------------------------------------------------------------------

def load_gold_facts(doc_id: str, gold_root: Path = GOLD,
                    include_boilerplate: bool = False,
                    facts_root: Path | None = None) -> list[dict]:
    """Flat list of gold facts for one document, in segment order.

    `facts_root` selects which gold root is read; it defaults to
    `gold_root / "facts"` for callers that predate the hi-res root. Scoring
    code should pass `GOLD_FACTS_HIRES` explicitly — the two roots share
    gold_ids, so the wrong one produces numbers rather than an error.

    `include_boilerplate` controls whether the formulaic closing clauses are
    scored (decision log D-03). Default off, because guideline v1 instructs the
    models not to extract them; turn it on to check that the choice does not
    drive the result.
    """
    p = (facts_root or gold_root / "facts") / f"{doc_id}.json"
    if not p.exists():
        return []
    payload = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for seg in payload["segments"]:
        for f in seg["facts"]:
            if f.get("boilerplate") and not include_boilerplate:
                continue
            out.append({**f, "doc_id": doc_id,
                        "segment_id": seg["segment_id"],
                        "section_path": seg["section_path"]})
    return out


def gold_triples_for_eval(doc_id: str, **kw) -> list[dict]:
    """Gold in the flat triple shape src.gold_compare already understands."""
    rows = []
    for f in load_gold_facts(doc_id, **kw):
        for t in f["triples"]:
            rows.append({
                "subject": t["subject"], "predicate": t["predicate"], "object": t["object"],
                "condition": f.get("condition", ""),
                "temporal_context": f.get("temporal_context", ""),
                "gold_id": f["gold_id"], "segment_id": f["segment_id"],
                "source_quote": f["source_quote"],
            })
    return rows


def main() -> None:
    rep = validate_all()
    print(f"docs={rep.n_docs} segments={rep.n_segments} facts={rep.n_facts} "
          f"triples={rep.n_triples} boilerplate={rep.n_boilerplate}")
    for w in rep.warnings:
        print("WARN ", w)
    for e in rep.errors:
        print("ERROR", e)
    print("OK" if rep.ok else f"{len(rep.errors)} error(s)")
    raise SystemExit(0 if rep.ok else 1)


if __name__ == "__main__":
    main()
