"""One substrate for both guideline-revision arms: the draft, its checkpoints,
and the log of why each edit was made.

RQ2 sets the diagnosed revision $g_{v2}$ against the no-feedback control
$g_{v2}'$. That comparison is only readable if the two arms leave the **same
kind of record** — same entry shape, same freeze, same history — differing in
one thing only: what an entry is allowed to cite. The control cites raw baseline
annotations (doc, section, annotator, group_id); the diagnosed arm cites what
the interface showed (a pair, a group, a conflict label, a review). If each arm
grew its own logger they would drift, and the write-up would be comparing two
formats as much as two revisions. So both go through here.

What this module does NOT decide: who may cite what. That is the caller's
business — `serve_control_console.py` serves nothing from `data/conflicts/` by
construction, and that isolation is a property of its routes, not of this file.

    log = DraftLog(draft=Path("prompts/extract_v2.md"),
                   seed=Path("prompts/extract_v1.md"))
    log.checkpoint(text, now, explicit=True)
    shot = log.add_shot(png_bytes, now, caption="the granularity split")
    log.add_entry({"op": "modify", "rule": "R7", "reason": "...", "cites": [...],
                   "shots": [{"file": shot["file"], "caption": shot["caption"]}]})
    log.history(current_text)
"""

from __future__ import annotations

import difflib
import json
from datetime import datetime, timezone
from pathlib import Path

OPS = ("modify", "delete", "new")

# An autosave fires a second or so after you stop typing, so one per keystroke
# burst. Within this window a new checkpoint replaces the previous one instead
# of adding to it, or a morning's writing buries the history it exists to show.
# A pressed button (Save, Commit, Restore) always keeps its own.
COALESCE_SECONDS = 90

SNAP_TIME = "%Y%m%dT%H%M%SZ"
STAMP = "%Y-%m-%d %H:%M:%SZ"

# What a citation may store. A single annotation resolves the same way in both
# arms: (doc_id, section, annotator, group_id) -> the facts file of that
# annotator and document -> the triples sharing that group_id.
#
# `version` is the guideline version the interface was showing — the suffix of
# the conflicts file (`data/conflicts/<doc>__<version>.json`; "v1" is the
# unsuffixed baseline). It is needed because `group_id` and `fact_id` hash
# (model, doc, section, index) and NOT the version: the first annotation of a
# model on a clause has the same id under v1 and v2, so a version-less citation
# of v2 output resolves to a different, v1 annotation without failing. Absent
# means v1 — every citation written before 2026-09-23, in both arms, cites
# baseline output. Resolve the facts root through that conflicts file's
# `params.facts_root`, not by name: the batch roots do not follow the
# interface's naming (`data/facts_v2/` vs `data/facts__v2/`).
CITE_FIELDS = ("doc_id", "section", "annotator", "group_id", "excerpt",
               "pair_key", "label", "kind", "version")

# A screenshot attached to an entry. A citation names what the interface showed;
# a shot shows it, for the cases a `pair_key` cannot carry — a layout, a colour
# pattern across a whole view, a graph shape. It is evidence *of the reason*,
# not a new thing to cite, so it hangs off the entry and never off a cite.
#
# The extension is decided by the bytes, never by the filename the browser sent:
# the file is written into a directory this app also serves, and an attacker-
# named ".html" served back from our own origin would be script in our page.
SHOT_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "png", "image/png"),
    (b"\xff\xd8\xff", "jpg", "image/jpeg"),
    (b"GIF87a", "gif", "image/gif"),
    (b"GIF89a", "gif", "image/gif"),
)
SHOT_MAX_BYTES = 8 * 1024 * 1024
SHOT_PREFIX = "shot_"


class DraftLog:
    """A guideline draft plus the two records that make its revision auditable."""

    def __init__(self, draft: Path, seed: Path, *,
                 snaps_dir: Path | None = None, prefix: str = "snap_",
                 log_path: Path | None = None, artifact: str = "guideline_change_log",
                 shots_dir: Path | None = None):
        self.draft = draft
        self.seed = seed
        self.snaps_dir = snaps_dir or (draft.parent / "_drafts" / draft.stem)
        self.prefix = prefix
        self.log_path = log_path or draft.with_name(draft.stem + "_rationale.json")
        self.artifact = artifact
        self.shots_dir = shots_dir or (draft.parent / "_shots" / draft.stem)

    # ----------------------------------------------------------- checkpoints
    def _key(self, p: Path) -> tuple:
        """(timestamp, n) — two checkpoints can share a second, so `-2`
        disambiguates and must not sort before the plain name, which is what a
        plain lexicographic sort does."""
        stem = p.stem[len(self.prefix):]
        stamp, _, n = stem.partition("-")
        return stamp, int(n or 0)

    def snaps(self) -> list[Path]:
        """Checkpoints, oldest first."""
        if not self.snaps_dir.exists():
            return []
        return sorted(self.snaps_dir.glob(self.prefix + "*.md"), key=self._key)

    def snap_dt(self, p: Path) -> datetime:
        return datetime.strptime(self._key(p)[0], SNAP_TIME).replace(
            tzinfo=timezone.utc)

    def checkpoint(self, text: str, now: datetime, explicit: bool) -> Path | None:
        self.snaps_dir.mkdir(parents=True, exist_ok=True)
        snaps = self.snaps()
        last = snaps[-1] if snaps else None
        if last is not None and last.read_text(encoding="utf-8") == text:
            return None                           # nothing moved
        if (last is not None and not explicit
                and (now - self.snap_dt(last)).total_seconds() < COALESCE_SECONDS):
            last.unlink()                         # roll this burst into one
        stamp = now.strftime(SNAP_TIME)
        path = self.snaps_dir / (self.prefix + stamp + ".md")
        n = 2
        while path.exists():                      # two saves in one second
            path = self.snaps_dir / ("%s%s-%d.md" % (self.prefix, stamp, n))
            n += 1
        # explicit newline: on Windows text-mode writing turns every LF back
        # into CRLF, which would make a checkpoint differ from the guideline it
        # is a checkpoint OF in every line (BUG-2026-09-20-D)
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        return path

    def prunable(self) -> list[Path]:
        """Checkpoints the coalescing rule would never have written.

        Applied to the past this collapses a typing burst to its result. It only
        ever drops a checkpoint whose successor is that close behind it, so the
        newest of any run — and every isolated save — survives.
        """
        snaps = self.snaps()
        doomed = []
        for a, b in zip(snaps, snaps[1:]):
            gap = (self.snap_dt(b) - self.snap_dt(a)).total_seconds()
            if gap < COALESCE_SECONDS or (a.read_text(encoding="utf-8")
                                          == b.read_text(encoding="utf-8")):
                doomed.append(a)
        return doomed

    def prune(self) -> list[str]:
        doomed = self.prunable()
        for p in doomed:
            p.unlink()
        return [p.name for p in doomed]

    # ----------------------------------------------------------- the history
    def _base_of(self, snaps: list, i: int) -> tuple:
        """What a checkpoint is a change *to*: the one before it, else the seed.

        Not "the distance to the current draft" — that answers how old a
        checkpoint is, which is not what a history of an edit is for.
        """
        if i > 0:
            return snaps[i - 1].name, snaps[i - 1].read_text(encoding="utf-8")
        if self.seed.exists():
            return self.seed.name, self.seed.read_text(encoding="utf-8")
        return "(empty)", ""

    def history(self, current: str | None) -> dict:
        snaps = self.snaps()
        items = []
        for i, p in enumerate(snaps):
            text = p.read_text(encoding="utf-8")
            base_name, base = self._base_of(snaps, i)
            add, rem = _linecount(base, text)
            items.append({"kind": "draft", "name": p.name,
                          "at": self.snap_dt(p).strftime(STAMP),
                          "chars": len(text), "words": len(text.split()),
                          "base": base_name, "add": add, "rem": rem,
                          "d_chars": len(text) - len(base),
                          "notes": len(self.notes_for(p.name)),
                          "current": text == current})
        entries = self.read_log().get("entries", [])
        for e in entries:
            items.append({"kind": "note", "at": e.get("at", ""),
                          "op": e.get("op", ""), "rule": e.get("rule", ""),
                          "cites": len(e.get("cites") or []),
                          "shots": len(e.get("shots") or [])})
        items.sort(key=lambda i: i["at"])
        return {"items": items, "coalesce": COALESCE_SECONDS,
                "drafts": len(snaps), "notes": len(entries),
                "prunable": len(self.prunable())}

    def detail(self, name: str, current: str) -> dict | None:
        snaps = self.snaps()
        names = [p.name for p in snaps]
        if name not in names:                     # also blocks ../ traversal
            return None
        i = names.index(name)
        text = snaps[i].read_text(encoding="utf-8")
        base_name, base = self._base_of(snaps, i)
        return {"name": name, "text": text, "base": base_name,
                "notes": self.notes_for(name),
                "diff": _diff(base, text, base_name, name)
                        or "(this save changed nothing)",
                "diff_current": _diff(text, current, name, "current draft")
                                or "(identical to the current draft)"}

    def notes_for(self, snap_name: str) -> list:
        """The change-log entries attached to one checkpoint.

        A checkpoint's diff says WHAT changed; only the log says why. Opening a
        save and seeing the lines without the reason is the half of the record
        that cannot be reconstructed later — the lines are still on disk, the
        reasoning is not. Two ways an entry attaches: it was filed with that
        save (`after`), or it was written afterwards to explain it
        (`documents`), which is how a late reason stays visible as late.
        """
        out = []
        for e in self.entries():
            if e.get("after") == snap_name or e.get("documents") == snap_name:
                out.append({"id": e.get("id", ""), "at": e.get("at", ""),
                            "op": e.get("op", ""), "rule": e.get("rule", ""),
                            "reason": e.get("reason", ""),
                            "add": e.get("add"), "rem": e.get("rem"),
                            "cites": e.get("cites") or [],
                            "shots": e.get("shots") or [],
                            "written_late": bool(e.get("written_late"))})
        return out

    # ------------------------------------------------------------- the shots
    #
    # A reason often points at something the interface *shows* and a citation
    # cannot carry: three annotators' spans side by side, the shape of a group,
    # a colour pattern across a whole view. A month later the view is gone —
    # re-running Phase-2 under a changed threshold does not reproduce the pixels
    # the decision was made on. So the picture is copied into the record next to
    # the reason, not linked to a view that will have moved on.
    #
    # Same rule as the rest of this module: both arms get the mechanism, the
    # caller decides what may go in it. The control console may attach a shot of
    # raw baseline output; it must not attach one of a conflict view, and that
    # is enforced by what its routes serve, not here.

    def shot_path(self, name: str) -> Path | None:
        """The stored file for `name`, or None. Membership in the listing is the
        check — it settles `../` and absolute paths without parsing them."""
        if name in {p.name for p in self.shots()}:
            return self.shots_dir / name
        return None

    def shots(self) -> list[Path]:
        if not self.shots_dir.exists():
            return []
        return sorted(self.shots_dir.glob(SHOT_PREFIX + "*.*"))

    def add_shot(self, data: bytes, now: datetime | None = None,
                 caption: str = "") -> dict:
        """Store one screenshot and return its record. Raises ValueError if the
        bytes are not an image we recognise, or if there are too many of them."""
        now = now or datetime.now(timezone.utc)
        if not data:
            raise ValueError("empty upload")
        if len(data) > SHOT_MAX_BYTES:
            raise ValueError("a screenshot may be at most %d MB, this is %.1f MB"
                             % (SHOT_MAX_BYTES // (1024 * 1024),
                                len(data) / (1024 * 1024)))
        kind = sniff_shot(data)
        if kind is None:
            raise ValueError("not a PNG, JPEG, GIF or WebP image "
                             "(the bytes decide, not the filename)")
        ext, mime = kind
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        stamp = now.strftime(SNAP_TIME)
        path = self.shots_dir / ("%s%s.%s" % (SHOT_PREFIX, stamp, ext))
        n = 2
        while path.exists():                      # two pastes in one second
            path = self.shots_dir / ("%s%s-%d.%s" % (SHOT_PREFIX, stamp, n, ext))
            n += 1
        path.write_bytes(data)
        return {"file": path.name, "caption": caption.strip(), "mime": mime,
                "bytes": len(data), "at": now.strftime(STAMP)}

    def _shot_refs(self, shots) -> list:
        """Keep only shots that are actually on disk.

        An entry naming a file that was never stored would render as a broken
        image in the record the thesis reports from, which is worse than an
        entry with no picture at all.
        """
        have = {p.name for p in self.shots()}
        out = []
        for s in shots or []:
            if isinstance(s, str):
                s = {"file": s}
            if not isinstance(s, dict):
                continue
            name = str(s.get("file") or "")
            if name in have:
                out.append({"file": name,
                            "caption": str(s.get("caption") or "").strip()})
        return out

    # -------------------------------------------------------------- the log
    def read_log(self) -> dict:
        if self.log_path.exists():
            return json.loads(self.log_path.read_text(encoding="utf-8"))
        return {"artifact": self.artifact, "of": self.draft.name, "entries": []}

    def _write_log(self, payload: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")

    def _entry_base(self, entries: list) -> tuple:
        """(name, text) the next entry's diff is measured against: the checkpoint
        the previous entry was logged at, else the seed.

        Not the checkpoint immediately before the current one — several saves can
        happen between two logged reasons, and what the reader needs is *what
        this logged change did*, not what the last autosave did.
        """
        for e in reversed(entries):
            name = e.get("after")
            if name:
                path = self.snaps_dir / name
                if path.exists():
                    return name, path.read_text(encoding="utf-8")
        if self.seed.exists():
            return self.seed.name, self.seed.read_text(encoding="utf-8")
        return "(empty)", ""

    def add_entry(self, entry: dict, now: datetime | None = None,
                  text: str | None = None) -> dict:
        """Append one change note. Raises ValueError on the two things that make
        an entry worthless: an unknown op, and no reason.

        `text` is the draft as it stands. Given it, the entry also carries the
        change itself: a checkpoint is written, and the unified diff against the
        previous entry's checkpoint is stored on the record. A reason without
        the lines it explains is an assertion; with them it is evidence, and the
        write-up reports this file.
        """
        now = now or datetime.now(timezone.utc)
        log = self.read_log()
        entries = log.setdefault("entries", [])
        op = entry.get("op")
        if op not in OPS:
            raise ValueError("op must be one of %s" % (OPS,))
        reason = (entry.get("reason") or "").strip()
        if not reason:
            raise ValueError("a reason is required")
        cites = []
        for c in entry.get("cites") or []:
            if isinstance(c, dict):
                cites.append({k: str(c.get(k) or "") for k in CITE_FIELDS
                              if c.get(k) not in (None, "")})
        record = {"id": "e%03d-%s" % (len(entries) + 1, now.strftime("%H%M%S")),
                  "at": now.strftime(STAMP), "op": op,
                  "rule": (entry.get("rule") or "").strip(),
                  "reason": reason, "cites": cites,
                  "shots": self._shot_refs(entry.get("shots"))}
        # A reason written after the edit is still worth having, but `at` must
        # stay the time it was *written*: back-dating it would make the record
        # claim a discipline the session did not have. `documents` says which
        # checkpoint it explains, so the gap is visible instead of hidden.
        if entry.get("documents"):
            record["documents"] = str(entry["documents"])
            record["written_late"] = True
        if text is not None:
            base_name, base = self._entry_base(entries)
            snap = self.checkpoint(text, now, explicit=True)
            add, rem = _linecount(base, text)
            record["base"] = base_name
            record["after"] = snap.name if snap is not None else (
                self.snaps()[-1].name if self.snaps() else "")
            record["add"] = add
            record["rem"] = rem
            record["diff"] = _diff(base, text, base_name, record["after"])
            record["unchanged"] = (add == 0 and rem == 0)
        entries.append(record)
        self._write_log(log)
        return record

    def remove_entry(self, entry_id: str) -> bool:
        log = self.read_log()
        entries = log.get("entries", [])
        kept = [e for e in entries if e.get("id") != str(entry_id)]
        if len(kept) == len(entries):
            return False
        log["entries"] = kept
        self._write_log(log)
        return True

    def entries(self) -> list:
        return self.read_log().get("entries", [])

    def digest(self) -> dict:
        """What a freeze records about the log: enough to detect a later edit."""
        import hashlib
        text = json.dumps(self.entries(), sort_keys=True, ensure_ascii=False)
        return {"path": self.log_path.name, "entries": len(self.entries()),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def sniff_shot(data: bytes):
    """(extension, media type) read off the bytes, or None if this is not an
    image format we serve."""
    for magic, ext, mime in SHOT_MAGIC:
        if data.startswith(magic):
            return ext, mime
    # WebP is the one that needs two windows: "RIFF" <4-byte size> "WEBP".
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    return None


def normalize_newlines(text: str) -> str:
    """LF, always.

    A browser textarea hands back CRLF, so a guideline saved through either
    console would differ from `extract_v1.md` in every line while saying the
    same thing. That is an unintended variation in the prompt (invariant 3:
    only $g_v$ varies) and it makes the $v1 \\to v2$ diff — the artefact that
    *is* the intervention — unreadable.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _diff(old: str, new: str, from_name: str, to_name: str) -> str:
    return "\n".join(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=from_name, tofile=to_name, lineterm="", n=2))


def _linecount(old: str, new: str) -> tuple:
    add = rem = 0
    for line in difflib.ndiff(old.splitlines(), new.splitlines()):
        add += line.startswith("+ ")
        rem += line.startswith("- ")
    return add, rem
