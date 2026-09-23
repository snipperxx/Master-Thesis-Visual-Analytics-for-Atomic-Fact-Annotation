"""Split the console's bundle export into the per-document files the importer wants.

`data/gold/gemini_console.html` can hand back either twelve downloads or one
bundle `{doc_id: reply}` --- the bundle is the path that works where a page is
not allowed to start a download. This turns the bundle back into
`data/gold/external_replies/<doc_id>.json`.

Nothing is validated here; that is `import_external_adjudication.py`'s job and it
does it against the gold. This only unpacks.

    python -m scripts.split_external_replies data/gold/gemini_replies_bundle.json
    python -m scripts.import_external_adjudication data/gold/external_replies \\
        --model gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.gold_store import GOLD, GOLD_FACTS  # noqa: E402

OUT_DEFAULT = GOLD / "external_replies"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path, help="the JSON exported by the console")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    payload = json.loads(args.bundle.read_text(encoding="utf-8"))
    if "replies" in payload and isinstance(payload["replies"], dict):
        payload = payload["replies"]          # a progress file, raw text per doc

    known = {p.stem for p in GOLD_FACTS.glob("*.json")}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise SystemExit(f"not gold documents: {unknown}")

    args.out.mkdir(parents=True, exist_ok=True)
    for doc_id, reply in sorted(payload.items()):
        text = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False, indent=2)
        (args.out / f"{doc_id}.json").write_text(text + "\n", encoding="utf-8")
        print(f"  + {args.out / f'{doc_id}.json'}")

    missing = sorted(known - set(payload))
    if missing:
        print(f"\n{len(missing)} document(s) not in the bundle: {', '.join(missing)}")
    print(f"\nwrote {len(payload)} file(s) to {args.out}")
    print("next:")
    print(f"    python -m scripts.import_external_adjudication {args.out} --model gemini-2.5-pro")


if __name__ == "__main__":
    main()
