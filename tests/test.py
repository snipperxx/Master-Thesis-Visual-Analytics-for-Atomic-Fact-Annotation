#!/usr/bin/env python3
"""
Reproduces the ~50s cutoff we see against the hosted endpoint (OLLAMA_CLOUD_URL).

This is the actual code path our extraction pipeline uses (requests.post to
/api/generate, client timeout 600s), reduced to a single file. It runs four
probes against one model and prints how long each took:

  1. load-only   : {"prompt": "", "keep_alive": "30m"}   -> loads the model, no generation
  2. short prompt: 13-token prompt, stream=False
  3. real prompt : our actual ~16.5 KB extraction prompt, stream=False
  4. real prompt : same, stream=True, timing when the FIRST chunk arrives

Usage:
    pip install requests
    OLLAMA_CLOUD_URL=<url> OLLAMA_CLOUD_API_KEY=<your key> python test.py
    OLLAMA_CLOUD_URL=<url> OLLAMA_CLOUD_API_KEY=<your key> python test.py --model qwen3.6:35b

What we observe (Nemotron-3-Nano-Omni-30B, 2026-09-04, two runs hours apart):

  earlier in the day, model reachable:
    1. load-only    HTTP 200 in 37.0s
    2. short prompt HTTP 200 in 16.0s   (load_duration 1.0s -> model WAS warm)
    3. real prompt  ConnectionError after 50.1s  (RemoteDisconnected)   x2
    4. real prompt  ConnectionError after 50.1s, ZERO chunks received

  later the same day, same script, nothing gets through at all:
    1. load-only    ConnectionError after 50.1s
    2. short prompt ConnectionError after 50.1s
    3. real prompt  ConnectionError after 50.1s
    4. real prompt  ConnectionError after 50.1s, ZERO chunks received

Three things this rules out:

  * Our client timing out. Its timeout is 600s, and a client-side timeout
    raises requests.exceptions.ReadTimeout / curl (28). We get
    RemoteDisconnected / curl (56) — the peer closing the connection — at
    50.1s, reproducibly, to within 0.1s.
  * Model loading being the cause. Probe 3 fails with the model demonstrably
    warm (probe 2 reported load_duration = 1.0s seconds earlier).
  * stream=false being the cause. Probe 4 sets stream=true and receives no
    chunk at all before the cut, so the silent stretch is prompt evaluation,
    before the first token exists. Streaming has nothing to keep alive.

And one more data point: this same model, this same prompt and this same code
completed 36 EUR-Lex documents on 2026-08-31 (7.8h, ~1000 facts). Nothing on
our side changed since.

Our reading is a ~50s read timeout on the proxy in front of Ollama, applied to
time-to-first-byte. Short prompts return inside it, so it is invisible to
anyone testing with a small request — which is why it reproduces for us and
not for you.
"""
import argparse
import json
import os
import sys
import time

import requests

BASE = os.environ.get("OLLAMA_CLOUD_URL", "").rstrip("/")
KEY = os.environ.get("OLLAMA_CLOUD_API_KEY", "").strip()
DEFAULT_MODEL = (
    "hf.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF:UD-Q4_K_M"
)
CLIENT_TIMEOUT = 600  # the same value our pipeline uses

# A prompt of the same size and shape as the real one: a ~13 KB annotation
# guideline + output schema, then one short EUR-Lex clause to annotate.
# The exact wording does not matter for the timing; the token count does.
FILLER_RULE = (
    "R{n}. Every clause of the section must be represented by at least one "
    "fact. A relation stated inside a noun phrase becomes its own triple; "
    "chain them so the object of one triple is repeated verbatim as the "
    "subject of the next. Quantities and dates go where they belong: a "
    "quantity is the object, a date that scopes the whole statement goes in "
    "temporal_context only. Never split inside a named entity, a measurement "
    "or a code list, and never add a participant the text does not state.\n\n"
)
CLAUSE = (
    "(3) The refund on raw sugar must be fixed in respect of the standard "
    "quality; the latter is defined in Article 1 of Council Regulation (EC) "
    "No 431/68 of 9 April 1968 determining the standard quality for raw sugar "
    "and fixing the Community frontier crossing point for calculating cif "
    "prices for sugar(3), as amended by Regulation (EC) No 3290/94(4); "
    "furthermore, this refund should be fixed in accordance with Article 19(4) "
    "of Regulation (EC) No 2038/1999."
)


def build_real_prompt() -> str:
    head = (
        "# Atomic Fact Extraction\n\n"
        "You are an expert legal-text annotator. You receive ONE section of a "
        "EUR-Lex regulatory document and output a single JSON object listing "
        "the atomic facts it contains. Output STRICT JSON ONLY.\n\n"
        '{"facts": [{"natural_language": "string", "source_quote": "string", '
        '"condition": "string", "temporal_context": "string", '
        '"triples": [{"subject": "s", "predicate": "p", "object": "o"}]}]}\n\n'
        "## Guideline\n\n"
    )
    rules = "".join(FILLER_RULE.format(n=i) for i in range(1, 26))
    tail = f'\n## Input\n\nSection text:\n"""\n{CLAUSE}\n"""\n\nProduce the JSON object.\n'
    return head + rules + tail


def probe(label: str, payload: dict, stream: bool = False) -> None:
    t0 = time.time()
    first_chunk = None
    try:
        resp = requests.post(
            f"{BASE}/api/generate",
            json=payload,
            headers={"Authorization": f"Bearer {KEY}"},
            timeout=CLIENT_TIMEOUT,
            stream=stream,
        )
        if stream:
            chunks = 0
            with resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if first_chunk is None:
                        first_chunk = time.time() - t0
                    chunks += 1
            print(
                f"{label:26s} HTTP {resp.status_code} in {time.time() - t0:5.1f}s"
                f"  chunks={chunks} first_chunk={first_chunk}"
            )
            return
        body = resp.json()
        print(
            f"{label:26s} HTTP {resp.status_code} in {time.time() - t0:5.1f}s"
            f"  load={body.get('load_duration', 0) / 1e9:.1f}s"
            f"  prompt_eval={body.get('prompt_eval_duration', 0) / 1e9:.1f}s"
            f"  resp_chars={len(body.get('response') or '')}"
        )
    except Exception as exc:
        extra = "" if not stream else f" (chunks received before cut: first_chunk={first_chunk})"
        print(
            f"{label:26s} FAILED after {time.time() - t0:5.1f}s"
            f"  {type(exc).__name__}: {str(exc)[:70]}{extra}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    if not BASE:
        sys.exit("set OLLAMA_CLOUD_URL")
    if not KEY:
        sys.exit("set OLLAMA_CLOUD_API_KEY")

    prompt = build_real_prompt()
    opts = {"temperature": 0.0, "num_ctx": 16384}
    print(f"endpoint     : {BASE}")
    print(f"model        : {args.model}")
    print(f"real prompt  : {len(prompt)} chars")
    print(f"client timeout: {CLIENT_TIMEOUT}s\n")

    probe("1 load-only", {"model": args.model, "prompt": "", "keep_alive": "30m"})
    probe(
        "2 short prompt",
        {"model": args.model, "prompt": "Say OK.", "stream": False,
         "keep_alive": "30m", "options": opts},
    )
    probe(
        "3 real prompt stream=False",
        {"model": args.model, "prompt": prompt, "stream": False, "format": "json",
         "keep_alive": "30m", "think": False, "options": opts},
    )
    probe(
        "4 real prompt stream=True",
        {"model": args.model, "prompt": prompt, "stream": True, "format": "json",
         "keep_alive": "30m", "think": False, "options": opts},
        stream=True,
    )


if __name__ == "__main__":
    main()
