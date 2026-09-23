"""
Load EUR-Lex documents from the HuggingFace `lex_glue` dataset.

Why this exists alongside `eurlex_fetch.py`:
    The HTTP path via eur-lex.europa.eu is blocked by AWS WAF, and
    publications.europa.eu (CELLAR) rejected our requests with HTTP 400.
    Rather than keep trying URL variants, we use the pre-cleaned
    `coastalcph/lex_glue` `eurlex` subset on HuggingFace, which contains
    ~57k English EUR-Lex documents as plain text.

Trade-off:
    + No WAF, no rate limits, fully offline after first download.
    + Larger corpus (57k vs. ~5 hand-picked CELEX numbers).
    - Plain text only — no HTML structure, so no annex tables as
      structured `<table>` nodes. Annex content is interleaved as prose.

The HTML fetcher (`eurlex_fetch.py`) and table linearizer are kept
intact. When annex-table extraction becomes necessary we can fetch a
handful of specific documents via browser export and feed them through
the existing HTML path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# HuggingFace dataset identifier.
# We use the `coastalcph/lex_glue` "eurlex" subset because it's well-
# maintained, English-only, and pre-cleaned. Each item has:
#     - text:   str   (the document body)
#     - labels: list  (EuroVoc descriptor IDs — not needed here)
DATASET_NAME = "coastalcph/lex_glue"
DATASET_CONFIG = "eurlex"


@dataclass
class EurlexDoc:
    """A single EUR-Lex document loaded from the HF dataset."""

    doc_id: str           # synthetic ID: "{split}-{index}"
    text: str
    n_words: int
    source_split: str     # "train" | "validation" | "test"
    source_index: int     # index into the original HF split

    @property
    def celex(self) -> str:
        """
        Compatibility alias. The HF dataset doesn't carry CELEX numbers,
        so we use the synthetic doc_id everywhere downstream code expects
        a CELEX-shaped identifier.
        """
        return self.doc_id


def _word_count(text: str) -> int:
    return len(text.split())


def load_short_docs(
    n: int = 5,
    min_words: int = 500,
    max_words: int = 2500,
    split: str = "train",
    cache_dir: Optional[Path] = None,
    seed_offset: int = 0,
) -> list[EurlexDoc]:
    """
    Return the first `n` documents whose word count is in [min_words, max_words].

    `seed_offset` lets you skip the first K matching documents — useful if
    you want a different sample without restarting the iterator from zero.

    The HF datasets library is imported lazily so that `import src.eurlex_dataset`
    doesn't fail on machines that haven't installed `datasets` yet.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency: `pip install datasets`. "
            "See requirements.txt."
        ) from e

    logger.info("Loading HF dataset %s/%s split=%s", DATASET_NAME, DATASET_CONFIG, split)
    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    selected: list[EurlexDoc] = []
    skipped = 0
    for idx, item in enumerate(ds):
        wc = _word_count(item["text"])
        if not (min_words <= wc <= max_words):
            continue
        if skipped < seed_offset:
            skipped += 1
            continue
        selected.append(
            EurlexDoc(
                doc_id=f"{split}-{idx:06d}",
                text=item["text"],
                n_words=wc,
                source_split=split,
                source_index=idx,
            )
        )
        if len(selected) >= n:
            break

    logger.info("Selected %d documents (%d–%d words)", len(selected), min_words, max_words)
    return selected


def iter_corpus(
    split: str = "train",
    cache_dir: Optional[Path] = None,
) -> Iterator[EurlexDoc]:
    """
    Full-corpus iterator. Phase-1 production code will use this once the
    dry run validates the extraction pipeline.
    """
    from datasets import load_dataset

    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    for idx, item in enumerate(ds):
        yield EurlexDoc(
            doc_id=f"{split}-{idx:06d}",
            text=item["text"],
            n_words=_word_count(item["text"]),
            source_split=split,
            source_index=idx,
        )
