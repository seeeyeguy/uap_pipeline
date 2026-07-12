#!/usr/bin/env python3
"""
Chunk-level text quality gate, shared by rebuild_chunks.py, main.py's chunk
path, and purge_junk_chunks.py.

Catches what the document-level text_quality_score() misses: that score
counts bare numbers as plausible tokens, so a page of OCR'd table noise like
"16 16 16 16 ..." scores 1.0 and sails into the index, where it sits close
to any short query in embedding space. Rules here are deliberately
conservative — flag only degenerate repetition and alphabet-free token
shred, not merely low-quality prose.
"""
import re

WORD = re.compile(r"[A-Za-z]{3,}")


def chunk_junk_reason(text: str) -> str | None:
    """None if the chunk looks fine, else a short reason tag."""
    tokens = text.split()
    n = len(tokens)
    if n < 8 and len(text.strip()) < 40:
        return "too_short"               # page-number / header fragments
    if n < 30:
        return None                      # too short to judge repetition; keep
    uniq = len(set(tokens)) / n
    if uniq < 0.12:
        return "repetition"              # "16 16 16 ...", "- - - -", etc.
    words = sum(1 for t in tokens if WORD.search(t))
    if n >= 50 and words / n < 0.15:
        return "no_words"                # digit/symbol shred, no real language
    return None
