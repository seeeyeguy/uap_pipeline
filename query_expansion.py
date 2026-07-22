"""
Lexical query expansion for the retriever's tsquery arms.
---------------------------------------------------------
A small, static domain synonym map for UFO/UAP vocabulary. Applied ONLY to
the text handed to websearch_to_tsquery — never to the embedding text, so
the dense arm keeps seeing the user's exact wording.

Design constraints (why variants, not inline rewriting):
  websearch_to_tsquery has no grouping parentheses, so rewriting
  "saucer over lake" into "(saucer OR disc) over lake" would parse as
  saucer | (disc & over & lake) — wrong precedence. Instead we emit
  whole-query VARIANTS, each differing from the original by one synonym
  substitution, and the SQL ORs the resulting tsqueries together:

      websearch_to_tsquery(cfg, 'saucer over lake')
   || websearch_to_tsquery(cfg, 'disc over lake')
   || websearch_to_tsquery(cfg, 'disk over lake')

  Each variant keeps full websearch semantics (phrases, OR, -negation),
  and tsquery || tsquery is a plain OR of the two parse trees.

Conservative expansion rules:
  * exact whole-token (or whole-phrase) matches only, case-insensitive
  * tokens inside double-quoted phrases are never touched
  * negated tokens (-saucer) are never expanded
  * total variants capped (MAX_VARIANTS) to bound tsquery size
"""

from __future__ import annotations

import re

# Each group is a set of interchangeable domain terms. Multi-word entries
# are matched as exact phrases and substituted as quoted phrases so
# websearch_to_tsquery keeps them adjacent.
SYNONYM_GROUPS: list[set[str]] = [
    {"saucer", "disc", "disk"},
    {"craft", "object", "vehicle"},
    {"foo fighter", "foo-fighter", "kraut fireball"},      # 1940s WWII term
    {"airship", "dirigible", "mystery airship"},           # 1890s wave term
    {"cigar", "cylindrical", "cylinder"},
    {"sphere", "orb", "globe"},
    {"triangle", "triangular", "delta"},
    {"entity", "being", "occupant", "humanoid"},
]

# Hard cap on the number of ADDITIONAL query variants (the original query is
# always variant zero). 3 lexical arms x (1 + 8) tsquery parses per request
# is still trivially cheap, but this stops multi-term queries from blowing
# the tsquery up combinatorially.
MAX_VARIANTS = 8

# token -> (group_index, canonical term) for O(1) lookup; phrases kept
# separately, longest first, so "foo fighter" wins before any single token.
_TERM_TO_GROUP: dict[str, int] = {}
_PHRASES: list[tuple[str, int]] = []
for _gi, _group in enumerate(SYNONYM_GROUPS):
    for _term in _group:
        _TERM_TO_GROUP[_term] = _gi
        if " " in _term or "-" in _term:
            _PHRASES.append((_term, _gi))
_PHRASES.sort(key=lambda t: -len(t[0]))

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z-]*")


def _outside_quotes_spans(text: str) -> list[tuple[int, int]]:
    """[start, end) spans of text lying OUTSIDE double-quoted phrases."""
    spans, pos, inside = [], 0, False
    for i, ch in enumerate(text):
        if ch == '"':
            if not inside:
                spans.append((pos, i))
            inside = not inside
            pos = i + 1
    if not inside:
        spans.append((pos, len(text)))
    # unbalanced quote: treat the trailing open segment as quoted (skip it)
    return [(a, b) for a, b in spans if a < b]


def _is_negated(text: str, start: int) -> bool:
    """True when the match at `start` is directly preceded by websearch '-'."""
    return start > 0 and text[start - 1] == "-"


def _find_matches(text: str) -> list[tuple[int, int, int]]:
    """(start, end, group_index) for every expandable exact match, outside
    quotes, unnegated, non-overlapping (phrases claim their span first)."""
    matches: list[tuple[int, int, int]] = []
    claimed: list[tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        return any(a < e and b > s for s, e in claimed)

    lowered = text.lower()
    for a, b in _outside_quotes_spans(text):
        segment = lowered[a:b]
        # phrases first (longest first)
        for phrase, gi in _PHRASES:
            for m in re.finditer(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])",
                                 segment):
                s, e = a + m.start(), a + m.end()
                if not _is_negated(text, s) and not overlaps(s, e):
                    matches.append((s, e, gi))
                    claimed.append((s, e))
        # then single tokens — exact whole-token matches only
        for m in _WORD_RE.finditer(segment):
            tok = m.group(0)
            gi = _TERM_TO_GROUP.get(tok)
            if gi is None:
                continue
            s, e = a + m.start(), a + m.end()
            if not _is_negated(text, s) and not overlaps(s, e):
                matches.append((s, e, gi))
                claimed.append((s, e))
    matches.sort()
    return matches


def _substitute(text: str, start: int, end: int, replacement: str) -> str:
    if " " in replacement:  # keep phrases adjacent under websearch parsing
        replacement = f'"{replacement}"'
    return text[:start] + replacement + text[end:]


def expand_query(text: str, max_variants: int = MAX_VARIANTS) -> list[str]:
    """Return [original, variant, ...] for websearch_to_tsquery.

    Each variant swaps exactly ONE matched term for one of its synonyms,
    so every variant remains a well-formed websearch query. The original
    text is always first; the list is deduplicated and capped at
    1 + max_variants entries. Pure function — safe to unit test.
    """
    out = [text]
    if not text.strip():
        return out
    seen = {text}
    for start, end, gi in _find_matches(text):
        matched = text[start:end].lower()
        for syn in sorted(SYNONYM_GROUPS[gi]):
            if syn == matched:
                continue
            if len(out) - 1 >= max_variants:
                return out
            variant = _substitute(text, start, end, syn)
            if variant not in seen:
                seen.add(variant)
                out.append(variant)
    return out
