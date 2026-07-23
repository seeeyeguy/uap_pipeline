#!/usr/bin/env python3
"""
Entity alias resolution — deterministic passes (1-3).

Builds corpus.entity_aliases, mapping raw extracted entity names to canonical
forms, so "Dr. J. Allen Hynek", "Dr Hynek", "J.A. Hynek" and "Hynek" all
resolve to one entity. The raw corpus.entities rows are never modified — the
API joins through the alias table (COALESCE(canonical, name)), so this script
is a post-pass that can re-run after every search_upgrade.sql rebuild.

Passes (all deterministic; the LLM adjudication pass is deliberately NOT
implemented yet — under-merging is the correct failure mode for an archive):

  1. Normalization: case, punctuation, honorific prefixes (Dr/Maj/Col/...),
     generation+corporate suffixes (Jr/Sr/Inc/Ltd), whitespace. Names sharing
     a normalized key merge outright. "A (B)" parentheticals mine an explicit
     alias pair for organizations (MUFON (Mutual UFO Network)).
  2. Structural person merging within a surname block: two names merge when
     their given-name token sequences are initial-compatible ("J. A." vs
     "J. Allen": each token equal, or one is the initial of the other) and no
     full given names conflict (Robert never merges with Richard).
  3. Bare-surname adoption: a single-token person name ("Hynek") becomes an
     alias of a canonical iff exactly ONE canonical person in the corpus
     carries that surname.

Hard rules: never merge across etype; organizations never merge by token
subset (Indiana MUFON / MUFON ERT stay distinct from MUFON); incidents only
merge on normalized-key equality.

Canonical selection: the variant with the most given-name expansion, ties
broken by document count, then alphabetically (stable across runs).

Usage:
  PG_DSN=postgresql://... python build_entity_aliases.py [--apply] [--report N]

Without --apply it prints the merge report and exits (dry run).
"""
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import psycopg

# Placeholder "names" from redaction-heavy documents — never merge these;
# they are not entities. (They stay in the graph as-is.)
PLACEHOLDER_RE = re.compile(
    r"redact|unknown|unnamed|illegible|witness|anonymous|"
    r"not (specified|stated|given)|no name|\[", re.I)

ROOT = Path(__file__).parent

HONORIFICS = {
    "dr", "doctor", "mr", "mrs", "ms", "prof", "professor", "sir", "rev",
    "fr", "maj", "major", "col", "colonel", "capt", "captain", "cpt",
    "lt", "lieut", "lieutenant", "sgt", "sergeant", "cpl", "corporal",
    "gen", "general", "adm", "admiral", "cmdr", "commander", "cdr",
    "pvt", "private", "ens", "ensign", "brig",
}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md", "esq",
            "inc", "llc", "ltd", "corp", "co"}

# Curated nickname / spelling-variant groups for given names. Tokens in the
# same group compare as equal. Deliberately small: only unambiguous,
# extremely common English pairs (plus Josef/Joseph, endemic in this corpus).
NICKNAME_GROUPS = [
    {"robert", "bob", "bobby", "rob"}, {"william", "bill", "billy", "will"},
    {"james", "jim", "jimmy"}, {"john", "jack", "johnny"},
    {"joseph", "josef", "joe"}, {"richard", "dick", "rick"},
    {"edward", "ed", "ted", "eddie"}, {"thomas", "tom", "tommy"},
    {"henry", "hank"}, {"donald", "don"}, {"david", "dave"},
    {"michael", "mike"}, {"steven", "stephen", "steve"},
    {"charles", "charlie", "chuck"}, {"kenneth", "ken"},
    {"ronald", "ron"}, {"lawrence", "larry"}, {"daniel", "dan", "danny"},
    {"samuel", "sam"}, {"benjamin", "ben"}, {"frederick", "fred"},
    {"albert", "al"}, {"walter", "walt"}, {"raymond", "ray"},
    {"margaret", "peggy", "meg"}, {"elizabeth", "beth", "liz", "betty"},
    {"katherine", "kate", "kathy", "catherine"}, {"patricia", "pat"},
]
NICKNAME_OF = {}
for _g in NICKNAME_GROUPS:
    _root = frozenset(_g)
    for _n in _g:
        NICKNAME_OF[_n] = _root

# Phonetically-spelled initials ("Dr. Jay Allen Hynek" for J. Allen Hynek).
PHONETIC_INITIAL = {"jay": "j", "kay": "k", "dee": "d", "bee": "b", "em": "m"}

PAREN_RE = re.compile(r"^(.*?)\s*\(([^)]{2,})\)\s*$")


def dsn():
    if os.environ.get("PG_DSN"):
        return os.environ["PG_DSN"]
    env = (ROOT.parent / "uap-api/.env").read_text()
    pw = re.search(r"^POSTGRES_PASSWORD=(.*)$", env, re.M).group(1).strip()
    user = re.search(r"^POSTGRES_USER=(.*)$", env, re.M).group(1).strip()
    db = re.search(r"^POSTGRES_DB=(.*)$", env, re.M).group(1).strip()
    return f"postgresql://{user}:{pw}@localhost:5439/{db}"


# Tokens that are titles/branches/noise anywhere in a name, not name parts.
NOISE = {"usaf", "usn", "usa", "usmc", "usarmy", "ret", "retired"}


def norm_tokens(name: str) -> list[str]:
    """Lowercased tokens with punctuation stripped; initials keep one letter.

    Comma segments that are ONLY suffixes/honorifics/branch noise are dropped
    ("Quintanilla, Jr., Major, USAF" -> "quintanilla"); exactly two real
    segments left means "Last, First" and re-orders ("Quintanilla, Hector").
    Honorific and branch tokens are filtered wherever they appear — comma
    flattening and parentheticals put them mid-name.
    """
    s = name.lower()
    drop = SUFFIXES | HONORIFICS | NOISE

    def toks_of(seg: str) -> list[str]:
        return re.sub(r"[^a-z0-9& ]+", " ", seg).split()

    if "," in s:
        segs = [seg for seg in s.split(",") if toks_of(seg)]
        keep = [seg for seg in segs
                if not all(t in drop for t in toks_of(seg))]
        if len(keep) == 2:
            s = keep[1] + " " + keep[0]  # Last, First -> First Last
        else:
            s = " ".join(keep) if keep else s
    toks = [t for t in toks_of(s) if t not in HONORIFICS | NOISE]
    while toks and toks[-1] in SUFFIXES:
        toks = toks[:-1]
    return [t for t in toks if t not in SUFFIXES]


def token_equiv(x: str, y: str) -> bool:
    """Token equality up to initials, phonetic initials and nicknames."""
    if x == y:
        return True
    x = PHONETIC_INITIAL.get(x, x)
    y = PHONETIC_INITIAL.get(y, y)
    if x == y:
        return True
    if len(x) == 1 and y.startswith(x):
        return True
    if len(y) == 1 and x.startswith(y):
        return True
    return NICKNAME_OF.get(x) is not None and NICKNAME_OF.get(x) is NICKNAME_OF.get(y)


def norm_key(name: str) -> str:
    return " ".join(norm_tokens(name))


def given_compatible(a: list[str], b: list[str]) -> bool:
    """Are two given-name token lists initial-compatible?

    Compared pairwise from the front; a shorter list may be a prefix of the
    longer ("j allen" vs "j"), because dropped middle names are common. Any
    positional pair must be equal or an initial of the other.
    """
    return all(token_equiv(x, y) for x, y in zip(a, b))


def full_names(toks: list[str]) -> list[str]:
    """The multi-letter (non-initial) tokens, in order."""
    return [t for t in toks if len(t) > 1]


def is_subseq(short: list[str], long: list[str]) -> bool:
    it = iter(long)
    return all(s in it for s in short)


def is_subseq_equiv(short: list[str], long: list[str]) -> bool:
    """Subsequence test under STRICT token matching: full names pair only
    with equivalent full names, initials only with equal initials. The
    initial-expands-to-full case belongs to the positional branch — allowing
    it here would let "John" embed into "J. Allen" through the initial."""
    def strict(x, y):
        x = PHONETIC_INITIAL.get(x, x)
        y = PHONETIC_INITIAL.get(y, y)
        if len(x) == 1 or len(y) == 1:
            return x == y
        return token_equiv(x, y)
    i = 0
    for t in long:
        if i < len(short) and strict(short[i], t):
            i += 1
    return i == len(short)


def person_mergeable(g1: list[str], g2: list[str]) -> bool:
    """Merge two given-name sequences only when nothing contradicts.

    Two independent alignments; either sufficing merges:
      * positional: every zipped pair is token_equiv, extra tail tokens are
        dropped middle names ("J. Allen" ~ "Joseph A.", "H. J." ~ "Hector").
      * embedding: ALL tokens of the shorter sequence appear in order in the
        longer ("Allen" ~ "Josef Allen" — goes-by-middle-name).
    Neither branch tolerates a real conflict: "J. Allen"/"John" fails both
    (allen vs john positionally, no embedding), as do "Roberta"/"Robert J"
    and "Robert J."/"Robert D." (j vs d kills position AND embedding).
    """
    if all(token_equiv(x, y) for x, y in zip(g1, g2)):
        # Dropping a FULL middle name from the longer form ("Josef Allen" ->
        # "Josef") is only credible when some zipped pair matches full-on-full
        # — otherwise "J. Allen" would positionally swallow "John" by pairing
        # the initial and discarding the evidence.
        short, long = (g1, g2) if len(g1) <= len(g2) else (g2, g1)
        dropped_full = any(len(t) > 1 for t in long[len(short):])
        anchored = any(len(x) > 1 and len(y) > 1 for x, y in zip(g1, g2))
        if not dropped_full or anchored:
            return True
    short, long = (g1, g2) if len(g1) <= len(g2) else (g2, g1)
    return len(short) < len(long) and is_subseq_equiv(short, long)


ACRO_STOP = {"of", "the", "and", "for", "de", "la", "del", "das", "dos"}


def acronym_of(inner: str, outer: str) -> bool:
    """Does `inner` look like an acronym of the multi-word name `outer`?

    Word-prefix strict: the acronym's letters must consume the outer's
    significant words in order, each used word contributing a PREFIX of
    itself (MUFON = MU-tual U-F-O N-etwork works; ACIC against "Air Force
    Office of Special Investigations" fails at the C). Trailing unused words
    (locations, "Wright-Patterson AFB") are fine. This is what prevents both
    qualifier chaining ("(SAC)" on a bomb wing) and hallucinated glosses
    from fusing sibling agencies.
    """
    inner_l = re.sub(r"[^a-z0-9]", "", inner.lower())
    if not (2 <= len(inner_l) <= 12) or " " in inner.strip():
        return False
    words = [w for w in norm_tokens(outer) if w not in ACRO_STOP]
    if len(words) < 2 or not words[0].startswith(inner_l[0]):
        return False

    # Backtracking prefix match: each consumed word contributes some prefix
    # of itself; greedy would fail MUFON (MUtual would eat UFO's U).
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def match(i: int, k: int, used: int) -> bool:
        if i == len(inner_l):
            return used >= 2
        if k == len(words):
            return False
        w = words[k]
        for j in range(1, min(len(w), len(inner_l) - i) + 1):
            if w[:j] == inner_l[i:i + j] and match(i + j, k + 1, used + 1):
                return True
        return False

    return match(0, 0, 0)


def expansion_score(toks: list[str]) -> tuple:
    """More tokens and longer (non-initial) tokens = more canonical."""
    return (len(toks), sum(len(t) for t in toks if len(t) > 1))


class Clusters:
    """Union-find over raw (name, etype) pairs."""

    def __init__(self):
        self.parent = {}

    def add(self, item):
        self.parent.setdefault(item, item)

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self):
        out = defaultdict(list)
        for item in self.parent:
            out[self.find(item)].append(item)
        return [sorted(v) for v in out.values() if len(v) > 1 or True]


def build(names_by_etype, doc_counts):
    """names_by_etype: {etype: set(raw names)} -> {(etype, alias): canonical}"""
    aliases = {}
    report = []

    for etype, names in names_by_etype.items():
        mergeable = [n for n in names if not PLACEHOLDER_RE.search(n)]
        uf = Clusters()
        for n in mergeable:
            uf.add((etype, n))

        # ── Pass 1: normalized-key equality (all etypes) ──
        # Persons whose key is a single token ("Dr. Smith" -> "smith") are
        # NOT merged here: an equal bare surname is no evidence of identity
        # (Capt Smith / Dr. Smith / Gen. Smith are different people). Those
        # forms wait for pass 3's unambiguous-surname rule.
        by_key = defaultdict(list)
        for n in mergeable:
            k = norm_key(n)
            if k:
                by_key[k].append(n)
        for k, group in by_key.items():
            if etype == "person" and " " not in k:
                continue
            for other in group[1:]:
                uf.union((etype, group[0]), (etype, other))

        # Pass 1b: parenthetical mining — orgs only, and only when the
        # parenthetical is an ACRONYM of the outer name (or vice versa).
        # "MUFON (Mutual UFO Network)" links all three spellings;
        # "22d Bombardment Wing (SAC)" links nothing — (SAC) is a parent-org
        # qualifier, and chaining on it would fuse the whole Air Force.
        if etype == "organization":
            for n in mergeable:
                m = PAREN_RE.match(n)
                if not m:
                    continue
                outer, inner = m.group(1).strip(), m.group(2).strip()
                if not (acronym_of(inner, outer) or acronym_of(outer, inner)):
                    continue
                for part_key in (norm_key(outer), norm_key(inner)):
                    if part_key in by_key:
                        uf.union((etype, n), (etype, by_key[part_key][0]))

        # ── Pass 2: person merging within surname blocks ──
        # Complete-linkage clustering, NOT transitive chaining: a name joins
        # a cluster only if it is mergeable with EVERY current member, and
        # only if exactly ONE cluster accepts it. This stops "Robert Friend"
        # from bridging "Robert J. Friend" and "Robert D. Friend", and an
        # ambiguous "J. Smith" from fusing Joseph, James and John Smith.
        # Pass-1 key-equality groups enter as single units (same key = same
        # name); most-complete names seed clusters first.
        if etype == "person":
            by_surname = defaultdict(dict)  # surname -> {key-root: givens}
            for n in mergeable:
                toks = norm_tokens(n)
                if len(toks) >= 2:
                    root = uf.find((etype, n))
                    by_surname[toks[-1]].setdefault(root, toks[:-1])
            for units in by_surname.values():
                ordered = sorted(units.items(),
                                 key=lambda kv: expansion_score(kv[1]),
                                 reverse=True)
                clusters = []  # list of [(root, givens), ...]
                for root, g in ordered:
                    fits = [c for c in clusters
                            if all(person_mergeable(g, gm) for _, gm in c)]
                    if len(fits) == 1:
                        fits[0].append((root, g))
                    else:  # no fit, or ambiguous between clusters: stay alone
                        clusters.append([(root, g)])
                for c in clusters:
                    for root, _ in c[1:]:
                        uf.union(c[0][0], root)

        # ── Choose canonicals ──
        # Persons: most complete name (full given tokens, then doc count).
        # Orgs: the best-known form — doc count first (MUFON beats
        # "Mutual UFO Network, Inc."), shortest name breaks ties.
        for group in uf.groups():
            if len(group) < 2:
                continue
            if etype == "person":
                # The best-documented form is what users recognize
                # ("J. Allen Hynek", 205 docs, beats "Josef Allen Hynek");
                # completeness breaks ties among the rare forms.
                best = max(group, key=lambda it: (
                    doc_counts.get(it, 0),
                    len(full_names(norm_tokens(it[1]))),
                    expansion_score(norm_tokens(it[1])), it[1]))
            else:
                best = max(group, key=lambda it: (
                    doc_counts.get(it, 0), -len(it[1]), it[1]))
            canonical = best[1]
            for _, raw in group:
                if raw != canonical:
                    aliases[(etype, raw)] = canonical
            report.append((etype, canonical,
                           sorted(r for _, r in group if r != canonical)))

        # ── Pass 3: bare-surname adoption (persons only) ──
        # "Hynek" merges into a canonical when that canonical either is the
        # ONLY holder of the surname, or utterly dominates it (>=90% of the
        # surname's documents and >=10 docs — bare "Hynek" means J. Allen,
        # not his son Paul). "Smith" stays alone: no dominant Smith exists.
        if etype == "person":
            canon_docs = defaultdict(int)
            for n in mergeable:
                if (etype, n) in aliases:
                    canon_docs[aliases[(etype, n)]] += doc_counts.get((etype, n), 0)
                else:
                    canon_docs[n] += doc_counts.get((etype, n), 0)
            canon_by_surname = defaultdict(set)
            for n in mergeable:
                if (etype, n) in aliases:
                    continue  # aliases resolve through their canonical
                toks = norm_tokens(n)
                if len(toks) >= 2:
                    canon_by_surname[toks[-1]].add(n)
            for n in mergeable:
                toks = norm_tokens(n)
                if len(toks) == 1 and (etype, n) not in aliases:
                    owners = canon_by_surname.get(toks[0], set())
                    if not owners:
                        continue
                    if len(owners) == 1:
                        winner = next(iter(owners))
                    else:
                        ranked = sorted(owners, key=lambda o: -canon_docs[o])
                        total = sum(canon_docs[o] for o in owners)
                        if total < 10 or canon_docs[ranked[0]] < 0.9 * total:
                            continue
                        winner = ranked[0]
                    aliases[(etype, n)] = winner
                    report.append((etype, winner, [n]))

    return aliases, report


# ════════════════════════════════════════════════════════════════════
# Pass 4 — LLM adjudication of the ambiguous residue (--llm).
# Deterministic passes leave clusters unmerged whenever evidence is
# ambiguous ("Maj H Quintanilla" vs "Hector Quintanilla"; NICAP vs its
# expansion with no parenthetical). A cheap model judges exactly those
# candidate pairs; verdicts cache on disk so re-runs only pay for new
# names. "different"/"unsure" verdicts are no-ops — under-merging stays
# the failure mode.
# ════════════════════════════════════════════════════════════════════
VERDICT_CACHE = ROOT / "data/entity_alias_verdicts.json"
LLM_MODEL = "claude-haiku-4-5-20251001"
PAIRS_PER_CALL = 25
MAX_PAIRS = 3000

ADJUDICATE_PROMPT = """You are resolving entity aliases for a UAP/UFO document archive \
(1947-present: Project Blue Book, NICAP, MUFON, AARO, international agencies).
For each numbered pair below, decide whether the two names refer to the SAME \
real-world {kind}. Each side lists known spelling variants, how many archive \
documents mention it, and sample source documents.

Answer with ONLY a JSON array, one object per pair:
  {{"id": <n>, "verdict": "same"|"different"|"unsure", "canonical": "<best display name>"}}

Rules:
- "same" ONLY when confident; historical figures in this domain (Blue Book \
officers, ufologists) are strong priors when ranks/eras align.
- Different middle initials, different sub-organizations, or a parent org vs \
its field office => "different".
- When evidence is thin, answer "unsure". Never guess.
- "canonical" must be copied exactly from one of the pair's listed names.

{pairs}"""


def cluster_info(etype, canonical, aliases, doc_counts, samples):
    members = [a for (t, a), c in aliases.items()
               if t == etype and c == canonical]
    docs = doc_counts.get((etype, canonical), 0) + \
        sum(doc_counts.get((etype, m), 0) for m in members)
    files = samples.get((etype, canonical), [])[:3]
    return {"canonical": canonical, "variants": sorted(members)[:4],
            "docs": docs, "samples": files}


def edit1(a: str, b: str) -> bool:
    """Levenshtein distance exactly 1, for full-name tokens of length >= 4
    (OCR corruptions: Hector/Nectar/Lector, Guintanilla/Quintanilla)."""
    if len(a) < 4 or len(b) < 4 or a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(long)):
        if short == long[:i] + long[i + 1:]:
            return True
    return False


def strict_initials(inner: str, outer: str) -> bool:
    """Exact initialism: every significant word of `outer` contributes exactly
    its first letter, in order. Cross-canonical pairing has no document
    asserting the link, so the loose prefix matcher (fine for parentheticals)
    would drown the judge in "AFIC ~ Army Finner Camp" noise."""
    inner_l = re.sub(r"[^a-z0-9]", "", inner.lower())
    words = [w for w in norm_tokens(outer) if w not in ACRO_STOP]
    return len(words) >= 2 and inner_l == "".join(w[0] for w in words)


def gen_candidates(aliases, names_by_etype, doc_counts):
    """Pairs of post-pass canonical clusters worth an LLM look."""
    person_pairs, org_pairs = [], []
    aliased = set(aliases)
    for etype, names in names_by_etype.items():
        canons = [n for n in names
                  if (etype, n) not in aliased and not PLACEHOLDER_RE.search(n)]
        if etype == "person":
            by_surname = defaultdict(list)
            for n in canons:
                toks = norm_tokens(n)
                if len(toks) >= 2:
                    by_surname[toks[-1]].append((n, toks[:-1]))
            for entries in by_surname.values():
                if len(entries) < 2:
                    continue
                block = []
                for i, (n1, g1) in enumerate(entries):
                    for n2, g2 in entries[i + 1:]:
                        # pairs the deterministic passes found plausible but
                        # unprovable: positionally compatible givens, or an
                        # OCR-corruption-distance first name (Nectar/Lector/
                        # Rector/Sector all ~ Hector).
                        if given_compatible(g1, g2) or (
                                g1 and g2 and edit1(g1[0], g2[0])):
                            block.append((etype, n1, n2))
                # Cap per block: a genuinely-ambiguous surname (Smith) yields
                # few plausible pairs anyway; runaway blocks get truncated
                # rather than dropped wholesale.
                person_pairs.extend(block[:60])
        elif etype == "organization":
            singles = [n for n in canons if " " not in norm_key(n)]
            multis = [n for n in canons if " " in norm_key(n)]
            for s in singles:
                hits = [m for m in multis if strict_initials(s, m)]
                if 0 < len(hits) <= 5:  # generic initialisms match everything
                    org_pairs.extend((etype, s, m) for m in hits)
    # Persons first: they are the higher-value merges if the cap bites.
    return (person_pairs + org_pairs)[:MAX_PAIRS]


def pair_key(etype, a, b):
    import hashlib
    lo, hi = sorted((a, b))
    return hashlib.sha256(f"{etype}|{lo}|{hi}".encode()).hexdigest()[:24]


def adjudicate(candidates, aliases, doc_counts, samples):
    import anthropic
    cache = {}
    if VERDICT_CACHE.exists():
        cache = json.loads(VERDICT_CACHE.read_text())
    todo = [c for c in candidates if pair_key(*c) not in cache]
    print(f"pass 4: {len(candidates)} candidate pairs "
          f"({len(candidates) - len(todo)} cached, {len(todo)} to adjudicate)")
    if todo:
        client = anthropic.Anthropic()
        for start in range(0, len(todo), PAIRS_PER_CALL):
            batch = todo[start:start + PAIRS_PER_CALL]
            lines = []
            for i, (etype, a, b) in enumerate(batch):
                ia = cluster_info(etype, a, aliases, doc_counts, samples)
                ib = cluster_info(etype, b, aliases, doc_counts, samples)
                lines.append(f"Pair {i}:\n  A: {json.dumps(ia)}\n  B: {json.dumps(ib)}")
            kind = batch[0][0]
            msg = client.messages.create(
                model=LLM_MODEL, max_tokens=4000,
                messages=[{"role": "user", "content": ADJUDICATE_PROMPT.format(
                    kind=kind, pairs="\n".join(lines))}])
            text = msg.content[0].text.strip()
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```")
            try:
                verdicts = json.loads(text.strip())
            except json.JSONDecodeError:
                print(f"  batch at {start}: unparseable response, skipping")
                continue
            for v in verdicts:
                try:
                    etype, a, b = batch[int(v["id"])]
                except (KeyError, ValueError, IndexError):
                    continue
                cache[pair_key(etype, a, b)] = {
                    "etype": etype, "a": a, "b": b,
                    "verdict": v.get("verdict", "unsure"),
                    "canonical": v.get("canonical", ""),
                }
            print(f"  adjudicated {min(start + PAIRS_PER_CALL, len(todo))}/{len(todo)}")
            VERDICT_CACHE.write_text(json.dumps(cache, indent=1))
    return cache


def initials_only(name):
    toks = norm_tokens(name)
    return not full_names(toks[:-1]) if len(toks) >= 2 else False


def apply_verdicts(aliases, candidates, cache, doc_counts):
    """Merge 'same' verdicts. Full-name pairs are hard edges (the judge saw
    both real names); an initials-only name attaches to exactly ONE partner
    cluster — "E. E. Brown" judged same as both Earl E. and Eugene E. Brown
    must not transitively fuse Earl with Eugene."""
    uf = Clusters()
    chosen = {}  # preferred canonical votes
    deferred = defaultdict(set)  # initials-only name -> same-partners
    merged = 0
    for etype, a, b in candidates:
        v = cache.get(pair_key(etype, a, b))
        if not v or v["verdict"] != "same":
            continue
        merged += 1
        if v.get("canonical") in (a, b):
            chosen[(etype, v["canonical"])] = True
        ia, ib = initials_only(a), initials_only(b)
        if etype == "person" and ia != ib:
            init, full = (a, b) if ia else (b, a)
            deferred[(etype, init)].add((etype, full))
            continue
        uf.union((etype, a), (etype, b))
    for init, partners in deferred.items():
        uf.add(init)
        for p in partners:
            uf.add(p)
        roots = {uf.find(p) for p in partners}
        if len(roots) == 1:
            uf.union(init, next(iter(roots)))
        # multiple distinct partner clusters: ambiguous, leave unmerged
    # Canonical choice weighs the CLUSTER (own docs + every name already
    # aliased to it) — raw-spelling counts made the OCR corruption "Rector
    # Quintanilla" outrank Hector, whose documents live under 29 variants.
    cluster_docs = defaultdict(int)
    for key, d in doc_counts.items():
        cluster_docs[(key[0], aliases.get(key, key[1]))] += d
    for group in uf.groups():
        if len(group) < 2:
            continue
        # Weight first, LLM canonical vote only as tie-break: a vote on an
        # OCR-variant pair must never outrank a 700-doc cluster.
        best = max(group, key=lambda it: (
            cluster_docs.get(it, 0), 1 if chosen.get(it) else 0, it[1]))
        etype, canonical = best
        for t, name in group:
            if name == canonical:
                continue
            aliases[(t, name)] = canonical
            # re-point this cluster's existing aliases at the new canonical
            for key, c in list(aliases.items()):
                if key[0] == t and c == name and key[1] != canonical:
                    aliases[key] = canonical
    # drop any accidental self-mappings
    for key in [k for k, c in aliases.items() if k[1] == c]:
        del aliases[key]
    print(f"pass 4: {merged} pairs judged same; aliases now {len(aliases)}")
    return aliases


def adopt_bare_surnames(aliases, names_by_etype, doc_counts):
    """Re-run pass 3 after LLM merges: clusters that became dominant can now
    adopt the bare/title-only surname forms ("Maj Quintanilla")."""
    etype = "person"
    canon_docs = defaultdict(int)
    for n in names_by_etype.get(etype, ()):
        if PLACEHOLDER_RE.search(n):
            continue
        target = aliases.get((etype, n), n)
        canon_docs[target] += doc_counts.get((etype, n), 0)
    canon_by_surname = defaultdict(set)
    for n in names_by_etype.get(etype, ()):
        if (etype, n) in aliases or PLACEHOLDER_RE.search(n):
            continue
        toks = norm_tokens(n)
        if len(toks) >= 2:
            canon_by_surname[toks[-1]].add(n)
    adopted = 0
    for n in names_by_etype.get(etype, ()):
        if (etype, n) in aliases or PLACEHOLDER_RE.search(n):
            continue
        toks = norm_tokens(n)
        if len(toks) != 1:
            continue
        owners = canon_by_surname.get(toks[0], set())
        if not owners:
            continue
        if len(owners) == 1:
            winner = next(iter(owners))
        else:
            ranked = sorted(owners, key=lambda o: -canon_docs[o])
            total = sum(canon_docs[o] for o in owners)
            if total < 10 or canon_docs[ranked[0]] < 0.9 * total:
                continue
            winner = ranked[0]
        aliases[(etype, n)] = winner
        adopted += 1
    if adopted:
        print(f"pass 4b: {adopted} bare-surname forms adopted post-merge")
    return aliases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write corpus.entity_aliases (default: dry-run report)")
    ap.add_argument("--report", type=int, default=25,
                    help="show the N largest merge clusters")
    ap.add_argument("--llm", action="store_true",
                    help="pass 4: LLM-adjudicate the ambiguous residue"
                         " (needs ANTHROPIC_API_KEY; verdicts cache on disk)")
    args = ap.parse_args()

    with psycopg.connect(dsn()) as pg:
        rows = pg.execute(
            "SELECT name, etype, count(DISTINCT filename),"
            " (array_agg(DISTINCT filename))[1:3]"
            " FROM corpus.entities GROUP BY 1, 2").fetchall()
    names_by_etype = defaultdict(set)
    doc_counts = {}
    samples = {}
    for name, etype, docs, files in rows:
        names_by_etype[etype].add(name)
        doc_counts[(etype, name)] = docs
        samples[(etype, name)] = files or []

    aliases, report = build(names_by_etype, doc_counts)

    if args.llm:
        candidates = gen_candidates(aliases, names_by_etype, doc_counts)
        cache = adjudicate(candidates, aliases, doc_counts, samples)
        aliases = apply_verdicts(aliases, candidates, cache, doc_counts)
        aliases = adopt_bare_surnames(aliases, names_by_etype, doc_counts)

    total = sum(len(v) for v in names_by_etype.values())
    merged_groups = [r for r in report if r[2]]
    print(f"{total} distinct (name, etype) pairs; "
          f"{len(aliases)} aliased into {len(merged_groups)} canonical entities")
    report.sort(key=lambda r: len(r[2]), reverse=True)
    for etype, canonical, others in report[:args.report]:
        print(f"  [{etype}] {canonical}  <=  {', '.join(others[:8])}"
              + (f" (+{len(others)-8} more)" if len(others) > 8 else ""))

    if not args.apply:
        print("\ndry run — pass --apply to write corpus.entity_aliases")
        return

    with psycopg.connect(dsn()) as pg:
        with pg.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS corpus.entity_aliases (
                    etype          TEXT NOT NULL,
                    alias          TEXT NOT NULL,
                    canonical_name TEXT NOT NULL
                )""")
            cur.execute("TRUNCATE corpus.entity_aliases")
            # Case-variant raw names ("CAPTAIN GREGORY"/"Captain Gregory")
            # are one lookup key; keep a single row per (etype, lower(alias)).
            unique = {}
            for (etype, alias), canonical in sorted(aliases.items()):
                unique.setdefault((etype, alias.lower()), (etype, alias, canonical))
            with cur.copy("COPY corpus.entity_aliases (etype, alias,"
                          " canonical_name) FROM STDIN") as cp:
                for etype, alias, canonical in unique.values():
                    cp.write_row((etype, alias, canonical))
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS entity_aliases_key"
                        " ON corpus.entity_aliases (etype, lower(alias))")
            cur.execute("ANALYZE corpus.entity_aliases")
        pg.commit()
    print(f"applied: {len(aliases)} alias rows written")


if __name__ == "__main__":
    main()
