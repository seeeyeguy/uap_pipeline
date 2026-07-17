#!/usr/bin/env python3
"""
MinHash near-duplicate detection over the corpus text files.
------------------------------------------------------------
Scans data/text_final/*.txt, builds 5-word shingles per document, MinHash
signatures (128 permutations), then LSH banding to find candidate pairs,
which are verified by signature similarity (>= 0.85 by default). Connected
components of verified pairs become groups.

Output: data/near_dups.json
    [{"filenames": ["a.txt", "b.txt", ...], "similarity": 0.93}, ...]
  similarity is the mean estimated Jaccard (fraction of equal MinHash
  values) over the verified pairs in the group — an approximation.

Implementation: uses `datasketch` when installed; otherwise a pure
stdlib+numpy MinHash/LSH (this venv has no datasketch, so that is the
path actually exercised). Word hashes come from Python's builtin hash()
(consistent within a run); set PYTHONHASHSEED for run-to-run
reproducibility.

Usage:
    .venv/bin/python scripts/near_dup.py \
        [--dir data/text_final] [--out data/near_dups.json] \
        [--threshold 0.85] [--perms 128] [--shingle 5] [--bands 16]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from datasketch import MinHash, MinHashLSH  # type: ignore
    HAVE_DATASKETCH = True
except ImportError:
    HAVE_DATASKETCH = False

U64 = np.uint64


def shingle_hashes(text: str, k: int) -> np.ndarray:
    """Unique 64-bit hashes of the k-word shingles of `text` (vectorized:
    per-word builtin hashes combined with position-weighted odd constants,
    so shingles are order-sensitive)."""
    words = text.lower().split()
    if len(words) < k:
        if not words:
            return np.empty(0, dtype=U64)
        # degenerate short doc: one shingle of everything
        return np.array([hash(" ".join(words)) & 0xFFFFFFFFFFFFFFFF], dtype=U64)
    wh = np.fromiter((hash(w) & 0xFFFFFFFFFFFFFFFF for w in words),
                     dtype=U64, count=len(words))
    n = len(words) - k + 1
    acc = np.zeros(n, dtype=U64)
    # fixed odd multipliers per position keep the combination order-sensitive
    consts = [U64(c) for c in (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F,
                               0x165667B19E3779F9, 0x27D4EB2F165667C5,
                               0x85EBCA77C2B2AE63, 0xFF51AFD7ED558CCD,
                               0xC4CEB9FE1A85EC53, 0x2545F4914F6CDD1D)]
    for j in range(k):
        acc ^= wh[j:j + n] * consts[j % len(consts)]
    return np.unique(acc)


class NumpyMinHasher:
    """(a*h + b) mod 2**64 with random odd `a` per permutation; the MinHash
    signature is the elementwise min over a document's shingle hashes."""

    def __init__(self, perms: int, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.a = (rng.integers(1, 1 << 63, size=perms, dtype=np.uint64)
                  << U64(1)) | U64(1)          # odd multipliers
        self.b = rng.integers(0, 1 << 63, size=perms, dtype=np.uint64)
        self.perms = perms

    def signature(self, hashes: np.ndarray, batch: int = 100_000) -> np.ndarray:
        if hashes.size == 0:
            return np.full(self.perms, np.iinfo(np.uint64).max, dtype=U64)
        sig = np.full(self.perms, np.iinfo(np.uint64).max, dtype=U64)
        for i in range(0, hashes.size, batch):
            h = hashes[i:i + batch]
            # perms x batch matrix; uint64 wraparound is intentional
            m = self.a[:, None] * h[None, :] + self.b[:, None]
            np.minimum(sig, m.min(axis=1), out=sig)
        return sig


def sig_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Estimated Jaccard = fraction of matching MinHash values."""
    return float(np.count_nonzero(a == b)) / len(a)


class UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def lsh_candidate_pairs(sigs: np.ndarray, bands: int) -> set[tuple[int, int]]:
    """LSH banding: docs sharing any (band, row-hash) bucket are candidates."""
    n_docs, perms = sigs.shape
    assert perms % bands == 0, "perms must be divisible by bands"
    rows = perms // bands
    pairs: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        block = sigs[:, band * rows:(band + 1) * rows]
        for doc, key in enumerate(map(bytes, block)):
            buckets[key].append(doc)
        for members in buckets.values():
            if len(members) > 1:
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        pairs.add((members[i], members[j]))
    return pairs


def run_numpy(files: list[Path], args) -> tuple[list[dict], dict]:
    hasher = NumpyMinHasher(args.perms)
    rng = np.random.default_rng(2)
    sigs = np.empty((len(files), args.perms), dtype=U64)
    empty = 0
    t0 = time.time()
    for i, path in enumerate(files):
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text.split()) < args.min_words:
            # unique random signature: empty/stub docs must never match
            # anything (identical junk-OCR stubs like a bare markdown fence
            # would otherwise dominate the report with meaningless groups)
            empty += 1
            sigs[i] = rng.integers(0, 1 << 63, size=args.perms, dtype=np.uint64)
            continue
        sigs[i] = hasher.signature(shingle_hashes(text, args.shingle))
        if (i + 1) % 1000 == 0:
            print(f"  hashed {i + 1}/{len(files)} docs "
                  f"({time.time() - t0:.0f}s)", flush=True)
    t_sig = time.time() - t0

    t0 = time.time()
    candidates = lsh_candidate_pairs(sigs, args.bands)
    t_lsh = time.time() - t0

    # verify candidates and group
    t0 = time.time()
    uf = UnionFind()
    pair_sims: dict[tuple[int, int], float] = {}
    for i, j in candidates:
        s = sig_similarity(sigs[i], sigs[j])
        if s >= args.threshold:
            uf.union(i, j)
            pair_sims[(i, j)] = s
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in {d for pair in pair_sims for d in pair}:
        groups[uf.find(idx)].append(idx)
    out = []
    for members in groups.values():
        mset = set(members)
        sims = [s for (i, j), s in pair_sims.items() if i in mset and j in mset]
        out.append({
            "filenames": sorted(files[m].name for m in members),
            "similarity": round(sum(sims) / len(sims), 4) if sims else None,
        })
    out.sort(key=lambda g: (-len(g["filenames"]), g["filenames"][0]))
    stats = {"skipped_short_docs": empty, "candidate_pairs": len(candidates),
             "verified_pairs": len(pair_sims),
             "t_signatures": t_sig, "t_lsh": t_lsh,
             "t_verify": time.time() - t0}
    return out, stats


def run_datasketch(files: list[Path], args) -> tuple[list[dict], dict]:
    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.perms)
    mhs: list[MinHash] = []
    t0 = time.time()
    for i, path in enumerate(files):
        m = MinHash(num_perm=args.perms)
        words = path.read_text(encoding="utf-8", errors="replace").lower().split()
        if len(words) >= args.min_words:   # skip junk-OCR stubs
            for s in range(max(1, len(words) - args.shingle + 1)):
                m.update(" ".join(words[s:s + args.shingle]).encode("utf-8"))
            lsh.insert(str(i), m)
        mhs.append(m)
        if (i + 1) % 1000 == 0:
            print(f"  hashed {i + 1}/{len(files)} docs", flush=True)
    t_sig = time.time() - t0
    uf = UnionFind()
    pair_sims: dict[tuple[int, int], float] = {}
    for i, m in enumerate(mhs):
        for other in lsh.query(m):
            j = int(other)
            if j <= i:
                continue
            s = mhs[i].jaccard(mhs[j])
            if s >= args.threshold:
                uf.union(i, j)
                pair_sims[(i, j)] = s
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in {d for pair in pair_sims for d in pair}:
        groups[uf.find(idx)].append(idx)
    out = []
    for members in groups.values():
        mset = set(members)
        sims = [s for (i, j), s in pair_sims.items() if i in mset and j in mset]
        out.append({
            "filenames": sorted(files[m].name for m in members),
            "similarity": round(sum(sims) / len(sims), 4) if sims else None,
        })
    out.sort(key=lambda g: (-len(g["filenames"]), g["filenames"][0]))
    return out, {"verified_pairs": len(pair_sims), "t_signatures": t_sig}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dir", default="data/text_final")
    ap.add_argument("--out", default="data/near_dups.json")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--perms", type=int, default=128)
    ap.add_argument("--shingle", type=int, default=5)
    ap.add_argument("--bands", type=int, default=16,
                    help="LSH bands (perms/bands rows each); 16x8 catches "
                         "pairs well below 0.85, verification filters them")
    ap.add_argument("--min-words", type=int, default=25,
                    help="skip docs shorter than this (junk-OCR stubs)")
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("*.txt"))
    if not files:
        sys.exit(f"no .txt files under {args.dir}")
    print(f"{len(files)} documents | {args.perms} perms | "
          f"{args.shingle}-word shingles | threshold {args.threshold} | "
          f"backend {'datasketch' if HAVE_DATASKETCH else 'stdlib+numpy'}")

    t0 = time.time()
    groups, stats = (run_datasketch if HAVE_DATASKETCH else run_numpy)(files, args)
    total = time.time() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(groups, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    dup_docs = sum(len(g["filenames"]) for g in groups)
    print(f"\nDone in {total:.0f}s ({stats})")
    print(f"{len(groups)} near-duplicate groups covering {dup_docs} documents "
          f"-> {out_path}")
    for g in groups[:10]:
        print(f"  [{len(g['filenames'])} docs, sim~{g['similarity']}] "
              f"{', '.join(g['filenames'][:4])}"
              f"{' …' if len(g['filenames']) > 4 else ''}")


if __name__ == "__main__":
    main()
