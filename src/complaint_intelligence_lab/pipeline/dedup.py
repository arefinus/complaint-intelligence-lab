"""Exact and near-duplicate narrative detection.

Exact duplicates share a SHA-256 of the normalised text. Near duplicates share enough
word shingles (Jaccard similarity at or above a threshold). Both are merged into groups
with union-find so that a group can be kept on one side of a split.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

_NON_WORD_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def normalise(text: str) -> str:
    lowered = text.lower()
    lowered = _NON_WORD_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", lowered).strip()


def exact_hash(text: str) -> str:
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 5) -> frozenset[str]:
    words = normalise(text).split()
    if len(words) < k:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(" ".join(words[i : i + k]) for i in range(len(words) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = int(self.parent[i])
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass(frozen=True)
class DedupResult:
    group_id: np.ndarray  # one integer per row; rows sharing a value are duplicates
    exact_pairs: int
    near_pairs: int

    @property
    def n_groups_with_duplicates(self) -> int:
        _, counts = np.unique(self.group_id, return_counts=True)
        return int((counts > 1).sum())

    def is_duplicate(self) -> np.ndarray:
        _, inverse, counts = np.unique(self.group_id, return_inverse=True, return_counts=True)
        return counts[inverse] > 1


def find_duplicates(texts: list[str], k: int = 5, threshold: float = 0.8) -> DedupResult:
    """Group exact and near-duplicate narratives.

    Empty narratives are never grouped with each other: an empty text carries no
    evidence of duplication. Candidate pairs for the near-duplicate check are found by
    shared shingles, so the comparison is linear in the number of shared shingles rather
    than quadratic in rows.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    n = len(texts)
    uf = _UnionFind(n)
    exact_pairs = 0
    near_pairs = 0

    by_hash: dict[str, int] = {}
    for i, text in enumerate(texts):
        if not normalise(text):
            continue
        h = exact_hash(text)
        if h in by_hash:
            uf.union(by_hash[h], i)
            exact_pairs += 1
        else:
            by_hash[h] = i

    sets = [shingles(t, k=k) if normalise(t) else frozenset() for t in texts]
    postings: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sets):
        for sh in s:
            postings[sh].append(i)
    seen: set[tuple[int, int]] = set()
    for members in postings.values():
        if len(members) < 2 or len(members) > 200:
            continue
        for a_idx in range(len(members)):
            for b_idx in range(a_idx + 1, len(members)):
                a, b = members[a_idx], members[b_idx]
                key = (a, b)
                if key in seen or uf.find(a) == uf.find(b):
                    continue
                seen.add(key)
                if jaccard(sets[a], sets[b]) >= threshold:
                    uf.union(a, b)
                    near_pairs += 1
    group_id = np.array([uf.find(i) for i in range(n)], dtype=int)
    return DedupResult(group_id=group_id, exact_pairs=exact_pairs, near_pairs=near_pairs)
