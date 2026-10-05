"""Breadth-first search over primitive compositions, verified against the training pairs."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .dsl import PRIMITIVES, Grid


def _key(grids: tuple[Grid, ...]) -> bytes:
    return b"|".join(g.tobytes() + bytes(g.shape) for g in grids)


def learn_color_map(preds: list[Grid], targets: list[Grid]) -> Optional[dict[int, int]]:
    """A single cell-wise color mapping that turns every pred into its target, or None."""
    cmap: dict[int, int] = {}
    for p, t in zip(preds, targets):
        if p.shape != t.shape:
            return None
        for a, b in zip(p.ravel().tolist(), t.ravel().tolist()):
            if cmap.setdefault(a, b) != b:
                return None
    return cmap


def apply_color_map(g: Grid, cmap: dict[int, int]) -> Optional[Grid]:
    out = g.copy()
    for a in np.unique(g).tolist():
        if a not in cmap:
            return None  # unseen color at test time
        out[g == a] = cmap[a]
    return out


@dataclass
class Candidate:
    program: list[str]
    test_outputs: list[Grid]
    uses_cmap: bool = False


@dataclass
class SearchResult:
    candidates: list[Candidate] = field(default_factory=list)
    nodes: int = 0
    seconds: float = 0.0


def search(train_in: list[Grid], train_out: list[Grid], test_in: list[Grid],
           max_depth: int = 3, time_limit: float = 20.0, max_candidates: int = 8) -> SearchResult:
    t0 = time.time()
    n_train = len(train_in)
    res = SearchResult()
    seen_pred: set[bytes] = set()

    def check(state: tuple[Grid, ...], prog: list[str]) -> None:
        tr, te = list(state[:n_train]), list(state[n_train:])
        if all(np.array_equal(p, t) for p, t in zip(tr, train_out)):
            outs, cm = te, False
        else:
            cmap = learn_color_map(tr, train_out)
            if cmap is None or all(a == b for a, b in cmap.items()):
                return
            outs = [apply_color_map(g, cmap) for g in te]
            if any(o is None for o in outs):
                return
            cm = True
        k = _key(tuple(outs))
        if k not in seen_pred:
            seen_pred.add(k)
            res.candidates.append(Candidate(prog + (["cmap"] if cm else []), outs, cm))

    start = tuple(train_in) + tuple(test_in)
    frontier = [(start, [])]
    visited = {_key(start)}
    check(start, [])

    for _ in range(max_depth):
        nxt = []
        for state, prog in frontier:
            for prim in PRIMITIVES:
                if time.time() - t0 > time_limit or len(res.candidates) >= max_candidates:
                    res.seconds = time.time() - t0
                    return res
                try:
                    new = tuple(prim(g) for g in state)
                except Exception:
                    continue
                if any(g is None or g.size == 0 for g in new):
                    continue
                k = _key(new)
                if k in visited:
                    continue
                visited.add(k)
                res.nodes += 1
                p2 = prog + [prim.__name__]
                check(new, p2)
                nxt.append((new, p2))
        frontier = nxt
    res.seconds = time.time() - t0
    return res
